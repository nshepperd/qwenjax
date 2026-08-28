"""Context distillation of a corpus into a cartridge.

The objective from arXiv 2506.06266 section 4.2:

    min_Z  sum_(x, c) sum_i  KL( F(. | c ⊕ x[:i])  ||  F_Z(. | x[:i]) )

The teacher is the frozen model with the chunk `c` in its system prompt; the
student is the same model with the cartridge `Z` in place of the system
prompt; `x` is a self-study conversation. Both see the identical suffix token
ids (see `qwen_jax.chat`), so position `i` of the student lines up with
position `|c| + i` of the teacher.

The teacher runs online, inside the training step: one extra forward per
step rather than a logit store on disk. That makes the KL exact over the full
vocabulary, and it is what makes the step self-contained -- the model is an
argument, nothing is closed over, so `jax.jit` captures no weights.

Logits are never materialised for a whole sequence. `lm_head` over 151936
outputs at 1k positions is 600 MB in float32 per sequence; instead the
positions are scanned in blocks, with the block function checkpointed so the
backward pass recomputes its logits rather than keeping them.
"""
from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Bool, Float, Int

from . import chat
from .cartridge import Cartridge
from .selfstudy import Example

# -----------------------------------------------------------------------------
# Batches
# -----------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Batch:
    """Teacher and student views of the same conversations.

    Teacher rows are `[left pad | system prefix | suffix | right pad]` with the
    suffix starting at column `context`; student rows are `[suffix | right
    pad]`. `loss_mask` marks suffix positions that exist.
    """

    teacher_ids: Int[Array, "batch context+seq"]
    teacher_mask: Int[Array, "batch context+seq"]
    student_ids: Int[Array, "batch seq"]
    student_mask: Int[Array, "batch seq"]
    loss_mask: Bool[Array, "batch seq"]

    @property
    def context(self) -> int:
        return self.teacher_ids.shape[1] - self.student_ids.shape[1]


def encode_example(tokenizer, ex: Example, description: str) -> chat.Tokens:
    chunk_text = tokenizer.decode(ex.chunk_ids, skip_special_tokens=False)
    system = f"{description}\n\n{chunk_text}" if description else chunk_text
    # `teacher_note` lands in the system text, which only the teacher reads --
    # the student's view is the suffix alone. So an instruction here is
    # distilled into the cartridge rather than required at inference.
    if ex.teacher_note:
        system = f"{system}\n\n{ex.teacher_note}"
    return chat.encode(tokenizer, system, [("user", ex.user), ("assistant", ex.assistant)])


def make_batch(
    tokenizer,
    examples: list[Example],
    *,
    description: str,
    context: int,
    seq: int,
    pad_id: int,
) -> Batch:
    """Lay `examples` out at fixed shapes `(context + seq)` and `(seq)`.

    A system prefix longer than `context` is truncated from its *front* (the
    header and first lines go, the conversation's subject is more likely to
    be late in the chunk); a suffix longer than `seq` is truncated at the end.
    Fixed shapes are what keep the jitted step from recompiling per batch.
    """
    b = len(examples)
    t_ids = np.full((b, context + seq), pad_id, dtype=np.int32)
    t_mask = np.zeros((b, context + seq), dtype=np.int32)
    s_ids = np.full((b, seq), pad_id, dtype=np.int32)
    s_mask = np.zeros((b, seq), dtype=np.int32)
    for i, ex in enumerate(examples):
        toks = encode_example(tokenizer, ex, description)
        system = toks.system[-context:]
        suffix = toks.suffix[:seq]
        t_ids[i, context - len(system):context] = system
        t_mask[i, context - len(system):context] = 1
        t_ids[i, context:context + len(suffix)] = suffix
        t_mask[i, context:context + len(suffix)] = 1
        s_ids[i, :len(suffix)] = suffix
        s_mask[i, :len(suffix)] = 1
    return Batch(
        teacher_ids=jnp.asarray(t_ids),
        teacher_mask=jnp.asarray(t_mask),
        student_ids=jnp.asarray(s_ids),
        student_mask=jnp.asarray(s_mask),
        loss_mask=jnp.asarray(s_mask.astype(bool)),
    )


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


def blockwise_kl(
    lm_head: Callable[[Array], Array],
    student_hidden: Float[Array, "batch seq hidden"],
    teacher_hidden: Float[Array, "batch seq hidden"],
    mask: Bool[Array, "batch seq"],
    *,
    block: int = 128,
) -> Float[Array, ""]:
    """Mean KL(teacher || student) over masked positions, `block` positions at a time."""
    b, s, h = student_hidden.shape
    if s % block:
        raise ValueError(f"seq {s} must be a multiple of block {block}")
    n = s // block

    def blocks(x):
        return x.reshape(b, n, block, *x.shape[2:]).swapaxes(0, 1)

    @jax.checkpoint
    def one(args):
        hs, ht, m = args
        log_s = jax.nn.log_softmax(lm_head(hs).astype(jnp.float32), axis=-1)
        log_t = jax.nn.log_softmax(lm_head(ht).astype(jnp.float32), axis=-1)
        kl = jnp.sum(jnp.exp(log_t) * (log_t - log_s), axis=-1)
        return jnp.sum(kl * m)

    total = jax.lax.map(one, (blocks(student_hidden), blocks(teacher_hidden), blocks(mask)))
    return jnp.sum(total) / jnp.maximum(jnp.sum(mask), 1)


def distill_loss(model, cartridge: Cartridge, batch: Batch, *, block: int = 128) -> Float[Array, ""]:
    """The objective for one batch. Differentiable in the cartridge only."""
    context = batch.context
    teacher_hidden, _, _ = model.model(
        input_ids=batch.teacher_ids, attention_mask=batch.teacher_mask,
    )
    teacher_hidden = jax.lax.stop_gradient(teacher_hidden[:, context:])
    student_hidden, _, _ = model.model(
        input_ids=batch.student_ids, attention_mask=batch.student_mask,
        prefix=cartridge.prefix(model.cache_dtype()),
    )
    return blockwise_kl(model.get_lm_head(), student_hidden, teacher_hidden,
                        batch.loss_mask, block=block)


# -----------------------------------------------------------------------------
# Training step
# -----------------------------------------------------------------------------


@dataclasses.dataclass
class Trainer:
    """Holds the optimiser and the jitted step.

    The step takes the model as an argument every call. It is a pytree of
    weights; passing it is free (no copy) and keeps it out of the compiled
    executable, which would otherwise bake several GB of constants in.
    """

    optimizer: optax.GradientTransformation
    block: int = 128

    def __post_init__(self):
        # The optimiser is a bundle of functions, not arrays: bind it rather
        # than pass it, so jit sees only the model, cartridge, state and batch.
        self._step = jax.jit(functools.partial(self._step_impl, self.optimizer),
                             static_argnames=("block",))

    def init(self, cartridge: Cartridge) -> Any:
        return self.optimizer.init(cartridge.params())

    @staticmethod
    def _step_impl(optimizer, model, cartridge: Cartridge, opt_state, batch: Batch, *, block: int):
        def loss_fn(params):
            return distill_loss(model, cartridge.with_params(params), batch, block=block)

        loss, grads = jax.value_and_grad(loss_fn)(cartridge.params())
        grads = cartridge.mask_grads(grads)
        updates, opt_state = optimizer.update(grads, opt_state, cartridge.params())
        params = optax.apply_updates(cartridge.params(), updates)
        return cartridge.with_params(params), opt_state, loss

    def step(self, model, cartridge: Cartridge, opt_state, batch: Batch):
        cartridge, opt_state, loss = self._step(model, cartridge, opt_state, batch, block=self.block)
        return cartridge.advance(1), opt_state, loss


def evaluate(model, cartridge: Cartridge, batches: list[Batch], *, block: int = 128) -> float:
    """Mean distillation loss over `batches`, no gradient."""
    f = jax.jit(distill_loss, static_argnames=("block",))
    return float(np.mean([float(f(model, cartridge, b, block=block)) for b in batches]))


__all__ = ["Batch", "Trainer", "blockwise_kl", "distill_loss", "encode_example", "evaluate", "make_batch"]
