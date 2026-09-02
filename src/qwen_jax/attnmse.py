"""Teacher-forced attention-output distillation of a corpus into a cartridge.

A cartridge's only causal pathway into the model is the attention output at
suffix positions: queries attend over `[cartridge KV | suffix KV]` where the
teacher's attend over `[corpus KV | suffix KV]`, and every other computation
is shared. If the attention outputs matched at every layer and position, the
logits would match identically. So instead of the logit KL of
`qwen_jax.distill`, this objective matches the attention outputs directly:

    min_Z  sum_l  || attn_l(h_l; Z) - attn_l(h_l; corpus KV) ||^2

with `h_l` the *teacher's* layer-l residual stream at the suffix positions
(teacher forcing). Feeding the teacher's own layer inputs to the student
makes the queries and suffix KV identical on both sides, so the layers
decouple: layer l's loss depends only on layer l's cartridge slots, and the
backward pass never crosses a layer boundary. No lm_head, no full-depth
backprop -- one attention op per layer each way.

The suffix tokens sit at positions `p + i` for the student versus
`|system| + i` for the teacher, exactly as in `distill_loss`; RoPE makes the
suffix-suffix interactions identical under that shift, and the prefix-query
interaction is the part being trained.

Each layer's error is normalised by the teacher's own output power (a
relative squared error), so layers with small attention outputs are not
drowned out by layers with large ones.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .cartridge import Cartridge
from .distill import Batch


def _rotary(model, position_ids):
    lm = model.model.language_model
    cos, sin = lm.rotary_emb(position_ids)
    dtype = lm.embed_tokens.weight().dtype
    return cos.astype(dtype), sin.astype(dtype)


def teacher_targets(
    model, batch: Batch,
) -> tuple[Float[Array, "layers batch seq hidden"], Float[Array, "layers batch seq hidden"]]:
    """Per-layer residual-stream inputs and attention outputs at suffix positions.

    Replays the text decoder loop of `Qwen3VLTextModel.__call__` (text-only:
    no DeepStack) so the attention block's output can be read off before the
    residual add. Position ids and the KV mask come from the same resolution
    the real forward uses, so the capture is the teacher of `distill_loss`
    to the bit.
    """
    lm = model.model.language_model
    ids, mask = batch.teacher_ids, batch.teacher_mask
    context = batch.context
    position_ids, _, _ = model.model._resolve_position_ids(
        ids, None, mask, None, None, None, None)
    pos_emb = _rotary(model, position_ids)
    kv_mask = mask.astype(jnp.bool)
    h = lm.embed_tokens(ids)
    ins, outs = [], []
    for layer in lm.layers:
        ins.append(h[:, context:])
        attn, _ = layer.self_attn(layer.input_layernorm(h), position_embeddings=pos_emb,
                                  kv_mask=kv_mask)
        outs.append(attn[:, context:])
        h = h + attn
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return jnp.stack(ins), jnp.stack(outs)


def attnmse_layers(
    model, cartridge: Cartridge, batch: Batch, *, eps: float = 1e-6, prefix=None,
) -> Float[Array, "layers"]:
    """Per-layer relative squared error of the student's attention outputs.

    Differentiable in the cartridge only; layer l's entry depends only on
    layer l's slots. `prefix` overrides the cartridge's own KV prefix, for
    graduated-optimization variants that perturb it during training.
    """
    if batch.note_ids is not None:
        raise ValueError("student notes are not supported by the teacher-forced loss: "
                         "the teacher has no hidden states for note positions")
    lm = model.model.language_model
    ins, targets = jax.lax.stop_gradient(teacher_targets(model, batch))
    if prefix is None:
        prefix = cartridge.prefix(model.cache_dtype())
    ids, mask = batch.student_ids, batch.student_mask
    position_ids, _, _ = model.model._resolve_position_ids(
        ids, None, mask, None, None, None, prefix.length)
    pos_emb = _rotary(model, position_ids)
    ones = jnp.ones((ids.shape[0], prefix.length), dtype=jnp.bool)
    kv_mask = jnp.concatenate([ones, mask.astype(jnp.bool)], axis=1)
    m = batch.loss_mask[..., None]
    losses = []
    gate = jnp.zeros((), jnp.float32)
    for i, layer in enumerate(lm.layers):
        # The layers are computationally independent -- that is the point of
        # teacher forcing -- which leaves XLA free to schedule many of them
        # concurrently, and the peak memory of the unrolled loop OOMs a 16 GB
        # card. The barrier chains each layer's inputs to the previous
        # layer's result, forcing one-at-a-time execution; the checkpoint
        # keeps the backward pass from holding every layer's q/k/v at once.
        h, target = jax.lax.optimization_barrier((ins[i], targets[i], gate))[:2]

        @jax.checkpoint
        def one(h, target, pfx, layer=layer):
            out, _ = layer.self_attn(layer.input_layernorm(h), position_embeddings=pos_emb,
                                     kv_mask=kv_mask, prefix=pfx)
            d = jnp.where(m, (out - target).astype(jnp.float32), 0.0)
            t = jnp.where(m, target.astype(jnp.float32), 0.0)
            return jnp.sum(d * d) / (jnp.sum(t * t) + eps)

        losses.append(one(h, target, prefix.layer(i)))
        gate = losses[-1]
    return jnp.stack(losses)


def attnmse_loss(model, cartridge: Cartridge, batch: Batch) -> Float[Array, ""]:
    """Mean over layers of the per-layer relative error. Drop-in for `distill_loss`."""
    return jnp.mean(attnmse_layers(model, cartridge, batch))


def attnmse_loss_graduated(
    model, cartridge: Cartridge, batch: Batch, *,
    tau0: float = 1.0, noise0: float = 0.0, anneal_steps: int = 1000,
) -> Float[Array, ""]:
    """Graduated variant against attention collapse onto few slots.

    Two annealed perturbations of the trainable slots, both linear in
    `cartridge.steps` and gone by `anneal_steps`:

    - temperature: dividing the slot *keys* by `tau` divides exactly the
      prefix logits by `tau` (the suffix is untouched), softening prefix
      attention so unread slots keep receiving gradient early on;
    - instance noise: Gaussian noise on the keys, scaled per layer by the
      keys' own RMS, gives cold slots random chances to win queries.

    Both act on the physical keys (parameters times the per-layer scale),
    so they mean the same thing under either parameterization. The frozen
    sink slot is exempt from both, so its behaviour is exact throughout. At
    tau=1, sigma=0 this is `attnmse_loss` to the bit.
    """
    t = jnp.minimum(cartridge.steps.astype(jnp.float32) / anneal_steps, 1.0)
    tau = tau0 + (1.0 - tau0) * t
    sigma = noise0 * (1.0 - t)
    K = cartridge.physical_keys
    m = cartridge.trainable[None, :, None, None]
    rms = jnp.sqrt(jnp.mean(jax.lax.stop_gradient(K) ** 2, axis=(1, 2, 3), keepdims=True))
    key = jax.random.fold_in(jax.random.key(17), cartridge.steps)
    noise = jax.random.normal(key, K.shape, jnp.float32) * rms * sigma
    Kp = jnp.where(m, (K + noise) / tau, K)
    from .cache import KVPrefix

    prefix = KVPrefix(keys=Kp.astype(model.cache_dtype()),
                      values=cartridge.physical_values.astype(model.cache_dtype()))
    return jnp.mean(attnmse_layers(model, cartridge, batch, prefix=prefix))


__all__ = ["attnmse_layers", "attnmse_loss", "attnmse_loss_graduated", "teacher_targets"]
