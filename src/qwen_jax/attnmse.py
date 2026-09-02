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
    model, batch: Batch, *, context_kv: bool = False,
):
    """Per-layer residual-stream inputs and attention outputs at suffix positions.

    Replays the text decoder loop of `Qwen3VLTextModel.__call__` (text-only:
    no DeepStack) so the attention block's output can be read off before the
    residual add. Position ids and the KV mask come from the same resolution
    the real forward uses, so the capture is the teacher of `distill_loss`
    to the bit.

    With `context_kv`, also returns the teacher's rotated keys and values over
    the context positions, `(layers, batch, context, kv_heads, head_dim)`
    each: the corpus KV, for attending the teacher's layer to a stream other
    than the teacher's own (`attnmse_onpolicy_layers`).
    """
    from .attention import apply_rotary_pos_emb

    lm = model.model.language_model
    ids, mask = batch.teacher_ids, batch.teacher_mask
    context = batch.context
    b, s = ids.shape
    position_ids, _, _ = model.model._resolve_position_ids(
        ids, None, mask, None, None, None, None)
    pos_emb = _rotary(model, position_ids)
    kv_mask = mask.astype(jnp.bool)
    h = lm.embed_tokens(ids)
    ins, outs, ks, vs = [], [], [], []
    for layer in lm.layers:
        ins.append(h[:, context:])
        x = layer.input_layernorm(h)
        if context_kv:
            a = layer.self_attn
            k = a.k_norm(a.k_proj(x).reshape(b, s, a.num_kv_heads, a.head_dim))
            v = a.v_proj(x).reshape(b, s, a.num_kv_heads, a.head_dim)
            _, k = apply_rotary_pos_emb(k, k, *pos_emb)
            ks.append(k[:, :context])
            vs.append(v[:, :context])
        attn, _ = layer.self_attn(x, position_embeddings=pos_emb, kv_mask=kv_mask)
        outs.append(attn[:, context:])
        h = h + attn
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    if context_kv:
        return jnp.stack(ins), jnp.stack(outs), (jnp.stack(ks), jnp.stack(vs))
    return jnp.stack(ins), jnp.stack(outs)


def student_inputs(model, cartridge: Cartridge, batch: Batch) -> Float[Array, "layers batch seq hidden"]:
    """The student's own residual stream entering each layer, with the
    cartridge in place: the queries the cartridge actually meets at inference.
    No gradient flows through it (DAgger: the learner's states are data)."""
    lm = model.model.language_model
    prefix = cartridge.prefix(model.cache_dtype())
    ids, mask = batch.student_ids, batch.student_mask
    position_ids, _, _ = model.model._resolve_position_ids(
        ids, None, mask, None, None, None, prefix.length)
    pos_emb = _rotary(model, position_ids)
    ones = jnp.ones((ids.shape[0], prefix.length), dtype=jnp.bool)
    kv_mask = jnp.concatenate([ones, mask.astype(jnp.bool)], axis=1)
    h = lm.embed_tokens(ids)
    ins = []
    for i, layer in enumerate(lm.layers):
        ins.append(h)
        attn, _ = layer.self_attn(layer.input_layernorm(h), position_embeddings=pos_emb,
                                  kv_mask=kv_mask, prefix=prefix.layer(i))
        h = h + attn
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return jax.lax.stop_gradient(jnp.stack(ins))


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


def attnmse_onpolicy_layers(
    model, cartridge: Cartridge, batch: Batch, rollout: Cartridge | None = None, *,
    alpha: float = 1.0, target: str = "dagger", beta: float = 0.0, eps: float = 1e-6,
) -> Float[Array, "layers"]:
    """Per-layer relative error with the STUDENT's residual stream as the query.

    Teacher forcing across depth (`attnmse_layers`) trains layer l on the
    teacher's stream; at inference layer l sees the student's stream, which
    carries the accumulated error of every layer below, and a few percent per
    layer compounds through 36. Here layer l's input is the student's own
    stream `h_l^s` (one no-gradient forward with the cartridge in place), and
    the per-layer losses remain decoupled given the captured streams.

    Two targets for the student's attention output `a_l^s`:

    - "dagger" (default): the teacher's layer on the student's stream over
      the corpus KV, `attn_l(h_l^s; corpus)` -- the expert's action in the
      state the learner is actually in. Drift at layer l+1 is drift at l plus
      the local on-policy error plus the model's own response, so this local
      error is the only thing that injects drift; zero everywhere reproduces
      the teacher's stream exactly from the shared embedding.
    - "corrective": `a_l^t + beta * (h_l^t - h_l^s)`, the teacher's output on
      the student's queries plus a fraction of the inherited drift. At
      `beta=1` it lands the post-attention residual exactly on the teacher's,
      which is right in the limit and 25x out of scale in practice (residual
      streams dwarf attention outputs; held-out KL rose above the untrained
      cartridge's). `beta=0` is the plain on-policy-query variant.

    Both are normalised by the teacher's attention-output power, like
    `attnmse_layers`, and coincide with it when the streams coincide.
    `alpha` mixes in the teacher-forced term: `alpha * on_policy +
    (1 - alpha) * teacher_forced` per layer; `alpha=1` is pure on-policy.

    `rollout` is the cartridge the student streams are computed with. With
    `None` they come from `cartridge` itself, recomputed every step, which is
    fully online: each step's gradient is taken on streams the previous step
    just moved, and the loss is measured on streams that will move again
    (over six Adam steps from a wrong cartridge it rose 0.104 -> 0.112).
    DAgger proper freezes the learner's states for an iteration and trains
    on the aggregate; pass a frozen copy, refreshed every N steps, and use
    `alpha < 1` as the stand-in for the aggregated teacher-forced data.
    """
    if target not in ("corrective", "dagger"):
        raise ValueError(f"unknown target {target!r}")
    from .cache import KVCacheLayer

    if batch.note_ids is not None:
        raise ValueError("student notes are not supported by the teacher-forced loss: "
                         "the teacher has no hidden states for note positions")
    lm = model.model.language_model
    ins_t, tgt_t, (k_ctx, v_ctx) = jax.lax.stop_gradient(
        teacher_targets(model, batch, context_kv=True))
    ins_s = student_inputs(model, cartridge if rollout is None else rollout, batch)
    prefix = cartridge.prefix(model.cache_dtype())
    ids, mask = batch.student_ids, batch.student_mask
    context = batch.context
    t_ids, t_mask = batch.teacher_ids, batch.teacher_mask
    # Student geometry: suffix at p + i over [cartridge | suffix].
    pos_s, _, _ = model.model._resolve_position_ids(ids, None, mask, None, None, None, prefix.length)
    pos_s_emb = _rotary(model, pos_s)
    ones = jnp.ones((ids.shape[0], prefix.length), dtype=jnp.bool)
    kv_mask_s = jnp.concatenate([ones, mask.astype(jnp.bool)], axis=1)
    # Teacher geometry for the same suffix: |context| + i over [corpus | suffix].
    pos_t, _, _ = model.model._resolve_position_ids(t_ids, None, t_mask, None, None, None, None)
    cos_t, sin_t = _rotary(model, pos_t)
    pos_tc_emb = (cos_t[:, context:], sin_t[:, context:])
    kv_mask_tc = jnp.concatenate([t_mask[:, :context].astype(jnp.bool), mask.astype(jnp.bool)],
                                 axis=1)
    m = batch.loss_mask[..., None]

    def rel(x, ref, norm):
        d = jnp.where(m, x.astype(jnp.float32) - ref.astype(jnp.float32), 0.0)
        r = jnp.where(m, norm.astype(jnp.float32), 0.0)
        return jnp.sum(d * d) / (jnp.sum(r * r) + eps)

    losses = []
    gate = jnp.zeros((), jnp.float32)
    for i, layer in enumerate(lm.layers):
        # See `attnmse_layers` for why the barrier and the checkpoint.
        h_t, t_t, h_s, kc, vc = jax.lax.optimization_barrier(
            (ins_t[i], tgt_t[i], ins_s[i], k_ctx[i], v_ctx[i], gate))[:5]

        @jax.checkpoint
        def one(h_t, t_t, h_s, kc, vc, pfx, layer=layer):
            attn = layer.self_attn
            x_s = layer.input_layernorm(h_s)
            if target == "corrective":
                t_op = t_t.astype(jnp.float32) + beta * (h_t.astype(jnp.float32)
                                                         - h_s.astype(jnp.float32))
            else:
                t_op, _ = attn(x_s, position_embeddings=pos_tc_emb, kv_mask=kv_mask_tc,
                               prefix=KVCacheLayer(keys=kc, values=vc))
            o_s, _ = attn(x_s, position_embeddings=pos_s_emb, kv_mask=kv_mask_s, prefix=pfx)
            loss = alpha * rel(o_s, jax.lax.stop_gradient(t_op), t_t)
            if alpha < 1.0:
                o_t, _ = attn(layer.input_layernorm(h_t), position_embeddings=pos_s_emb,
                              kv_mask=kv_mask_s, prefix=pfx)
                loss = loss + (1.0 - alpha) * rel(o_t, t_t, t_t)
            return loss

        losses.append(one(h_t, t_t, h_s, kc, vc, prefix.layer(i)))
        gate = losses[-1]
    return jnp.stack(losses)


def attnmse_onpolicy_loss(model, cartridge: Cartridge, batch: Batch, rollout: Cartridge | None = None,
                          *, alpha: float = 1.0, target: str = "dagger", beta: float = 0.0):
    """Mean over layers of `attnmse_onpolicy_layers`. Drop-in for `distill_loss`;
    `rollout` arrives as the trainer's pass-through argument."""
    return jnp.mean(attnmse_onpolicy_layers(model, cartridge, batch, rollout, alpha=alpha,
                                            target=target, beta=beta))


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


__all__ = ["attnmse_layers", "attnmse_loss", "attnmse_loss_graduated", "attnmse_onpolicy_layers",
           "attnmse_onpolicy_loss", "student_inputs", "teacher_targets"]
