"""Attention probe: where do the input tokens look, inside a KV prefix?

The flash kernel never materialises attention weights, so this re-runs the
text decoder with a dense float32 softmax per layer -- the same maths as
`attention_xla.masked_attention`, with the weights kept -- and reduces them
to the two views that matter for a composed prefix:

- `by_segment`: for every layer, head and query token, the attention mass
  landing on each labelled segment of the key sequence (each cartridge in a
  composed prefix, and the input itself).
- `by_slot`: the mass on every key slot, averaged over heads and queries, for
  a "which slots light up" picture.

Text only, batch of one, no cache. Exactness is checked by returning the
final logits, which must match the model's own prefix path.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .cache import KVPrefix
from .rope import apply_rotary_pos_emb
from .utils.pjit import pjit


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ProbeOutput:
    by_segment: Float[Array, "layers heads seq segments"]
    by_slot: Float[Array, "layers kv_seq"]
    logits: Float[Array, "seq vocab"]


def _layer_attention(attn, x, cos, sin, pk, pv, query_offset: int):
    """One layer's attention with the weights kept. `x` is post-norm (1, s, hidden).

    Returns (output (1, s, hidden-ish pre-o_proj reshaped), weights (heads, s, kv)).
    """
    _, s, _ = x.shape
    q = attn.q_proj(x).reshape(1, s, attn.num_heads, attn.head_dim)
    k = attn.k_proj(x).reshape(1, s, attn.num_kv_heads, attn.head_dim)
    v = attn.v_proj(x).reshape(1, s, attn.num_kv_heads, attn.head_dim)
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    k = jnp.concatenate([pk[None].astype(k.dtype), k], axis=1)[0]  # (kv, kvH, d)
    v = jnp.concatenate([pv[None].astype(v.dtype), v], axis=1)[0]
    q = q[0]  # (s, H, d)
    kv_len = k.shape[0]
    groups = attn.num_heads // attn.num_kv_heads

    qf = q.astype(jnp.float32).reshape(s, attn.num_kv_heads, groups, attn.head_dim)
    scores = jnp.einsum("qhgd,khd->hgqk", qf, k.astype(jnp.float32)) * attn.head_dim**-0.5
    causal = jnp.arange(kv_len)[None, :] <= (query_offset + jnp.arange(s))[:, None]
    scores = jnp.where(causal[None, None], scores, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(scores, axis=-1)  # (kvH, g, s, kv)
    out = jnp.einsum("hgqk,khd->qhgd", weights, v.astype(jnp.float32))
    out = out.reshape(1, s, attn.num_heads * attn.head_dim).astype(x.dtype)
    return attn.o_proj(out), weights.reshape(attn.num_heads, s, kv_len)


@pjit(static_argnames=("num_segments",))
def probe(
    model,
    input_ids: Int[Array, "seq"],
    prefix: KVPrefix,
    segments: Int[Array, "kv_seq"],
    *,
    num_segments: int,
) -> ProbeOutput:
    """Run `input_ids` after `prefix`, keeping every layer's attention weights.

    `segments` labels each of the `prefix.length + seq` key slots with an
    integer in `[0, num_segments)`; `by_segment` sums the attention mass per
    label.
    """
    lm = model.model.language_model
    ids = input_ids[None]
    s = ids.shape[1]
    p = prefix.length
    if segments.shape != (p + s,):
        raise ValueError(f"segments {segments.shape} must label {p + s} key slots")

    h = lm.embed_tokens(ids)
    pos = jnp.arange(s) + p
    cos, sin = lm.rotary_emb(jnp.broadcast_to(pos[None, None, :], (3, 1, s)))
    cos = cos.astype(h.dtype)
    sin = sin.astype(h.dtype)
    onehot = jax.nn.one_hot(segments, num_segments, dtype=jnp.float32)  # (kv, nseg)

    by_segment = []
    by_slot = []
    for i, layer in enumerate(lm.layers):
        x = layer.input_layernorm(h)
        a, w = _layer_attention(
            layer.self_attn, x, cos, sin, prefix.keys[i], prefix.values[i], p,
        )
        h = h + a
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        by_segment.append(jnp.einsum("hqk,kn->hqn", w, onehot))
        by_slot.append(w.mean(axis=(0, 1)))

    h = lm.norm(h)
    logits = model.get_lm_head()(h)[0]
    return ProbeOutput(
        by_segment=jnp.stack(by_segment),
        by_slot=jnp.stack(by_slot),
        logits=logits,
    )


__all__ = ["ProbeOutput", "probe"]
