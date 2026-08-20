"""Attention in plain XLA ops, for when the flash kernel is unavailable.

`flash_mha_varlen` is a CUDA custom call, so the model cannot run at all on a
host-only backend. That matters for exactly one thing: a bf16 reference of a
model too large to fit in VRAM has to run somewhere, and the honest place is
the same model code the quantized runs use, on CPU. A separate reference
implementation (HF, say) would fold implementation differences into every
measured quantization gap.

These are the same maths as the varlen calls they stand in for, written as a
dense masked softmax: O(seq^2) where flash is O(seq), which is why they are the
fallback and not the default.

Selected by `QWEN_JAX_ATTENTION`: `flash` or `xla`, defaulting to `flash` when
the default backend is a GPU and `xla` otherwise.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


def use_flash() -> bool:
    """Whether to dispatch attention to the flash kernel."""
    default = "flash" if jax.default_backend() == "gpu" else "xla"
    choice = os.environ.get("QWEN_JAX_ATTENTION", default)
    if choice not in ("flash", "xla"):
        raise ValueError(f"QWEN_JAX_ATTENTION must be 'flash' or 'xla', got {choice!r}")
    return choice == "flash"


def masked_attention(
    q: Float[Array, "seq heads dim"],
    k: Float[Array, "kv_seq kv_heads dim"],
    v: Float[Array, "kv_seq kv_heads dim"],
    mask: Bool[Array, "seq kv_seq"],
    *,
    scale: float | None = None,
) -> Float[Array, "seq heads dim"]:
    """Single-sequence attention under an explicit boolean mask.

    Grouped-query is handled by folding the group axis out of the head axis, so
    `heads` need only be a multiple of `kv_heads`. Scores are accumulated in
    float32 regardless of input dtype -- this is the reference path, and the
    cost of the upcast is irrelevant next to the cost of the dense mask.

    Fully-masked query rows attend uniformly rather than producing NaN. They
    only arise at padded positions, whose outputs are discarded, and a NaN
    there would otherwise poison the whole batch item downstream.
    """
    seq, heads, dim = q.shape
    kv_heads = k.shape[1]
    groups = heads // kv_heads
    scale = dim**-0.5 if scale is None else scale

    qf = q.astype(jnp.float32).reshape(seq, kv_heads, groups, dim)
    scores = jnp.einsum("qhgd,khd->hgqk", qf, k.astype(jnp.float32)) * scale
    scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("hgqk,khd->qhgd", weights, v.astype(jnp.float32))
    return out.reshape(seq, heads, dim).astype(q.dtype)


def causal_mask(
    kv_mask: Bool[Array, "kv_seq"],
    seq_len: int,
    query_offset: Int[Array, ""] | int = 0,
) -> Bool[Array, "seq kv_seq"]:
    """Causal mask for `seq_len` queries starting at absolute position `query_offset`.

    A query attends a key when the key is unpadded and does not come after it.
    The flash path expresses the same thing by compacting the valid tokens to
    the front and running two varlen segments over them; because that
    compaction is a stable sort it preserves relative order, so causality among
    the valid tokens is the same relation either way.
    """
    kv_pos = jnp.arange(kv_mask.shape[0])
    q_pos = query_offset + jnp.arange(seq_len)
    return kv_mask[None, :] & (kv_pos[None, :] <= q_pos[:, None])


def segment_mask(
    cu_seqlens: Int[Array, "num_seqs_plus_1"], seq_len: int
) -> Bool[Array, "seq seq"]:
    """Block-diagonal mask: tokens attend within their own packed sequence.

    `cu_seqlens` is the same cumulative-length vector the varlen kernel takes,
    so the segment a token belongs to is how many boundaries precede it.
    """
    pos = jnp.arange(seq_len)
    segment = jnp.searchsorted(cu_seqlens, pos, side="right") - 1
    return segment[:, None] == segment[None, :]


__all__ = ["causal_mask", "masked_attention", "segment_mask", "use_flash"]
