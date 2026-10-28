"""Split-KV decode attention (flash-decoding) as a cutejax kernel.

For a few queries against a long KV cache -- decode -- attention is a
memory-bound read of the filled part of the cache. fa4 runs it one CTA per
head over the whole cache, which leaves most of the GPU idle; XLA's dense path
keeps the SMs busy but reads the whole *allocated* cache, filled or not.

This kernel splits the filled length across many warps:

- Each warp is one split: a contiguous range of keys of one (batch, KV head).
  All the query heads that share that KV head (GQA), times all the queries,
  ride in the same warp, so each K/V row is read once.
- A key row is read by the whole warp, each lane taking D/32 contiguous
  elements -- 16 bytes at head_dim 256 -- so every load is one coalesced
  512-byte row (which also suits a cache in pinned host memory, read over
  PCIe). The row's dot products are finished with a butterfly reduction, after
  which every lane holds every score and runs the same online softmax.
- The split boundaries come from the runtime filled length, so the cost
  follows the fill rather than the allocation and nothing recompiles as the
  cache grows.

Each split writes its unnormalized output, running max and sum; `combine`
merges them (a few MB, in XLA).
"""
from __future__ import annotations

import math

import cutejax
import cutlass
import jax
import jax.numpy as jnp
from cutlass import cute
from jaxtyping import Array, Bool, Float, Int

_WARPS = 4  # per CTA; each warp is one split
_UNROLL = 4  # keys whose loads are issued together
_LOG2E = math.log2(math.e)
_NEG_INF = -math.inf


def _min(i, hi):
    """min(i, hi) for Int32 (no integer min in the DSL's operator set)."""
    d = i - hi
    return i - (d & ~(d >> 31))


def _unpack_bf16(words, dst, n_words: int):
    """Little-endian bf16 pairs in Int32 `words` -> Float32 `dst` (2 per word)."""
    w16 = cute.make_tensor(cute.recast_ptr(words.iterator, dtype=cutlass.BFloat16),
                           cute.make_layout(2 * n_words))
    for i in range(2 * n_words):  # a plain helper: unrolled by Python itself
        dst[i] = cutlass.Float32(w16[i])


@cute.kernel
def _decode_kernel(
    gK: cute.Tensor, gV: cute.Tensor, gQ: cute.Tensor, gMask: cute.Tensor, gN: cute.Tensor,
    gAcc: cute.Tensor, gM: cute.Tensor, gL: cute.Tensor,
    kv_len: cutlass.Int32,
    KH: cutlass.Constexpr[int], R: cutlass.Constexpr[int], G: cutlass.Constexpr[int],
    SQ: cutlass.Constexpr[int], NS: cutlass.Constexpr[int], W: cutlass.Constexpr[int],
):
    tid, _, _ = cute.arch.thread_idx()
    bx, g, b = cute.arch.block_idx()
    warp = tid >> 5
    lane = tid & 31
    split = bx * _WARPS + warp
    E = 2 * W  # elements per lane

    n = gN[b]
    chunk = (n + NS - 1) // NS
    t0 = split * chunk
    t1 = _min(t0 + chunk, n)

    # this lane's slice of every query row (pre-scaled by log2(e)/sqrt(d))
    q = cute.make_rmem_tensor(R * E, cutlass.Float32)
    for r in cutlass.range_constexpr(R):
        for i in cutlass.range_constexpr(E):
            q[r * E + i] = gQ[b, g, r, lane * E + i]

    m = cute.make_rmem_tensor(R, cutlass.Float32)
    lsum = cute.make_rmem_tensor(R, cutlass.Float32)
    acc = cute.make_rmem_tensor(R * E, cutlass.Float32)
    for r in cutlass.range_constexpr(R):
        m[r] = cutlass.Float32(_NEG_INF)
        lsum[r] = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(E):
            acc[r * E + i] = cutlass.Float32(0.0)

    kw = cute.make_rmem_tensor(cute.make_layout((_UNROLL, W), stride=(W, 1)), cutlass.Int32)
    vw = cute.make_rmem_tensor(cute.make_layout((_UNROLL, W), stride=(W, 1)), cutlass.Int32)
    kf = cute.make_rmem_tensor(E, cutlass.Float32)
    vf = cute.make_rmem_tensor(E, cutlass.Float32)
    last = kv_len - 1

    for base in cutlass.range(t0, t1, _UNROLL):
        # issue every load of the group before using any of them
        for u in cutlass.range_constexpr(_UNROLL):
            t = _min(base + u, last)
            row = (b * kv_len + t) * KH + g
            cute.autovec_copy(gK[row * 32 + lane, None], kw[u, None])
            cute.autovec_copy(gV[row * 32 + lane, None], vw[u, None])
        for u in cutlass.range_constexpr(_UNROLL):
            t = base + u
            valid = (t < t1) & (gMask[b, _min(t, last)] != 0)
            _unpack_bf16(kw[u, None], kf, W)
            _unpack_bf16(vw[u, None], vf, W)
            for r in cutlass.range_constexpr(R):
                s = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(E):
                    s = s + q[r * E + i] * kf[i]
                s = s + cute.arch.shuffle_sync_bfly(s, 16)
                s = s + cute.arch.shuffle_sync_bfly(s, 8)
                s = s + cute.arch.shuffle_sync_bfly(s, 4)
                s = s + cute.arch.shuffle_sync_bfly(s, 2)
                s = s + cute.arch.shuffle_sync_bfly(s, 1)
                # query row r is query r // G of SQ, at position n - SQ + r // G
                if valid & (t <= n - SQ + r // G):
                    m_new = cute.arch.fmax(m[r], s)
                    scale = cute.math.exp2(m[r] - m_new, fastmath=True)
                    p = cute.math.exp2(s - m_new, fastmath=True)
                    lsum[r] = lsum[r] * scale + p
                    for i in cutlass.range_constexpr(E):
                        acc[r * E + i] = acc[r * E + i] * scale + p * vf[i]
                    m[r] = m_new

    for r in cutlass.range_constexpr(R):
        for i in cutlass.range_constexpr(E):
            gAcc[b, g, split, r, lane * E + i] = acc[r * E + i]
        if lane == 0:
            gM[b, g, split, r] = m[r]
            gL[b, g, split, r] = lsum[r]


@cute.jit
def _decode_launch(
    stream, gKc: cute.Tensor, gVc: cute.Tensor, gQ: cute.Tensor, gMask: cute.Tensor,
    gN: cute.Tensor, gAcc: cute.Tensor, gM: cute.Tensor, gL: cute.Tensor,
    G: cutlass.Constexpr[int], SQ: cutlass.Constexpr[int], NS: cutlass.Constexpr[int],
):
    batch, kv_len, kh, d = gKc.shape
    W = d // 64  # Int32 words per lane: D/32 bf16 elements
    rows = batch * kv_len * kh * 32
    # the caches as (row * 32 + lane, word): the static W makes each lane's
    # piece one vector load
    gK = cute.make_tensor(cute.recast_ptr(gKc.iterator, dtype=cutlass.Int32),
                          cute.make_layout((rows, W), stride=(W, 1)))
    gV = cute.make_tensor(cute.recast_ptr(gVc.iterator, dtype=cutlass.Int32),
                          cute.make_layout((rows, W), stride=(W, 1)))
    R = gQ.shape[2]
    _decode_kernel(gK, gV, gQ, gMask, gN, gAcc, gM, gL, kv_len, kh, R, G, SQ, NS, W).launch(
        grid=[NS // _WARPS, kh, batch], block=[_WARPS * 32, 1, 1], stream=stream)


def decode_attention(
    q: Float[Array, "batch seq heads head_dim"],
    k: Float[Array, "batch kv_seq kv_heads head_dim"],
    v: Float[Array, "batch kv_seq kv_heads head_dim"],
    kv_mask: Bool[Array, "batch kv_seq"],
    seqused: Int[Array, "batch"],
    *,
    num_splits: int = 128,
) -> Float[Array, "batch seq heads head_dim"]:
    """Causal attention of a few queries against the first `seqused` cache slots.

    The `seq` queries sit at positions `seqused - seq ..` of each row; slots
    from `seqused` on are never read. `kv_mask` drops padded slots. k/v must
    be bfloat16 with head_dim 128 or 256; they may live in pinned host memory,
    which the kernel reads in place.
    """
    batch, sq, heads, d = q.shape
    kh = k.shape[2]
    g = heads // kh
    assert k.dtype == v.dtype == jnp.bfloat16 and d in (128, 256), (k.dtype, d)
    assert num_splits % _WARPS == 0
    r = sq * g
    qr = (
        q.reshape(batch, sq, kh, g, d).transpose(0, 2, 1, 3, 4).reshape(batch, kh, r, d)
        .astype(jnp.float32) * (_LOG2E / math.sqrt(d))
    )
    acc, mx, lsum = cutejax.call(
        _decode_launch,
        (
            jax.ShapeDtypeStruct((batch, kh, num_splits, r, d), jnp.float32),
            jax.ShapeDtypeStruct((batch, kh, num_splits, r), jnp.float32),
            jax.ShapeDtypeStruct((batch, kh, num_splits, r), jnp.float32),
        ),
        k, v, qr, kv_mask.astype(jnp.int32), seqused.astype(jnp.int32),
        in_specs=[
            cutejax.ArraySpec(static_dims=(2, 3)),
            cutejax.ArraySpec(static_dims=(2, 3)),
            cutejax.ArraySpec(static_dims=(1, 2, 3)),
            None,
            None,
        ],
        G=g, SQ=sq, NS=num_splits,
    )
    out = combine(acc, mx, lsum)  # (batch, kh, r, d)
    return (
        out.reshape(batch, kh, sq, g, d).transpose(0, 2, 1, 3, 4)
        .reshape(batch, sq, heads, d).astype(q.dtype)
    )


def combine(acc, mx, lsum):
    """Merge per-split (unnormalized output, max, sum) along axis 2.

    The maxima are in log2 units. A row no split saw (every key masked) comes
    out zero, as fa4 returns it.
    """
    top = jnp.max(mx, axis=2, keepdims=True)
    w = jnp.exp2(mx - jnp.where(jnp.isfinite(top), top, 0.0))  # 0 for empty splits
    num = jnp.sum(w[..., None] * acc, axis=2)
    den = jnp.sum(w * lsum, axis=2)
    return num / jnp.where(den > 0, den, 1.0)[..., None]


__all__ = ["combine", "decode_attention"]
