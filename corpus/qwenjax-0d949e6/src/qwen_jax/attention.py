"""Attention layers for Qwen3-VL."""
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from fa4_jax import MaskMod, document_mask, flash_attn
from jaxtyping import Array, Bool, Float, Int

from qwen_jax.config import Qwen3VLTextConfig

from . import attention_xla as axla
from .cache import KVCacheLayer
from .linear import Linear, RMSNorm
from .rope import apply_rotary_pos_emb, apply_rotary_pos_emb_vision


@dataclass(frozen=True)
class PaddedCausalMask(MaskMod):
    """Causal attention over a padded key sequence.

    A query at absolute position ``offset[b] + q_idx`` attends a key when the
    key is unpadded and does not come after it -- the flash-kernel form of
    `attention_xla.causal_mask`. Both fields are runtime data (int32, since the
    kernel codegen indexes them as tensors), so every prompt shape and cache
    position reuses one compiled kernel.
    """

    key_valid: Int[Array, "batch kv_seq"]
    offset: Int[Array, "batch"]

    def __call__(self, b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx + self.offset[b]) & (self.key_valid[b, kv_idx] != 0)


class Qwen3VLVisionAttention(eqx.Module):
    """Vision attention with variable-length sequence support.

    Uses flash attention with cu_seqlens for efficient processing of
    packed sequences (multiple images/videos of different sizes).
    """

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    qkv: Linear  # Fused QKV projection
    proj: Linear

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Fused QKV: projects to 3 * hidden_size
        self.qkv = Linear(hidden_size, hidden_size * 3, use_bias=True)
        self.proj = Linear(hidden_size, hidden_size, use_bias=True)

    def __call__(
        self,
        hidden_states: Float[Array, "seq hidden"],
        cu_seqlens: Int[Array, "num_seqs_plus_1"],
        position_embeddings: tuple[
            Float[Array, "seq head_dim"], Float[Array, "seq head_dim"]
        ],
    ) -> Float[Array, "seq hidden"]:
        """Forward pass with variable-length flash attention.

        Args:
            hidden_states: Packed sequences (total_tokens, hidden_size)
            cu_seqlens: Cumulative sequence lengths, e.g., [0, 100, 250, 400]
                       means 3 sequences of lengths 100, 150, 150
            position_embeddings: (cos, sin) for RoPE, each (total_tokens, head_dim)

        Returns:
            Output tensor (total_tokens, hidden_size)
        """
        seq_len = hidden_states.shape[0]

        # Fused QKV projection
        qkv = self.qkv(hidden_states)  # (seq, 3 * hidden)

        # Reshape to (seq, 3, heads, head_dim) and split
        q, k, v = rearrange(
            qkv, "s (p h d) -> p s h d", p=3, h=self.num_heads, d=self.head_dim
        )

        # Apply RoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # The flash kernel requires float16 or bfloat16
        orig_dtype = q.dtype
        if orig_dtype == jnp.float32 and axla.use_flash():
            q = q.astype(jnp.float16)
            k = k.astype(jnp.float16)
            v = v.astype(jnp.float16)

        if axla.use_flash():
            # Packed sequences attend block-diagonally: a token's segment is
            # how many cu_seqlens boundaries precede it (same relation as
            # axla.segment_mask), expressed as a document id per token.
            pos = jnp.arange(seq_len, dtype=jnp.int32)
            segment = jnp.searchsorted(cu_seqlens, pos, side="right").astype(jnp.int32)
            attn_output = flash_attn(
                q[None],
                k[None],
                v[None],
                mask_mod=document_mask(segment[None]),
                scale=self.head_dim**-0.5,
                backend="cute",
            )[0]
        else:
            attn_output = axla.masked_attention(
                q, k, v,
                axla.segment_mask(cu_seqlens, seq_len),
                scale=self.head_dim**-0.5,
            )

        # Convert back to original dtype
        attn_output = attn_output.astype(orig_dtype)

        # Reshape back: (seq, heads, head_dim) -> (seq, hidden)
        attn_output = attn_output.reshape(seq_len, -1)

        # Output projection
        return self.proj(attn_output)


def causal_attention(
    q: Float[Array, "batch seq heads head_dim"],
    k: Float[Array, "batch kv_seq kv_heads head_dim"],
    v: Float[Array, "batch kv_seq kv_heads head_dim"],
    kv_mask: Bool[Array, "batch kv_seq"],
    query_offset: Int[Array, ""] | int,
) -> Float[Array, "batch seq heads head_dim"]:
    """Causal attention where the queries are the last `seq` of the `kv_seq` positions.

    One routine for every way keys can reach attention -- computed alongside
    the queries, read back from a cache, or concatenated from a prefix -- since
    they all reduce to: here are `kv_seq` key slots, `kv_mask` says which hold a
    real token, and the queries occupy slots `[query_offset, query_offset + seq)`.
    A query attends a key iff the key is real and its slot is not later.

    The flash path hands the same relation to the kernel as a mask mod
    (`PaddedCausalMask` with `query_offset` as the offset). When the queries
    are statically known to be the tail of the key sequence -- prefill, or a
    prefix concatenated ahead of the input -- that is exactly the kernel's
    native bottom-right causal alignment, so `causal=True` lets it skip the
    future KV blocks outright; a cache (traced offset, capacity beyond the
    queries) goes through the mask alone.
    """
    batch_size, seq_len, _, _ = q.shape
    kv_len = k.shape[1]
    if kv_mask.shape != (batch_size, kv_len):
        raise ValueError(f"kv_mask {kv_mask.shape} does not match kv shape ({batch_size}, {kv_len})")

    dtype = q.dtype
    if not axla.use_flash():
        mask = jax.vmap(axla.causal_mask, in_axes=(0, None, None))(
            kv_mask, seq_len, query_offset
        )
        return jax.vmap(axla.masked_attention)(q, k, v, mask).astype(dtype)

    if dtype == jnp.float32:
        # The flash kernel only takes 16-bit inputs; the XLA path above does
        # not need the downcast and a float32 reference should not pay it.
        q = q.astype(jnp.float16)
        k = k.astype(jnp.float16)
        v = v.astype(jnp.float16)

    mask = PaddedCausalMask(
        key_valid=kv_mask.astype(jnp.int32),
        offset=jnp.full((batch_size,), query_offset, jnp.int32),
    )
    tail = isinstance(query_offset, int) and query_offset + seq_len == kv_len
    out = flash_attn(q, k, v, causal=tail, mask_mod=mask, backend="cute")
    return out.astype(dtype)


class Qwen3VLTextAttention(eqx.Module):
    """Text attention with QK normalization and MRoPE.

    Key features:
    - QK normalization: RMSNorm applied to Q and K after projection
    - Grouped Query Attention (GQA) support
    - KV cache (in-place, fixed capacity) or KV prefix (concatenated)
    """

    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    q_norm: RMSNorm
    k_norm: RMSNorm

    def __init__(
        self,
        config: Qwen3VLTextConfig,
    ):
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            use_bias=config.attention_bias,
        )
        self.k_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            use_bias=config.attention_bias,
        )
        self.v_proj = Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            use_bias=config.attention_bias,
        )
        self.o_proj = Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            use_bias=config.attention_bias,
        )

        # QK normalization (per head)
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    @jax.remat
    def __call__(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: tuple[
            Float[Array, "batch seq head_dim"], Float[Array, "batch seq head_dim"]
        ],
        *,
        kv_mask: Bool[Array, "batch kv_seq"],
        cache: KVCacheLayer | None = None,
        cache_position: Int[Array, ""] | None = None,
        prefix: KVCacheLayer | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], KVCacheLayer | None]:
        """Forward pass with optional KV cache or KV prefix.

        Args:
            hidden_states: Input tensor (batch, seq, hidden)
            position_embeddings: (cos, sin) from MRoPE, each (batch, seq, head_dim)
            kv_mask: Which key slots hold a real token, over the full key
                sequence attention will see: the cache's capacity, the prefix
                plus the input, or just the input.
            cache: Optional KV cache layer; new K/V are written at cache_position.
            cache_position: Position in cache for new tokens.
            prefix: Optional K/V (batch-or-1, p, kv_heads, head_dim) attended to
                ahead of the input. Mutually exclusive with `cache` -- to serve a
                prefix through a cache, write it in first (`KVCache.write_prefix`).

        Returns:
            (output, new_cache) tuple
        """
        batch_size, seq_len, _ = hidden_states.shape
        if cache is not None and prefix is not None:
            raise ValueError("pass either a cache or a prefix, not both")

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: (batch, seq, heads * head_dim) -> (batch, seq, heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # QK normalization (applied per head, before RoPE)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply MRoPE
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        new_cache = None
        query_offset: Int[Array, ""] | int = 0
        if cache is not None:
            assert cache_position is not None
            new_cache = cache.update(cache_position, k, v)
            k, v = new_cache.get()
            query_offset = cache_position
        elif prefix is not None:
            pk, pv = prefix.keys, prefix.values
            shape = (batch_size, *pk.shape[1:])
            k = jnp.concatenate([jnp.broadcast_to(pk.astype(k.dtype), shape), k], axis=1)
            v = jnp.concatenate([jnp.broadcast_to(pv.astype(v.dtype), shape), v], axis=1)
            query_offset = pk.shape[1]

        attn_output = causal_attention(q, k, v, kv_mask, query_offset)

        # Reshape to (batch, seq, hidden) - already in (batch, seq, heads, dim)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(attn_output)

        return output, new_cache


__all__ = ["Qwen3VLTextAttention", "Qwen3VLVisionAttention", "causal_attention"]
