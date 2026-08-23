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


class Qwen3VLTextAttention(eqx.Module):
    """Text attention with QK normalization and MRoPE.

    Key features:
    - QK normalization: RMSNorm applied to Q and K after projection
    - Grouped Query Attention (GQA) support
    - Standard KV cache integration
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
        attention_mask: Float[Array, "batch 1 seq kv_seq"] | None = None,
        cache: KVCacheLayer | None = None,
        cache_position: Int[Array, ""] | None = None,
        kv_mask: Bool[Array, "batch kv_seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], KVCacheLayer | None]:
        """Forward pass with optional KV cache.

        Args:
            hidden_states: Input tensor (batch, seq, hidden)
            position_embeddings: (cos, sin) from MRoPE, each (batch, seq, head_dim)
            attention_mask: Causal mask (batch, 1, seq, kv_seq)
            cache: Optional KV cache layer
            cache_position: Position in cache for new tokens

        Returns:
            (output, new_cache) tuple
        """
        batch_size, seq_len, _ = hidden_states.shape

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

        # Handle KV cache
        new_cache = None
        if cache is not None:
            # Update cache with new K, V
            new_cache = cache.update(cache_position, k, v)
            # Get full K, V from cache
            k, v = new_cache.get()

        # Compute attention
        # jax.nn.dot_product_attention expects (batch, seq, heads, head_dim) - NTHD order
        dtype = q.dtype
        if dtype == jnp.float32 and axla.use_flash():
            # The flash kernel only takes 16-bit inputs; the XLA path does not
            # need the downcast and a float32 reference should not pay it.
            q = q.astype(jnp.float16)
            k = k.astype(jnp.float16)
            v = v.astype(jnp.float16)

        # Use JAX's dot product attention
        if not axla.use_flash():
            if kv_mask is None:
                raise NotImplementedError(
                    "Causal attention without mask is not implemented."
                )
            if cache is None:
                mask = jax.vmap(axla.causal_mask, in_axes=(0, None))(kv_mask, seq_len)
            else:
                assert cache_position is not None
                mask = jax.vmap(axla.causal_mask, in_axes=(0, None, None))(
                    kv_mask, seq_len, cache_position
                )
            attn_output = jax.vmap(axla.masked_attention)(q, k, v, mask)
        elif kv_mask is not None and cache is None:
            # Prefill without a cache: queries and keys share positions, so
            # the kernel's native causal path (which skips future KV blocks)
            # does the causality and the mask mod only drops padded keys.
            assert kv_mask.shape == (batch_size, seq_len)
            mask = PaddedCausalMask(
                key_valid=kv_mask.astype(jnp.int32),
                offset=jnp.zeros((batch_size,), jnp.int32),
            )
            attn_output = flash_attn(
                q, k, v, causal=True, mask_mod=mask, backend="cute"
            )
        elif kv_mask is not None and cache is not None:
            # Queries sit at cache_position.. within the cache, which is not
            # the bottom-right alignment the native causal path assumes, so
            # the mask mod carries the offset instead. Padded and not-yet-
            # written cache slots are both dropped by the mask: the former by
            # kv_mask, the latter by causality.
            assert cache_position is not None
            assert kv_mask.shape == (batch_size, cache.keys.shape[1])
            mask = PaddedCausalMask(
                key_valid=kv_mask.astype(jnp.int32),
                offset=jnp.full((batch_size,), cache_position, jnp.int32),
            )
            attn_output = flash_attn(q, k, v, mask_mod=mask, backend="cute")
        else:
            raise NotImplementedError(
                "Causal attention without mask is not implemented."
            )

        # Reshape to (batch, seq, hidden) - already in (batch, seq, heads, dim)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = attn_output.astype(dtype)

        # Output projection
        output = self.o_proj(attn_output)

        return output, new_cache


__all__ = ["Qwen3VLTextAttention", "Qwen3VLVisionAttention"]
