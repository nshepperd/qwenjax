"""Attention layers for Qwen3-VL."""

from einops import rearrange
from dataclasses import field
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from flash_attn_jax.varlen import flash_mha_varlen
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from . import equinox_utils as eu
from .cache import KVCacheLayer
from .utils.debug import debugpy_pm
from .utils.rng import split
from qwen_jax.config import Qwen3VLTextConfig

from .linear import Linear, RMSNorm
from .rope import apply_rotary_pos_emb, apply_rotary_pos_emb_vision


class Qwen3VLVisionAttention(eqx.Module):
    """Vision attention with variable-length sequence support.

    Uses flash attention with cu_seqlens for efficient processing of
    packed sequences (multiple images/videos of different sizes).
    """

    dim: int = field(metadata=dict(static=True))
    num_heads: int = field(metadata=dict(static=True))
    head_dim: int = field(metadata=dict(static=True))

    qkv: Linear  # Fused QKV projection
    proj: Linear

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
        self.dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Fused QKV: projects to 3 * hidden_size
        self.qkv = Linear(hidden_size, hidden_size * 3, use_bias=True)
        self.proj = Linear(hidden_size, hidden_size, use_bias=True)

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        """Load state dict into module."""
        qkv = self.qkv.load_state_dict(state_dict, prefix + "qkv.")
        proj = self.proj.load_state_dict(state_dict, prefix + "proj.")
        return eu.replace(self, qkv=qkv, proj=proj)

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

        # flash_mha_varlen expects (total_seq, heads, head_dim) - no batch dimension
        # q, k, v are already (seq, heads, head_dim)

        # flash_mha_varlen requires float16 or bfloat16
        orig_dtype = q.dtype
        if orig_dtype == jnp.float32:
            q = q.astype(jnp.float16)
            k = k.astype(jnp.float16)
            v = v.astype(jnp.float16)

        # Compute max sequence length for flash attention
        max_seqlen = hidden_states.shape[0]

        # Variable-length flash attention
        attn_output = flash_mha_varlen(
            q,
            k,
            v,
            seqlens_q=cu_seqlens,
            seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=self.head_dim**-0.5,
            is_causal=False,
        )

        # Convert back to original dtype
        if orig_dtype == jnp.float32:
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

    hidden_size: int = field(metadata=dict(static=True))
    num_heads: int = field(metadata=dict(static=True))
    num_kv_heads: int = field(metadata=dict(static=True))
    head_dim: int = field(metadata=dict(static=True))
    num_kv_groups: int = field(metadata=dict(static=True))

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
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_attention_heads // config.num_key_value_heads

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

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        """Load state dict into module."""
        return eu.replace(
            self,
            q_proj=self.q_proj.load_state_dict(state_dict, prefix + "q_proj."),
            k_proj=self.k_proj.load_state_dict(state_dict, prefix + "k_proj."),
            v_proj=self.v_proj.load_state_dict(state_dict, prefix + "v_proj."),
            o_proj=self.o_proj.load_state_dict(state_dict, prefix + "o_proj."),
            q_norm=self.q_norm.load_state_dict(state_dict, prefix + "q_norm."),
            k_norm=self.k_norm.load_state_dict(state_dict, prefix + "k_norm."),
        )

    @jax.remat
    def __call__(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: tuple[
            Float[Array, "batch seq head_dim"], Float[Array, "batch seq head_dim"]
        ],
        attention_mask: Optional[Float[Array, "batch 1 seq kv_seq"]] = None,
        cache: Optional[KVCacheLayer] = None,
        cache_position: Optional[Int[Array, ""]] = None,
        kv_mask: Optional[Bool[Array, "batch kv_seq"]] = None,
    ) -> tuple[Float[Array, "batch seq hidden"], Optional[KVCacheLayer]]:
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
        scale = self.head_dim**-0.5

        dtype = q.dtype
        if dtype == jnp.float32:
            q = q.astype(jnp.float16)
            k = k.astype(jnp.float16)
            v = v.astype(jnp.float16)

        # Use JAX's dot product attention
        if kv_mask is not None and cache is None:
            # no cache and we have mask. use varlen attn
            assert kv_mask.shape == (batch_size, seq_len)

            def varlen_fwd(q, k, v, kv_mask):
                # idx: [seq_len] = index such that kv_mask[idx] has all the 1s at the front
                idx = jnp.argsort(~kv_mask, stable=True, axis=0)
                vlen = jnp.sum(kv_mask, axis=0, dtype=jnp.int32)
                q = q[idx]  # s h d
                k = k[idx]  # s h d
                v = v[idx]  # s h d
                # staggered seqlens produces sparsity without needing seqused_k
                seqlens_q = jnp.array([0, vlen, vlen, seq_len], dtype=jnp.int32)
                seqlens_k = jnp.array([0, vlen, seq_len, seq_len], dtype=jnp.int32)
                o = flash_mha_varlen(q, k, v, seqlens_q, seqlens_k, is_causal=True)
                inv_idx = jnp.argsort(idx, axis=0)
                o = o[inv_idx]
                return o

            attn_output = jax.vmap(varlen_fwd)(q, k, v, kv_mask)
        elif kv_mask is not None and cache is not None:
            assert cache_position is not None
            assert kv_mask.shape == (batch_size, cache.keys.shape[1])
            cache_len = cache.keys.shape[1]

            def varlen_fwd(q, k, v, kv_mask):
                assert cache_position is not None
                kv_mask = kv_mask & (jnp.arange(cache_len) < cache_position + seq_len)
                kv_idx = jnp.argsort(~kv_mask, stable=True, axis=0)
                q_mask = jax.lax.dynamic_slice(
                    kv_mask, start_indices=(cache_position,), slice_sizes=(seq_len,)
                )
                q_idx = jnp.argsort(~q_mask, stable=True, axis=0)
                kvlen = kv_mask.sum(dtype=jnp.int32)
                qlen = q_mask.sum(dtype=jnp.int32)
                seqlens_q = jnp.array([0, qlen, qlen, seq_len])
                seqlens_k = jnp.array([0, kvlen, cache_len, cache_len])
                o = flash_mha_varlen(
                    q[q_idx],
                    k[kv_idx],
                    v[kv_idx],
                    seqlens_q=seqlens_q,
                    seqlens_k=seqlens_k,
                    is_causal=True,
                )
                inv_idx = jnp.argsort(q_idx, axis=0)
                return o[inv_idx]

            attn_output = jax.vmap(varlen_fwd)(q, k, v, kv_mask)
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


__all__ = ["Qwen3VLVisionAttention", "Qwen3VLTextAttention"]
