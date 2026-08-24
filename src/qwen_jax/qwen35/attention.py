"""Gated full attention for Qwen3.5 (every fourth layer).

Differs from Qwen3-VL attention in two ways, both inherited from Qwen3-Next:

  - The query projection is fused with an output gate: q_proj emits
    2*head_dim per head, laid out per-head as [query | gate]. The gate is
    applied as sigmoid(gate) on the attention output, before o_proj. The
    fused layout is exactly how both HF and GGUF store the weight, so the
    projection loads without surgery (quantized included).

  - RoPE is partial: only the first quarter of each 256-dim head rotates.
    That arrives here for free -- cos/sin are sized to the rotary dim and
    `apply_rotary_pos_emb` passes the tail through.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from qwen_jax.attention import masked_causal_attention
from qwen_jax.cache import KVCacheLayer
from qwen_jax.linear import Linear, RMSNorm
from qwen_jax.rope import apply_rotary_pos_emb

from .config import Qwen35TextConfig


class Qwen35Attention(eqx.Module):
    """Full attention with QK norm, partial MRoPE, and an output gate."""

    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    q_norm: RMSNorm
    k_norm: RMSNorm

    def __init__(self, config: Qwen35TextConfig):
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        assert config.attn_output_gate, "Qwen3.5 attention is always gated"
        self.q_proj = Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim * 2,
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
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    @jax.remat
    def __call__(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: tuple[
            Float[Array, "batch seq rotary_dim"], Float[Array, "batch seq rotary_dim"]
        ],
        cache: KVCacheLayer | None = None,
        cache_position: Int[Array, ""] | None = None,
        kv_mask: Bool[Array, "batch kv_seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], KVCacheLayer | None]:
        batch_size, seq_len, _ = hidden_states.shape

        # Fused query+gate: per head, the first head_dim lanes are the query
        # and the second head_dim lanes the gate.
        qg = self.q_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_heads, 2 * self.head_dim
        )
        q, gate = qg[..., : self.head_dim], qg[..., self.head_dim :]
        k = self.k_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        new_cache = None
        if cache is not None:
            assert cache_position is not None
            new_cache = cache.update(cache_position, k, v)
            k, v = new_cache.get()
        else:
            cache_position = None

        attn_output = masked_causal_attention(
            q, k, v, cache_position=cache_position, kv_mask=kv_mask
        )

        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = attn_output * jax.nn.sigmoid(
            gate.reshape(batch_size, seq_len, -1)
        )
        output = self.o_proj(attn_output)
        return output, new_cache


__all__ = ["Qwen35Attention"]
