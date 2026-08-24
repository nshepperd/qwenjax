"""Qwen3.5 text decoder: hybrid DeltaNet / full-attention stack."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from qwen_jax.cache import KVCacheLayer
from qwen_jax.linear import Embedding, RMSNorm
from qwen_jax.mlp import Qwen3VLTextMLP
from qwen_jax.rope import Qwen3VLTextRotaryEmbedding

from .attention import Qwen35Attention
from .cache import DeltaNetCacheLayer, Qwen35Cache
from .config import Qwen35TextConfig
from .deltanet import Qwen35GatedDeltaNet


class Qwen35DecoderLayer(eqx.Module):
    """Pre-norm decoder layer whose token mixer is set by layer_type.

    Exactly one of linear_attn / self_attn is present, mirroring how HF (and
    the state dict keys) name the two variants.
    """

    layer_type: str = eqx.field(static=True)
    input_layernorm: RMSNorm
    linear_attn: Qwen35GatedDeltaNet | None
    self_attn: Qwen35Attention | None
    post_attention_layernorm: RMSNorm
    mlp: Qwen3VLTextMLP

    def __init__(self, config: Qwen35TextConfig, layer_type: str):
        assert layer_type in ("linear_attention", "full_attention")
        self.layer_type = layer_type
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.linear_attn = (
            Qwen35GatedDeltaNet(config) if layer_type == "linear_attention" else None
        )
        self.self_attn = (
            Qwen35Attention(config) if layer_type == "full_attention" else None
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3VLTextMLP(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: tuple[Array, Array],
        cache: KVCacheLayer | DeltaNetCacheLayer | None = None,
        cache_position: Int[Array, ""] | None = None,
        kv_mask: Bool[Array, "batch kv_seq"] | None = None,
        token_mask: Bool[Array, "batch seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], KVCacheLayer | DeltaNetCacheLayer | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.linear_attn is not None:
            assert cache is None or isinstance(cache, DeltaNetCacheLayer)
            hidden_states, new_cache = self.linear_attn(
                hidden_states, cache=cache, token_mask=token_mask
            )
        else:
            assert self.self_attn is not None
            assert cache is None or isinstance(cache, KVCacheLayer)
            hidden_states, new_cache = self.self_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                cache=cache,
                cache_position=cache_position,
                kv_mask=kv_mask,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache


class Qwen35TextModel(eqx.Module):
    """Qwen3.5 text decoder."""

    config: Qwen35TextConfig = eqx.field(static=True)

    embed_tokens: Embedding
    layers: tuple[Qwen35DecoderLayer, ...]
    norm: RMSNorm
    rotary_emb: Qwen3VLTextRotaryEmbedding

    def __init__(self, config: Qwen35TextConfig):
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = tuple(
            Qwen35DecoderLayer(config, layer_type)
            for layer_type in config.resolved_layer_types()
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Duck-typed on head_dim / rope_theta / rope_scaling; the partial
        # rotary factor in rope_scaling sizes cos/sin to the rotary dim.
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)  # type: ignore[arg-type]

    def __call__(
        self,
        input_ids: Int[Array, "batch seq"] | None = None,
        inputs_embeds: Float[Array, "batch seq hidden"] | None = None,
        position_ids: Int[Array, "3 batch seq"] | None = None,
        cache: Qwen35Cache | None = None,
        cache_position: Int[Array, ""] | None = None,
        kv_mask: Bool[Array, "batch kv_seq"] | None = None,
        token_mask: Bool[Array, "batch seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], Qwen35Cache | None]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq). Either this or inputs_embeds.
            inputs_embeds: Pre-computed embeddings (batch, seq, hidden).
            position_ids: 3D MRoPE positions (3, batch, seq).
            cache: Hybrid cache to continue from.
            cache_position: Where these tokens sit in the KV cache.
            kv_mask: Validity over the whole KV length (full-attention mask).
            token_mask: Validity of the current tokens (DeltaNet padding).

        Returns:
            (hidden_states, new_cache) tuple.
        """
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape

        if position_ids is None:
            if cache_position is not None:
                pos = jnp.arange(seq_len) + cache_position
            else:
                pos = jnp.arange(seq_len)
            position_ids = jnp.broadcast_to(pos[None, None, :], (3, batch_size, seq_len))

        position_embeddings = self.rotary_emb(position_ids)
        position_embeddings = jax.tree.map(
            lambda x: x.astype(inputs_embeds.dtype), position_embeddings
        )

        hidden_states = inputs_embeds

        new_cache_layers = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = cache.layers[layer_idx] if cache is not None else None
            hidden_states, new_layer_cache = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cache=layer_cache,
                cache_position=cache_position,
                kv_mask=kv_mask,
                token_mask=token_mask,
            )
            if new_layer_cache is not None:
                new_cache_layers.append(new_layer_cache)

        hidden_states = self.norm(hidden_states)

        new_cache = None
        if cache is not None:
            assert len(new_cache_layers) == len(self.layers)
            new_cache = Qwen35Cache(
                layers=tuple(new_cache_layers), position=cache.position + seq_len
            )

        return hidden_states, new_cache


__all__ = ["Qwen35DecoderLayer", "Qwen35TextModel"]
