"""Text model components for Qwen3-VL."""
from qwen_jax.config import Qwen3VLTextConfig
from einops import rearrange
from dataclasses import field
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from . import equinox_utils as eu
from .cache import KVCache, KVCacheLayer
from .utils.rng import split

from .attention import Qwen3VLTextAttention
from .linear import Embedding, Linear, RMSNorm
from .mlp import Qwen3VLTextMLP
from .rope import Qwen3VLTextRotaryEmbedding


class Qwen3VLTextDecoderLayer(eqx.Module):
    """Qwen3-VL text decoder layer with pre-norm."""
    hidden_size: int = field(metadata=dict(static=True))
    layer_idx: int = field(metadata=dict(static=True))

    input_layernorm: RMSNorm
    self_attn: Qwen3VLTextAttention
    post_attention_layernorm: RMSNorm
    mlp: Qwen3VLTextMLP

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        layer_idx: int = 0,
    ):
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3VLTextAttention(
            config=config,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3VLTextMLP(config.hidden_size, config.intermediate_size)

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        return eu.replace(
            self,
            input_layernorm=self.input_layernorm.load_state_dict(
                state_dict, prefix + "input_layernorm."
            ),
            self_attn=self.self_attn.load_state_dict(state_dict, prefix + "self_attn."),
            post_attention_layernorm=self.post_attention_layernorm.load_state_dict(
                state_dict, prefix + "post_attention_layernorm."
            ),
            mlp=self.mlp.load_state_dict(state_dict, prefix + "mlp."),
        )

    @jax.remat
    def __call__(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: tuple[Float[Array, "batch seq head_dim"], Float[Array, "batch seq head_dim"]],
        attention_mask: Optional[Float[Array, "batch 1 seq kv_seq"]] = None,
        cache: Optional[KVCacheLayer] = None,
        cache_position: Optional[Int[Array, ""]] = None,
        kv_mask: Optional[Bool[Array, "batch kv_seq"]] = None,
    ) -> tuple[Float[Array, "batch seq hidden"], Optional[KVCacheLayer]]:
        """Forward pass.

        Args:
            hidden_states: Input (batch, seq, hidden)
            position_embeddings: (cos, sin) from MRoPE
            attention_mask: Causal attention mask
            cache: Optional KV cache layer
            cache_position: Position in cache

        Returns:
            (output, new_cache) tuple
        """
        # Pre-norm self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_cache = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            cache=cache,
            cache_position=cache_position,
            kv_mask=kv_mask,
        )
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_cache


class Qwen3VLTextModel(eqx.Module):
    """Qwen3-VL text decoder with DeepStack integration.

    DeepStack injects intermediate visual features from the vision encoder
    into early layers of the text model.
    """
    # Config
    config: Qwen3VLTextConfig = field(metadata=dict(static=True))

    # Layers
    embed_tokens: Embedding
    layers: tuple[Qwen3VLTextDecoderLayer, ...]
    norm: RMSNorm
    rotary_emb: Qwen3VLTextRotaryEmbedding

    def __init__(
        self,
        config: Qwen3VLTextConfig,
    ):
        self.config = config
        self.embed_tokens = Embedding(self.config.vocab_size, self.config.hidden_size)
        layers = []
        for layer_idx in range(self.config.num_hidden_layers):
            layers.append(
                Qwen3VLTextDecoderLayer(
                    config=self.config,
                    layer_idx=layer_idx,
                )
            )
        self.layers = tuple(layers)
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(
            config=self.config,
        )

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str = ""):
        return eu.replace(
            self,
            embed_tokens=self.embed_tokens.load_state_dict(state_dict, prefix + "embed_tokens."),
            layers=tuple(
                self.layers[i].load_state_dict(state_dict, prefix + f"layers.{i}.")
                for i in range(len(self.layers))
            ),
            norm=self.norm.load_state_dict(state_dict, prefix + "norm."),
            # rotary emb has no parameters
        )

    def _deepstack_process(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        visual_pos_masks: Bool[Array, "batch seq"],
        visual_embeds: Float[Array, "num_visual_tokens hidden"],
    ) -> Float[Array, "batch seq hidden"]:
        """Inject visual embeddings at visual token positions.

        Args:
            hidden_states: Current hidden states (batch, seq, hidden)
            visual_pos_masks: Boolean mask for visual positions (batch, seq)
            visual_embeds: Visual embeddings to add (num_visual_tokens, hidden)

        Returns:
            Updated hidden states
        """
        visual_idx = jnp.cumsum(visual_pos_masks.flatten()) - 1
        gathered = rearrange(
            visual_embeds[visual_idx],
            '(batch seq) hidden -> batch seq hidden',
            batch=hidden_states.shape[0],
        )
        # Add visual embeddings at masked positions
        # This is element-wise addition
        hidden_states = jnp.where(
            visual_pos_masks[..., None],
            hidden_states + gathered,
            hidden_states
        )
        return hidden_states

    # @jax.remat
    def __call__(
        self,
        input_ids: Optional[Int[Array, "batch seq"]] = None,
        inputs_embeds: Optional[Float[Array, "batch seq hidden"]] = None,
        position_ids: Optional[Int[Array, "3 batch seq"]] = None,
        attention_mask: Optional[Float[Array, "batch 1 seq kv_seq"]] = None,
        cache: Optional[KVCache] = None,
        cache_position: Optional[Int[Array, ""]] = None,
        visual_pos_masks: Optional[Float[Array, "batch seq"]] = None,
        deepstack_visual_embeds: Optional[tuple[Float[Array, "..."], ...]] = None,
        kv_mask: Optional[Bool[Array, "batch kv_seq"]] = None,
    ) -> tuple[Float[Array, "batch seq hidden"], Optional[KVCache]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq). Either this or inputs_embeds required.
            inputs_embeds: Pre-computed embeddings (batch, seq, hidden)
            position_ids: 3D position IDs for MRoPE (3, batch, seq)
            attention_mask: Causal mask (batch, 1, seq, kv_seq)
            cache: Optional KV cache
            cache_position: Position in cache
            visual_pos_masks: Boolean mask for visual token positions (batch, seq)
            deepstack_visual_embeds: Tuple of visual embeddings for DeepStack

        Returns:
            (hidden_states, new_cache) tuple
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_len, _ = inputs_embeds.shape

        # Handle position IDs
        if position_ids is None:
            if cache_position is not None:
                # Use cache position for all 3 dimensions
                pos = jnp.arange(seq_len) + cache_position
            else:
                pos = jnp.arange(seq_len)
            position_ids = jnp.broadcast_to(pos[None, None, :], (3, batch_size, seq_len))

        # Compute position embeddings (MRoPE)
        position_embeddings = self.rotary_emb(position_ids)
        position_embeddings = jax.tree.map(lambda x: x.astype(self.embed_tokens.weight.dtype), position_embeddings)

        hidden_states = inputs_embeds

        # Process through layers
        new_cache_layers = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = cache.layers[layer_idx] if cache is not None else None

            hidden_states, new_layer_cache = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                cache=layer_cache,
                cache_position=cache_position,
                kv_mask=kv_mask,
            )

            if new_layer_cache is not None:
                new_cache_layers.append(new_layer_cache)

            # DeepStack: inject visual features after early layers
            if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)

        # Build new cache
        new_cache = None
        if cache is not None:
            new_position = cache.position + seq_len
            new_cache = KVCache(layers=tuple(new_cache_layers), position=new_position)

        return hidden_states, new_cache


__all__ = ["Qwen3VLTextDecoderLayer", "Qwen3VLTextModel"]
