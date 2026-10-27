"""Llama-architecture decoder (HF `LlamaForCausalLM`).

Structurally the Qwen3 text decoder without q/k norm and with plain 1-D RoPE:
GQA attention, SwiGLU MLP, pre-norm RMSNorm blocks. The module tree mirrors
HF's names, so an HF state dict loads with no key translation.

RoPE uses HF's rotate-half convention. llama.cpp's converter permutes the
q/k projection rows of `llama`-arch GGUFs into its interleaved-pairs
convention, so a GGUF loader has to undo that before the weights land here.
"""
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from qwen_jax import causal_lm
from qwen_jax.attention import masked_causal_attention
from qwen_jax.cache import KVCache, KVCacheLayer
from qwen_jax.linear import Embedding, Linear, RMSNorm
from qwen_jax.mlp import Qwen3VLTextMLP
from qwen_jax.rope import apply_rotary_pos_emb
from qwen_jax.utils.pjit import pjit

from .config import LlamaConfig


@jax.tree_util.register_dataclass
@dataclass
class LlamaOutput:
    """Output from LlamaForCausalLM.

    With ``last_logit_only=True`` only ``last_logits`` is populated (logits at
    the last non-padded position per batch item); otherwise ``logits`` holds
    the full ``(batch, seq, vocab)`` tensor.
    """

    logits: Float[Array, "batch seq vocab"] | None
    hidden_states: Float[Array, "batch seq hidden"]
    cache: KVCache | None = None
    last_logits: Float[Array, "batch vocab"] | None = None


class LlamaRotaryEmbedding(eqx.Module):
    """Standard (unscaled) RoPE over 1-D positions."""

    # Derived from config, not loaded: a bare Array rather than a Param.
    inv_freq: Float[Array, "head_dim_half"]

    def __init__(self, config: LlamaConfig):
        dim = config.head_dim
        self.inv_freq = 1.0 / (
            config.rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim)
        )

    def __call__(
        self, position_ids: Int[Array, "batch seq"]
    ) -> tuple[Float[Array, "batch seq head_dim"], Float[Array, "batch seq head_dim"]]:
        freqs = position_ids.astype(jnp.float32)[..., None] * self.inv_freq
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return jnp.cos(emb), jnp.sin(emb)


class LlamaAttention(eqx.Module):
    """Grouped-query attention with RoPE and an optional KV cache."""

    num_heads: int = eqx.field(static=True)
    num_kv_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear

    def __init__(self, config: LlamaConfig):
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        bias = config.attention_bias
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = Linear(config.hidden_size, q_dim, use_bias=bias)
        self.k_proj = Linear(config.hidden_size, kv_dim, use_bias=bias)
        self.v_proj = Linear(config.hidden_size, kv_dim, use_bias=bias)
        self.o_proj = Linear(q_dim, config.hidden_size, use_bias=bias)

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: tuple[
            Float[Array, "batch seq head_dim"], Float[Array, "batch seq head_dim"]
        ],
        cache: KVCacheLayer | None = None,
        cache_position: Int[Array, ""] | None = None,
        kv_mask: Bool[Array, "batch kv_seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], KVCacheLayer | None]:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

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
        output = self.o_proj(attn_output.reshape(batch_size, seq_len, -1))
        return output, new_cache


class LlamaDecoderLayer(eqx.Module):
    """Pre-norm decoder layer: attention then SwiGLU MLP, each residual."""

    input_layernorm: RMSNorm
    self_attn: LlamaAttention
    post_attention_layernorm: RMSNorm
    mlp: Qwen3VLTextMLP

    def __init__(self, config: LlamaConfig):
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3VLTextMLP(config.hidden_size, config.intermediate_size)

    @jax.remat
    def __call__(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        position_embeddings: tuple[Array, Array],
        cache: KVCacheLayer | None = None,
        cache_position: Int[Array, ""] | None = None,
        kv_mask: Bool[Array, "batch kv_seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], KVCacheLayer | None]:
        residual = hidden_states
        hidden_states, new_cache = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            cache=cache,
            cache_position=cache_position,
            kv_mask=kv_mask,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, new_cache


class LlamaModel(eqx.Module):
    """The decoder stack: embeddings, layers, final norm."""

    config: LlamaConfig = eqx.field(static=True)

    embed_tokens: Embedding
    layers: tuple[LlamaDecoderLayer, ...]
    norm: RMSNorm
    rotary_emb: LlamaRotaryEmbedding

    def __init__(self, config: LlamaConfig):
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = tuple(LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)

    def __call__(
        self,
        input_ids: Int[Array, "batch seq"] | None = None,
        inputs_embeds: Float[Array, "batch seq hidden"] | None = None,
        position_ids: Int[Array, "batch seq"] | None = None,
        cache: KVCache | None = None,
        cache_position: Int[Array, ""] | None = None,
        kv_mask: Bool[Array, "batch kv_seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], KVCache | None]:
        """Forward pass.

        Positions default to the index (offset by `cache_position`). RoPE
        scores depend only on position differences, so left padding needs no
        per-row offset: the padded slots are masked out by `kv_mask`.
        """
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_len, _ = inputs_embeds.shape

        if position_ids is None:
            pos = jnp.arange(seq_len)
            if cache_position is not None:
                pos = pos + cache_position
            position_ids = jnp.broadcast_to(pos[None, :], (batch_size, seq_len))

        position_embeddings = jax.tree.map(
            lambda x: x.astype(inputs_embeds.dtype), self.rotary_emb(position_ids)
        )

        hidden_states = inputs_embeds
        new_layers = []
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, new_layer = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                cache=cache.layers[layer_idx] if cache is not None else None,
                cache_position=cache_position,
                kv_mask=kv_mask,
            )
            new_layers.append(new_layer)
        hidden_states = self.norm(hidden_states)

        new_cache = None
        if cache is not None:
            new_cache = KVCache(layers=tuple(new_layers), position=cache.position + seq_len)
        return hidden_states, new_cache


class LlamaForCausalLM(eqx.Module):
    """Llama decoder with a language-model head."""

    config: LlamaConfig = eqx.field(static=True)

    model: LlamaModel
    # None when weights are tied; get_lm_head() resolves that.
    lm_head: Linear | None

    def __init__(self, config: LlamaConfig):
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = (
            None
            if config.tie_word_embeddings
            else Linear(config.hidden_size, config.vocab_size, use_bias=False)
        )

    def get_lm_head(self) -> Linear:
        return causal_lm.lm_head(self.lm_head, self.model.embed_tokens)

    def create_cache(self, batch_size: int, max_seq_len: int, dtype=None) -> KVCache:
        config = self.config
        return KVCache.create(
            num_layers=config.num_hidden_layers,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=dtype or causal_lm.cache_dtype(self.model.embed_tokens),
        )

    @pjit(static_argnames=("use_cache", "last_logit_only"))
    def __call__(
        self,
        input_ids: Int[Array, "batch seq"],
        attention_mask: Int[Array, "batch seq"] | None = None,
        position_ids: Int[Array, "batch seq"] | None = None,
        cache: KVCache | None = None,
        use_cache: bool = False,
        last_logit_only: bool = False,
    ) -> LlamaOutput:
        """Forward pass.

        Args:
            attention_mask: padding mask (1 = real token). With a cache it may
                cover the cache's whole capacity; shorter masks are extended
                with ones.
            cache: KV cache to continue from; the input is written at
                ``cache.position``.
            use_cache: create a cache sized to the input when none is given.
            last_logit_only: compute logits only at the last non-padded
                position (``LlamaOutput.last_logits``), skipping the lm_head
                over the whole prompt.
        """
        batch_size, seq_len = input_ids.shape
        if use_cache and cache is None:
            cache = self.create_cache(batch_size, seq_len)

        kv_len = cache.max_seq_len if cache is not None else seq_len
        kv_mask = causal_lm.pad_mask(attention_mask, batch_size, kv_len)[:, :kv_len]

        hidden_states, new_cache = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            cache=cache,
            cache_position=cache.position if cache is not None else None,
            kv_mask=kv_mask.astype(jnp.bool),
        )

        if last_logit_only:
            last = causal_lm.last_hidden(hidden_states, attention_mask)
            return LlamaOutput(
                logits=None,
                hidden_states=hidden_states,
                cache=new_cache,
                last_logits=self.get_lm_head()(last),
            )
        return LlamaOutput(
            logits=self.get_lm_head()(hidden_states),
            hidden_states=hidden_states,
            cache=new_cache,
        )

    @pjit(static_argnames=("max_new_tokens", "progress_bar", "return_logits"))
    def generate(
        self,
        input_ids: Int[Array, "batch seq"],
        attention_mask: Int[Array, "batch seq"] | None = None,
        *,
        cache: KVCache | None = None,
        max_new_tokens: int,
        key: PRNGKeyArray,
        temperature: float = 1.0,
        progress_bar: bool = True,
        return_logits: bool = False,
        stop_token_id: int = -1,
        pad_token_id: int = 0,
    ) -> causal_lm.GenerateOutput:
        """Sample `max_new_tokens` tokens after the prompt; see `causal_lm.decode`."""
        batch_size, prompt_len = input_ids.shape
        max_seq_len = prompt_len + max_new_tokens
        if cache is None:
            cache = self.create_cache(batch_size, max_seq_len)
        attention_mask = causal_lm.pad_mask(attention_mask, batch_size, max_seq_len)

        prefill = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache=cache,
            last_logit_only=True,
        )
        assert prefill.last_logits is not None and prefill.cache is not None

        def step(cache, token):
            out = self(input_ids=token[:, None], attention_mask=attention_mask, cache=cache)
            return out.logits[:, -1, :], out.cache

        tokens, cache, logits = causal_lm.decode(
            step,
            prefill.last_logits,
            prefill.cache,
            key=key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_token_id=stop_token_id,
            pad_token_id=pad_token_id,
            return_logits=return_logits,
            progress_bar=progress_bar,
        )
        return causal_lm.GenerateOutput(
            tokens=jnp.concatenate([input_ids, tokens], axis=1), cache=cache, logits=logits
        )


__all__ = [
    "LlamaAttention",
    "LlamaDecoderLayer",
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaOutput",
    "LlamaRotaryEmbedding",
]
