"""Qwen3-VL model implementation in JAX/Equinox."""
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from transformers import Qwen3VLConfig

from . import causal_lm
from .cache import KVCache
from .config import (
    Qwen3VLConfig as Qwen3VLConfigModel,
)
from .linear import Linear
from .mrope import get_rope_index
from .text import Qwen3VLTextModel
from .utils.pjit import pjit
from .vision import Qwen3VLVisionModel


@jax.tree_util.register_dataclass
@dataclass
class Qwen3VLOutput:
    """Output from Qwen3VL model.

    When ``last_logit_only=True`` was passed to the forward call, only
    ``last_logits`` is populated — logits gathered at the last non-padded
    position per batch item, which avoids the lm_head matmul over the whole
    prompt. Otherwise ``logits`` holds the full ``(batch, seq, vocab)`` tensor
    and ``last_logits`` is None.
    """
    logits: Float[Array, "batch seq vocab"] | None
    hidden_states: Float[Array, "batch seq hidden"]
    rope_deltas: Int[Array, "batch 1"]
    cache: KVCache | None = None
    last_logits: Float[Array, "batch vocab"] | None = None


Qwen3VLGenerateOutput = causal_lm.GenerateOutput

class Qwen3VLModel(eqx.Module):
    """Qwen3-VL multimodal model.

    Combines vision encoder with text decoder. Vision embeddings replace
    placeholder tokens in the text sequence, and DeepStack injects
    intermediate vision features into early text layers.
    """
    # Config
    config: Qwen3VLConfigModel = eqx.field(static=True)

    # Models
    visual: Qwen3VLVisionModel
    language_model: Qwen3VLTextModel

    def __init__(
        self,
        config: Qwen3VLConfigModel,
    ):
        self.config = config

        self.visual = Qwen3VLVisionModel(
            config.vision_config,
        )

        self.language_model = Qwen3VLTextModel(
            config.text_config,
        )

    def get_image_features(
        self,
        pixel_values: Float[Array, "total_patches C*T*H*W"],
        image_grid_thw: Int[Array, "num_images 3"],
    ) -> tuple[Float[Array, "total_merged_tokens hidden"], tuple[Float[Array, "..."], ...]]:
        """Encode images into embeddings.

        Args:
            pixel_values: Raw image patches
            image_grid_thw: Grid dimensions for each image

        Returns:
            (image_embeds, deepstack_embeds) tuple
        """
        image_embeds, deepstack_embeds = self.visual(pixel_values, image_grid_thw)
        return image_embeds, deepstack_embeds

    def get_rope_index(
        self,
        input_ids: Int[Array, "batch seq"],
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        attention_mask: Float[Array, "batch seq"] | None = None,
        mm_token_type_ids: Int[Array, "batch seq"] | None = None,
    ) -> tuple[Int[Array, "3 batch seq"], Int[Array, "batch 1"]]:
        """Compute 3D position IDs for MRoPE. See `qwen_jax.mrope`."""
        return get_rope_index(
            input_ids,
            image_grid_thw,
            attention_mask,
            mm_token_type_ids,
            spatial_merge_size=self.config.vision_config.spatial_merge_size,
            image_token_id=self.config.image_token_id,
        )

    def _splice_image_embeds(
        self,
        input_ids: Int[Array, "batch seq"],
        pixel_values: Float[Array, "total_patches C*T*H*W"] | None,
        image_grid_thw: Int[Array, "num_images 3"] | None,
        mm_token_type_ids: Int[Array, "batch seq"] | None,
    ) -> tuple[
        Float[Array, "batch seq hidden"],
        Bool[Array, "batch seq"] | None,
        tuple[Float[Array, "..."], ...] | None,
    ]:
        """Embed text tokens and splice image embeddings at image positions."""
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        if pixel_values is None or image_grid_thw is None:
            return inputs_embeds, None, None

        # Prefer mm_token_type_ids when supplied so this matches get_rope_index.
        if mm_token_type_ids is not None:
            image_mask = (mm_token_type_ids == 1)
        else:
            image_mask = (input_ids == self.config.image_token_id)

        image_embeds, deepstack_image_embeds = self.get_image_features(
            pixel_values, image_grid_thw
        )
        batch_size = input_ids.shape[0]
        image_idx = jnp.cumsum(jnp.reshape(image_mask, [-1])) - 1
        image_embeds_i = rearrange(
            image_embeds[image_idx],
            "(b seq) hidden -> b seq hidden",
            b=batch_size,
        )
        inputs_embeds = jnp.where(image_mask[..., None], image_embeds_i, inputs_embeds)
        return inputs_embeds, image_mask, deepstack_image_embeds

    def _resolve_position_ids(
        self,
        input_ids: Int[Array, "batch seq"],
        image_grid_thw: Int[Array, "num_images 3"] | None,
        attention_mask: Float[Array, "batch seq"] | None,
        mm_token_type_ids: Int[Array, "batch seq"] | None,
        position_ids: Int[Array, "3 batch seq"] | None,
        rope_deltas: Int[Array, "batch 1"] | None,
        cache_position: Int[Array, ""] | None,
    ) -> tuple[Int[Array, "3 batch seq"], Int[Array, "batch 1"], Int[Array, "batch 1"]]:
        """Resolve 3D position IDs.

        Returns (position_ids, prior_rope_deltas, new_rope_deltas). The two delta
        terms are returned separately so the caller can thread/sum as needed.
        """
        batch_size = input_ids.shape[0]
        if rope_deltas is None:
            rope_deltas = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        new_rope_deltas = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        if position_ids is None:
            position_ids, new_rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, attention_mask, mm_token_type_ids
            )
            position_ids += rope_deltas
            if cache_position is not None:
                position_ids += cache_position
        return position_ids, rope_deltas, new_rope_deltas

    def _build_attention_mask(
        self,
        attention_mask: Float[Array, "batch seq"] | None,
        cache: KVCache | None,
        cache_position: Int[Array, ""] | None,
        batch_size: int,
        seq_len: int,
    ) -> tuple[Bool[Array, "batch 1 seq kv_seq"], Bool[Array, "batch kv_seq"]]:
        """Build the 4-D causal+padding mask and a 2-D kv_mask for varlen attention.

        attention_mask defaults to all-ones when None — both downstream consumers
        (causal mask and varlen kv_mask) require it.
        """
        if cache is not None:
            kv_len = cache.max_seq_len
            past_seen_tokens = cache_position if cache_position is not None else 0
        else:
            kv_len = seq_len
            past_seen_tokens = 0

        q_positions = jnp.arange(seq_len) + past_seen_tokens
        k_positions = jnp.arange(kv_len)
        causal_mask = (k_positions[None, :] <= q_positions[:, None])[None, None, :, :]

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, kv_len), dtype=jnp.int32)
        elif attention_mask.shape[-1] < kv_len:
            attention_mask = jnp.pad(
                attention_mask,
                ((0, 0), (0, kv_len - attention_mask.shape[-1])),
                constant_values=1,
            )

        kv_mask = attention_mask[:, :kv_len].astype(jnp.bool)
        causal_mask = causal_mask & kv_mask[:, None, None, :]
        return causal_mask, kv_mask

    def __call__(
        self,
        input_ids: Int[Array, "batch seq"],
        pixel_values: Float[Array, "total_patches C*T*H*W"] | None = None,
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        attention_mask: Float[Array, "batch seq"] | None = None,
        position_ids: Int[Array, "3 batch seq"] | None = None,
        cache: KVCache | None = None,
        cache_position: Int[Array, ""] | None = None,
        rope_deltas: Int[Array, "batch 1"] | None = None,
        mm_token_type_ids: Int[Array, "batch seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], KVCache | None, Int[Array, "batch 1"]]:
        """Forward pass.

        Returns:
            (hidden_states, new_cache, rope_deltas) tuple
        """
        batch_size, seq_len = input_ids.shape

        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._splice_image_embeds(
            input_ids, pixel_values, image_grid_thw, mm_token_type_ids,
        )

        position_ids, rope_deltas, new_rope_deltas = self._resolve_position_ids(
            input_ids, image_grid_thw, attention_mask, mm_token_type_ids,
            position_ids, rope_deltas, cache_position,
        )

        causal_mask, kv_mask = self._build_attention_mask(
            attention_mask, cache, cache_position, batch_size, seq_len,
        )

        hidden_states, new_cache = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=causal_mask,
            cache=cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            kv_mask=kv_mask,
        )

        return hidden_states, new_cache, (rope_deltas + new_rope_deltas)


class Qwen3VLForConditionalGeneration(eqx.Module):
    """Qwen3-VL for conditional text generation.

    Wraps Qwen3VLModel with a language model head for next-token prediction.
    """
    # Config
    config: Qwen3VLConfigModel = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)

    # Models
    model: Qwen3VLModel

    # None when weights are tied; get_lm_head() resolves that.
    lm_head: Linear | None

    def __init__(
        self,
        config: Qwen3VLConfig | Qwen3VLConfigModel,
    ):
        if isinstance(config, Qwen3VLConfig):
            config = Qwen3VLConfigModel.model_validate(config.to_dict())
        self.config = config
        tc = config.text_config
        self.vocab_size = tc.vocab_size

        self.model = Qwen3VLModel(config)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(self.model.language_model.config.hidden_size, self.vocab_size, use_bias=False)

    def get_lm_head(self) -> Linear:
        """The output projection, borrowing the input embedding when tied."""
        return causal_lm.lm_head(self.lm_head, self.model.language_model.embed_tokens)

    def create_cache(self, batch_size: int, max_seq_len: int, dtype=None) -> KVCache:
        text_config = self.config.text_config
        return KVCache.create(
            num_layers=text_config.num_hidden_layers,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=text_config.num_key_value_heads,
            head_dim=text_config.head_dim,
            dtype=dtype or causal_lm.cache_dtype(self.model.language_model.embed_tokens),
        )

    @pjit(static_argnames=("use_cache", "last_logit_only"))
    def __call__(
        self,
        input_ids: Int[Array, "batch seq"],
        pixel_values: Float[Array, "total_patches C*T*H*W"] | None = None,
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        attention_mask: Float[Array, "batch seq"] | None = None,
        position_ids: Int[Array, "3 batch seq"] | None = None,
        cache: KVCache | None = None,
        rope_deltas: Int[Array, "batch 1"] | None = None,
        use_cache: bool = False,
        mm_token_type_ids: Int[Array, "batch seq"] | None = None,
        last_logit_only: bool = False,
    ) -> Qwen3VLOutput:
        """Forward pass.

        Args:
            last_logit_only: If True, skip the full lm_head projection over the
                prompt and only compute logits at the last non-padded position
                per batch item — populates ``Qwen3VLOutput.last_logits`` and
                leaves ``Qwen3VLOutput.logits`` as None.

        Returns:
            Qwen3VLOutput with hidden_states, cache, rope_deltas, plus either
            logits (full) or last_logits ((batch, vocab)) depending on the flag.
        """
        if use_cache and cache is None:
            cache = self.create_cache(input_ids.shape[0], input_ids.shape[1])

        hidden_states, new_cache, new_rope_deltas = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            cache_position=cache.position if cache is not None else None,
            rope_deltas=rope_deltas,
            mm_token_type_ids=mm_token_type_ids,
        )

        if last_logit_only:
            last_hidden = causal_lm.last_hidden(hidden_states, attention_mask)
            last_logits = self.get_lm_head()(last_hidden)
            return Qwen3VLOutput(
                logits=None,
                hidden_states=hidden_states,
                cache=new_cache,
                rope_deltas=new_rope_deltas,
                last_logits=last_logits,
            )

        logits = self.get_lm_head()(hidden_states)
        return Qwen3VLOutput(
            logits=logits,
            hidden_states=hidden_states,
            cache=new_cache,
            rope_deltas=new_rope_deltas,
        )

    @pjit(static_argnames=("max_new_tokens", "progress_bar", "return_logits"))
    def generate(
        self,
        input_ids: Int[Array, "batch seq"],
        pixel_values: Float[Array, "total_patches C*T*H*W"] | None = None,
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        attention_mask: Float[Array, "batch seq"] | None = None,
        *,
        cache: KVCache | None = None,
        rope_deltas: Int[Array, "batch 1"] | None = None,
        max_new_tokens: int,
        key: PRNGKeyArray,
        temperature: float = 1.0,
        progress_bar: bool = True,
        return_logits: bool = False,
        stop_token_id: int = -1,
        pad_token_id: int = 0,
        mm_token_type_ids: Int[Array, "batch seq"] | None = None,
    ) -> Qwen3VLGenerateOutput:
        """Sample `max_new_tokens` tokens after the prompt; see `causal_lm.decode`.

        Args:
            input_ids: Initial token IDs (prompt).
            pixel_values: Image patches (required if images in prompt).
            image_grid_thw: Grid dimensions for each image (T, H, W).
            attention_mask: Padding mask for prompt (batch, seq).
            max_new_tokens: Number of tokens to generate.
            key: PRNG key for sampling.
            temperature: Sampling temperature (0 = greedy).
            progress_bar: Print each decode step.

        Returns:
            GenerateOutput: prompt + generated tokens, the final cache, and
            (with return_logits) the generated positions' logits.
        """
        batch_size, prompt_len = input_ids.shape
        max_seq_len = prompt_len + max_new_tokens
        if cache is None:
            cache = self.create_cache(batch_size, max_seq_len)
        attention_mask = causal_lm.pad_mask(attention_mask, batch_size, max_seq_len)

        # Prefill: the whole prompt, images included, projecting only the last
        # real position through the lm_head.
        prefill = self(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            cache=cache,
            rope_deltas=rope_deltas,
            mm_token_type_ids=mm_token_type_ids,
            last_logit_only=True,
        )
        assert prefill.last_logits is not None and prefill.cache is not None

        def step(cache, token):
            # Images are already embedded in the cache.
            out = self(
                input_ids=token[:, None],
                attention_mask=attention_mask,
                cache=cache,
                rope_deltas=prefill.rope_deltas,
            )
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
        return Qwen3VLGenerateOutput(
            tokens=jnp.concatenate([input_ids, tokens], axis=1), cache=cache, logits=logits
        )


__all__ = ["Qwen3VLForConditionalGeneration", "Qwen3VLModel", "Qwen3VLOutput"]
