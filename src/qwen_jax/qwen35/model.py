"""Qwen3.5 model: hybrid text decoder plus the (DeepStack-free) vision tower.

Structurally a sibling of `qwen_jax.model.Qwen3VLForConditionalGeneration`;
the differences are the hybrid cache, the extra token_mask plumbed to the
DeltaNet layers, and the absence of DeepStack injection.
"""
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from qwen_jax import equinox_utils as eu
from qwen_jax.linear import Linear
from qwen_jax.mrope import get_rope_index
from qwen_jax.utils.indexing import gather
from qwen_jax.utils.pjit import pjit
from qwen_jax.vision import Qwen3VLVisionModel

from .cache import Qwen35Cache
from .config import Qwen35Config
from .text import Qwen35TextModel


@jax.tree_util.register_dataclass
@dataclass
class Qwen35Output:
    """Output from the Qwen3.5 model; see Qwen3VLOutput for the field notes."""

    logits: Float[Array, "batch seq vocab"] | None
    hidden_states: Float[Array, "batch seq hidden"]
    rope_deltas: Int[Array, "batch 1"]
    cache: Qwen35Cache | None = None
    last_logits: Float[Array, "batch vocab"] | None = None


@jax.tree_util.register_dataclass
@dataclass
class Qwen35GenerateOutput:
    tokens: Int[Array, "batch seq"]
    cache: Qwen35Cache | None = None
    logits: Float[Array, "batch seq vocab"] | None = None


class Qwen35Model(eqx.Module):
    """Qwen3.5 multimodal model (vision encoder + hybrid text decoder)."""

    config: Qwen35Config = eqx.field(static=True)

    visual: Qwen3VLVisionModel
    language_model: Qwen35TextModel

    def __init__(self, config: Qwen35Config):
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen35TextModel(config.text_config)

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
    ) -> Float[Array, "batch seq hidden"]:
        """Embed text tokens and splice image embeddings at image positions."""
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        if pixel_values is None or image_grid_thw is None:
            return inputs_embeds

        if mm_token_type_ids is not None:
            image_mask = (mm_token_type_ids == 1)
        else:
            image_mask = (input_ids == self.config.image_token_id)

        # Qwen3.5 has no DeepStack; the intermediate features come back empty.
        image_embeds, _deepstack = self.visual(pixel_values, image_grid_thw)
        batch_size = input_ids.shape[0]
        image_idx = jnp.cumsum(jnp.reshape(image_mask, [-1])) - 1
        image_embeds_i = rearrange(
            image_embeds[image_idx],
            "(b seq) hidden -> b seq hidden",
            b=batch_size,
        )
        return jnp.where(image_mask[..., None], image_embeds_i, inputs_embeds)

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

    def _build_masks(
        self,
        attention_mask: Float[Array, "batch seq"] | None,
        cache: Qwen35Cache | None,
        cache_position: Int[Array, ""] | None,
        batch_size: int,
        seq_len: int,
    ) -> tuple[Bool[Array, "batch kv_seq"], Bool[Array, "batch seq"]]:
        """The 2-D kv_mask over the KV length, and the current tokens' mask.

        kv_mask feeds the full-attention layers (varlen flash attention);
        token_mask is the slice of it covering just these positions, which is
        what the DeltaNet layers need to zero out padding.
        """
        if cache is not None:
            kv_len = cache.max_seq_len
        else:
            kv_len = seq_len

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, kv_len), dtype=jnp.int32)
        elif attention_mask.shape[-1] < kv_len:
            attention_mask = jnp.pad(
                attention_mask,
                ((0, 0), (0, kv_len - attention_mask.shape[-1])),
                constant_values=1,
            )

        kv_mask = attention_mask[:, :kv_len].astype(jnp.bool)
        if cache is not None and cache_position is not None:
            token_mask = jax.lax.dynamic_slice(
                kv_mask, (0, cache_position), (batch_size, seq_len)
            )
        else:
            token_mask = kv_mask[:, :seq_len]
        return kv_mask, token_mask

    def __call__(
        self,
        input_ids: Int[Array, "batch seq"],
        pixel_values: Float[Array, "total_patches C*T*H*W"] | None = None,
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        attention_mask: Float[Array, "batch seq"] | None = None,
        position_ids: Int[Array, "3 batch seq"] | None = None,
        cache: Qwen35Cache | None = None,
        cache_position: Int[Array, ""] | None = None,
        rope_deltas: Int[Array, "batch 1"] | None = None,
        mm_token_type_ids: Int[Array, "batch seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], Qwen35Cache | None, Int[Array, "batch 1"]]:
        batch_size, seq_len = input_ids.shape

        inputs_embeds = self._splice_image_embeds(
            input_ids, pixel_values, image_grid_thw, mm_token_type_ids,
        )

        position_ids, rope_deltas, new_rope_deltas = self._resolve_position_ids(
            input_ids, image_grid_thw, attention_mask, mm_token_type_ids,
            position_ids, rope_deltas, cache_position,
        )

        kv_mask, token_mask = self._build_masks(
            attention_mask, cache, cache_position, batch_size, seq_len,
        )

        hidden_states, new_cache = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cache=cache,
            cache_position=cache_position,
            kv_mask=kv_mask,
            token_mask=token_mask,
        )

        return hidden_states, new_cache, (rope_deltas + new_rope_deltas)


class Qwen35ForConditionalGeneration(eqx.Module):
    """Qwen3.5 with a language model head."""

    config: Qwen35Config = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)

    model: Qwen35Model
    lm_head: Linear | None

    def __init__(self, config: Qwen35Config):
        self.config = config
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen35Model(config)
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(
                config.text_config.hidden_size, self.vocab_size, use_bias=False
            )

    def get_lm_head(self) -> Linear:
        if self.config.tie_word_embeddings:
            return eu.replace(
                Linear(
                    in_features=self.config.text_config.hidden_size,
                    out_features=self.vocab_size,
                    use_bias=False,
                ),
                weight=self.model.language_model.embed_tokens.weight,
                bias=None,
            )
        assert self.lm_head is not None
        return self.lm_head

    def _create_cache(self, batch_size: int, max_seq_len: int) -> Qwen35Cache:
        cache_dtype = self.model.language_model.embed_tokens.weight().dtype
        if cache_dtype == jnp.float32:
            cache_dtype = jnp.bfloat16
        return Qwen35Cache.create(
            self.config.text_config,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            dtype=cache_dtype,
        )

    @pjit(static_argnames=("use_cache", "last_logit_only"))
    def __call__(
        self,
        input_ids: Int[Array, "batch seq"],
        pixel_values: Float[Array, "total_patches C*T*H*W"] | None = None,
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        attention_mask: Float[Array, "batch seq"] | None = None,
        position_ids: Int[Array, "3 batch seq"] | None = None,
        cache: Qwen35Cache | None = None,
        rope_deltas: Int[Array, "batch 1"] | None = None,
        use_cache: bool = False,
        mm_token_type_ids: Int[Array, "batch seq"] | None = None,
        last_logit_only: bool = False,
    ) -> Qwen35Output:
        if use_cache and cache is None:
            cache = self._create_cache(input_ids.shape[0], input_ids.shape[1])

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
            batch_size, seq_len = input_ids.shape
            if attention_mask is not None:
                mask_slice = attention_mask[:, :seq_len]
                positions = jnp.arange(seq_len)
                last_idx = jnp.max(
                    positions[None, :] * (mask_slice == 1).astype(jnp.int32),
                    axis=-1,
                )
            else:
                last_idx = jnp.full((batch_size,), seq_len - 1, dtype=jnp.int32)
            last_hidden = gather('b s h, b [s] -> b h', hidden_states, last_idx)
            last_logits = self.get_lm_head()(last_hidden)
            return Qwen35Output(
                logits=None,
                hidden_states=hidden_states,
                cache=new_cache,
                rope_deltas=new_rope_deltas,
                last_logits=last_logits,
            )

        logits = self.get_lm_head()(hidden_states)
        return Qwen35Output(
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
        cache: Qwen35Cache | None = None,
        rope_deltas: Int[Array, "batch 1"] | None = None,
        max_new_tokens: int,
        key: PRNGKeyArray,
        temperature: float = 1.0,
        progress_bar: bool = True,
        return_logits: bool = False,
        stop_token_id: int = -1,
        pad_token_id: int = 0,
        mm_token_type_ids: Int[Array, "batch seq"] | None = None,
    ) -> Qwen35GenerateOutput:
        """Generate tokens; see Qwen3VLForConditionalGeneration.generate."""
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        max_seq_len = prompt_len + max_new_tokens

        if cache is None:
            cache = self._create_cache(batch_size, max_seq_len)

        if attention_mask is None:
            attention_mask = jnp.ones([batch_size, max_seq_len], dtype=jnp.int32)
        elif attention_mask.shape[1] < max_seq_len:
            attention_mask = jnp.pad(
                attention_mask,
                ((0, 0), (0, max_seq_len - attention_mask.shape[1])),
                constant_values=1,
            )

        output = self(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            cache=cache,
            rope_deltas=rope_deltas,
            mm_token_type_ids=mm_token_type_ids,
            last_logit_only=True,
        )

        def sample(logits, key):
            return jnp.where(
                temperature > 0,
                jax.random.categorical(key, logits / temperature, axis=-1),
                jnp.argmax(logits, axis=-1),
            )

        assert output.last_logits is not None
        first_token_logits = output.last_logits
        key, subkey = jax.random.split(key)
        first_token = sample(first_token_logits, subkey)
        assert output.cache is not None
        cache = output.cache

        stop_active = stop_token_id >= 0
        done_init = stop_active & (first_token == stop_token_id)
        keys = jax.random.split(key, max_new_tokens)

        def while_step(carry):
            cache, token, step_idx, tokens, all_logits, done_mask = carry
            if progress_bar:
                jax.debug.print("Decoding step {}/{}", step_idx, max_new_tokens - 1)

            out = self(
                input_ids=token[:, None],
                pixel_values=None,
                image_grid_thw=None,
                attention_mask=attention_mask,
                cache=cache,
                rope_deltas=output.rope_deltas,
            )

            assert out.logits.shape[1] == 1
            logits = out.logits[:, -1, :]
            sampled = jnp.where(
                temperature > 0,
                jax.random.categorical(keys[step_idx], logits / temperature, axis=-1),
                jnp.argmax(logits, axis=-1),
            )
            next_token = jnp.where(done_mask, pad_token_id, sampled)
            new_done_mask = done_mask | (stop_active & (next_token == stop_token_id))

            if all_logits is not None:
                all_logits = all_logits.at[:, step_idx, :].set(logits)
            tokens = tokens.at[:, step_idx].set(next_token)

            return (out.cache, next_token, step_idx + 1, tokens, all_logits, new_done_mask)

        def while_cond(carry):
            _, _, step_idx, _, _, done_mask = carry
            all_done = stop_active & jnp.all(done_mask)
            return (~all_done) & (step_idx < max_new_tokens - 1)

        if max_new_tokens > 1:
            tokens = jnp.full((batch_size, max_new_tokens - 1), pad_token_id, dtype=jnp.int32)
            if return_logits:
                all_logits = jnp.zeros(
                    (batch_size, max_new_tokens - 1, self.vocab_size), dtype=jnp.float32
                )
            else:
                all_logits = None
            init = (cache, first_token, 0, tokens, all_logits, done_init)
            final_carry = jax.lax.while_loop(while_cond, while_step, init)
            final_cache, _, _, gen_tokens, gen_logits, _ = final_carry
            all_tokens = jnp.concatenate(
                [input_ids, first_token[:, None], gen_tokens], axis=1
            )
            if return_logits:
                all_logits = jnp.concatenate(
                    [first_token_logits[:, None, :], gen_logits], axis=1
                )
        else:
            final_cache = cache
            all_tokens = jnp.concatenate([input_ids, first_token[:, None]], axis=1)
            all_logits = first_token_logits[:, None, :] if return_logits else None

        return Qwen35GenerateOutput(tokens=all_tokens, cache=final_cache, logits=all_logits)


__all__ = [
    "Qwen35ForConditionalGeneration",
    "Qwen35GenerateOutput",
    "Qwen35Model",
    "Qwen35Output",
]
