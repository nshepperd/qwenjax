"""Qwen3-VL model implementation in JAX/Equinox."""
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from transformers import Qwen3VLConfig

from . import equinox_utils as eu
from .cache import KVCache, KVPrefix
from .config import (
    Qwen3VLConfig as Qwen3VLConfigModel,
)
from .linear import Linear
from .mrope import get_rope_index
from .text import Qwen3VLTextModel
from .utils.indexing import gather
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


@jax.tree_util.register_dataclass
@dataclass
class Qwen3VLGenerateOutput:
    """Output from Qwen3VL generate method."""
    tokens: Int[Array, "batch seq"]
    cache: KVCache | None = None
    logits: Float[Array, "batch seq vocab"] | None = None

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
        past_len: Int[Array, ""] | int | None,
    ) -> tuple[Int[Array, "3 batch seq"], Int[Array, "batch 1"], Int[Array, "batch 1"]]:
        """Resolve 3D position IDs.

        `past_len` is how many positions precede the input: the cache position,
        or a prefix's length. Returns (position_ids, prior_rope_deltas,
        new_rope_deltas). The two delta terms are returned separately so the
        caller can thread/sum as needed.
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
            if past_len is not None:
                position_ids += past_len
        return position_ids, rope_deltas, new_rope_deltas

    @staticmethod
    def _kv_mask(
        attention_mask: Bool[Array, "batch seq"],
        cache: KVCache | None,
        prefix: KVPrefix | None,
    ) -> tuple[Bool[Array, "batch kv_seq"], KVCache | None]:
        """The mask over every key slot attention will see, and the cache with
        the incoming tokens' validity recorded.

        `attention_mask` only ever describes the tokens being passed in now. A
        cache remembers the validity of what was written before; a prefix is
        always entirely valid.
        """
        batch_size = attention_mask.shape[0]
        if cache is not None:
            cache = cache.mark(attention_mask)
            return cache.valid, cache
        if prefix is not None:
            ones = jnp.ones((batch_size, prefix.length), dtype=jnp.bool)
            return jnp.concatenate([ones, attention_mask], axis=1), None
        return attention_mask, None

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
        prefix: KVPrefix | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], KVCache | None, Int[Array, "batch 1"]]:
        """Forward pass.

        `attention_mask` is (batch, seq) over `input_ids` only -- never over the
        cache's capacity. `prefix` is attended ahead of the input (see
        `KVPrefix`); it is exclusive with `cache`.

        Returns:
            (hidden_states, new_cache, rope_deltas) tuple
        """
        batch_size, seq_len = input_ids.shape
        if cache is not None and prefix is not None:
            raise ValueError("pass either a cache or a prefix, not both")
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask {attention_mask.shape} must match input_ids "
                f"{(batch_size, seq_len)}: it describes the new tokens only"
            )

        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._splice_image_embeds(
            input_ids, pixel_values, image_grid_thw, mm_token_type_ids,
        )

        past_len: Int[Array, ""] | int | None = cache_position
        if past_len is None and prefix is not None:
            past_len = prefix.length
        position_ids, rope_deltas, new_rope_deltas = self._resolve_position_ids(
            input_ids, image_grid_thw, attention_mask, mm_token_type_ids,
            position_ids, rope_deltas, past_len,
        )

        kv_mask, cache = self._kv_mask(attention_mask.astype(jnp.bool), cache, prefix)

        hidden_states, new_cache = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            kv_mask=kv_mask,
            cache=cache,
            cache_position=cache_position,
            prefix=prefix,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
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
        if self.config.tie_word_embeddings:
            # weight is [out_features, in_features] == [vocab, dim] same as embed
            return eu.replace(
                Linear(
                    in_features=self.model.language_model.config.hidden_size,
                    out_features=self.vocab_size,
                    use_bias=False,
                ),
                weight=self.model.language_model.embed_tokens.weight,
                bias=None,
            )
        else:
            assert self.lm_head is not None
            return self.lm_head

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
        prefix: KVPrefix | None = None,
    ) -> Qwen3VLOutput:
        """Forward pass.

        Args:
            attention_mask: (batch, seq) padding mask over ``input_ids`` only.
            cache: KV cache to read from and write into. Its own ``valid`` mask
                covers everything written earlier.
            prefix: K/V attended ahead of the input (a cartridge, say). With
                ``use_cache`` and no cache, the new cache is seeded with it;
                otherwise it is concatenated in each layer.
            last_logit_only: If True, skip the full lm_head projection over the
                prompt and only compute logits at the last non-padded position
                per batch item — populates ``Qwen3VLOutput.last_logits`` and
                leaves ``Qwen3VLOutput.logits`` as None.

        Returns:
            Qwen3VLOutput with hidden_states, cache, rope_deltas, plus either
            logits (full) or last_logits ((batch, vocab)) depending on the flag.
        """
        if use_cache and cache is None:
            cache_dtype = self.cache_dtype()
            cache = KVCache.for_model(
                self, input_ids.shape[0], input_ids.shape[1],
                dtype=cache_dtype, prefix=prefix,
            )
            prefix = None

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
            prefix=prefix,
        )

        if last_logit_only:
            batch_size, seq_len = input_ids.shape
            if attention_mask is not None:
                positions = jnp.arange(seq_len)
                last_idx = jnp.max(
                    positions[None, :] * (attention_mask == 1).astype(jnp.int32),
                    axis=-1,
                )
            else:
                last_idx = jnp.full((batch_size,), seq_len - 1, dtype=jnp.int32)
            last_hidden = gather('b s h, b [s] -> b h', hidden_states, last_idx)
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

    def cache_dtype(self):
        """The dtype K/V are cached in: the model's, except never float32."""
        dtype = self.model.language_model.embed_tokens.weight().dtype
        return jnp.bfloat16 if dtype == jnp.float32 else dtype

    @pjit(static_argnames=("max_new_tokens", "progress_bar", "return_logits"))
    def generate(
        self,
        input_ids: Int[Array, "batch seq"],
        pixel_values: Float[Array, "total_patches C*T*H*W"] | None = None,
        image_grid_thw: Int[Array, "num_images 3"] | None = None,
        attention_mask: Float[Array, "batch seq"] | None = None,
        *,
        cache: KVCache | None = None,
        prefix: KVPrefix | None = None,
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
        """Generate tokens using jax.lax.scan for efficiency.

        Args:
            input_ids: Initial token IDs (prompt).
            pixel_values: Image patches (required if images in prompt).
            image_grid_thw: Grid dimensions for each image (T, H, W).
            attention_mask: Padding mask for prompt (batch, seq).
            cache: A cache to continue from. Created if absent.
            prefix: K/V to attend ahead of the prompt, e.g. a cartridge. Written
                into the cache before the prefill.
            max_new_tokens: Number of tokens to generate.
            key: PRNG key for sampling.
            temperature: Sampling temperature (0 = greedy).
            progress_bar: Show tqdm progress bar.

        Returns:
            Tuple of (all_token_ids, final_cache).
        """
        batch_size = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        max_seq_len = prompt_len + max_new_tokens

        # === Step 2: Create KV cache ===
        if cache is None:
            cache = KVCache.for_model(
                self, batch_size, max_seq_len, dtype=jnp.bfloat16, prefix=prefix,
            )
        elif prefix is not None:
            cache = cache.write_prefix(prefix)

        if attention_mask is None:
            attention_mask = jnp.ones([batch_size, prompt_len], dtype=jnp.int32)
        decode_mask = jnp.ones([batch_size, 1], dtype=jnp.int32)

        # === Step 3: Prefill - process entire prompt with images ===
        # last_logit_only skips the lm_head matmul over the full prompt and only
        # projects the last non-padded position — what we need to sample from.
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

        # Sample first token from the gathered last-position logits.
        assert output.last_logits is not None, "last_logit_only=True must populate last_logits"
        first_token_logits = output.last_logits  # (batch, vocab)
        key, subkey = jax.random.split(key)
        first_token = sample(first_token_logits, subkey)
        assert output.cache is not None, "Cache should not be None after forward pass"
        cache = output.cache

        # === Step 4: Decode loop ===
        # stop_token_id < 0 disables early stopping entirely. Track which batch
        # items have already emitted the stop token so each item halts on its own
        # schedule and emits pad afterwards.
        stop_active = stop_token_id >= 0
        done_init = stop_active & (first_token == stop_token_id)
        keys = jax.random.split(key, max_new_tokens)

        def while_step(carry):
            cache, token, step_idx, tokens, all_logits, done_mask = carry
            if progress_bar:
                jax.debug.print("Decoding step {}/{}", step_idx, max_new_tokens-1)

            # Single token forward (no images - already embedded in cache)
            out = self(
                input_ids=token[:, None],
                pixel_values=None,
                image_grid_thw=None,
                attention_mask=decode_mask,
                cache=cache,
                rope_deltas=output.rope_deltas,
            )

            assert out.logits.shape[1] == 1, "Logits should have sequence length 1 during decode"
            logits = out.logits[:, -1, :]  # (batch, vocab)
            sampled = jnp.where(
                temperature > 0,
                jax.random.categorical(keys[step_idx], logits / temperature, axis=-1),
                jnp.argmax(logits, axis=-1),
            )
            # Items that already stopped emit pad; everyone else gets the sample.
            next_token = jnp.where(done_mask, pad_token_id, sampled)
            new_done_mask = done_mask | (stop_active & (next_token == stop_token_id))

            if all_logits is not None:
                all_logits = all_logits.at[:, step_idx, :].set(logits)
            tokens = tokens.at[:, step_idx].set(next_token)

            return (out.cache, next_token, step_idx+1, tokens, all_logits, new_done_mask)

        def while_cond(carry):
            _, _, step_idx, _, _, done_mask = carry
            all_done = stop_active & jnp.all(done_mask)
            return (~all_done) & (step_idx < max_new_tokens - 1)

        # === Step 5: Run decode loop ===
        if max_new_tokens > 1:
            tokens = jnp.full((batch_size, max_new_tokens - 1), pad_token_id, dtype=jnp.int32)
            if return_logits:
                all_logits = jnp.zeros((batch_size, max_new_tokens - 1, self.vocab_size), dtype=jnp.float32)
            else:
                all_logits = None
            init = (cache, first_token, 0, tokens, all_logits, done_init)
            final_carry = jax.lax.while_loop(
                while_cond,
                while_step,
                init
            )
            final_cache, _, _, gen_tokens, gen_logits, _ = final_carry
            gen_logits: Array
            all_tokens = jnp.concatenate(
                [input_ids, first_token[:, None], gen_tokens], axis=1
            )
            if return_logits:
                # gen_logits: (steps, batch, vocab) -> (batch, steps, vocab)
                all_logits = jnp.concatenate(
                    [first_token_logits[:, None, :], gen_logits], axis=1
                )
        else:
            final_cache = cache
            all_tokens = jnp.concatenate([input_ids, first_token[:, None]], axis=1)
            if return_logits:
                all_logits = first_token_logits[:, None, :]
            else:
                all_logits = None

        return Qwen3VLGenerateOutput(tokens=all_tokens, cache=final_cache, logits=all_logits)


__all__ = ["Qwen3VLForConditionalGeneration", "Qwen3VLModel", "Qwen3VLOutput"]
