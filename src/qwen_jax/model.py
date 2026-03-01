"""Qwen3-VL model implementation in JAX/Equinox."""
from .utils.pjit import pjit
from dataclasses import dataclass, field
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, Int, PRNGKeyArray
from transformers import Qwen3VLConfig

from . import equinox_utils as eu
from .utils.indexing import gather
from .cache import KVCache
from .utils.rng import split

from .config import (
    Qwen3VLConfig as Qwen3VLConfigModel,
)
from .linear import Embedding, Linear
from .text import Qwen3VLTextModel
from .vision import Qwen3VLVisionModel


@jax.tree_util.register_dataclass
@dataclass
class Qwen3VLOutput:
    """Output from Qwen3VL model."""
    logits: Float[Array, "batch seq vocab"]
    hidden_states: Float[Array, "batch seq hidden"]
    rope_deltas: Int[Array, "batch 1"]
    cache: Optional[KVCache] = None


@jax.tree_util.register_dataclass
@dataclass
class Qwen3VLGenerateOutput:
    """Output from Qwen3VL generate method."""
    tokens: Int[Array, "batch seq"]
    cache: Optional[KVCache] = None
    logits: Optional[Float[Array, "batch seq vocab"]] = None

class Qwen3VLModel(eqx.Module):
    """Qwen3-VL multimodal model.

    Combines vision encoder with text decoder. Vision embeddings replace
    placeholder tokens in the text sequence, and DeepStack injects
    intermediate vision features into early text layers.
    """
    # Config
    config: Qwen3VLConfigModel = field(metadata=dict(static=True))

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

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str = ""):
        return eu.replace(
            self,
            visual=self.visual.load_state_dict(state_dict, prefix + "visual."),
            language_model=self.language_model.load_state_dict(state_dict, prefix + "language_model."),
        )

    def get_input_embeddings(self) -> Embedding:
        return self.language_model.embed_tokens

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
        image_grid_thw: Optional[Int[Array, "num_images 3"]] = None,
        attention_mask: Optional[Float[Array, "batch seq"]] = None,
    ) -> tuple[Int[Array, "3 batch seq"], Int[Array, "batch 1"]]:
        """Compute 3D position IDs for MRoPE (JIT-compatible).

        For text tokens: all three dimensions have the same position
        For image tokens: positions reflect (T, H, W) grid coordinates

        Args:
            input_ids: Token IDs (batch, seq)
            image_grid_thw: Grid dimensions for each image
            attention_mask: Attention mask

        Returns:
            (position_ids, rope_deltas) tuple
        """
        batch_size, seq_len = input_ids.shape
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id

        attention_mask = attention_mask[:, -seq_len:] if attention_mask is not None else None

        # Handle text-only case (already JIT-compatible)
        if image_grid_thw is None:
            if attention_mask is not None:
                position_ids = jnp.cumsum(attention_mask, axis=-1) - 1
                position_ids = jnp.where(attention_mask == 0, 1, position_ids)
                position_ids = jnp.broadcast_to(position_ids[None, ...], (3, batch_size, seq_len))
                max_position = position_ids.max(axis=(0, 2))  # (batch,)
                rope_deltas = (max_position + 1 - seq_len)[:, None]
            else:
                position_ids = jnp.broadcast_to(
                    jnp.arange(seq_len)[None, None, :],
                    (3, batch_size, seq_len)
                )
                rope_deltas = jnp.zeros((batch_size, 1), dtype=jnp.int32)
            return position_ids, rope_deltas

        # With images: JIT-compatible vectorized implementation
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        # Mask invalid tokens
        valid_ids = jnp.where(attention_mask == 1, input_ids, -1)

        # Identify image tokens
        image_mask = (valid_ids == image_token_id)  # (batch, seq)

        # Compute LLM grid dimensions for each image
        llm_grid_t = image_grid_thw[:, 0]
        llm_grid_h = image_grid_thw[:, 1] // spatial_merge_size
        llm_grid_w = image_grid_thw[:, 2] // spatial_merge_size
        tokens_per_image = llm_grid_t * llm_grid_h * llm_grid_w  # (num_images,)
        num_images = image_grid_thw.shape[0]

        # Count images per batch item
        # Images are detected by vision_start_token followed by image_token
        vision_start_mask = (valid_ids == vision_start_token_id)
        shifted_ids = jnp.roll(valid_ids, -1, axis=-1)
        image_start_mask = vision_start_mask & (shifted_ids == image_token_id)
        images_per_batch = jnp.sum(image_start_mask, axis=-1)  # (batch,)

        # Cumulative images before each batch item -> global image index offset
        batch_image_offset = jnp.cumsum(jnp.pad(images_per_batch, (1, 0)))[:-1]  # (batch,)

        # Create fenceposts for image tokens (global)
        image_token_fenceposts = jnp.cumsum(jnp.pad(tokens_per_image, (1, 0)))  # (num_images + 1,)

        # For each position, compute the cumulative image token count (within batch item)
        cumsum_image_tokens = jnp.cumsum(image_mask.astype(jnp.int32), axis=-1)  # (batch, seq)

        # For image tokens, determine which image they belong to (global index)
        global_image_token_idx = (cumsum_image_tokens - 1) + batch_image_offset[:, None]  # (batch, seq)

        # Use searchsorted to find which image each image token belongs to
        flat_global_idx = global_image_token_idx.ravel()
        image_idx = jnp.searchsorted(image_token_fenceposts, flat_global_idx, side='right') - 1
        image_idx = image_idx.reshape(batch_size, seq_len)
        image_idx = jnp.clip(image_idx, 0, num_images - 1)  # Safety clamp

        # Local index within the image token block
        local_idx = flat_global_idx - image_token_fenceposts[image_idx.ravel()]
        local_idx = local_idx.reshape(batch_size, seq_len)

        # Compute grid coordinates for image tokens
        H = llm_grid_h[image_idx]  # (batch, seq)
        W = llm_grid_w[image_idx]  # (batch, seq)

        t_coord = local_idx // (H * W)
        h_coord = (local_idx // W) % H
        w_coord = local_idx % W

        # Compute cumulative text count (ignoring images)
        text_mask = ~image_mask & (attention_mask == 1)
        text_cumsum = jnp.cumsum(text_mask.astype(jnp.int32), axis=-1)  # (batch, seq)

        # Image position contribution: max(T, H, W) for each image
        max_grid_dim = jnp.maximum(jnp.maximum(llm_grid_t, llm_grid_h), llm_grid_w)  # (num_images,)

        def process_batch_item(b):
            """Process a single batch item."""
            mask = attention_mask[b]
            img_mask_b = image_mask[b]
            text_cumsum_b = text_cumsum[b]
            img_cumsum_b = cumsum_image_tokens[b]

            n_images_b = images_per_batch[b]
            img_offset_b = batch_image_offset[b]

            # Shifted cumsum for "before j"
            img_cumsum_shifted = jnp.roll(img_cumsum_b, 1).at[0].set(0)

            # For each position, find how many images are complete
            # Use global fenceposts with adjustment
            adjusted_img_cumsum = img_cumsum_shifted + image_token_fenceposts[img_offset_b]
            num_complete_images = jnp.searchsorted(image_token_fenceposts, adjusted_img_cumsum, side='right') - 1 - img_offset_b
            num_complete_images = jnp.clip(num_complete_images, 0, n_images_b)

            # Sum of max_grid_dim contributions from complete images
            max_grid_cumsum = jnp.cumsum(jnp.pad(max_grid_dim, (1, 0)))  # (num_images + 1,)
            img_offset = max_grid_cumsum[img_offset_b + num_complete_images] - max_grid_cumsum[img_offset_b]

            # For text tokens: position = text_cumsum - 1 + img_offset
            text_pos = text_cumsum_b - 1 + img_offset

            # For image tokens: base_offset + grid_coord
            # Image segment starts where: img_mask_b is True and previous position is not image
            img_seg_start = img_mask_b & ~jnp.roll(img_mask_b, 1).at[0].set(True)

            # Base offset at image segment start positions
            # = text count before this image + sum of max_grid_dim for images before this one
            shifted_text_cumsum = jnp.roll(text_cumsum_b, 1).at[0].set(0)
            shifted_img_offset = jnp.roll(img_offset, 1).at[0].set(0)
            base_offset_at_img_start = shifted_text_cumsum + shifted_img_offset

            # Propagate base_offset to all image tokens using cummax
            base_offset_masked = jnp.where(img_seg_start, base_offset_at_img_start, -1)
            base_offset_propagated = jnp.maximum.accumulate(base_offset_masked)
            base_offset_propagated = jnp.where(img_mask_b, base_offset_propagated, 0)

            # Compute image token positions with grid coordinates
            t_c = t_coord[b]
            h_c = h_coord[b]
            w_c = w_coord[b]

            img_pos_t = base_offset_propagated + t_c
            img_pos_h = base_offset_propagated + h_c
            img_pos_w = base_offset_propagated + w_c

            # Combine text and image positions
            pos_t = jnp.where(img_mask_b, img_pos_t, text_pos)
            pos_h = jnp.where(img_mask_b, img_pos_h, text_pos)
            pos_w = jnp.where(img_mask_b, img_pos_w, text_pos)

            # Handle masked positions (attention_mask == 0)
            pos_t = jnp.where(mask == 1, pos_t, 1)
            pos_h = jnp.where(mask == 1, pos_h, 1)
            pos_w = jnp.where(mask == 1, pos_w, 1)

            pos_ids = jnp.stack([pos_t, pos_h, pos_w], axis=0)  # (3, seq)

            # Rope delta = max_position + 1 - seq_len
            max_pos = jnp.max(pos_ids)
            rope_delta = max_pos + 1 - seq_len

            return pos_ids, rope_delta

        # vmap over batch dimension
        position_ids, rope_deltas = jax.vmap(process_batch_item)(jnp.arange(batch_size))

        # position_ids shape: (batch, 3, seq) -> need (3, batch, seq)
        position_ids = jnp.transpose(position_ids, (1, 0, 2))
        rope_deltas = rope_deltas[:, None]  # (batch, 1)

        return position_ids, rope_deltas

    def __call__(
        self,
        input_ids: Int[Array, "batch seq"],
        pixel_values: Optional[Float[Array, "total_patches C*T*H*W"]] = None,
        image_grid_thw: Optional[Int[Array, "num_images 3"]] = None,
        attention_mask: Optional[Float[Array, "batch seq"]] = None,
        position_ids: Optional[Int[Array, "3 batch seq"]] = None,
        cache: Optional[KVCache] = None,
        cache_position: Optional[Int[Array, ""]] = None,
        rope_deltas: Optional[Int[Array, "batch 1"]] = None,
    ) -> tuple[Float[Array, "batch seq hidden"], Optional[KVCache], Int[Array, "batch 1"]]:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            pixel_values: Image patches (if images present)
            image_grid_thw: Grid dimensions for each image
            attention_mask: Attention mask (batch, seq)
            position_ids: Pre-computed position IDs (3, batch, seq)
            cache: KV cache
            cache_position: Position in cache

        Returns:
            (hidden_states, new_cache, rope_deltas) tuple
        """
        batch_size, seq_len = input_ids.shape

        # Get text embeddings
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        # Process images
        image_mask = None
        deepstack_visual_embeds = None
        visual_pos_masks = None

        if pixel_values is not None and image_grid_thw is not None:
            # Get image features
            image_embeds, deepstack_image_embeds = self.get_image_features(
                pixel_values, image_grid_thw
            )

            # Find image token positions
            image_mask = input_ids == self.config.image_token_id
            image_idx = jnp.cumsum(jnp.reshape(image_mask, [-1])) - 1
            image_embeds_i = rearrange(image_embeds[image_idx],
                "(b seq) hidden -> b seq hidden", b=batch_size
            )

            # Replace image tokens with image embeddings
            # Note: This assumes image_embeds matches the number of image tokens
            inputs_embeds = jnp.where(
                image_mask[..., None],
                image_embeds_i,
                inputs_embeds
            )

            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds

        # Compute position IDs if not provided
        if rope_deltas is None:
            rope_deltas = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        new_rope_deltas = jnp.zeros((batch_size, 1), dtype=jnp.int32)
        if position_ids is None:
            position_ids, new_rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, attention_mask
            )
            position_ids += rope_deltas
            if cache is not None:
                position_ids += cache_position if cache_position is not None else 0

        # Create causal attention mask (always needed for causal language modeling)
        # attention_mask is a padding mask: (batch, seq) with 1 for valid tokens, 0 for padding
        if cache is not None:
            kv_len = cache.max_seq_len
            # When using cache, need to account for cache position
            past_seen_tokens = cache_position if cache_position is not None else 0
        else:
            kv_len = seq_len
            past_seen_tokens = 0

        # Create causal mask: can only attend to positions <= current position
        # Shape: (seq, kv_seq)
        q_positions = jnp.arange(seq_len) + past_seen_tokens
        k_positions = jnp.arange(kv_len)
        causal_mask = (k_positions[None, :] <= q_positions[:, None])
        # Expand to (batch, 1, seq, kv_seq) for broadcasting
        causal_mask = causal_mask[None, None, :, :]

        # Apply padding mask if provided
        # attention_mask: (batch, seq) with 1 for valid tokens, 0 for padding
        if attention_mask is not None:
            # Expand padding mask: (batch, seq) -> (batch, 1, 1, kv_seq)
            # Mask out positions where attention_mask is 0 (padding)
            if attention_mask.shape[-1] < kv_len:
                attention_mask = jnp.pad(
                    attention_mask,
                    ((0, 0), (0, kv_len - attention_mask.shape[-1])),
                    constant_values=1,
                )
            padding_mask = attention_mask[:, None, None, :kv_len].astype(jnp.bool)
            causal_mask = causal_mask & padding_mask

        # Forward through language model
        hidden_states, new_cache = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=causal_mask,
            cache=cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            kv_mask=attention_mask.astype(jnp.bool)
        )

        return hidden_states, new_cache, (rope_deltas + new_rope_deltas)


class Qwen3VLForConditionalGeneration(eqx.Module):
    """Qwen3-VL for conditional text generation.

    Wraps Qwen3VLModel with a language model head for next-token prediction.
    """
    # Config
    config: Qwen3VLConfigModel = field(metadata=dict(static=True))
    vocab_size: int = field(metadata=dict(static=True))

    # Models
    model: Qwen3VLModel

    _lm_head: Linear | None

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
            self._lm_head = None
        else:
            self._lm_head = Linear(self.model.language_model.config.hidden_size, self.vocab_size, use_bias=False)

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str = ""):
        lm_head = None
        if not self.config.tie_word_embeddings:
            assert self._lm_head is not None
            lm_head = self._lm_head.load_state_dict(state_dict, prefix + "lm_head.")
        return eu.replace(
            self,
            model=self.model.load_state_dict(state_dict, prefix + "model."),
            _lm_head=lm_head,
        )

    @property
    def lm_head(self) -> Linear:
        if self.config.tie_word_embeddings:
            # weight is [out_features, in_features] == [vocab, dim] same as embed
            return eu.replace(
                Linear(
                    in_features=self.model.language_model.config.hidden_size,
                    out_features=self.vocab_size,
                    use_bias=False,
                ),
                weight=self.model.language_model.embed_tokens.weight,
            )
        else:
            assert self._lm_head is not None
            return self._lm_head

    @pjit(static_argnames=("use_cache",))
    def __call__(
        self,
        input_ids: Int[Array, "batch seq"],
        pixel_values: Optional[Float[Array, "total_patches C*T*H*W"]] = None,
        image_grid_thw: Optional[Int[Array, "num_images 3"]] = None,
        attention_mask: Optional[Float[Array, "batch seq"]] = None,
        position_ids: Optional[Int[Array, "3 batch seq"]] = None,
        cache: Optional[KVCache] = None,
        rope_deltas: Optional[Int[Array, "batch 1"]] = None,
        use_cache: bool = False,
    ) -> Qwen3VLOutput:
        """Forward pass.

        Args:
            input_ids: Token IDs (batch, seq)
            pixel_values: Image patches
            image_grid_thw: Grid dimensions for each image
            attention_mask: Attention mask
            position_ids: Pre-computed position IDs
            cache: KV cache
            cache_position: Position in cache

        Returns:
            Qwen3VLOutput with logits, hidden_states, cache, rope_deltas
        """
        if use_cache and cache is None:
            batch_size = input_ids.shape[0]
            max_seq_len = input_ids.shape[1]
            text_config = self.model.config.text_config
            cache_dtype = self.model.language_model.embed_tokens.weight.dtype
            if cache_dtype == jnp.float32:
                cache_dtype = jnp.bfloat16
            cache = KVCache.create(
                num_layers=text_config.num_hidden_layers,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=text_config.num_key_value_heads,
                head_dim=text_config.head_dim,
                dtype=cache_dtype,
            )

        hidden_states, new_cache, new_rope_deltas = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache=cache,
            cache_position=cache.position if cache is not None else None,
            rope_deltas=rope_deltas,
        )

        logits = self.lm_head(hidden_states)

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
        pixel_values: Optional[Float[Array, "total_patches C*T*H*W"]] = None,
        image_grid_thw: Optional[Int[Array, "num_images 3"]] = None,
        attention_mask: Optional[Float[Array, "batch seq"]] = None,
        *,
        cache: Optional[KVCache] = None,
        rope_deltas: Optional[Int[Array, "batch 1"]] = None,
        max_new_tokens: int,
        key: PRNGKeyArray,
        temperature: float = 1.0,
        progress_bar: bool = True,
        return_logits: bool = False,
        stop_token_id: int = -1,
        pad_token_id: int = 0,
    ) -> Qwen3VLGenerateOutput:
        """Generate tokens using jax.lax.scan for efficiency.

        Args:
            input_ids: Initial token IDs (prompt).
            pixel_values: Image patches (required if images in prompt).
            image_grid_thw: Grid dimensions for each image (T, H, W).
            attention_mask: Padding mask for prompt (batch, seq).
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
        text_config = self.model.config.text_config

        # === Step 1: Compute position IDs for prefill (handles images) ===

        # === Step 2: Create KV cache ===
        if cache is None:
            cache = KVCache.create(
                num_layers=text_config.num_hidden_layers,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=text_config.num_key_value_heads,
                head_dim=text_config.head_dim,
                dtype=jnp.bfloat16,
            )

        if attention_mask is None:
            attention_mask = jnp.ones([batch_size, max_seq_len], dtype=jnp.int32)
        elif attention_mask.shape[1] < max_seq_len:
            attention_mask = jnp.pad(
                attention_mask,
                ((0, 0), (0, max_seq_len - attention_mask.shape[1])),
                constant_values=1,
            )

        # === Step 3: Prefill - process entire prompt with images ===
        output = self(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
            cache=cache,
            rope_deltas=rope_deltas,
        )

        def sample(logits, key):
            return jnp.where(
                temperature > 0,
                jax.random.categorical(key, logits / temperature, axis=-1),
                jnp.argmax(logits, axis=-1),
            )

        # Sample first token
        last_input_idx = jnp.max(jnp.arange(prompt_len)[None, :] * (attention_mask[:, :prompt_len] == 1), axis=-1)  # (batch,)
        first_token_logits = gather('b s v, b [s] -> b v', output.logits, last_input_idx)
        key, subkey = jax.random.split(key)
        first_token = sample(first_token_logits, subkey)
        assert output.cache is not None, "Cache should not be None after forward pass"
        cache = output.cache

        # === Step 4: Decode loop setup ===
        keys = jax.random.split(key, max_new_tokens)

        # === Step 5: Scan step function ===
        def while_step(carry):
            cache, token, rng, step_idx, tokens, all_logits = carry
            if progress_bar:
                jax.debug.print("Decoding step {}/{}", step_idx, max_new_tokens-1)

            # Single token forward (no images - already embedded in cache)
            out = self(
                input_ids=token[:, None],
                pixel_values=None,
                image_grid_thw=None,
                attention_mask=attention_mask,
                cache=cache,
                rope_deltas=output.rope_deltas
            )

            # Sample next token
            assert out.logits.shape[1] == 1, "Logits should have sequence length 1 during decode"
            logits = out.logits[:, -1, :]  # (batch, vocab)
            rng, subkey = jax.random.split(rng)
            next_token = jnp.where(
                temperature > 0,
                jax.random.categorical(keys[step_idx], logits / temperature, axis=-1),
                jnp.argmax(logits, axis=-1),
            )

            if all_logits is not None:
                all_logits = all_logits.at[:, step_idx, :].set(logits)
            tokens = tokens.at[:, step_idx].set(next_token)

            return (out.cache, next_token, rng, step_idx+1, tokens, all_logits)

        def while_cond(carry):
            _, token, _, step_idx, tokens, _ = carry
            not_done = (stop_token_id < 0) | jnp.any(token != stop_token_id)
            return not_done & (step_idx < max_new_tokens - 1)

        # === Step 6: Run decode loop ===
        if max_new_tokens > 1:
            tokens = jnp.full((batch_size, max_new_tokens - 1), pad_token_id, dtype=jnp.int32)
            if return_logits:
                all_logits = jnp.zeros((batch_size, max_new_tokens - 1, self.vocab_size), dtype=jnp.float32)
            else:
                all_logits = None
            init = (cache, first_token, key, 0, tokens, all_logits)
            final_carry = jax.lax.while_loop(
                while_cond,
                while_step,
                init
            )
            final_cache, _, _, _, gen_tokens, gen_logits = final_carry
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


__all__ = ["Qwen3VLModel", "Qwen3VLForConditionalGeneration", "Qwen3VLOutput"]
