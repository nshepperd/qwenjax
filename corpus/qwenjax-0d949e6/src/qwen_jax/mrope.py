"""3D position indexing for MRoPE.

Qwen3-VL gives every token three position coordinates instead of one. Text
tokens get the same value in all three; image tokens get their (t, h, w)
coordinate within the image's post-merge grid. An image therefore advances the
position counter by `max(T, H, W)` rather than by its token count, so the text
that follows it resumes from a position that does not depend on the image's
aspect ratio.

Everything here is written to run under `jit` with `num_images` and the batch
shape static but the *contents* of `image_grid_thw` traced, which is why image
boundaries are located with cumsum/searchsorted over fenceposts rather than by
slicing per image.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int


def _text_only_rope_index(
    input_ids: Int[Array, "batch seq"],
    attention_mask: Int[Array, "batch seq"] | None,
) -> tuple[Int[Array, "3 batch seq"], Int[Array, "batch 1"]]:
    """Position ids when there are no images: all three dims are the same."""
    batch_size, seq_len = input_ids.shape

    if attention_mask is None:
        position_ids = jnp.broadcast_to(
            jnp.arange(seq_len)[None, None, :], (3, batch_size, seq_len)
        )
        return position_ids, jnp.zeros((batch_size, 1), dtype=jnp.int32)

    # Padded positions do not advance the counter, and are parked at 1.
    position_ids = jnp.cumsum(attention_mask, axis=-1) - 1
    position_ids = jnp.where(attention_mask == 0, 1, position_ids)
    position_ids = jnp.broadcast_to(position_ids[None, ...], (3, batch_size, seq_len))
    max_position = position_ids.max(axis=(0, 2))  # (batch,)
    rope_deltas = (max_position + 1 - seq_len)[:, None]
    return position_ids, rope_deltas


class _ImageLayout(NamedTuple):
    """Where each image token sits, both within its image and along the batch.

    Produced by `_locate_image_tokens`, consumed by the per-batch-item pass that
    turns these into actual positions.
    """

    t_coord: Int[Array, "batch seq"]
    """Temporal grid coordinate of each image token (0 for non-image tokens)."""
    h_coord: Int[Array, "batch seq"]
    """Height grid coordinate of each image token."""
    w_coord: Int[Array, "batch seq"]
    """Width grid coordinate of each image token."""
    max_grid_cumsum: Int[Array, "num_images_plus_1"]
    """Cumulative sum of each image's max(T, H, W) -- how far it advances positions."""
    images_per_batch: Int[Array, "batch"]
    """Number of images in each batch item."""
    batch_image_offset: Int[Array, "batch"]
    """Global index of each batch item's first image."""
    token_fenceposts: Int[Array, "num_images_plus_1"]
    """Cumulative token counts per image, for locating a global token index."""
    cumsum_image_tokens: Int[Array, "batch seq"]
    """Running count of image tokens seen so far within each batch item."""


def _locate_image_tokens(
    image_mask: Bool[Array, "batch seq"],
    image_grid_thw: Int[Array, "num_images 3"],
    spatial_merge_size: int,
) -> _ImageLayout:
    """Work out which image each image token belongs to, and where inside it.

    Image tokens are numbered globally across the batch so that one set of
    fenceposts -- the cumulative token counts of `image_grid_thw` -- identifies
    the owning image of any token by a single searchsorted.
    """
    batch_size, seq_len = image_mask.shape
    num_images = image_grid_thw.shape[0]

    # Post-merge grid: the patch merger collapses each spatial_merge_size square.
    llm_grid_t = image_grid_thw[:, 0]
    llm_grid_h = image_grid_thw[:, 1] // spatial_merge_size
    llm_grid_w = image_grid_thw[:, 2] // spatial_merge_size
    tokens_per_image = llm_grid_t * llm_grid_h * llm_grid_w  # (num_images,)

    # Count images per batch item by counting segment-starts of image_mask.
    # A segment starts where image_mask is True and the previous position is not.
    shifted_image_mask = jnp.pad(image_mask[:, :-1], ((0, 0), (1, 0)))
    image_start_mask = image_mask & ~shifted_image_mask
    images_per_batch = jnp.sum(image_start_mask, axis=-1)  # (batch,)

    # Cumulative images before each batch item -> global image index offset
    batch_image_offset = jnp.cumsum(jnp.pad(images_per_batch, (1, 0)))[:-1]  # (batch,)

    # Fenceposts for image tokens (global)
    token_fenceposts = jnp.cumsum(jnp.pad(tokens_per_image, (1, 0)))  # (num_images + 1,)

    # For each position, the cumulative image token count (within batch item)
    cumsum_image_tokens = jnp.cumsum(image_mask.astype(jnp.int32), axis=-1)  # (batch, seq)

    # For image tokens, which image they belong to (global index)
    global_image_token_idx = (cumsum_image_tokens - 1) + batch_image_offset[:, None]
    flat_global_idx = global_image_token_idx.ravel()
    image_idx = jnp.searchsorted(token_fenceposts, flat_global_idx, side='right') - 1
    image_idx = image_idx.reshape(batch_size, seq_len)
    image_idx = jnp.clip(image_idx, 0, num_images - 1)  # Safety clamp

    # Index within the owning image's token block
    local_idx = flat_global_idx - token_fenceposts[image_idx.ravel()]
    local_idx = local_idx.reshape(batch_size, seq_len)

    # Grid coordinates for image tokens
    H = llm_grid_h[image_idx]  # (batch, seq)
    W = llm_grid_w[image_idx]  # (batch, seq)
    t_coord = local_idx // (H * W)
    h_coord = (local_idx // W) % H
    w_coord = local_idx % W

    # How far each image advances the shared position counter.
    max_grid_dim = jnp.maximum(jnp.maximum(llm_grid_t, llm_grid_h), llm_grid_w)

    return _ImageLayout(
        t_coord=t_coord,
        h_coord=h_coord,
        w_coord=w_coord,
        max_grid_cumsum=jnp.cumsum(jnp.pad(max_grid_dim, (1, 0))),
        images_per_batch=images_per_batch,
        batch_image_offset=batch_image_offset,
        token_fenceposts=token_fenceposts,
        cumsum_image_tokens=cumsum_image_tokens,
    )


def _positions_for_batch_item(
    b: Int[Array, ""],
    layout: _ImageLayout,
    image_mask: Bool[Array, "batch seq"],
    attention_mask: Int[Array, "batch seq"],
    text_cumsum: Int[Array, "batch seq"],
    seq_len: int,
) -> tuple[Int[Array, "3 seq"], Int[Array, ""]]:
    """Lay one batch item's tokens out along the shared position counter.

    Text tokens advance it by one each; each image advances it by its max grid
    dimension in one jump, and its own tokens sit at that jump's base plus their
    grid coordinate. Called under `vmap` over `b`.
    """
    mask = attention_mask[b]
    img_mask_b = image_mask[b]
    text_cumsum_b = text_cumsum[b]
    img_cumsum_b = layout.cumsum_image_tokens[b]

    n_images_b = layout.images_per_batch[b]
    img_offset_b = layout.batch_image_offset[b]

    # Shifted cumsum for "before j"
    img_cumsum_shifted = jnp.roll(img_cumsum_b, 1).at[0].set(0)

    # For each position, how many images are complete.
    # Use global fenceposts with adjustment.
    adjusted_img_cumsum = img_cumsum_shifted + layout.token_fenceposts[img_offset_b]
    num_complete_images = (
        jnp.searchsorted(layout.token_fenceposts, adjusted_img_cumsum, side='right')
        - 1
        - img_offset_b
    )
    num_complete_images = jnp.clip(num_complete_images, 0, n_images_b)

    # Sum of max_grid_dim contributions from complete images
    img_offset = (
        layout.max_grid_cumsum[img_offset_b + num_complete_images]
        - layout.max_grid_cumsum[img_offset_b]
    )

    # For text tokens: position = text_cumsum - 1 + img_offset
    text_pos = text_cumsum_b - 1 + img_offset

    # For image tokens: base_offset + grid_coord.
    # An image segment starts where img_mask_b is True and the previous position is not.
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

    # Combine text and image positions
    pos_t = jnp.where(img_mask_b, base_offset_propagated + layout.t_coord[b], text_pos)
    pos_h = jnp.where(img_mask_b, base_offset_propagated + layout.h_coord[b], text_pos)
    pos_w = jnp.where(img_mask_b, base_offset_propagated + layout.w_coord[b], text_pos)

    # Handle masked positions (attention_mask == 0)
    pos_t = jnp.where(mask == 1, pos_t, 1)
    pos_h = jnp.where(mask == 1, pos_h, 1)
    pos_w = jnp.where(mask == 1, pos_w, 1)

    pos_ids = jnp.stack([pos_t, pos_h, pos_w], axis=0)  # (3, seq)

    # Rope delta = max_position + 1 - seq_len
    rope_delta = jnp.max(pos_ids) + 1 - seq_len

    return pos_ids, rope_delta


def get_rope_index(
    input_ids: Int[Array, "batch seq"],
    image_grid_thw: Int[Array, "num_images 3"] | None = None,
    attention_mask: Int[Array, "batch seq"] | None = None,
    mm_token_type_ids: Int[Array, "batch seq"] | None = None,
    *,
    spatial_merge_size: int,
    image_token_id: int,
) -> tuple[Int[Array, "3 batch seq"], Int[Array, "batch 1"]]:
    """Compute 3D position IDs for MRoPE (JIT-compatible).

    For text tokens: all three dimensions have the same position
    For image tokens: positions reflect (T, H, W) grid coordinates

    Args:
        input_ids: Token IDs (batch, seq)
        image_grid_thw: Grid dimensions for each image. None for text-only input.
        attention_mask: Attention mask
        mm_token_type_ids: Per-token modality (0=text, 1=image, 2=video).
            If None, derived from input_ids using image_token_id.
        spatial_merge_size: Vision patch merge factor, from the vision config.
        image_token_id: The placeholder token that image embeddings replace.

    Returns:
        (position_ids, rope_deltas) tuple
    """
    batch_size, seq_len = input_ids.shape

    attention_mask = attention_mask[:, -seq_len:] if attention_mask is not None else None

    if image_grid_thw is None:
        return _text_only_rope_index(input_ids, attention_mask)

    # With images: JIT-compatible vectorized implementation
    if attention_mask is None:
        attention_mask = jnp.ones_like(input_ids)

    # Derive image_mask from mm_token_type_ids (preferred) or input_ids (fallback)
    if mm_token_type_ids is not None:
        image_mask = (mm_token_type_ids == 1) & (attention_mask == 1)
    else:
        valid_ids = jnp.where(attention_mask == 1, input_ids, -1)
        image_mask = (valid_ids == image_token_id)

    layout = _locate_image_tokens(image_mask, image_grid_thw, spatial_merge_size)

    # Cumulative text count (ignoring images)
    text_mask = ~image_mask & (attention_mask == 1)
    text_cumsum = jnp.cumsum(text_mask.astype(jnp.int32), axis=-1)  # (batch, seq)

    position_ids, rope_deltas = jax.vmap(
        _positions_for_batch_item, in_axes=(0, None, None, None, None, None)
    )(jnp.arange(batch_size), layout, image_mask, attention_mask, text_cumsum, seq_len)

    # position_ids shape: (batch, 3, seq) -> need (3, batch, seq)
    position_ids = jnp.transpose(position_ids, (1, 0, 2))
    rope_deltas = rope_deltas[:, None]  # (batch, 1)

    return position_ids, rope_deltas


__all__ = ["get_rope_index"]
