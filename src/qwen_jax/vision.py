"""Vision model components for Qwen3-VL."""
from qwen_jax.config import Qwen3VLVisionConfig
from dataclasses import field
from typing import Optional
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from einops import rearrange

from .linear import Linear, LayerNorm, Embedding
from .mlp import Qwen3VLVisionMLP
from .attention import Qwen3VLVisionAttention
from .rope import Qwen3VLVisionRotaryEmbedding
from . import equinox_utils as eu
from .utils.rng import split
from jax.scipy.interpolate import RegularGridInterpolator



class PatchEmbedProj(eqx.Module):
    """This is effectively a Conv3d except we don't actually need a conv
    because the stride is equal to the kernel size. And the input is already
    reshaped to separate patches."""
    weight: Float[Array, "out_channels in_channels kernel_t kernel_h kernel_w"] | None
    bias: Float[Array, "out_channels"] | None
    in_channels: int = field(metadata=dict(static=True))
    out_channels: int = field(metadata=dict(static=True))
    kernel_size: tuple[int, int, int] = field(metadata=dict(static=True))
    def __init__(self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int],
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        kt, kh, kw = kernel_size
        self.weight = None
        self.bias = None

    def init_weights(self, key: PRNGKeyArray):
        kt, kh, kw = self.kernel_size
        weight = jax.random.normal(
            key, (self.out_channels, self.in_channels, kt, kh, kw)
        ) / np.sqrt(self.in_channels * kt * kh * kw)
        bias = jnp.zeros((self.out_channels,))
        return eu.replace(self, weight=weight, bias=bias)

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        """Load state dict into module."""
        weight_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
        )
        weight = state_dict.pop(prefix + "weight")

        assert weight.shape == weight_shape, \
            f"Weight shape mismatch: expected {weight_shape}, got {weight.shape}"
        bias = state_dict.pop(prefix + "bias")
        assert bias.shape == (self.out_channels,), \
            f"Bias shape mismatch: expected {(self.out_channels,)}, got {bias.shape}"
        return eu.replace(self, weight=weight, bias=bias)


    def __call__(
        self,
        hidden_states: Float[Array, "total_patches in_channels temporal patch patch"],
    ) -> Float[Array, "total_patches out_channels"]:
        assert self.weight is not None and self.bias is not None, "Weights are not initialized."
        hidden_states = jnp.einsum(
            'bctpq,octpq->bo', hidden_states, self.weight
        )
        hidden_states += self.bias  # (total_patches, out_channels)
        return hidden_states

class Qwen3VLVisionPatchEmbed(eqx.Module):
    """3D patch embedding for video/image.

    The input is in the format [num_patches x patch_rgb].
    Where patch_rgb = (in_channels=3 patch_t=2 patch_h=16 patch_w=16)
    """
    patch_size: int = field(metadata=dict(static=True))
    temporal_patch_size: int = field(metadata=dict(static=True))
    in_channels: int = field(metadata=dict(static=True))
    embed_dim: int = field(metadata=dict(static=True))

    proj: PatchEmbedProj

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
    ):
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = PatchEmbedProj(
            self.in_channels, self.embed_dim, kernel_size
        )

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        """Load state dict into module."""
        proj = self.proj.load_state_dict(state_dict, prefix + "proj.")
        return eu.replace(self, proj=proj)

    def __call__(
        self,
        hidden_states: Float[Array, "total_patches in_channels*temporal*patch*patch"],
    ) -> Float[Array, "total_patches embed_dim"]:
        hidden_states = rearrange(
            hidden_states,
            "b (c t p1 p2) -> b c t p1 p2",
            c=self.in_channels,
            t=self.temporal_patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        hidden_states = self.proj(hidden_states)
        return hidden_states


class Qwen3VLVisionPatchMerger(eqx.Module):
    """Spatial patch merger for downsampling and projection.

    Merges 2x2 spatial patches and projects to output dimension.
    """
    hidden_size: int = field(metadata=dict(static=True))
    spatial_merge_size: int = field(metadata=dict(static=True))
    out_hidden_size: int = field(metadata=dict(static=True))
    use_postshuffle_norm: bool = field(metadata=dict(static=True))

    norm: LayerNorm
    linear_fc1: Linear
    linear_fc2: Linear

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
        *,
        use_postshuffle_norm: bool = False,
    ):
        self.spatial_merge_size = config.spatial_merge_size
        self.out_hidden_size = config.out_hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm

        # After merging 2x2 patches, hidden size multiplied by 4
        merged_hidden = config.hidden_size * (config.spatial_merge_size ** 2)
        self.hidden_size = merged_hidden

        # Norm applied either before or after spatial shuffle
        norm_size = merged_hidden if use_postshuffle_norm else config.hidden_size
        self.norm = LayerNorm(norm_size, eps=1e-6)

        self.linear_fc1 = Linear(merged_hidden, merged_hidden, use_bias=True)
        self.linear_fc2 = Linear(merged_hidden, config.out_hidden_size, use_bias=True)

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        return eu.replace(
            self,
            norm=self.norm.load_state_dict(state_dict, prefix + "norm."),
            linear_fc1=self.linear_fc1.load_state_dict(state_dict, prefix + "linear_fc1."),
            linear_fc2=self.linear_fc2.load_state_dict(state_dict, prefix + "linear_fc2."),
        )

    def __call__(
        self,
        hidden_states: Float[Array, "seq hidden"],
    ) -> Float[Array, "seq_merged out_hidden"]:
        """Merge patches and project.

        seq = (n_img, t, bh, bw, mh, mw)
        hidden = (embedded patch dim)
        seq_merged = (n_img, t, bh, bw)
        dim_merged = (mh, mw, dim)

        Args:
            hidden_states: (seq hidden)

        Returns:
            Merged output ((n_img, t, bh, bw) (mh, mw, dim))
        """
        if self.use_postshuffle_norm:
            # Norm after reshaping to merged hidden size
            hidden_states = hidden_states.reshape(-1, self.hidden_size)
            hidden_states = self.norm(hidden_states)
        else:
            # Norm before merging, then reshape
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.reshape(-1, self.hidden_size)

        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states, approximate=False)  # Standard GELU
        hidden_states = self.linear_fc2(hidden_states)

        return hidden_states


class Qwen3VLVisionBlock(eqx.Module):
    """Vision transformer block with pre-norm."""
    hidden_size: int = field(metadata=dict(static=True))

    norm1: LayerNorm
    norm2: LayerNorm
    attn: Qwen3VLVisionAttention
    mlp: Qwen3VLVisionMLP

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
    ):
        self.hidden_size = hidden_size

        self.norm1 = LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = LayerNorm(hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(hidden_size, num_heads)
        self.mlp = Qwen3VLVisionMLP(hidden_size, intermediate_size)

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        return eu.replace(
            self,
            norm1 = self.norm1.load_state_dict(state_dict, prefix + "norm1."),
            norm2 = self.norm2.load_state_dict(state_dict, prefix + "norm2."),
            attn = self.attn.load_state_dict(state_dict, prefix + "attn."),
            mlp = self.mlp.load_state_dict(state_dict, prefix + "mlp.")
        )

    def __call__(
        self,
        hidden_states: Float[Array, "seq hidden"],
        cu_seqlens: Int[Array, "num_seqs_plus_1"],
        position_embeddings: tuple[Float[Array, "seq head_dim"], Float[Array, "seq head_dim"]],
    ) -> Float[Array, "seq hidden"]:
        """Forward pass with residual connections.

        Args:
            hidden_states: Input (total_tokens, hidden_size)
            cu_seqlens: Cumulative sequence lengths
            position_embeddings: (cos, sin) for RoPE

        Returns:
            Output (total_tokens, hidden_size)
        """
        # Attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, cu_seqlens, position_embeddings)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3VLVisionModel(eqx.Module):
    """Qwen3-VL Vision Encoder with DeepStack feature extraction.

    Processes images/videos through patch embedding, position embedding,
    transformer blocks, and patch merging. Extracts intermediate features
    at specified layers for DeepStack fusion in the text model.
    """
    # Config values
    config: Qwen3VLVisionConfig = field(metadata=dict(static=True))

    # Layers
    patch_embed: Qwen3VLVisionPatchEmbed
    pos_embed: Embedding
    rotary_pos_emb: Qwen3VLVisionRotaryEmbedding
    blocks: tuple[Qwen3VLVisionBlock, ...]
    merger: Qwen3VLVisionPatchMerger
    deepstack_merger_list: tuple[Qwen3VLVisionPatchMerger, ...]

    def __init__(
        self,
        config: Qwen3VLVisionConfig,
    ):
        self.config = config


        # Patch embedding
        self.patch_embed = Qwen3VLVisionPatchEmbed(
            config=config,
        )

        # Position embedding
        self.pos_embed = Embedding(config.num_position_embeddings, config.hidden_size)

        # Rotary embedding
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        # Transformer blocks
        blocks = []
        for _ in range(config.depth):
            blocks.append(Qwen3VLVisionBlock(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_heads,
            ))
        self.blocks = tuple(blocks)

        # Main merger (no postshuffle norm)
        self.merger = Qwen3VLVisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,
        )

        # DeepStack mergers (with postshuffle norm)
        deepstack_mergers = []
        for _ in range(len(config.deepstack_visual_indexes)):
            deepstack_mergers.append(Qwen3VLVisionPatchMerger(
                config=config,
                use_postshuffle_norm=True,
            ))
        self.deepstack_merger_list = tuple(deepstack_mergers)

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str = ""):
        return eu.replace(
            self,
            patch_embed=self.patch_embed.load_state_dict(state_dict, prefix + "patch_embed."),
            pos_embed=self.pos_embed.load_state_dict(state_dict, prefix + "pos_embed."),
            # rotary emb has no parameters
            blocks=tuple(
                self.blocks[i].load_state_dict(state_dict, prefix + f"blocks.{i}.")
                for i in range(len(self.blocks))
            ),
            merger=self.merger.load_state_dict(state_dict, prefix + "merger."),
            deepstack_merger_list=tuple(
                self.deepstack_merger_list[i].load_state_dict(
                    state_dict, prefix + f"deepstack_merger_list.{i}."
                )
                for i in range(len(self.deepstack_merger_list))
            ),
        )

    def rot_pos_emb(
        self,
        grid_thw: Int[Array, "num_images 3"],
        total_patches: int,
    ) -> Float[Array, "total_patches head_dim"]:
        """Compute rotary position embeddings based on grid coordinates.

        Args:
            grid_thw: Grid dimensions (num_images, 3) where each row is (T, H, W)

        Returns:
            Position embeddings (total_patches, head_dim)
        """
        # total_patches = (n_img t bh bw mh mw)
        merge_size = self.config.spatial_merge_size

        fenceposts = jnp.cumsum(jnp.pad(jnp.prod(grid_thw, axis=1), (1, 0)))  # (num_images + 1,)
        patch_image = jnp.searchsorted(fenceposts, jnp.arange(total_patches), side="right") - 1
        patch_idx = jnp.arange(total_patches) - fenceposts[patch_image]

        # T = grid_thw[patch_image, 0]
        H = grid_thw[patch_image, 1] # bh * mh
        W = grid_thw[patch_image, 2] # bw * mw
        # assume image is (t h/m w/m m1 m2)
        # t = patch_idx // (H*W)
        bh = (patch_idx // (W*merge_size)) % (H // merge_size)
        bw = (patch_idx // (merge_size*merge_size)) % (W // merge_size)
        mh = (patch_idx // merge_size) % merge_size
        mw = patch_idx % merge_size
        h = bh * merge_size + mh
        w = bw * merge_size + mw
        pos_ids = jnp.stack([h,w], axis=-1)  # (total_patches, 2)

        # Just directly compute the outer product, avoids needing to choose a static table size.
        embeddings = self.rotary_pos_emb.table(pos_ids)  # (total_patches, 2, dim // 2)
        embeddings = embeddings.reshape(total_patches, -1)

        return embeddings

    def fast_pos_embed_interpolate(
        self,
        grid_thw: Int[Array, "num_images 3"],
        total_patches: int,
    ) -> Float[Array, "total_patches hidden"]:
        """Interpolate position embeddings for variable image sizes.

        Uses bilinear interpolation of learned position embeddings.

        Args:
            grid_thw: Grid dimensions (num_images, 3) where each row is (T, H, W)

        Returns:
            Position embeddings (total_patches, hidden_size)
        """
        # total_patches = (n_img t bh bw mh mw)
        num_grid_per_side = round(np.sqrt(self.config.num_position_embeddings))
        merge_size = self.config.spatial_merge_size

        fenceposts = jnp.cumsum(jnp.pad(jnp.prod(grid_thw, axis=1), (1, 0)))  # (num_images + 1,)
        patch_image = jnp.searchsorted(fenceposts, jnp.arange(total_patches), method='sort', side="right") - 1
        patch_idx = jnp.arange(total_patches) - fenceposts[patch_image]
        # index within each image's patches

        assert self.pos_embed.weight is not None
        interp = RegularGridInterpolator(
            (jnp.linspace(0, 1, num_grid_per_side),
             jnp.linspace(0, 1, num_grid_per_side)),
            self.pos_embed.weight.reshape(num_grid_per_side, num_grid_per_side, -1),
            method="linear",
        )

        def embed(im, idx):
            H = grid_thw[im, 1]  # bh * mh
            W = grid_thw[im, 2]  # bw * mw
            bh = (idx // (W*merge_size)) % (H // merge_size)
            bw = (idx // (merge_size*merge_size)) % (W // merge_size)
            mh = (idx // merge_size) % merge_size
            mw = idx % merge_size
            h = bh * merge_size + mh
            w = bw * merge_size + mw
            coord = jnp.array([[h / (H - 1), w / (W - 1)]])
            return interp(coord).squeeze(0)
        return jax.vmap(embed)(patch_image, patch_idx)


    def __call__(
        self,
        hidden_states: Float[Array, "total_patches in_channels*temporal*patch*patch"],
        grid_thw: Int[Array, "num_images 3"],
    ) -> tuple[Float[Array, "tokens out_hidden"], tuple[Float[Array, "tokens out_hidden"], ...]]:
        """Forward pass.

        Args:
            hidden_states: Raw image/video patches
                          (total_patches, in_channels, temporal_patch_size, patch_size, patch_size)
            grid_thw: Grid dimensions for each image/video (num_images, 3)
                     Each row is (temporal_frames, height_patches, width_patches)

        Returns:
            Tuple of (main_features, deepstack_features):
            - main_features: Final vision features (total_merged_tokens, out_hidden_size)
            - deepstack_features: Tuple of intermediate features at deepstack layers
        """
        # Patch embedding
        hidden_states = self.patch_embed(hidden_states)

        # Position embedding (interpolated)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw, total_patches=hidden_states.shape[0])
        hidden_states = hidden_states + pos_embeds

        # Rotary position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw, total_patches=hidden_states.shape[0])
        emb = jnp.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)
        position_embeddings = (jnp.cos(emb).astype(hidden_states.dtype), jnp.sin(emb).astype(hidden_states.dtype))

        # Compute cu_seqlens for variable-length attention
        # Each grid produces (T * H * W) tokens
        seq_lens = grid_thw[:, 1] * grid_thw[:, 2]  # H * W per frame
        cu_seqlens = jnp.concatenate([jnp.array([0]), jnp.cumsum(seq_lens)])
        cu_seqlens = cu_seqlens.astype(jnp.int32)

        # Process through transformer blocks
        deepstack_features = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings)

            # Extract DeepStack features at specified layers
            if layer_num in self.config.deepstack_visual_indexes:
                idx = self.config.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[idx](hidden_states)
                deepstack_features.append(deepstack_feature)

        # Final merger
        main_features = self.merger(hidden_states)

        return main_features, tuple(deepstack_features)


__all__ = [
    "Qwen3VLVisionPatchEmbed",
    "Qwen3VLVisionPatchMerger",
    "Qwen3VLVisionBlock",
    "Qwen3VLVisionModel",
]
