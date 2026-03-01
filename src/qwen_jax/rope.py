"""Rotary Position Embeddings for Qwen3-VL.

Contains:
- Qwen3VLVisionRotaryEmbedding: Simple 1D RoPE for vision attention
- Qwen3VLTextRotaryEmbedding: Multimodal RoPE (MRoPE) with 3D positions
"""
from qwen_jax.config import Qwen3VLTextConfig
from einops.einops import rearrange
from .utils.buffer import Buffer
from typing import Annotated
from dataclasses import field
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from . import equinox_utils as eu


class Qwen3VLVisionRotaryEmbedding(eqx.Module):
    """Simple 1D Rotary Position Embedding for vision transformer.

    Computes frequency table that can be indexed by position.
    """
    inv_freq: Annotated[Float[Array, "dim_half"], Buffer(persistent=False)]
    theta: float = field(metadata=dict(static=True))
    dim: int = field(metadata=dict(static=True))

    def __init__(self, dim: int, theta: float = 10000.0):
        self.dim = dim
        self.theta = theta
        self.inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))

    def table(self, seq: Array):
        return jnp.outer(seq, self.inv_freq)

    def __call__(self, seqlen: int) -> Float[Array, "seqlen dim_half"]:
        """Compute frequency table for given sequence length.

        Args:
            seqlen: Maximum sequence length

        Returns:
            Frequency table of shape (seqlen, dim // 2)
        """
        seq = jnp.arange(seqlen, dtype=jnp.float32)
        freqs = jnp.outer(seq, self.inv_freq)
        return freqs


def rotate_half(x: Array) -> Array:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb_vision(
    q: Float[Array, "seq heads head_dim"],
    k: Float[Array, "seq heads head_dim"],
    cos: Float[Array, "seq head_dim"],
    sin: Float[Array, "seq head_dim"],
) -> tuple[Float[Array, "seq heads head_dim"], Float[Array, "seq heads head_dim"]]:
    """Apply rotary position embedding to vision query and key tensors.

    Vision attention doesn't have batch dimension, works on packed sequences.

    Args:
        q: Query tensor (seq, heads, head_dim)
        k: Key tensor (seq, heads, head_dim)
        cos: Cosine embeddings (seq, head_dim)
        sin: Sine embeddings (seq, head_dim)

    Returns:
        Tuple of (q_embed, k_embed) with RoPE applied
    """
    orig_dtype = q.dtype

    # Convert to float32 for numerical stability
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)

    # Add heads dimension to cos/sin: (seq, head_dim) -> (seq, 1, head_dim)
    cos = cos[:, None, :].astype(jnp.float32)
    sin = sin[:, None, :].astype(jnp.float32)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed.astype(orig_dtype), k_embed.astype(orig_dtype)


class Qwen3VLTextRotaryEmbedding(eqx.Module):
    """Multimodal Rotary Position Embedding (MRoPE) for text model.

    Uses 3D positions (temporal, height, width) and interleaves frequencies
    across the head dimension.

    mrope_section = [24, 20, 20] means:
    - First 24 dims use temporal frequencies
    - Next 20 dims use height frequencies
    - Last 20 dims use width frequencies

    The frequencies are interleaved in the pattern [T, H, W, T, H, W, ...]
    """
    inv_freq: Annotated[Float[Array, "head_dim_half"], Buffer(persistent=False)]
    mrope_section: tuple[int, int, int] = field(metadata=dict(static=True))
    attention_scaling: float = field(metadata=dict(static=True))
    head_dim: int = field(metadata=dict(static=True))

    def __init__(
        self,
        config: Qwen3VLTextConfig,
    ):
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            assert config.rope_scaling.rope_type == 'default'

        assert config.rope_scaling
        self.head_dim = config.head_dim
        self.mrope_section = config.rope_scaling.mrope_section or (24, 20, 20)
        self.attention_scaling = 1.0
        # Standard RoPE inv_freq
        rope_theta = config.rope_theta
        dim = config.head_dim
        self.inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))

    def apply_interleaved_mrope(
        self,
        freqs: Float[Array, "3 batch seq head_dim/2"],
    ) -> Float[Array, "batch seq head_dim/2"]:
        """Apply interleaved MRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHW...] pattern within each section.

        The mrope_section = [t_dim, h_dim, w_dim] specifies how many frequency
        pairs each dimension uses. Within the interleaved region (3 * min(section)),
        frequencies alternate: position 0,3,6,... use T; 1,4,7,... use H; 2,5,8,... use W.

        Args:
            freqs: Frequencies for each dimension (3, batch, seq, head_dim // 2)
                   freqs[0] = temporal, freqs[1] = height, freqs[2] = width

        Returns:
            Interleaved frequencies (batch, seq, head_dim // 2)
        """
        dim_half = freqs.shape[-1]
        mrope_array = jnp.asarray(self.mrope_section)
        idx = jnp.arange(dim_half)
        sec = idx % 3
        sec = jnp.where(idx < 3*mrope_array[sec], sec, 0)
        uwu = freqs[sec, :, :, idx]
        return rearrange(uwu, "dim b s -> b s dim")

    def __call__(
        self,
        position_ids: Int[Array, "3 batch seq"] | Int[Array, "batch seq"],
    ) -> tuple[Float[Array, "batch seq head_dim"], Float[Array, "batch seq head_dim"]]:
        """Compute cos and sin embeddings for MRoPE.

        Args:
            position_ids: 3D position IDs (3, batch, seq) or 2D (batch, seq)
                         For 2D input, expands to 3D with same positions

        Returns:
            (cos, sin) embeddings each of shape (batch, seq, head_dim)
        """
        # Expand 2D to 3D if needed
        if position_ids.ndim == 2:
            position_ids = jnp.broadcast_to(
                position_ids[None, ...],
                (3, position_ids.shape[0], position_ids.shape[1])
            )

        # Shape: (3, batch, seq)
        batch_size = position_ids.shape[1]
        seq_len = position_ids.shape[2]

        # Compute frequencies for each dimension
        # inv_freq: (head_dim // 2,)
        # position_ids: (3, batch, seq)
        # We want: freqs (3, batch, seq, head_dim // 2)
        position_ids_float = position_ids.astype(jnp.float32)

        # freqs[d, b, s, :] = position_ids[d, b, s] * inv_freq
        freqs = jnp.einsum("dbs,h->dbsh", position_ids_float, self.inv_freq)

        # Apply interleaving
        freqs = self.apply_interleaved_mrope(freqs)

        # Double the frequencies: (batch, seq, head_dim // 2) -> (batch, seq, head_dim)
        emb = jnp.concatenate([freqs, freqs], axis=-1)

        cos = jnp.cos(emb) * self.attention_scaling
        sin = jnp.sin(emb) * self.attention_scaling

        return cos, sin


def apply_rotary_pos_emb(
    q: Float[Array, "batch seq heads head_dim"],
    k: Float[Array, "batch seq kv_heads head_dim"],
    cos: Float[Array, "batch seq head_dim"],
    sin: Float[Array, "batch seq head_dim"],
) -> tuple[Float[Array, "batch seq heads head_dim"], Float[Array, "batch seq kv_heads head_dim"]]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor (batch, seq, heads, head_dim)
        k: Key tensor (batch, seq, kv_heads, head_dim)
        cos: Cosine embeddings (batch, seq, head_dim)
        sin: Sine embeddings (batch, seq, head_dim)

    Returns:
        Tuple of (q_embed, k_embed) with RoPE applied
    """
    # Add heads dimension: (batch, seq, head_dim) -> (batch, seq, 1, head_dim)
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


__all__ = [
    "Qwen3VLVisionRotaryEmbedding",
    "Qwen3VLTextRotaryEmbedding",
    "apply_rotary_pos_emb_vision",
    "apply_rotary_pos_emb",
    "rotate_half",
]
