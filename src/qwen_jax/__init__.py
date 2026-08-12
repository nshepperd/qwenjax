"""Qwen3-VL model implementation in JAX/Equinox."""

# Re-export common utilities
# Attention
from __future__ import annotations

from .attention import Qwen3VLTextAttention, Qwen3VLVisionAttention
from .cache import KVCache, KVCacheLayer
from .equinox_utils import mapmod, mapmod_with_path, replace

# Linear layers
from .linear import Embedding, LayerNorm, Linear, RMSNorm

# MLP layers
from .mlp import Qwen3VLTextMLP, Qwen3VLVisionMLP

# Full model
from .model import Qwen3VLForConditionalGeneration, Qwen3VLModel, Qwen3VLOutput

# RoPE
from .rope import (
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionRotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
)

# Text model
from .text import Qwen3VLTextDecoderLayer, Qwen3VLTextModel

# Vision model
from .vision import (
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
)

__all__ = [
    # Cache
    "KVCache",
    "KVCacheLayer",
    # Utilities
    "replace",
    "mapmod",
    "mapmod_with_path",
    # Linear layers
    "Linear",
    "Embedding",
    "LayerNorm",
    "RMSNorm",
    # MLP
    "Qwen3VLVisionMLP",
    "Qwen3VLTextMLP",
    # RoPE
    "Qwen3VLVisionRotaryEmbedding",
    "Qwen3VLTextRotaryEmbedding",
    "apply_rotary_pos_emb_vision",
    "apply_rotary_pos_emb",
    # Attention
    "Qwen3VLVisionAttention",
    "Qwen3VLTextAttention",
    # Vision model
    "Qwen3VLVisionPatchEmbed",
    "Qwen3VLVisionPatchMerger",
    "Qwen3VLVisionBlock",
    "Qwen3VLVisionModel",
    # Text model
    "Qwen3VLTextDecoderLayer",
    "Qwen3VLTextModel",
    # Full model
    "Qwen3VLModel",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLOutput",
]
