"""Qwen3-VL model implementation in JAX/Equinox."""

# Re-export common utilities
from .cache import KVCache, KVCacheLayer
from .equinox_utils import replace, mapmod, mapmod_with_path

# Linear layers
from .linear import Linear, Embedding, LayerNorm, RMSNorm

# MLP layers
from .mlp import Qwen3VLVisionMLP, Qwen3VLTextMLP

# RoPE
from .rope import (
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLTextRotaryEmbedding,
    apply_rotary_pos_emb_vision,
    apply_rotary_pos_emb,
)

# Attention
from .attention import Qwen3VLVisionAttention, Qwen3VLTextAttention

# Vision model
from .vision import (
    Qwen3VLVisionPatchEmbed,
    Qwen3VLVisionPatchMerger,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
)

# Text model
from .text import Qwen3VLTextDecoderLayer, Qwen3VLTextModel

# Full model
from .model import Qwen3VLModel, Qwen3VLForConditionalGeneration, Qwen3VLOutput

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
