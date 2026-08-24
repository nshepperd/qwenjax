"""Qwen3.5: hybrid Gated DeltaNet / gated-attention decoder, natively multimodal.

Model classes are separate from the Qwen3-VL set (the architectures diverge
structurally); the leaf primitives, vision tower, mRoPE machinery, and GGUF
plumbing are shared with the rest of `qwen_jax`.
"""
from __future__ import annotations

from .attention import Qwen35Attention
from .cache import DeltaNetCacheLayer, Qwen35Cache
from .config import Qwen35Config, Qwen35TextConfig
from .deltanet import (
    CausalConv1d,
    GatedRMSNorm,
    Qwen35GatedDeltaNet,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule_step,
)
from .gguf import load_qwen35_gguf
from .loading import convert_hf_state_dict, load_qwen35_jax
from .model import (
    Qwen35ForConditionalGeneration,
    Qwen35GenerateOutput,
    Qwen35Model,
    Qwen35Output,
)
from .text import Qwen35DecoderLayer, Qwen35TextModel

__all__ = [
    "CausalConv1d",
    "DeltaNetCacheLayer",
    "GatedRMSNorm",
    "Qwen35Attention",
    "Qwen35Cache",
    "Qwen35Config",
    "Qwen35DecoderLayer",
    "Qwen35ForConditionalGeneration",
    "Qwen35GatedDeltaNet",
    "Qwen35GenerateOutput",
    "Qwen35Model",
    "Qwen35Output",
    "Qwen35TextConfig",
    "Qwen35TextModel",
    "chunk_gated_delta_rule",
    "convert_hf_state_dict",
    "load_qwen35_gguf",
    "load_qwen35_jax",
    "recurrent_gated_delta_rule_step",
]
