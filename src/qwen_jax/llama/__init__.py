"""Llama-architecture decoders (HF `LlamaForCausalLM`).

Shares the leaf modules (Linear, RMSNorm, SwiGLU MLP), the KV cache, the
attention dispatch, and the causal-LM plumbing with the rest of `qwen_jax`.
"""
from __future__ import annotations

from .config import LlamaConfig
from .gguf import load_llama_gguf
from .loading import load_llama_jax
from .model import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaOutput,
    LlamaRotaryEmbedding,
)

__all__ = [
    "LlamaAttention",
    "LlamaConfig",
    "LlamaDecoderLayer",
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaOutput",
    "LlamaRotaryEmbedding",
    "load_llama_gguf",
    "load_llama_jax",
]
