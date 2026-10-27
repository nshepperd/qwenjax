"""Loading Llama-architecture models from GGUF (llama.cpp `llama` arch).

The same pipeline as `qwen_jax.gguf` with the llama name table, plus one
transform: llama.cpp's converter permutes the rows of the q and k projections
(weights and biases) so that each head's rotary pairs sit adjacent -- its
`llama` arch applies RoPE to interleaved pairs, where HF (and this package)
rotates the two halves of the head. `unpermute_rows` is the inverse. A GGUF
row is a self-contained run of blocks, so undoing it is a gather over the
packed rows, quantized or not.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from gguf_jax import QuantizedArray, load_gguf

from qwen_jax.gguf import dense_or_quantized, load_into, rename

from .config import LlamaConfig
from .model import LlamaForCausalLM

_TEXT_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
_TEXT_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_output.bias": "self_attn.o_proj.bias",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

_RE_Q = re.compile(r"\.self_attn\.q_proj\.(weight|bias)$")
_RE_K = re.compile(r"\.self_attn\.k_proj\.(weight|bias)$")


def text_key(name: str) -> str:
    """The state dict key for a llama-arch GGUF tensor name."""
    return rename(name, _TEXT_MAP, _TEXT_LAYER_MAP, "model.layers")


def unpermute_rows(num_heads: int, rows: int) -> np.ndarray:
    """Row order that undoes llama.cpp's q/k permute: `hf = gguf[perm]`.

    llama.cpp reshapes each head's rows (2, head_dim/2) -> (head_dim/2, 2),
    interleaving the two rotary halves; this is the gather back.
    """
    head_dim = rows // num_heads
    return np.arange(rows).reshape(num_heads, head_dim // 2, 2).swapaxes(1, 2).reshape(-1)


def _take_rows(x: jax.Array | QuantizedArray, perm: np.ndarray) -> jax.Array | QuantizedArray:
    if isinstance(x, QuantizedArray):
        return dataclasses.replace(x, data=x.data[perm])
    return x[perm]


def gguf_state_dict(
    gguf_path: str | Path, config: LlamaConfig, dtype=jnp.bfloat16
) -> dict[str, jax.Array | QuantizedArray]:
    """Read a llama-arch GGUF into a state dict keyed the way the model tree is."""
    q_rows = config.num_attention_heads * config.head_dim
    kv_rows = config.num_key_value_heads * config.head_dim
    q_perm = unpermute_rows(config.num_attention_heads, q_rows)
    k_perm = unpermute_rows(config.num_key_value_heads, kv_rows)

    state_dict: dict[str, jax.Array | QuantizedArray] = {}
    for name, tensor in load_gguf(str(gguf_path), dtype=dtype).tensors.items():
        key = text_key(name)
        value = dense_or_quantized(tensor, dtype)
        if _RE_Q.search(key):
            value = _take_rows(value, q_perm)
        elif _RE_K.search(key):
            value = _take_rows(value, k_perm)
        state_dict[key] = value
    return state_dict


def load_llama_gguf(
    config_path: str | Path,
    gguf_path: str | Path,
    *,
    dtype=jnp.bfloat16,
    fused: bool = True,
    error_on_unused: bool = True,
) -> LlamaForCausalLM:
    """Load a Llama model from a GGUF file.

    Args:
        config_path: an HF model directory (or hub id) to take config.json
            from; the tokenizer lives there too.
        gguf_path: the .gguf file.
        fused: use the fused dequant-matmul kernels where one exists.
    """
    config = LlamaConfig.from_pretrained(config_path)
    return load_into(
        LlamaForCausalLM(config),
        gguf_state_dict(gguf_path, config, dtype=dtype),
        fused=fused,
        error_on_unused=error_on_unused,
    )


__all__ = ["gguf_state_dict", "load_llama_gguf", "text_key", "unpermute_rows"]
