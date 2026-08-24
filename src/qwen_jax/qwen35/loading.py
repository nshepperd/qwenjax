"""Loading Qwen3.5 from HF safetensors checkpoints.

HF stores DeltaNet value heads grouped by key head and A as A_log; this
package (like llama.cpp GGUFs) stores heads tiled and A as -exp(A_log), and
its RMSNorms are ordinary rather than zero-centered. `convert_hf_state_dict`
is the single place those conventions meet: it takes HF-named tensors and
returns tensors keyed and laid out the way the model tree expects. The GGUF
path needs none of this -- llama.cpp's converter already applied the same
transforms when the GGUF was written.
"""
from __future__ import annotations

import re
from pathlib import Path

import jax.numpy as jnp
import safetensors.flax as st
from jaxtyping import Array
from tqdm import tqdm

from qwen_jax.loading import load_state_dict

from .config import Qwen35Config, Qwen35TextConfig
from .model import Qwen35ForConditionalGeneration

# Text-side zero-centered RMSNorms, stored in HF as (weight - 1). The gated
# DeltaNet norm (linear_attn.norm) is an ordinary RMSNorm and is excluded;
# vision norms are LayerNorms and never match these patterns.
_RE_CENTERED_NORM = re.compile(
    r"(?:^|\.)(?:input_layernorm|post_attention_layernorm|q_norm|k_norm|norm)\.weight$"
)
_RE_LINEAR_ATTN = re.compile(r"(?:^|\.)linear_attn\.(?P<field>[A-Za-z0-9_.]+)$")


def tiled_head_order(num_k_heads: int, num_v_heads: int) -> "jnp.ndarray":
    """Index i of the grouped-order head that lands at tiled position i.

    Grouped (HF): heads sorted by (k_head, replica). Tiled (llama.cpp/here):
    sorted by (replica, k_head), so elementwise ops against k-head-sized
    tensors broadcast as a plain tile.
    """
    ratio = num_v_heads // num_k_heads
    return jnp.arange(num_v_heads).reshape(num_k_heads, ratio).T.flatten()


def _row_perm(order: Array, block: int) -> Array:
    """Expand a head permutation to row indices with `block` rows per head."""
    return (order[:, None] * block + jnp.arange(block)[None, :]).flatten()


def convert_hf_state_dict(
    hf_state_dict: dict[str, Array],
    text_config: Qwen35TextConfig,
) -> dict[str, Array]:
    """Convert an HF Qwen3.5 state dict to this package's keys and layouts.

    Accepts both Qwen3_5ForCausalLM keys (model.layers...) and
    Qwen3_5ForConditionalGeneration keys (model.language_model.layers...);
    the former are lifted to the latter. MTP weights are dropped.
    """
    num_k = text_config.linear_num_key_heads
    num_v = text_config.linear_num_value_heads
    head_k = text_config.linear_key_head_dim
    head_v = text_config.linear_value_head_dim
    key_dim = num_k * head_k
    order = tiled_head_order(num_k, num_v)

    has_lm_prefix = any(k.startswith("model.language_model.") for k in hf_state_dict)

    out: dict[str, Array] = {}
    for key, value in hf_state_dict.items():
        if key.startswith("mtp.") or ".mtp." in key or key.startswith("model.mtp."):
            continue
        if not has_lm_prefix and key.startswith("model."):
            key = "model.language_model." + key.removeprefix("model.")

        if m := _RE_LINEAR_ATTN.search(key):
            field = m.group("field")
            if field == "A_log":
                key = key.removesuffix("A_log") + "A"
                value = -jnp.exp(value.astype(jnp.float32))[order]
            elif field == "dt_bias":
                value = value[order]
            elif field == "conv1d.weight":
                value = value.squeeze(1)  # (conv_dim, 1, K) -> (conv_dim, K)
                qk, v_part = value[: 2 * key_dim], value[2 * key_dim :]
                value = jnp.concatenate([qk, v_part[_row_perm(order, head_v)]], axis=0)
            elif field == "in_proj_qkv.weight":
                qk, v_part = value[: 2 * key_dim], value[2 * key_dim :]
                value = jnp.concatenate([qk, v_part[_row_perm(order, head_v)]], axis=0)
            elif field == "in_proj_z.weight":
                value = value[_row_perm(order, head_v)]
            elif field in ("in_proj_a.weight", "in_proj_b.weight"):
                value = value[order]
            elif field == "out_proj.weight":
                value = value[:, _row_perm(order, head_v)]
            # norm.weight and the projections' absent biases pass through.
            out[key] = value
            continue

        if _RE_CENTERED_NORM.search(key) and ".visual." not in key:
            value = value + 1.0
        out[key] = value

    return out


def load_qwen35_jax(
    model_path: str | Path, error_on_unused: bool = True
) -> Qwen35ForConditionalGeneration:
    """Load Qwen3.5 from an HF model directory of safetensors."""
    model_path = Path(model_path)
    config = Qwen35Config.from_pretrained(model_path)

    hf_state_dict = {}
    for filename in tqdm(sorted(model_path.glob("*.safetensors")), desc="Loading safetensors"):
        hf_state_dict.update(st.load_file(filename))
    state_dict = convert_hf_state_dict(hf_state_dict, config.text_config)

    model = Qwen35ForConditionalGeneration(config)
    model = load_state_dict(model, state_dict)
    if error_on_unused:
        assert len(state_dict) == 0, f"Unused keys in state_dict: {sorted(state_dict)}"
    return model


__all__ = ["convert_hf_state_dict", "load_qwen35_jax", "tiled_head_order"]
