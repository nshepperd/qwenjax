"""Loading Qwen3.5 from GGUF (llama.cpp `qwen35` arch) checkpoints.

Same machinery as `qwen_jax.gguf` -- GGUFParam-holding modules swapped in
where the file holds a quantized tensor, dense tensors dequantized at load --
with the qwen35 name tables. llama.cpp's converter already baked in this
package's conventions (tiled DeltaNet value heads, A = -exp(A_log), +1 on the
zero-centered norms, squeezed conv kernels), so every tensor is a plain
rename; that was the point of adopting them.

The one wrinkle is dtype: `ssm_a` and `ssm_dt.bias` parameterize per-head
decay rates and are stored F32; they stay F32 here rather than joining the
other dense tensors in the activation dtype, since the recurrence consumes
them in float32 anyway.
"""
from __future__ import annotations

import re
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from gguf_jax import QuantizedArray, load_gguf

from qwen_jax import equinox_utils as eu
from qwen_jax.gguf import (
    DENSE_TYPES,
    EmbeddingGGUF,
    LinearGGUF,
    UnmappedTensor,
    vision_key,
)
from qwen_jax.linear import Embedding, Linear
from qwen_jax.param import AbstractParam, path_to_key

from .config import Qwen35Config
from .model import Qwen35ForConditionalGeneration

_TEXT_PREFIX = "model.language_model"

_TEXT_MAP = {
    "token_embd.weight": f"{_TEXT_PREFIX}.embed_tokens.weight",
    "output_norm.weight": f"{_TEXT_PREFIX}.norm.weight",
    "output.weight": "lm_head.weight",
}
_TEXT_LAYER_MAP = {
    # Shared
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    # Full attention (attn_q is the fused query+gate projection)
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    # Gated DeltaNet
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}

# Dense F32 tensors that stay float32 instead of the activation dtype.
_KEEP_F32 = re.compile(r"\.linear_attn\.(A|dt_bias)$")

_RE_TEXT_LAYER = re.compile(r"^blk\.(\d+)\.(.*)$")


def text_key(name: str) -> str:
    """The state dict key for a qwen35 text-model GGUF tensor name."""
    if name in _TEXT_MAP:
        return _TEXT_MAP[name]
    if m := _RE_TEXT_LAYER.match(name):
        n, rest = m.group(1), m.group(2)
        if rest in _TEXT_LAYER_MAP:
            return f"{_TEXT_PREFIX}.layers.{n}.{_TEXT_LAYER_MAP[rest]}"
    raise UnmappedTensor(name)


def gguf_state_dict(
    gguf_path: str | Path,
    mmproj_path: str | Path | None,
    dtype=jnp.bfloat16,
) -> dict[str, jax.Array | QuantizedArray]:
    """Read a GGUF (pair) into a state dict keyed the way the model tree is."""
    state_dict: dict[str, jax.Array | QuantizedArray] = {}

    def add(key: str, tensor: QuantizedArray) -> None:
        if tensor.qtype in DENSE_TYPES:
            target = jnp.float32 if _KEEP_F32.search(key) else dtype
            state_dict[key] = tensor.dequantize(target)
        else:
            state_dict[key] = tensor

    for name, tensor in load_gguf(str(gguf_path), dtype=dtype).tensors.items():
        add(text_key(name), tensor)

    if mmproj_path is not None:
        vision = load_gguf(str(mmproj_path), dtype=dtype).tensors
        patch = [vision.pop(k) for k in ("v.patch_embd.weight", "v.patch_embd.weight.1")]
        state_dict["model.visual.patch_embed.proj.weight"] = jnp.stack(
            [p.dequantize(dtype) for p in patch], axis=2
        )
        for name, tensor in vision.items():
            add(vision_key(name, deepstack_indexes=()), tensor)

    return state_dict


def load_qwen35_gguf(
    config_path: str | Path,
    gguf_path: str | Path,
    mmproj_path: str | Path | None = None,
    *,
    dtype=jnp.bfloat16,
    fused: bool = True,
    error_on_unused: bool = True,
) -> Qwen35ForConditionalGeneration:
    """Load a Qwen3.5 model from a GGUF file.

    Args:
        config_path: an HF model directory (or hub id) to take config.json
            from; the tokenizer lives there too.
        gguf_path: the text model .gguf.
        mmproj_path: the matching mmproj .gguf; without it the vision tower
            is left unloaded, which is enough for text-only work.
        fused: use the fused dequant-matmul kernels where one exists.
    """
    config = Qwen35Config.from_pretrained(config_path)
    model = Qwen35ForConditionalGeneration(config)

    state_dict = gguf_state_dict(gguf_path, mmproj_path, dtype=dtype)
    quantized = {k for k, v in state_dict.items() if isinstance(v, QuantizedArray)}

    def visit(path: jax.tree_util.KeyPath, module: eqx.Module) -> eqx.Module:
        if f"{path_to_key(path)}.weight" not in quantized:
            return module
        if isinstance(module, Linear):
            return LinearGGUF(
                module.in_features, module.out_features,
                use_bias=module.use_bias, fused=fused,
            )
        if isinstance(module, Embedding):
            return EmbeddingGGUF(module.num_embeddings, module.embedding_dim)
        return module

    model = eu.mapmod_with_path(visit, model)

    allow_missing = mmproj_path is None

    def fill(path: jax.tree_util.KeyPath, module: eqx.Module) -> eqx.Module:
        if not isinstance(module, AbstractParam):
            return module
        if allow_missing and path_to_key(path) not in state_dict:
            return module
        return module.load_state_dict(state_dict, path)

    model = eu.mapmod_with_path(fill, model)

    if error_on_unused and state_dict:
        raise AssertionError(f"Unused keys in state dict: {sorted(state_dict)}")
    return model


__all__ = ["gguf_state_dict", "load_qwen35_gguf", "text_key"]
