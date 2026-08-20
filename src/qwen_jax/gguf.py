"""Loading Qwen3-VL from GGUF (llama.cpp) checkpoints.

The shape of this mirrors `linear4bit`: a weight whose on-disk layout differs
is a different `AbstractParam` subclass -- here `GGUFParam`, holding a
`gguf_jax.QuantizedArray` -- and the modules that hold one are ordinary modules
that happen to hold an unusual param. Everything else (the tree walk in
`loading.load_state_dict`, the shape checks, the "every key must be consumed"
rule) is reused unchanged.

Two things about GGUF make it more than a rename of `linear4bit`:

  - Tensor names are llama.cpp's, not HF's, so they are translated up front
    into the state dict keys the model tree already expects. The translation
    is a pure function of the name, kept in one table each for the text model
    and the mmproj, so a mismatch shows up as an unclaimed key rather than a
    silently skipped weight.

  - A GGUF is mixed precision by construction. Norms and biases are stored
    dense (F32) even in a Q4_K file, and the whole vision tower lives in a
    separate BF16 mmproj. Only tensors in an actually-quantized qtype become
    `GGUFParam`s; the rest are dequantized at load time into ordinary `Param`s,
    so the model keeps a single activation dtype rather than promoting the
    residual stream to f32 wherever an F32 norm weight lands.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Self

import equinox as eqx
import gguf_jax
import jax
import jax.numpy as jnp
from gguf.constants import GGMLQuantizationType as QT
from gguf_jax import QuantizedArray, load_gguf
from jaxtyping import Array

from . import equinox_utils as eu
from .config import Qwen3VLConfig
from .linear import Embedding, Linear
from .model import Qwen3VLForConditionalGeneration
from .param import AbstractParam, Param, path_to_key

# Stored as plain floats: nothing to decode at inference time, so these are
# dequantized once at load and held as ordinary arrays.
DENSE_TYPES = frozenset({QT.F32, QT.F16, QT.BF16, QT.F64})

_TEXT_PREFIX = "model.language_model"
_VISION_PREFIX = "model.visual"

# llama.cpp text-model tensor names -> our state dict keys. `{n}` is the layer.
_TEXT_MAP = {
    "token_embd.weight": f"{_TEXT_PREFIX}.embed_tokens.weight",
    "output_norm.weight": f"{_TEXT_PREFIX}.norm.weight",
    "output.weight": "lm_head.weight",
}
_TEXT_LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

# mmproj (`general.architecture = clip`) names -> our state dict keys.
_VISION_MAP = {
    "v.patch_embd.bias": f"{_VISION_PREFIX}.patch_embed.proj.bias",
    "v.position_embd.weight": f"{_VISION_PREFIX}.pos_embed.weight",
    "v.post_ln.weight": f"{_VISION_PREFIX}.merger.norm.weight",
    "v.post_ln.bias": f"{_VISION_PREFIX}.merger.norm.bias",
    "mm.0.weight": f"{_VISION_PREFIX}.merger.linear_fc1.weight",
    "mm.0.bias": f"{_VISION_PREFIX}.merger.linear_fc1.bias",
    "mm.2.weight": f"{_VISION_PREFIX}.merger.linear_fc2.weight",
    "mm.2.bias": f"{_VISION_PREFIX}.merger.linear_fc2.bias",
}
_VISION_BLOCK_MAP = {
    "ln1": "norm1",
    "ln2": "norm2",
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
}
_DEEPSTACK_MAP = {"norm": "norm", "fc1": "linear_fc1", "fc2": "linear_fc2"}

_RE_TEXT_LAYER = re.compile(r"^blk\.(\d+)\.(.*)$")
_RE_VISION_BLOCK = re.compile(r"^v\.blk\.(\d+)\.(\w+)\.(weight|bias)$")
_RE_DEEPSTACK = re.compile(r"^v\.deepstack\.(\d+)\.(\w+)\.(weight|bias)$")


class UnmappedTensor(KeyError):
    """A GGUF tensor whose name has no counterpart in the model tree."""


def text_key(name: str) -> str:
    """The state dict key for a text-model GGUF tensor name."""
    if name in _TEXT_MAP:
        return _TEXT_MAP[name]
    if m := _RE_TEXT_LAYER.match(name):
        n, rest = m.group(1), m.group(2)
        if rest in _TEXT_LAYER_MAP:
            return f"{_TEXT_PREFIX}.layers.{n}.{_TEXT_LAYER_MAP[rest]}"
    raise UnmappedTensor(name)


def vision_key(name: str, deepstack_indexes: tuple[int, ...]) -> str:
    """The state dict key for an mmproj GGUF tensor name.

    `deepstack_indexes` are the vision block indexes that feed a deepstack
    merger (config order); the mmproj names its mergers by block index while
    the model holds them in a list, so the position in this tuple is the
    translation between the two.
    """
    if name in _VISION_MAP:
        return _VISION_MAP[name]
    if m := _RE_VISION_BLOCK.match(name):
        n, part, kind = m.groups()
        if part in _VISION_BLOCK_MAP:
            return f"{_VISION_PREFIX}.blocks.{n}.{_VISION_BLOCK_MAP[part]}.{kind}"
    if m := _RE_DEEPSTACK.match(name):
        block, part, kind = int(m.group(1)), m.group(2), m.group(3)
        if part in _DEEPSTACK_MAP and block in deepstack_indexes:
            i = deepstack_indexes.index(block)
            return f"{_VISION_PREFIX}.deepstack_merger_list.{i}.{_DEEPSTACK_MAP[part]}.{kind}"
    raise UnmappedTensor(name)


class GGUFParam(AbstractParam):
    """A GGUF-quantized weight, held as raw block data until the forward pass.

    Distinct from `Param` because what it holds is a `QuantizedArray`, not an
    Array: it has a logical shape but no dense values until something calls
    `.dequantize()` (or hands it to a fused kernel that never does).
    """

    array: QuantizedArray | None
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, *shape: int, array: QuantizedArray | None = None):
        self.shape = tuple(shape)
        self.array = array

    def __call__(self) -> QuantizedArray:
        if self.array is None:
            raise ValueError(
                f"GGUF parameter of shape {self.shape} has no value: the model "
                "has no weights yet. Load them with "
                "qwen_jax.gguf.load_qwen3_gguf()."
            )
        return self.array

    def is_set(self) -> bool:
        return self.array is not None

    def load_state_dict(self, state_dict: dict[str, Array], path: jax.tree_util.KeyPath) -> Self:
        key = path_to_key(path)
        if key not in state_dict:
            raise KeyError(f"Missing from state dict: {key}")
        array = state_dict.pop(key)
        if not isinstance(array, QuantizedArray):
            raise TypeError(f"{key}: expected a QuantizedArray, got {type(array).__name__}")
        if tuple(array.shape) != self.shape:
            raise ValueError(
                f"{key}: shape mismatch, expected {self.shape}, got {tuple(array.shape)}"
            )
        return eu.replace(self, array=array)


class LinearGGUF(eqx.Module):
    """`Linear` over a GGUF-quantized weight."""

    weight: GGUFParam
    bias: Param | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    fused: bool = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, *, use_bias=True, fused=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = GGUFParam(out_features, in_features)
        self.bias = Param(out_features) if use_bias else None
        self.fused = fused

    @property
    def use_bias(self) -> bool:
        return self.bias is not None

    @jax.remat  # type: ignore  # should be exported
    def __call__(self, x: Array):
        w = self.weight()
        # gguf_jax picks the route: a fused kernel where one covers this qtype
        # and batch size, otherwise dequantize-then-matmul, blocked if the
        # weight is too large to materialize at once. `fused=False` forces the
        # second, which is what makes the two comparable in a benchmark.
        y = (gguf_jax.matmul(x, w) if self.fused
             else gguf_jax.dequant_matmul(x, w))
        if self.bias is not None:
            y += self.bias()
        return y


class EmbeddingGGUF(eqx.Module):
    """`Embedding` over a GGUF-quantized table.

    Gathers the quantized *rows* and decodes only those, rather than decoding
    a 151936 x 4096 table (1.2 GB in bf16) to throw all but a few hundred rows
    of it away. GGUF rows are self-contained runs of blocks, so a gather on the
    byte axis is the same row selection as a gather on the decoded one.
    """

    weight: GGUFParam
    num_embeddings: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = GGUFParam(num_embeddings, embedding_dim)

    def __call__(self, input_ids: jax.Array):
        w = self.weight()
        rows = w.data[input_ids]  # (*input_ids.shape, row_bytes)
        return QuantizedArray(
            data=rows,
            qtype=w.qtype,
            shape=(*input_ids.shape, self.embedding_dim),
            dtype=w.dtype,
        ).dequantize()


def gguf_state_dict(
    gguf_path: str | Path,
    mmproj_path: str | Path | None,
    deepstack_indexes: tuple[int, ...],
    dtype=jnp.bfloat16,
) -> dict[str, Array | QuantizedArray]:
    """Read a GGUF pair into a state dict keyed the way the model tree is.

    Dense tensors come back as arrays of `dtype`; quantized ones stay
    `QuantizedArray`. The vision patch embedding is the one tensor that is not
    a rename: the mmproj splits the conv3d kernel across its two temporal
    slices (`v.patch_embd.weight` and `.weight.1`), which are stacked back into
    the single (out, in, t, h, w) weight the model declares.
    """
    state_dict: dict[str, Array | QuantizedArray] = {}

    def add(key: str, tensor: QuantizedArray) -> None:
        state_dict[key] = tensor.dequantize(dtype) if tensor.qtype in DENSE_TYPES else tensor

    for name, tensor in load_gguf(str(gguf_path), dtype=dtype).tensors.items():
        add(text_key(name), tensor)

    if mmproj_path is not None:
        vision = load_gguf(str(mmproj_path), dtype=dtype).tensors
        patch = [vision.pop(k) for k in ("v.patch_embd.weight", "v.patch_embd.weight.1")]
        state_dict[f"{_VISION_PREFIX}.patch_embed.proj.weight"] = jnp.stack(
            [p.dequantize(dtype) for p in patch], axis=2
        )
        for name, tensor in vision.items():
            add(vision_key(name, deepstack_indexes), tensor)

    return state_dict


def load_qwen3_gguf(
    config_path: str | Path,
    gguf_path: str | Path,
    mmproj_path: str | Path | None = None,
    *,
    dtype=jnp.bfloat16,
    fused: bool = True,
    error_on_unused: bool = True,
) -> Qwen3VLForConditionalGeneration:
    """Load a Qwen3-VL model from a GGUF file.

    Args:
        config_path: an HF model directory to take `config.json` from. A GGUF
            carries its own hyperparameters, but they are llama.cpp's names for
            llama.cpp's subset of them, and a tokenizer/processor is needed from
            the HF side anyway; reading the config from there keeps the GGUF and
            safetensors paths configured identically, which is the point when
            the two are being compared.
        gguf_path: the text model .gguf.
        mmproj_path: the matching mmproj .gguf. Without it the vision tower is
            left unloaded, which is enough for text-only work.
        fused: use the fused dequant-matmul kernels where one exists for the
            qtype. Turning this off dequantizes each weight to a dense array
            per use -- slower and far more memory, but a useful A/B.
    """
    config = Qwen3VLConfig.from_pretrained(pretrained_model_name_or_path=Path(config_path))
    model = Qwen3VLForConditionalGeneration(config)

    state_dict = gguf_state_dict(
        gguf_path,
        mmproj_path,
        tuple(config.vision_config.deepstack_visual_indexes),
        dtype=dtype,
    )
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

    # Without an mmproj the vision params have nothing to load from; leave
    # them unset rather than fail the walk over them.
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


__all__ = [
    "EmbeddingGGUF",
    "GGUFParam",
    "LinearGGUF",
    "UnmappedTensor",
    "gguf_state_dict",
    "load_qwen3_gguf",
    "text_key",
    "vision_key",
]
