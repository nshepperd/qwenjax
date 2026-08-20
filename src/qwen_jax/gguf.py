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

import os
import re
from pathlib import Path
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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


def _fused_matmul(x: Array, w: QuantizedArray) -> Array | None:
    """`x @ w.T` via a fused dequant-matmul kernel, or None if there isn't one.

    The kernels keep the weight quantized in HBM, which is the difference
    between a 7 GB resident model and one that transiently materializes a
    dense copy of whichever tensor it is multiplying by.

    Each kernel only fuses up to some batch size and above it falls back to
    dequantizing the whole weight -- exactly the allocation the fused path
    exists to avoid, and enough to OOM a 16 GB card on the lm_head during
    prefill. Past that point this returns None so the caller can dequantize in
    bounded blocks instead.
    """
    if x.dtype != jnp.bfloat16:
        return None
    try:
        from gguf_jax import cute
        from gguf_jax.cute import iq4_xs as _iq4_xs
        from gguf_jax.cute import q4_k as _q4_k
        from gguf_jax.cute import q5_k as _q5_k
        from gguf_jax.cute import q6_k as _q6_k
    except ImportError:
        return None
    batch = int(np.prod(x.shape[:-1]))
    if w.qtype == QT.Q4_K:
        # The only type with a tensor-core GEMM as well, and it only applies
        # when the output rows tile evenly.
        fused_max = max(_q4_k._GEMV_MAX_M,
                        _q4_k._GEMM_MAX_M if w.shape[0] % _q4_k_gemm_bn() == 0 else 0)
        return cute.matmul_q4_k(x, w) if batch <= fused_max else None
    # The warp-GEMV-only types. Missing one of these is expensive out of
    # proportion to its share of the weights: an unfused tensor does not merely
    # skip the fast path, it adds a dense round-trip that the fused ones never
    # pay, which is what made UD-Q4_K_XL decode at 59 tok/s against Q4_K_M's 91
    # on 18% unfused bytes.
    for qtype, fn, mod in (
        (QT.Q6_K, cute.matmul_q6_k, _q6_k),
        (QT.Q5_K, cute.matmul_q5_k, _q5_k),
        (QT.IQ4_XS, cute.matmul_iq4_xs, _iq4_xs),
    ):
        if w.qtype == qtype:
            return fn(x, w) if batch <= mod._GEMV_MAX_M else None
    return None


def _q4_k_gemm_bn() -> int:
    from gguf_jax.cute.q4_k_gemm import _BN

    return _BN


# Cap on the temporary a single dequantize may allocate, so that one weight too
# large to sit in scratch is split across output rows -- they are independent,
# so splitting them bounds the temporary without changing the result. The
# 151936 x 4096 lm_head is the case that needs it: 1.24 GB dequantized.
#
# Sized as 4*N*K, i.e. as though the buffer were float32. It is not -- decoding
# is fused into the convert, so what reaches memory is bf16 at 2*N*K, and this
# budget is a 2x conservative one. Left as is deliberately; the headroom is
# cheap and the number of buffers matters more than the size of any one of them
# (see the optimization barrier in `_blocked_matmul`).
DEQUANT_BLOCK_BYTES = int(os.environ.get("QWEN_JAX_DEQUANT_BLOCK_BYTES", 512 << 20))


def _block_count(out_rows: int, in_features: int) -> int:
    """How many row blocks keep one dequantize under `DEQUANT_BLOCK_BYTES`.

    Constrained to divisors of `out_rows` so the blocks are uniform, which is
    what lets the split be a `scan`.
    """
    needed = 4 * out_rows * in_features
    if needed <= DEQUANT_BLOCK_BYTES:
        return 1
    target = -(-needed // DEQUANT_BLOCK_BYTES)
    for n in range(target, out_rows + 1):
        if out_rows % n == 0:
            return n
    return out_rows


def _blocked_matmul(x: Array, w: QuantizedArray) -> Array:
    """`x @ w.T`, dequantizing the weight in bounded blocks of output rows.

    A loop rather than a Python loop over slices: unrolled blocks are
    independent, so XLA schedules them concurrently and every block's temporary
    is live at once -- which is the allocation this is trying to avoid. The
    loop makes the sequencing explicit, so only one block is resident at a
    time. The same hazard across *different* weights is what the optimization
    barrier below handles; this loop only orders the blocks within one.

    Each block writes its columns into the output in place. Stacking the blocks
    and transposing instead would be shorter, but it materializes the whole
    result twice and the transpose of that is large enough that XLA fails to
    find a config for it.
    """
    out_rows, in_features = w.shape
    blocks = _block_count(out_rows, in_features)

    # Pin the dequantize to this layer's place in the network.
    #
    # A dequantize depends only on the weight, which is a parameter and so is
    # available from the first instruction of the executable. XLA is therefore
    # free to hoist every one of them arbitrarily early, and it does: profiling
    # a Q6_K prefill (where M is above every fused cap, so all ~252 matmuls take
    # this path) found 108 dequantized gate/up weights laid out at 68 distinct
    # offsets in the temp arena -- 68 live at once, 3.66 GiB of scratch on top
    # of 6.7 GB of weights, which does not fit on a 16 GB card.
    #
    # Routing the weight bytes through an optimization barrier together with x
    # gives the dequantize an artificial dependency on the activation arriving
    # at this layer, so it cannot be scheduled before the layers feeding it.
    # That chains the dequantizes into the network's own sequential order and
    # takes the arena to 193 MiB -- about two live, which is the real floor
    # here: gate_proj and up_proj consume the same x, so those two legitimately
    # overlap. Semantically a no-op; verified bitwise identical output.
    data, x = jax.lax.optimization_barrier((w.data, x))
    w = QuantizedArray(data=data, qtype=w.qtype, shape=w.shape, dtype=w.dtype)

    if blocks == 1:
        return x @ w.dequantize().T

    rows = out_rows // blocks
    flat = x.reshape(-1, in_features)
    data = w.data.reshape(blocks, rows, -1)
    out = jnp.zeros((flat.shape[0], out_rows), dtype=w.dtype)

    def body(i, out):
        sub = QuantizedArray(
            data=jax.lax.dynamic_index_in_dim(data, i, keepdims=False),
            qtype=w.qtype, shape=(rows, in_features), dtype=w.dtype,
        )
        return jax.lax.dynamic_update_slice(out, flat @ sub.dequantize().T, (0, i * rows))

    out = jax.lax.fori_loop(0, blocks, body, out)
    return out.reshape(*x.shape[:-1], out_rows)


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
        y = _fused_matmul(x, w) if self.fused else None
        if y is None:
            y = _blocked_matmul(x, w)
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
