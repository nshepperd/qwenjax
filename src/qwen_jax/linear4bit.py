from __future__ import annotations

from typing import Self

import equinox as eqx
import jax
from bnb_jax.dequantize import QuantizedArray
from jaxtyping import Array

from . import equinox_utils as eu
from .param import AbstractParam, Param, path_to_key


def _strip_prefix(s: str, state_dict: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Strip prefix s from all keys in state_dict that start with s."""
    return {k[len(s):]: state_dict.pop(k) for k in list(state_dict.keys()) if k.startswith(s)}


class QuantizedParam(AbstractParam):
    """A 4-bit weight: a packed array plus bitsandbytes quantization state.

    Not a `Param`, because it holds neither an Array nor a single state dict
    entry -- the packed data sits at its own path and the quant state under
    `<path>.*`. Being its own AbstractParam is what keeps Linear4bit free of
    special cases: an ordinary module that happens to hold an unusual param.
    """

    array: QuantizedArray | None
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, *shape: int, array: QuantizedArray | None = None):
        self.shape = tuple(shape)
        self.array = array

    def __call__(self) -> QuantizedArray:
        if self.array is None:
            raise ValueError(
                f"Quantized parameter of shape {self.shape} has no value: the "
                "model has no weights yet. Load them with "
                "qwen_jax.loading.load_qwen3_jax()."
            )
        return self.array

    def is_set(self) -> bool:
        return self.array is not None

    def load_state_dict(self, state_dict: dict[str, jax.Array], path: jax.tree_util.KeyPath) -> Self:
        key = path_to_key(path)
        if key not in state_dict:
            raise KeyError(f"Missing from state dict: {key}")
        packed = state_dict.pop(key)
        quantized = QuantizedArray.from_dict(packed, _strip_prefix(key + ".", state_dict))
        if tuple(quantized.shape) != self.shape:
            raise ValueError(
                f"{key}: shape mismatch, expected {self.shape}, got {tuple(quantized.shape)}"
            )
        return eu.replace(self, array=quantized)


class Linear4bit(eqx.Module):
    weight: QuantizedParam
    bias: Param | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, *, use_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = QuantizedParam(out_features, in_features)
        self.bias = Param(out_features) if use_bias else None

    @property
    def use_bias(self) -> bool:
        return self.bias is not None

    @jax.remat  # type: ignore  # should be exported
    def __call__(self, x: Array):
        y = x @ self.weight().dequantize().T
        if self.bias is not None:
            y += self.bias()
        return y
