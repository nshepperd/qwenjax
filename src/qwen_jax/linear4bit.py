from __future__ import annotations

import equinox as eqx
import jax
from bnb_jax.dequantize import QuantizedArray
from jaxtyping import Array

from . import equinox_utils as eu


def _strip_prefix(s: str, state_dict: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Strip prefix s from all keys in state_dict that start with s."""
    return {k[len(s):]: state_dict.pop(k) for k in list(state_dict.keys()) if k.startswith(s)}

class Linear4bit(eqx.Module):
    weight: QuantizedArray | None
    bias: jax.Array | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, *, use_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight = None
        self.bias = None

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        """Load state dict with quantized weight."""
        assert f'{prefix}weight' in state_dict, f"Missing in state dict: {prefix}weight"
        weight_array = state_dict.pop(prefix + "weight")
        weight = QuantizedArray.from_dict(weight_array, _strip_prefix(prefix + "weight.", state_dict))
        assert weight.shape == (self.out_features, self.in_features), \
            f"Weight shape mismatch: expected {(self.out_features, self.in_features)}, got {weight.shape}"
        if self.use_bias:
            bias = state_dict.pop(prefix + "bias")
            assert bias.shape == (self.out_features,), \
                f"Bias shape mismatch: expected {(self.out_features,)}, got {bias.shape}"
        else:
            bias = None
        return eu.replace(self, weight=weight, bias=bias)

    @jax.remat # type: ignore [should be exported]
    def __call__(self, x: Array):
        if self.weight is None:
            raise ValueError("Linear4bit weight is not initialized.")
        y = x @ self.weight.dequantize().T
        if self.bias is not None:
            y += self.bias
        return y
