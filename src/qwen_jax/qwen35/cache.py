"""Hybrid cache for Qwen3.5.

A Qwen3.5 stack is mostly Gated DeltaNet layers with a full-attention layer
every fourth position, so the cache is heterogeneous: DeltaNet layers carry a
small fixed-size recurrent state (conv context + SSM state), full-attention
layers carry an ordinary KV cache. `layers` holds one entry per decoder layer
in layer order, typed by what that layer is -- the model's layer loop threads
each layer's own cache through untouched, so nothing outside the layer needs
to know which is which.
"""
from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from qwen_jax.cache import KVCacheLayer

from .config import Qwen35TextConfig


class DeltaNetCacheLayer(NamedTuple):
    """Recurrent state for one Gated DeltaNet layer.

    conv_state holds the last kernel-1 inputs to the short causal conv (what
    the next step needs as left context). recurrent_state is the delta-rule
    SSM state, kept in float32 per the checkpoint's mamba_ssm_dtype.
    """

    conv_state: Float[Array, "batch kernel_minus_1 conv_dim"]
    recurrent_state: Float[Array, "batch num_v_heads head_k_dim head_v_dim"]

    @classmethod
    def create(
        cls,
        batch_size: int,
        config: Qwen35TextConfig,
        dtype=jnp.bfloat16,
        ssm_dtype=jnp.float32,
    ) -> DeltaNetCacheLayer:
        conv_dim = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        return cls(
            conv_state=jnp.zeros(
                (batch_size, config.linear_conv_kernel_dim - 1, conv_dim), dtype=dtype
            ),
            recurrent_state=jnp.zeros(
                (
                    batch_size,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                ),
                dtype=ssm_dtype,
            ),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Qwen35Cache:
    """Per-layer cache entries (in layer order) plus the write position.

    The position only advances KV writes and mask offsets; DeltaNet layers
    fold everything they need into their own state.
    """

    layers: tuple[KVCacheLayer | DeltaNetCacheLayer, ...]
    position: Int[Array, ""]

    @classmethod
    def create(
        cls,
        config: Qwen35TextConfig,
        batch_size: int,
        max_seq_len: int,
        dtype=jnp.bfloat16,
    ) -> Qwen35Cache:
        layers: list[KVCacheLayer | DeltaNetCacheLayer] = []
        for layer_type in config.resolved_layer_types():
            if layer_type == "full_attention":
                layers.append(
                    KVCacheLayer.create(
                        batch_size,
                        max_seq_len,
                        config.num_key_value_heads,
                        config.head_dim,
                        dtype,
                    )
                )
            else:
                layers.append(DeltaNetCacheLayer.create(batch_size, config, dtype))
        return cls(layers=tuple(layers), position=jnp.array(0, dtype=jnp.int32))

    @property
    def max_seq_len(self) -> int:
        """The KV capacity (DeltaNet layers have no length to run out of)."""
        for layer in self.layers:
            if isinstance(layer, KVCacheLayer):
                return layer.keys.shape[1]
        raise ValueError("No full-attention layers in this cache.")

    def resize(self, new_max_seq_len: int) -> Qwen35Cache:
        return Qwen35Cache(
            layers=tuple(
                layer.resize(new_max_seq_len) if isinstance(layer, KVCacheLayer) else layer
                for layer in self.layers
            ),
            position=jnp.minimum(self.position, new_max_seq_len),
        )


__all__ = ["DeltaNetCacheLayer", "Qwen35Cache"]
