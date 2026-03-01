"""KV Cache implementation for Mllama with fixed-size arrays for JIT compatibility."""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class KVCacheLayer(NamedTuple):
    """Single layer's KV cache with fixed-size pre-allocated arrays.

    Uses in-place updates via .at[].set() for JIT compatibility.
    """

    keys: Float[Array, "batch max_seq kv_heads head_dim"]
    values: Float[Array, "batch max_seq kv_heads head_dim"]

    @classmethod
    def create(
        cls,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype=jnp.bfloat16,
    ) -> KVCacheLayer:
        """Create a new cache layer with pre-allocated zero arrays."""
        shape = (batch_size, max_seq_len, num_kv_heads, head_dim)
        return cls(
            keys=jnp.zeros(shape, dtype=dtype),
            values=jnp.zeros(shape, dtype=dtype),
        )

    def update(
        self,
        position: Int[Array, ""],
        new_keys: Float[Array, "batch seq kv_heads head_dim"],
        new_values: Float[Array, "batch seq kv_heads head_dim"],
    ) -> KVCacheLayer:
        """Write new K/V at position, return new cache.

        Args:
            position: Scalar position to write at.
            new_keys: Keys to write, shape (batch, seq, kv_heads, head_dim).
            new_values: Values to write, shape (batch, seq, kv_heads, head_dim).

        Returns:
            New KVCacheLayer with updated keys and values.
        """
        seq_len = new_keys.shape[1]
        # Use dynamic_update_slice for writing multiple positions
        keys = jax.lax.dynamic_update_slice(
            self.keys, new_keys.astype(self.keys.dtype), (0, position, 0, 0)
        )
        values = jax.lax.dynamic_update_slice(
            self.values, new_values.astype(self.values.dtype), (0, position, 0, 0)
        )
        return KVCacheLayer(keys=keys, values=values)

    def get(
        self,
    ) -> tuple[
        Float[Array, "batch max_seq kv_heads head_dim"],
        Float[Array, "batch max_seq kv_heads head_dim"],
    ]:
        """Get the full keys/values arrays.

        Returns the full pre-allocated arrays. Attention masking handles
        ignoring positions beyond the current sequence length.

        Returns:
            Tuple of (keys, values) - full arrays.
        """
        return self.keys, self.values

    def resize(self, new_max_seq_len: int) -> KVCacheLayer:
        """Resize the cache to a new maximum sequence length.

        Copies existing K/V values to a new larger array.

        Args:
            new_max_seq_len: New maximum sequence length (must be >= current).

        Returns:
            New KVCacheLayer with resized arrays.
        """
        batch, old_max_seq, kv_heads, head_dim = self.keys.shape

        # Create new larger arrays and copy existing values
        if new_max_seq_len < old_max_seq:
            new_keys = self.keys[:, :new_max_seq_len, :, :]
            new_values = self.values[:, :new_max_seq_len, :, :]
            return KVCacheLayer(keys=new_keys, values=new_values)
        new_keys = jnp.zeros(
            (batch, new_max_seq_len, kv_heads, head_dim), dtype=self.keys.dtype
        )
        new_values = jnp.zeros(
            (batch, new_max_seq_len, kv_heads, head_dim), dtype=self.values.dtype
        )
        new_keys = new_keys.at[:, :old_max_seq, :, :].set(self.keys)
        new_values = new_values.at[:, :old_max_seq, :, :].set(self.values)
        return KVCacheLayer(keys=new_keys, values=new_values)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class KVCache:
    """Full KV cache for all self-attention layers.

    Stores a tuple of KVCacheLayer (one per self-attention layer) and
    tracks the current write position.
    """

    layers: tuple[KVCacheLayer, ...]
    position: Int[Array, ""]  # Scalar: current sequence position

    @classmethod
    def create(
        cls,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype=jnp.bfloat16,
    ) -> "KVCache":
        """Create a new cache with pre-allocated arrays for all layers.

        Args:
            num_layers: Number of self-attention layers (not total layers).
            batch_size: Batch size.
            max_seq_len: Maximum sequence length to allocate.
            num_kv_heads: Number of key-value heads.
            head_dim: Dimension of each head.
            dtype: Data type for arrays.

        Returns:
            New KVCache with all layers initialized to zeros.
        """
        layers = tuple(
            KVCacheLayer.create(batch_size, max_seq_len, num_kv_heads, head_dim, dtype)
            for _ in range(num_layers)
        )
        return cls(layers=layers, position=jnp.array(0, dtype=jnp.int32))

    @property
    def max_seq_len(self) -> int:
        """Get the maximum sequence length this cache can hold."""
        return self.layers[0].keys.shape[1]

    def resize(self, new_max_seq_len: int) -> KVCache:
        return KVCache(
            layers=tuple(layer.resize(new_max_seq_len) for layer in self.layers),
            position=jnp.minimum(self.position, new_max_seq_len),
        )

    def extend(self, extend_by: int) -> KVCache:
        """Extend the cache by a given number of positions."""
        new_max_seq_len = self.max_seq_len + extend_by
        return self.resize(new_max_seq_len)
