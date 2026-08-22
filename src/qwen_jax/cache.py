"""KV cache and KV prefix.

Two ways of handing attention some keys and values it did not compute itself:

- `KVCache`: fixed-capacity arrays written in place with `dynamic_update_slice`,
  so that a decode loop has static shapes. Tracks its own `position` and
  `valid` mask: a slot is attended to iff something has been written there and
  that something was not padding. Callers therefore only ever describe the
  tokens they are passing in *now*; the cache remembers the rest.

- `KVPrefix`: exact-length keys and values that attention *concatenates* in
  front of the ones it computes. No capacity, no position, no in-place update.
  This is the training-time form of a cartridge (arXiv 2506.06266) -- the K/V
  are parameters and the gradient flows into them through the concatenate --
  and also the general "here is some context in KV form" object: a cache can be
  seeded from one with `KVCache.write_prefix`, and two can be joined with
  `KVPrefix.concat`.

Both hold keys in the same space the model caches them in: after QK-norm and
after RoPE, so a prefix carries the positions it was rotated at. `KVPrefix.shift`
moves it to other positions, which RoPE makes exact.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


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
class KVPrefix:
    """Keys and values for every layer, to be attended to ahead of the input.

    Layers are stacked into one array each rather than held as a tuple, so a
    prefix is two leaves however deep the model: that is what makes it cheap to
    hand to an optimizer, save, or broadcast. The batch axis is absent -- the
    same prefix is attended by every batch item -- and is broadcast in attention.
    """

    keys: Float[Array, "layers p kv_heads head_dim"]
    values: Float[Array, "layers p kv_heads head_dim"]

    @property
    def length(self) -> int:
        return self.keys.shape[1]

    @property
    def num_layers(self) -> int:
        return self.keys.shape[0]

    def layer(self, i: int) -> KVCacheLayer:
        """This prefix's K/V for layer `i`, with a broadcastable batch axis."""
        return KVCacheLayer(keys=self.keys[i][None], values=self.values[i][None])

    def astype(self, dtype) -> KVPrefix:
        return KVPrefix(keys=self.keys.astype(dtype), values=self.values.astype(dtype))

    @classmethod
    def from_cache(cls, cache: KVCache, batch_index: int = 0) -> KVPrefix:
        """The first `cache.position` slots of one batch item, as a prefix.

        `position` must be concrete (not traced); this is a host-side
        operation for turning a prefill into something reusable.
        """
        n = int(cache.position)
        return cls(
            keys=jnp.stack([layer.keys[batch_index, :n] for layer in cache.layers]),
            values=jnp.stack([layer.values[batch_index, :n] for layer in cache.layers]),
        )

    @classmethod
    def concat(cls, *prefixes: KVPrefix) -> KVPrefix:
        """Join prefixes along the sequence axis, in order.

        Does not re-rotate anything: each piece keeps the positions it was
        built at. Use `shift` first if the pieces should sit at their new
        absolute positions.
        """
        return cls(
            keys=jnp.concatenate([p.keys for p in prefixes], axis=1),
            values=jnp.concatenate([p.values for p in prefixes], axis=1),
        )

    def shift(self, delta: int, rotary) -> KVPrefix:
        """Re-rotate the keys as if they had been computed `delta` positions later.

        Keys are cached after RoPE, so a key built at position `t` carries the
        rotation for `t`. RoPE rotations compose, so applying the rotation for
        `delta` on top gives exactly the key that would have been cached at
        `t + delta`. Values carry no position and are untouched.

        `rotary` is the model's text rotary embedding; all three MRoPE axes
        take the same position for text, so this is a single rotation.
        """
        from .rope import apply_rotary_pos_emb

        pos = jnp.full((1, self.length), delta, dtype=jnp.int32)
        cos, sin = rotary(pos)  # (1, p, head_dim)
        cos = cos.astype(jnp.float32)
        sin = sin.astype(jnp.float32)

        def rotate(k):  # (p, kv_heads, head_dim)
            kf = k.astype(jnp.float32)[None]
            rotated, _ = apply_rotary_pos_emb(kf, kf, cos, sin)
            return rotated[0].astype(k.dtype)

        return KVPrefix(keys=jax.vmap(rotate)(self.keys), values=self.values)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class KVCache:
    """Full KV cache for all self-attention layers.

    Stores a tuple of KVCacheLayer (one per self-attention layer), the current
    write position, and which slots hold real (non-padding) tokens.
    """

    layers: tuple[KVCacheLayer, ...]
    position: Int[Array, ""]  # Scalar: current sequence position
    valid: Bool[Array, "batch max_seq"]

    @classmethod
    def create(
        cls,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype=jnp.bfloat16,
    ) -> KVCache:
        """Create a new cache with pre-allocated arrays for all layers.

        Args:
            num_layers: Number of self-attention layers (not total layers).
            batch_size: Batch size.
            max_seq_len: Maximum sequence length to allocate.
            num_kv_heads: Number of key-value heads.
            head_dim: Dimension of each head.
            dtype: Data type for arrays.

        Returns:
            New KVCache with all layers initialized to zeros, nothing valid.
        """
        layers = tuple(
            KVCacheLayer.create(batch_size, max_seq_len, num_kv_heads, head_dim, dtype)
            for _ in range(num_layers)
        )
        return cls(
            layers=layers,
            position=jnp.array(0, dtype=jnp.int32),
            valid=jnp.zeros((batch_size, max_seq_len), dtype=jnp.bool),
        )

    @classmethod
    def for_model(
        cls,
        model,
        batch_size: int,
        max_seq_len: int,
        dtype=jnp.bfloat16,
        prefix: KVPrefix | None = None,
    ) -> KVCache:
        """A cache shaped for `model`'s text decoder, optionally seeded with a prefix.

        `max_seq_len` is the number of *new* tokens the cache should hold; the
        prefix's own length is added on top.
        """
        cfg = model.model.config.text_config
        total = max_seq_len + (prefix.length if prefix is not None else 0)
        cache = cls.create(
            num_layers=cfg.num_hidden_layers,
            batch_size=batch_size,
            max_seq_len=total,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            dtype=dtype,
        )
        return cache if prefix is None else cache.write_prefix(prefix)

    @property
    def max_seq_len(self) -> int:
        """Get the maximum sequence length this cache can hold."""
        return self.layers[0].keys.shape[1]

    @property
    def batch_size(self) -> int:
        return self.valid.shape[0]

    def mark(self, new_valid: Bool[Array, "batch seq"]) -> KVCache:
        """Record the validity of the `seq` tokens about to be written at `position`."""
        valid = jax.lax.dynamic_update_slice(
            self.valid, new_valid.astype(jnp.bool), (0, self.position)
        )
        return dataclasses.replace(self, valid=valid)

    def write_prefix(self, prefix: KVPrefix) -> KVCache:
        """Copy `prefix` into the slots starting at `position`, all valid, and advance.

        This is how a prefix is *served*: written once, then decoded against
        through the ordinary cached path with no per-step concatenation.
        """
        if prefix.num_layers != len(self.layers):
            raise ValueError(
                f"prefix has {prefix.num_layers} layers, cache has {len(self.layers)}"
            )
        batch = self.batch_size
        layers = tuple(
            layer.update(
                self.position,
                jnp.broadcast_to(prefix.keys[i][None], (batch, *prefix.keys.shape[1:])),
                jnp.broadcast_to(prefix.values[i][None], (batch, *prefix.values.shape[1:])),
            )
            for i, layer in enumerate(self.layers)
        )
        marked = self.mark(jnp.ones((batch, prefix.length), dtype=jnp.bool))
        return KVCache(
            layers=layers,
            position=self.position + prefix.length,
            valid=marked.valid,
        )

    def resize(self, new_max_seq_len: int) -> KVCache:
        old = self.max_seq_len
        if new_max_seq_len < old:
            valid = self.valid[:, :new_max_seq_len]
        else:
            valid = jnp.zeros((self.batch_size, new_max_seq_len), dtype=jnp.bool)
            valid = valid.at[:, :old].set(self.valid)
        return KVCache(
            layers=tuple(layer.resize(new_max_seq_len) for layer in self.layers),
            position=jnp.minimum(self.position, new_max_seq_len),
            valid=valid,
        )


__all__ = ["KVCache", "KVCacheLayer", "KVPrefix"]
