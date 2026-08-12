"""Parameter slots.

A module declares each array it loads from a checkpoint as an `AbstractParam`
rather than a bare `jax.Array`. This buys three things:

  - The expected shape lives next to the array, so loading validates itself
    instead of every layer repeating the same pop/assert/replace dance.
  - An unfilled slot is distinguishable from an absent submodule (`None`),
    which a bare `Array | None` field cannot express.
  - Loading is uniform. `loading.load_state_dict` walks the tree and calls
    `load_state_dict` on every param it finds, so containers need no load
    method of their own. A weight whose on-disk layout differs is a different
    AbstractParam subclass (see `linear4bit.QuantizedParam`) rather than a
    special case in whichever module happens to hold it.

Arrays that are *derived* rather than loaded -- rotary `inv_freq`, say -- stay
bare Arrays. Not being a param is what marks them as not-from-the-checkpoint.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax
from jaxtyping import Array

from . import equinox_utils as eu


def path_to_key(path: jax.tree_util.KeyPath) -> str:
    """Render a tree path as its state dict key, e.g. `layers.0.mlp.up_proj.weight`."""
    return jax.tree_util.keystr(path, simple=True, separator=".")


class AbstractParam(eqx.Module):
    """A single slot in a module that is filled from a checkpoint.

    Subclasses differ in what they hold and how many state dict keys they span;
    they agree on knowing their shape, yielding their value when called, and
    being able to load themselves given their own path in the tree.
    """

    shape: eqx.AbstractVar[tuple[int, ...]]

    @abstractmethod
    def __call__(self):
        """The value held here, or a ValueError if nothing is loaded yet."""

    @abstractmethod
    def is_set(self) -> bool:
        """Whether a value has been loaded."""

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Array], path: jax.tree_util.KeyPath) -> Self:
        """Take this parameter's value out of `state_dict`.

        `path` is this param's own position in the model tree, which is how it
        derives its key(s). Every key it uses must be consumed, so that the
        caller can treat whatever remains as unclaimed.
        """


class Param(AbstractParam):
    """An ordinary array parameter, stored under one state dict key.

    Access it by calling it -- `self.weight()` -- which raises a useful error
    rather than letting a `None` propagate into a traceback further downstream.
    """

    array: Array | None
    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, *shape: int, array: Array | None = None):
        self.shape = tuple(shape)
        self.array = array

    def __call__(self) -> Array:
        if self.array is None:
            raise ValueError(
                f"Parameter of shape {self.shape} has no value: the model has no "
                "weights yet. Load them with qwen_jax.loading.load_qwen3_jax(), or "
                "initialise them with .init_weights(key)."
            )
        return self.array

    def is_set(self) -> bool:
        return self.array is not None

    def set(self, array: Array) -> Self:
        """Return a copy holding `array`, checked against the declared shape."""
        if tuple(array.shape) != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape}, got {tuple(array.shape)}"
            )
        return eu.replace(self, array=array)

    def load_state_dict(self, state_dict: dict[str, Array], path: jax.tree_util.KeyPath) -> Self:
        key = path_to_key(path)
        if key not in state_dict:
            raise KeyError(f"Missing from state dict: {key}")
        try:
            return self.set(state_dict.pop(key))
        except ValueError as e:
            raise ValueError(f"{key}: {e}") from None


def params_with_paths(module) -> list[tuple[jax.tree_util.KeyPath, AbstractParam]]:
    """Every param in `module`, paired with the tree path that names it.

    Passing `is_leaf` here is what keeps paths reading `...up_proj.weight`
    rather than `...up_proj.weight.array`; traversals that forget it will not
    line up with state dict keys.
    """
    leaves, _ = jax.tree_util.tree_flatten_with_path(
        module, is_leaf=lambda x: isinstance(x, AbstractParam)
    )
    return [(path, leaf) for path, leaf in leaves if isinstance(leaf, AbstractParam)]


__all__ = ["AbstractParam", "Param", "params_with_paths", "path_to_key"]
