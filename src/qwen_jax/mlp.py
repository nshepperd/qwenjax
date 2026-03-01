"""MLP layers for Qwen3-VL."""
from __future__ import annotations
from dataclasses import field

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from . import equinox_utils as eu

from .linear import Linear


class Qwen3VLVisionMLP(eqx.Module):
    hidden_size: int = eqx.field(static=True)
    intermediate_size: int = eqx.field(static=True)
    linear_fc1: Linear
    linear_fc2: Linear

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.linear_fc1 = Linear(hidden_size, intermediate_size, use_bias=True)
        self.linear_fc2 = Linear(intermediate_size, hidden_size, use_bias=True)

    def init_weights(self, key: PRNGKeyArray) -> Qwen3VLVisionMLP:
        key1, key2 = jax.random.split(key)
        linear_fc1 = self.linear_fc1.init_weights(key1)
        linear_fc2 = self.linear_fc2.init_weights(key2)
        return eu.replace(self, linear_fc1=linear_fc1, linear_fc2=linear_fc2)

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        fc1 = self.linear_fc1.load_state_dict(state_dict, prefix + "linear_fc1.")
        fc2 = self.linear_fc2.load_state_dict(state_dict, prefix + "linear_fc2.")
        return eu.replace(self, linear_fc1=fc1, linear_fc2=fc2)

    def __call__(self, hidden_states: Float[Array, "... hidden"]) -> Float[Array, "... hidden"]:
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states, approximate=True)
        hidden_states = self.linear_fc2(hidden_states)
        return hidden_states


class Qwen3VLTextMLP(eqx.Module):
    """Text MLP with SwiGLU activation.

    SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    hidden_size: int = field(metadata=dict(static=True))
    intermediate_size: int = field(metadata=dict(static=True))

    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = Linear(hidden_size, intermediate_size, use_bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, use_bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, use_bias=False)

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        """Load state dict into module."""
        gate = self.gate_proj.load_state_dict(state_dict, prefix + "gate_proj.")
        up = self.up_proj.load_state_dict(state_dict, prefix + "up_proj.")
        down = self.down_proj.load_state_dict(state_dict, prefix + "down_proj.")
        return eu.replace(self, gate_proj=gate, up_proj=up, down_proj=down)

    def __call__(self, hidden_states: Float[Array, "... hidden"]) -> Float[Array, "... hidden"]:
        gate = jax.nn.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


__all__ = ["Qwen3VLVisionMLP", "Qwen3VLTextMLP"]
