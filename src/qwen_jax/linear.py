"""Linear layers for Qwen3-VL.

Contains Linear, Embedding, LayerNorm (ported from llama_jax) and RMSNorm.
"""
from dataclasses import field
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import numpy as np

from . import equinox_utils as eu


class Linear(eqx.Module):
    weight: jax.Array | None
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

    def init_weights(self, key: PRNGKeyArray):
        weight = jax.random.normal(
            key, (self.out_features, self.in_features)
        ) / np.sqrt(self.in_features)
        bias = jnp.zeros((self.out_features,)) if self.use_bias else None
        return eu.replace(self, weight=weight, bias=bias)

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        """Load state dict into module."""
        weight = state_dict.pop(prefix + "weight")
        assert weight.shape == (self.out_features, self.in_features), (
            f"Weight shape mismatch: expected {(self.out_features, self.in_features)}, got {weight.shape}"
        )
        if self.use_bias:
            bias = state_dict.pop(prefix + "bias")
            assert bias.shape == (self.out_features,), (
                f"Bias shape mismatch: expected {(self.out_features,)}, got {bias.shape}"
            )
        else:
            bias = None
        return eu.replace(self, weight=weight, bias=bias)

    def __call__(self, x: Array):
        assert self.weight is not None, "Weights are not initialized."
        y = x @ self.weight.T
        if self.use_bias:
            assert self.bias is not None, "Bias is not initialized."
            y += self.bias
        return y


class Embedding(eqx.Module):
    weight: jax.Array | None
    num_embeddings: int = field(metadata=dict(static=True))
    embedding_dim: int = field(metadata=dict(static=True))

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = None

    def init_weights(self, key: PRNGKeyArray):
        weight = jax.random.normal(
            key, (self.num_embeddings, self.embedding_dim)
        ) / np.sqrt(self.embedding_dim)
        return eu.replace(self, weight=weight)

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        """Load state dict into module."""
        weight = state_dict.pop(prefix + "weight")
        assert weight.shape == (self.num_embeddings, self.embedding_dim), (
            f"Weight shape mismatch: expected {(self.num_embeddings, self.embedding_dim)}, got {weight.shape}"
        )
        return eu.replace(self, weight=weight)

    def __call__(self, input_ids: jax.Array):
        assert self.weight is not None, "Weights are not initialized."
        return self.weight[input_ids]


class LayerNorm(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    eps: float = field(metadata=dict(static=True))

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = jnp.ones((hidden_size,))
        self.bias = jnp.zeros((hidden_size,))
        self.eps = eps

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        """Load state dict into module."""
        weight = state_dict.pop(prefix + "weight")
        bias = state_dict.pop(prefix + "bias")
        assert weight.shape == self.weight.shape, (
            f"Weight shape mismatch: expected {(self.weight.shape,)}, got {weight.shape}"
        )
        assert bias.shape == self.bias.shape, (
            f"Bias shape mismatch: expected {(self.bias.shape,)}, got {bias.shape}"
        )
        return eu.replace(self, weight=weight, bias=bias)

    def __call__(self, hidden_states: Array):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        mean = hidden_states.mean(-1, keepdims=True)
        variance = jnp.var(hidden_states, axis=-1, keepdims=True)
        hidden_states = (hidden_states - mean) * jax.lax.rsqrt(variance + self.eps)
        return (self.weight * hidden_states + self.bias).astype(dtype)


class RMSNorm(eqx.Module):
    """RMSNorm layer for Qwen3-VL text model."""
    weight: jax.Array
    variance_epsilon: float = field(metadata=dict(static=True))

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = jnp.ones((hidden_size,))
        self.variance_epsilon = eps

    def load_state_dict(self, state_dict: dict[str, jax.Array], prefix: str):
        """Load state dict into module."""
        weight = state_dict.pop(prefix + 'weight')
        assert weight.shape == self.weight.shape, \
            f"Weight shape mismatch: expected {self.weight.shape}, got {weight.shape}"
        return eu.replace(self, weight=weight)

    def __call__(self, hidden_states: Array) -> Array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.square(hidden_states).mean(-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


__all__ = [
    "Linear",
    "Embedding",
    "LayerNorm",
    "RMSNorm",
]
