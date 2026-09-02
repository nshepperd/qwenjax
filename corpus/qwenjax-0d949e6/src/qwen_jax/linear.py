"""Linear layers for Qwen3-VL.

Contains Linear, Embedding, LayerNorm (ported from llama_jax) and RMSNorm.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from . import equinox_utils as eu
from .param import Param


class Linear(eqx.Module):
    weight: Param
    bias: Param | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, *, use_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Param(out_features, in_features)
        self.bias = Param(out_features) if use_bias else None

    @property
    def use_bias(self) -> bool:
        return self.bias is not None

    def init_weights(self, key: PRNGKeyArray):
        weight = self.weight.set(
            jax.random.normal(key, self.weight.shape) / np.sqrt(self.in_features)
        )
        bias = self.bias.set(jnp.zeros(self.bias.shape)) if self.bias is not None else None
        return eu.replace(self, weight=weight, bias=bias)

    def __call__(self, x: Array):
        y = x @ self.weight().T
        if self.bias is not None:
            y += self.bias()
        return y


class Embedding(eqx.Module):
    weight: Param
    num_embeddings: int = eqx.field(static=True)
    embedding_dim: int = eqx.field(static=True)

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Param(num_embeddings, embedding_dim)

    def init_weights(self, key: PRNGKeyArray):
        weight = self.weight.set(
            jax.random.normal(key, self.weight.shape) / np.sqrt(self.embedding_dim)
        )
        return eu.replace(self, weight=weight)

    def __call__(self, input_ids: jax.Array):
        return self.weight()[input_ids]


class LayerNorm(eqx.Module):
    weight: Param
    bias: Param
    eps: float = eqx.field(static=True)

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = Param(hidden_size, array=jnp.ones((hidden_size,)))
        self.bias = Param(hidden_size, array=jnp.zeros((hidden_size,)))
        self.eps = eps

    def __call__(self, hidden_states: Array):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        mean = hidden_states.mean(-1, keepdims=True)
        variance = jnp.var(hidden_states, axis=-1, keepdims=True)
        hidden_states = (hidden_states - mean) * jax.lax.rsqrt(variance + self.eps)
        return (self.weight() * hidden_states + self.bias()).astype(dtype)


class RMSNorm(eqx.Module):
    """RMSNorm layer for Qwen3-VL text model."""
    weight: Param
    variance_epsilon: float = eqx.field(static=True)

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = Param(hidden_size, array=jnp.ones((hidden_size,)))
        self.variance_epsilon = eps

    def __call__(self, hidden_states: Array) -> Array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.square(hidden_states).mean(-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return self.weight() * hidden_states.astype(input_dtype)


__all__ = [
    "Embedding",
    "LayerNorm",
    "Linear",
    "RMSNorm",
]
