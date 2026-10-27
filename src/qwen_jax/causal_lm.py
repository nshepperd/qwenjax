"""Pieces every causal-LM wrapper in this package shares.

The model families differ in what a forward pass takes (images, MRoPE deltas,
hybrid caches) but not in what sits around it: the lm head, picking the last
real position's logits, padding the attention mask out to the cache, and the
sampling loop. Those live here as free functions so each wrapper keeps its own
`__call__`/`generate` signature and only delegates the common part.

`decode` takes the per-token forward as a `step(cache, token)` closure, which
is what lets one loop serve every model: whatever extra state a model threads
through decoding (rope deltas, say) is captured by its closure.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from . import equinox_utils as eu
from .linear import Embedding, Linear
from .utils.indexing import gather

C = TypeVar("C")


@jax.tree_util.register_dataclass
@dataclass
class GenerateOutput:
    """Output from a `generate` method.

    ``tokens`` is the prompt followed by the generated tokens. ``logits``, when
    requested, covers the generated positions only.
    """

    tokens: Int[Array, "batch seq"]
    cache: Any = None
    logits: Float[Array, "batch seq vocab"] | None = None


def lm_head(head: Linear | None, embed_tokens: Embedding) -> Linear:
    """The output projection: `head`, or the input embedding when tied (None).

    The embedding table is [vocab, dim], which is already Linear's
    [out_features, in_features] layout.
    """
    if head is not None:
        return head
    return eu.replace(
        Linear(embed_tokens.embedding_dim, embed_tokens.num_embeddings, use_bias=False),
        weight=embed_tokens.weight,
        bias=None,
    )


def cache_dtype(embed_tokens: Embedding):
    """The dtype K/V are cached in: the model's, except never float32."""
    dtype = embed_tokens.weight().dtype
    return jnp.bfloat16 if dtype == jnp.float32 else dtype


def pad_mask(
    attention_mask: Int[Array, "batch seq"] | None, batch_size: int, length: int
) -> Int[Array, "batch length"]:
    """Extend a padding mask to `length` with ones (all-ones when None).

    The mask is over the cache's whole capacity, so slots not yet written count
    as valid; causality is what keeps queries off them.
    """
    if attention_mask is None:
        return jnp.ones((batch_size, length), dtype=jnp.int32)
    if attention_mask.shape[-1] < length:
        return jnp.pad(
            attention_mask,
            ((0, 0), (0, length - attention_mask.shape[-1])),
            constant_values=1,
        )
    return attention_mask


def last_hidden(
    hidden_states: Float[Array, "batch seq hidden"],
    attention_mask: Int[Array, "batch kv_seq"] | None,
) -> Float[Array, "batch hidden"]:
    """Hidden state at each batch item's last non-padded input position.

    `attention_mask` may extend past the input (over a cache); only its first
    `seq` columns are read.
    """
    batch_size, seq_len = hidden_states.shape[:2]
    if attention_mask is not None:
        positions = jnp.arange(seq_len)
        last_idx = jnp.max(
            positions[None, :] * (attention_mask[:, :seq_len] == 1).astype(jnp.int32),
            axis=-1,
        )
    else:
        last_idx = jnp.full((batch_size,), seq_len - 1, dtype=jnp.int32)
    return gather("b s h, b [s] -> b h", hidden_states, last_idx)


def sample(
    logits: Float[Array, "batch vocab"], key: PRNGKeyArray, temperature: float
) -> Int[Array, "batch"]:
    """Sample at `temperature`; 0 is greedy."""
    return jnp.where(
        temperature > 0,
        jax.random.categorical(key, logits / temperature, axis=-1),
        jnp.argmax(logits, axis=-1),
    )


def decode(
    step: Callable[[C, Int[Array, "batch"]], tuple[Float[Array, "batch vocab"], C]],
    first_logits: Float[Array, "batch vocab"],
    cache: C,
    *,
    key: PRNGKeyArray,
    max_new_tokens: int,
    temperature: float = 1.0,
    stop_token_id: int = -1,
    pad_token_id: int = 0,
    return_logits: bool = False,
    progress_bar: bool = True,
) -> tuple[Int[Array, "batch new"], C, Float[Array, "batch new vocab"] | None]:
    """Sample `max_new_tokens` tokens, starting from the prefill's last logits.

    `step(cache, token)` runs one token through the model and returns its
    logits and the updated cache. Returns (new_tokens, final_cache, logits).

    `stop_token_id < 0` disables early stopping. Otherwise each batch item
    halts on its own schedule, emitting `pad_token_id` once it has emitted the
    stop token; the loop exits once every item has.
    """
    batch_size, vocab_size = first_logits.shape
    key, subkey = jax.random.split(key)
    first_token = sample(first_logits, subkey, temperature)

    stop_active = stop_token_id >= 0
    done_init = stop_active & (first_token == stop_token_id)
    keys = jax.random.split(key, max_new_tokens)

    if max_new_tokens == 1:
        logits = first_logits[:, None, :] if return_logits else None
        return first_token[:, None], cache, logits

    def body(carry):
        cache, token, step_idx, tokens, all_logits, done_mask = carry
        if progress_bar:
            jax.debug.print("Decoding step {}/{}", step_idx, max_new_tokens - 1)
        logits, cache = step(cache, token)
        sampled = sample(logits, keys[step_idx], temperature)
        # Items that already stopped emit pad; everyone else gets the sample.
        next_token = jnp.where(done_mask, pad_token_id, sampled)
        done_mask = done_mask | (stop_active & (next_token == stop_token_id))
        if all_logits is not None:
            all_logits = all_logits.at[:, step_idx, :].set(logits)
        tokens = tokens.at[:, step_idx].set(next_token)
        return (cache, next_token, step_idx + 1, tokens, all_logits, done_mask)

    def cond(carry):
        _, _, step_idx, _, _, done_mask = carry
        all_done = stop_active & jnp.all(done_mask)
        return (~all_done) & (step_idx < max_new_tokens - 1)

    tokens = jnp.full((batch_size, max_new_tokens - 1), pad_token_id, dtype=jnp.int32)
    all_logits = (
        jnp.zeros((batch_size, max_new_tokens - 1, vocab_size), dtype=jnp.float32)
        if return_logits
        else None
    )
    init = (cache, first_token, 0, tokens, all_logits, done_init)
    cache, _, _, tokens, all_logits, _ = jax.lax.while_loop(cond, body, init)

    tokens = jnp.concatenate([first_token[:, None], tokens], axis=1)
    if all_logits is not None:
        all_logits = jnp.concatenate([first_logits[:, None, :], all_logits], axis=1)
    return tokens, cache, all_logits


__all__ = [
    "GenerateOutput",
    "cache_dtype",
    "decode",
    "last_hidden",
    "lm_head",
    "pad_mask",
    "sample",
]
