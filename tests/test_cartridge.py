"""Tests for KV prefixes and cartridges.

The load-bearing property is that a prefix is *exactly* a stand-in for the
tokens it was computed from: the model's distribution after `[prefix] + B`
must equal its distribution after `A + B` when the prefix is the KV of `A`,
through both the concatenating path (training) and the cache path (serving).
Everything else -- gradients, shifting, composition -- is built on that.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from transformers import AutoTokenizer

from qwen_jax import chat
from qwen_jax.cache import KVCache, KVPrefix
from qwen_jax.cartridge import Cartridge, compose
from qwen_jax.distill import Trainer, distill_loss, make_batch
from qwen_jax.selfstudy import Example
from qwen_jax.utils.pjit import pjit

QWEN3VL_MODEL_PATH = "/data/models/Qwen3-VL-2B-Instruct"


@pytest.fixture(scope="module")
def model():
    from qwen_jax.loading import load_qwen3_jax

    return load_qwen3_jax(QWEN3VL_MODEL_PATH)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(QWEN3VL_MODEL_PATH)


def assert_same_dist(logits1, logits2):
    """Jensen-Shannon divergence per position: tiny everywhere, and negligible on average.

    Two routes through bf16 attention disagree by ~1e-4 nats at most positions
    and occasionally ~1e-2 at a high-entropy one; a real bug is orders larger.
    """
    div = np.asarray(jsdiv(logits1, logits2))
    assert div.max() < 5e-2, div
    assert div.mean() < 3e-3, div


def assert_keys_close(actual, desired, deep_tol=2e-2):
    """Per-layer relative L2 error.

    Layer 0 keys depend only on the token and its rotation, so they isolate
    the rotation itself: bf16 cos/sin put it at ~5e-4. Deeper layers also
    carry the bf16 drift of re-running attention at other positions, which
    grows to ~1e-2 by the last layer. A wrong rotation is O(1) everywhere.
    """
    a = np.asarray(actual, np.float32)
    d = np.asarray(desired, np.float32)
    rel = np.linalg.norm((a - d).reshape(a.shape[0], -1), axis=1) / np.linalg.norm(
        d.reshape(d.shape[0], -1), axis=1)
    assert rel[0] < 2e-3, rel
    assert rel.max() < deep_tol, rel


def _H(logits):
    return -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1)


@pjit(device=jax.devices("cpu")[0])
def jsdiv(logits1, logits2):
    logits1 = jnp.float32(logits1)
    logits2 = jnp.float32(logits2)
    M = jnp.log(0.5 * (jax.nn.softmax(logits1) + jax.nn.softmax(logits2)))
    return _H(M) - 0.5 * (_H(logits1) + _H(logits2))


SYSTEM = ("The following is an excerpt from a manual about tea. Green tea is "
          "brewed at 80 degrees Celsius for two minutes; black tea at 95 degrees "
          "for four minutes. Oolong sits between the two.")
TURNS = [("user", "At what temperature should I brew oolong?"),
         ("assistant", "Somewhere between 80 and 95 degrees Celsius.")]


@pytest.fixture(scope="module")
def tokens(tokenizer) -> chat.Tokens:
    return chat.encode(tokenizer, SYSTEM, TURNS)


def _ids(xs):
    return jnp.asarray(np.asarray(xs, dtype=np.int32))[None, :]


@pytest.fixture(scope="module")
def prefix(model, tokens) -> KVPrefix:
    a = _ids(tokens.system)
    out = model(input_ids=a, attention_mask=jnp.ones_like(a), use_cache=True,
                last_logit_only=True)
    return KVPrefix.from_cache(out.cache)


def test_prefix_path_matches_full_forward(model, tokens, prefix):
    a, b = _ids(tokens.system), _ids(tokens.suffix)
    full = model(input_ids=jnp.concatenate([a, b], axis=1))
    with_prefix = model(input_ids=b, prefix=prefix)
    assert_same_dist(full.logits[:, a.shape[1]:], with_prefix.logits)


def test_cache_path_matches_prefix_path(model, tokens, prefix):
    b = _ids(tokens.suffix)
    cache = KVCache.for_model(model, 1, b.shape[1], prefix=prefix)
    assert int(cache.position) == prefix.length
    assert bool(cache.valid[0, :prefix.length].all())
    via_cache = model(input_ids=b, cache=cache)
    via_prefix = model(input_ids=b, prefix=prefix)
    assert_same_dist(via_cache.logits, via_prefix.logits)
    # and `use_cache` without a cache seeds one from the prefix
    seeded = model(input_ids=b, prefix=prefix, use_cache=True)
    assert int(seeded.cache.position) == prefix.length + b.shape[1]
    assert_same_dist(seeded.logits, via_prefix.logits)


def test_prefix_with_right_padding(model, tokens, prefix):
    """Batch of two, the second one right-padded: valid positions must agree."""
    b = np.asarray(tokens.suffix, dtype=np.int32)
    n = len(b)
    ids = np.zeros((2, n + 7), dtype=np.int32)
    mask = np.zeros((2, n + 7), dtype=np.int32)
    ids[0, :n] = b
    mask[0, :n] = 1
    ids[1, :n - 5] = b[:n - 5]
    mask[1, :n - 5] = 1
    ids[:, n:] = 0
    out = model(input_ids=jnp.asarray(ids), attention_mask=jnp.asarray(mask), prefix=prefix)
    ref = model(input_ids=_ids(b), prefix=prefix)
    assert_same_dist(out.logits[0:1, :n], ref.logits)
    assert_same_dist(out.logits[1:2, :n - 5], ref.logits[:, :n - 5])


def test_generate_with_prefix_matches_inline(model, tokenizer, tokens, prefix):
    """Greedy decoding after the prefix == greedy decoding after the real tokens."""
    a, b = _ids(tokens.system), _ids(tokens.suffix[:-6])  # stop inside the answer
    key = jax.random.key(0)
    inline = model.generate(input_ids=jnp.concatenate([a, b], axis=1), max_new_tokens=8,
                            key=key, temperature=0.0, progress_bar=False)
    carted = model.generate(input_ids=b, prefix=prefix, max_new_tokens=8,
                            key=key, temperature=0.0, progress_bar=False)
    np.testing.assert_array_equal(inline.tokens[:, a.shape[1] + b.shape[1]:],
                                  carted.tokens[:, b.shape[1]:])


def test_shift_reproduces_later_positions(model, tokens):
    """Rotating cached keys by `delta` == caching them at positions `+delta`."""
    a = _ids(tokens.system)
    n = a.shape[1]
    delta = 37
    base = KVPrefix.from_cache(model(input_ids=a, use_cache=True, last_logit_only=True).cache)
    pos = jnp.broadcast_to((jnp.arange(n) + delta)[None, None, :], (3, 1, n))
    later = KVPrefix.from_cache(
        model(input_ids=a, position_ids=pos, use_cache=True, last_logit_only=True).cache
    )
    rotary = model.model.language_model.rotary_emb
    shifted = base.shift(delta, rotary)
    assert_keys_close(shifted.keys, later.keys)
    # Values are untouched by shift, so any difference is the model re-run at
    # other positions: zero at layer 0, a few percent of bf16 drift by the end.
    assert_keys_close(shifted.values, later.values, deep_tol=5e-2)


def test_compose_repositions(model, tokens, tokenizer):
    """Two cartridges composed with repositioning == the two texts back to back."""
    first = _ids(tokens.system)
    second_text = chat.suffix([("user", "And how long does black tea steep?")])
    second = _ids(tokenizer.encode(second_text, add_special_tokens=False))
    both = jnp.concatenate([first, second], axis=1)
    c1 = Cartridge.init_from_tokens(model, first[0])
    # The second piece's KV at positions 0..m-1, i.e. as a standalone cartridge.
    c2 = Cartridge.init_from_tokens(model, second[0])
    joined = compose(model, c1, c2, reposition=True)
    reference = KVPrefix.from_cache(model(input_ids=both, use_cache=True, last_logit_only=True).cache)
    assert joined.length == reference.length
    # Keys agree only where attention context agrees: the first piece exactly,
    # the second piece only in *position* (its values were computed without
    # the first piece in context). So test what composition promises -- the
    # rotation -- via the first piece's exactness and the second's alignment
    # with a directly-offset computation.
    assert_keys_close(joined.keys[:, :first.shape[1]], reference.keys[:, :first.shape[1]])
    m = second.shape[1]
    pos = jnp.broadcast_to((jnp.arange(m) + first.shape[1])[None, None, :], (3, 1, m))
    offset = KVPrefix.from_cache(
        model(input_ids=second, position_ids=pos, use_cache=True, last_logit_only=True).cache
    )
    assert_keys_close(joined.keys[:, first.shape[1]:], offset.keys)


def test_cartridge_init_is_icl(model, tokens):
    """Untrained, a cartridge is in-context learning over its tokens."""
    cart = Cartridge.init_from_tokens(model, np.asarray(tokens.system))
    assert cart.length == len(tokens.system)
    assert not bool(cart.trainable[0]) and bool(cart.trainable[1:].all())
    a, b = _ids(tokens.system), _ids(tokens.suffix)
    full = model(input_ids=jnp.concatenate([a, b], axis=1))
    out = model(input_ids=b, prefix=cart.prefix(model.cache_dtype()))
    assert_same_dist(full.logits[:, a.shape[1]:], out.logits)


def test_save_load_roundtrip(model, tokens, tmp_path):
    cart = Cartridge.init_from_tokens(model, np.asarray(tokens.system)).advance(3)
    cart.save(tmp_path / "c.safetensors")
    back = Cartridge.load(tmp_path / "c.safetensors")
    np.testing.assert_array_equal(np.asarray(back.keys), np.asarray(cart.keys))
    np.testing.assert_array_equal(np.asarray(back.values), np.asarray(cart.values))
    np.testing.assert_array_equal(np.asarray(back.trainable), np.asarray(cart.trainable))
    assert back.meta == cart.meta and int(back.steps) == 3


def test_distillation_step(model, tokenizer, tokens):
    """Gradient reaches the cartridge, spares the sink, and lowers the loss.

    The cartridge is initialised from a *different* system text than the
    teacher sees, so there is something to learn.
    """
    description = "Below is an excerpt from a manual."
    wrong = chat.encode(tokenizer, "This document is about the history of bicycles.", [])
    cart = Cartridge.init_from_tokens(model, np.asarray(wrong.system))
    ex = Example(
        chunk_ids=tokenizer.encode(SYSTEM, add_special_tokens=False),
        user=TURNS[0][1], assistant=TURNS[1][1],
    )
    batch = make_batch(tokenizer, [ex, ex], description=description, context=128, seq=64,
                       pad_id=tokenizer.pad_token_id)
    assert batch.context == 128

    loss0 = float(jax.jit(distill_loss, static_argnames=("block",))(model, cart, batch, block=32))
    assert np.isfinite(loss0) and loss0 > 0

    trainer = Trainer(optax.adam(1e-2), block=32)
    state = trainer.init(cart)
    losses = []
    c = cart
    for _ in range(6):
        c, state, loss = trainer.step(model, c, state, batch)
        losses.append(float(loss))
    assert int(c.steps) == 6
    assert losses[-1] < losses[0], losses
    np.testing.assert_array_equal(np.asarray(c.keys[:, 0]), np.asarray(cart.keys[:, 0]))
    np.testing.assert_array_equal(np.asarray(c.values[:, 0]), np.asarray(cart.values[:, 0]))
    assert not np.array_equal(np.asarray(c.keys[:, 1]), np.asarray(cart.keys[:, 1]))


def test_probe_matches_prefix_path(model, tokens, prefix):
    """The dense-attention probe is the model: same logits, weights sum to one."""
    from qwen_jax.probe import probe

    b = np.asarray(tokens.suffix, np.int32)
    p = prefix.length
    segments = np.concatenate([np.zeros(p // 2, np.int32), np.ones(p - p // 2, np.int32),
                               np.full(len(b), 2, np.int32)])
    res = probe(model, jnp.asarray(b), prefix, jnp.asarray(segments), num_segments=3)
    ref = model(input_ids=_ids(b), prefix=prefix)
    assert_same_dist(ref.logits[0], res.logits)
    seg = np.asarray(res.by_segment)
    assert seg.shape == (prefix.num_layers, model.config.text_config.num_attention_heads, len(b), 3)
    np.testing.assert_allclose(seg.sum(-1), 1.0, atol=2e-3)
    slot = np.asarray(res.by_slot)
    assert slot.shape == (prefix.num_layers, p + len(b))
    np.testing.assert_allclose(slot.sum(-1), 1.0, atol=2e-3)
