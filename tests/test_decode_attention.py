"""The split-KV decode kernel against the XLA reference attention."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import SingleDeviceSharding

from qwen_jax import attention_xla as axla
from qwen_jax.decode_attention import decode_attention


def reference(q, k, v, kv_mask, used):
    """XLA attention over only the filled slots, queries at used - seq.."""
    seq = q.shape[1]
    k, v, kv_mask = k[:, :used], v[:, :used], kv_mask[:, :used]
    mask = jax.vmap(axla.causal_mask, in_axes=(0, None, None))(kv_mask, seq, used - seq)
    return jax.vmap(axla.masked_attention)(q, k, v, mask)


def make(batch, seq, heads, kv_heads, d, alloc, used, pad, seed=0):
    ks = jax.random.split(jax.random.key(seed), 3)
    q = jax.random.normal(ks[0], (batch, seq, heads, d), jnp.bfloat16)
    k = jax.random.normal(ks[1], (batch, alloc, kv_heads, d), jnp.bfloat16)
    v = jax.random.normal(ks[2], (batch, alloc, kv_heads, d), jnp.bfloat16)
    # The unfilled tail holds garbage that would dominate any output it leaked into.
    k = k.at[:, used:].set(50.0)
    v = v.at[:, used:].set(50.0)
    kv_mask = jnp.ones((batch, alloc), bool)
    if pad:
        kv_mask = kv_mask.at[-1, :pad].set(False)
    return q, k, v, kv_mask


@pytest.mark.parametrize(
    "batch,seq,heads,kv_heads,d,alloc,used,pad",
    [
        (1, 1, 24, 4, 256, 8192, 8000, 0),  # Qwen3.5 shape
        (1, 1, 24, 4, 256, 65536, 3000, 0),  # mostly empty allocation
        (2, 1, 24, 4, 256, 4096, 2500, 77),  # left-padded second row
        (1, 2, 24, 4, 256, 4096, 3001, 0),  # two queries: causal between them
        (1, 1, 16, 8, 128, 4096, 1000, 0),  # head_dim 128, group 2
        (3, 4, 16, 4, 128, 2048, 517, 9),  # 16 rows, odd fill, padding
        (1, 1, 24, 4, 256, 4096, 1, 0),  # a single filled slot
    ],
)
def test_matches_reference(batch, seq, heads, kv_heads, d, alloc, used, pad):
    q, k, v, kv_mask = make(batch, seq, heads, kv_heads, d, alloc, used, pad)
    seqused = jnp.full((batch,), used, jnp.int32)
    out = jax.jit(decode_attention)(q, k, v, kv_mask, seqused)
    ref = reference(q, k, v, kv_mask, used)
    np.testing.assert_allclose(
        np.asarray(out, np.float32), np.asarray(ref, np.float32), atol=2e-3, rtol=1e-2
    )


def test_fully_masked_row_is_zero():
    q, k, v, kv_mask = make(2, 1, 24, 4, 256, 1024, 600, 0)
    kv_mask = kv_mask.at[1].set(False)
    out = jax.jit(decode_attention)(q, k, v, kv_mask, jnp.full((2,), 600, jnp.int32))
    assert np.all(np.asarray(out[1]) == 0)
    assert np.all(np.isfinite(np.asarray(out, np.float32)))


def test_pinned_host_cache_read_in_place():
    dev = jax.devices()[0]
    host = SingleDeviceSharding(dev, memory_kind="pinned_host")
    device = SingleDeviceSharding(dev, memory_kind="device")
    q, k, v, kv_mask = make(1, 1, 24, 4, 256, 8192, 5000, 0)
    seqused = jnp.full((1,), 5000, jnp.int32)
    on_device = jax.jit(decode_attention)(q, k, v, kv_mask, seqused)
    f = jax.jit(
        decode_attention,
        in_shardings=(device, host, host, device, device),
        out_shardings=device,
    )
    k_h, v_h = jax.device_put(k, host), jax.device_put(v, host)
    hlo = f.lower(q, k_h, v_h, kv_mask, seqused).compile().as_text()
    assert "copy-start" not in hlo  # the kernel reads the host buffers itself
    np.testing.assert_array_equal(np.asarray(f(q, k_h, v_h, kv_mask, seqused)), np.asarray(on_device))
