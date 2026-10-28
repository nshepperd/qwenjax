"""Probe: how long a context Saluki (qwen35 GGUF) holds, and what it costs.

For each length L: allocate a cache of L + decode steps (or --alloc), prefill L random
tokens in fixed-size chunks through the cache (cache donated, so only one
copy of the KV is live), then time single-token decode steps at that depth.
Stops at the first length that fails to allocate.

    env -u TF_FORCE_UNIFIED_MEMORY XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.84 \
      uv run python bench/context_probe.py --chunk 512 --lengths 32768,65536,77824
    # with unified memory, beyond the card:
    TF_FORCE_UNIFIED_MEMORY=1 XLA_PYTHON_CLIENT_ALLOCATOR=bfc \
      XLA_PYTHON_CLIENT_MEM_FRACTION=3.0 uv run python bench/context_probe.py ...

The environment is read as given (phenex's shell defaults to unified memory,
so say which you mean). Prefill temp memory grows with --chunk, not with the
context (0.84 GB at 512, 2.4 GB at 2048 for Saluki); part of it is the
dequantized-weight scratch of the prefill matmuls, a fixed ~0.36 GB per MLP
weight. --alloc fixes the cache allocation across lengths, as a server would.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.84")

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp

from qwen_jax.qwen35 import load_qwen35_gguf

CONFIG = "/data/models/saluki/qwen3.8-27b-config"
GGUF = "/data/models/saluki/Underdog-Saluki-27B-1.0-IQ2-mix.gguf"


@eqx.filter_jit(donate="all-except-first")
def step(model, input_ids, cache):
    out = model(input_ids=input_ids, cache=cache, last_logit_only=True)
    return out.last_logits, out.cache


def gb(n: int) -> str:
    return f"{n / 1e9:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--gguf", default=GGUF)
    parser.add_argument("--lengths", default="16384,32768,49152,65536,81920,98304")
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--alloc", type=int, default=None,
                        help="allocate every cache at this many slots (default: length + steps)")
    args = parser.parse_args()

    dev = jax.devices()[0]
    unified = bool(os.environ.get("TF_FORCE_UNIFIED_MEMORY"))
    print(f"jax {jax.__version__}, {dev.device_kind}, unified memory {'on' if unified else 'off'}, "
          f"allocator {os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']}, "
          f"fraction {os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']}")
    t0 = time.perf_counter()
    model = load_qwen35_gguf(args.config, args.gguf)
    jax.block_until_ready(model)
    print(f"loaded in {time.perf_counter() - t0:.0f}s, in use {gb(dev.memory_stats().get('bytes_in_use', 0))} GB")

    key = jax.random.key(0)
    vocab = model.config.text_config.vocab_size
    print(f"\n{'length':>8s} {'KV GB':>6s} {'prefill s':>9s} {'tok/s':>7s} {'decode ms':>9s} {'in use GB':>9s}")
    for length in (int(x) for x in args.lengths.split(",")):
        assert length % args.chunk == 0, "lengths must be multiples of --chunk"
        try:
            cache = model.create_cache(1, args.alloc or length + args.decode_steps)
            kv_bytes = sum(
                x.nbytes for layer in cache.layers if hasattr(layer, "keys")
                for x in (layer.keys, layer.values)
            )
            ids = jax.random.randint(key, (1, length), 0, vocab)
            t0 = time.perf_counter()
            for i in range(0, length, args.chunk):
                logits, cache = step(model, ids[:, i : i + args.chunk], cache)
            jax.block_until_ready(logits)
            prefill = time.perf_counter() - t0  # includes the chunk program's compile

            tok = jnp.argmax(logits, -1)[:, None]
            logits, cache = step(model, tok, cache)  # compile
            jax.block_until_ready(logits)
            t0 = time.perf_counter()
            for _ in range(args.decode_steps - 1):
                logits, cache = step(model, jnp.argmax(logits, -1)[:, None], cache)
            jax.block_until_ready(logits)
            decode = (time.perf_counter() - t0) / (args.decode_steps - 1)
            in_use = dev.memory_stats().get("bytes_in_use", 0)
            print(f"{length:8d} {gb(kv_bytes):>6s} {prefill:9.1f} {length / prefill:7.0f} "
                  f"{decode * 1e3:9.1f} {gb(in_use):>9s}", flush=True)
            del cache, logits, ids
        except Exception as e:  # noqa: BLE001 -- the first failure is the answer
            print(f"{length:8d} failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}")
            break


if __name__ == "__main__":
    main()
