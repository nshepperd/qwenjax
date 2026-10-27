"""Probe: stream decoder-layer weights from pinned_host memory during decode.

Loads a run of real decoder layers from a llama-arch GGUF and times one decode
step (batch 1, one token, no KV cache) through them, with the layers' weights:

  device      resident in device memory (the compute-only baseline);
  host        in pinned_host, device_put inside jit with nothing gating it --
              the scheduler is free to hoist every copy to the start;
  gate-k      in pinned_host, the copy of layer i+k gated by an
              optimization_barrier on layer i's input, so at most k+1 layers'
              weights are on the device and each copy can overlap the compute
              of the layers before it.

For each it reports the step time, XLA's temp-memory estimate (an ungated
program that hoisted its copies needs room for every layer at once), and the
scheduled HLO's copy-start -> copy-done spans: how many compute ops XLA put
inside each async copy, i.e. what it could overlap the copy with. The raw
host->device bandwidth outside jit is measured too, as the ceiling.

Runs with ordinary device memory: the unified-memory pool (phenex's zshenv
default) would blur the line between "device" and "pinned_host".

    uv run python bench/stream_probe.py --layers 12
"""
from __future__ import annotations

import os

os.environ.pop("TF_FORCE_UNIFIED_MEMORY", None)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "bfc"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
os.environ.setdefault("QWEN_JAX_ATTENTION", "xla")  # seq 1: attention is negligible

import argparse
import re
import statistics
import time

import jax
import jax.numpy as jnp
from gguf_jax import load_gguf
from jax.sharding import SingleDeviceSharding

from qwen_jax.gguf import dense_or_quantized, load_into
from qwen_jax.llama import LlamaConfig, LlamaForCausalLM
from qwen_jax.llama.gguf import text_key

CONFIG = "/data/models/Nanbeige4.3-33B-Base"
GGUF = "/data/models/Nanbeige4.3-33B-Base-GGUF/nanbeige4.3-33b-base-q4_k_m.gguf"


def load_layers(config_path: str, gguf_path: str, start: int, count: int):
    """Decoder layers start..start+count-1, renumbered from 0, on the device."""
    config = LlamaConfig.from_pretrained(config_path).model_copy(
        update={"num_hidden_layers": count}
    )
    wanted = tuple(f"blk.{start + i}." for i in range(count))
    tensors = load_gguf(gguf_path, tensor_filter=lambda n: n.startswith(wanted)).tensors
    state_dict = {}
    for name, tensor in tensors.items():
        _, n, rest = name.split(".", 2)
        state_dict[text_key(f"blk.{int(n) - start}.{rest}")] = dense_or_quantized(
            tensor, jnp.bfloat16
        )
    # q/k are left in llama.cpp's permuted order: irrelevant to timing.
    model = load_into(LlamaForCausalLM(config), state_dict, allow_missing=True)
    return config, model.model.layers, model.model.rotary_emb


def nbytes(tree) -> int:
    return sum(x.nbytes for x in jax.tree.leaves(tree))


def make_forward(n: int, streamed: bool, lookahead: int | None, s_dev):
    """A decode step through n layers.

    streamed=False: layers are used where they are. Otherwise each layer is
    device_put to s_dev inside the program; with `lookahead` set, the copy of
    layer i+lookahead is gated on layer i's input.
    """

    def to_device(layer):
        return jax.device_put(layer, s_dev) if streamed else layer

    def forward(layers, h, position_embeddings, kv_mask):
        on_device = [None] * n
        first = n if lookahead is None else min(lookahead, n)
        for j in range(first):
            on_device[j] = to_device(layers[j])
        for i in range(n):
            j = i + lookahead if lookahead is not None else n
            if j < n:
                h, nxt = jax.lax.optimization_barrier((h, layers[j]))
                on_device[j] = to_device(nxt)
            h, _ = on_device[i](h, position_embeddings, kv_mask=kv_mask)
            on_device[i] = None
        return h

    return jax.jit(forward)


def copy_spans(hlo: str) -> list[int]:
    """Compute ops scheduled between each copy-start and its copy-done."""
    entry = hlo[hlo.index("ENTRY") :]
    entry = entry[: entry.index("\n}")]
    lines = entry.splitlines()
    starts: dict[str, int] = {}
    spans = []
    compute = re.compile(r"= .*\b(fusion|custom-call|dot|convolution)\(")
    compute_seen = 0
    for line in lines:
        if m := re.search(r"%(copy-start[\w.]*) = ", line):
            starts[m.group(1)] = compute_seen
        elif m := re.search(r"copy-done\(.*%(copy-start[\w.]*)\)", line):
            spans.append(compute_seen - starts.pop(m.group(1), compute_seen))
        elif compute.search(line):
            compute_seen += 1
    return spans


def time_step(fn, args, repeats: int) -> float:
    jax.block_until_ready(fn(*args))  # compile + warm
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--gguf", default=GGUF)
    parser.add_argument("--start", type=int, default=6, help="first GGUF layer to load")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--lookaheads", default="1,2")
    args = parser.parse_args()

    dev = jax.devices()[0]
    s_dev = SingleDeviceSharding(dev, memory_kind="device")
    s_host = SingleDeviceSharding(dev, memory_kind="pinned_host")
    print(f"jax {jax.__version__}, {dev.device_kind}")

    config, layers_dev, rotary = load_layers(args.config, args.gguf, args.start, args.layers)
    n = args.layers
    layer_bytes = nbytes(layers_dev)
    print(f"{n} layers, {layer_bytes / 1e9:.2f} GB ({layer_bytes / n / 1e6:.0f} MB/layer)")

    layers_host = jax.device_put(layers_dev, s_host)
    jax.block_until_ready(layers_host)

    # Ceiling: plain host->device DMA of the same bytes, outside jit.
    def h2d():
        return jax.block_until_ready(jax.device_put(layers_host, s_dev))

    h2d()
    t0 = time.perf_counter()
    for _ in range(3):
        h2d()
    h2d_s = (time.perf_counter() - t0) / 3
    print(f"h2d outside jit: {h2d_s * 1e3:.0f} ms, {layer_bytes / h2d_s / 1e9:.1f} GB/s")

    key = jax.random.key(0)
    h = jax.random.normal(key, (1, 1, config.hidden_size), jnp.bfloat16)
    position_embeddings = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16), rotary(jnp.zeros((1, 1), jnp.int32))
    )
    kv_mask = jnp.ones((1, 1), jnp.bool)

    modes = [("device", layers_dev, False, None), ("host", layers_host, True, None)]
    modes += [
        (f"gate-{k}", layers_host, True, int(k)) for k in args.lookaheads.split(",")
    ]
    print(f"\n{'mode':8s} {'step ms':>8s} {'GB/s':>6s} {'temp GB':>8s}  copy spans (compute ops inside each copy)")
    for name, layers, streamed, lookahead in modes:
        fn = make_forward(n, streamed, lookahead, s_dev)
        fn_args = (layers, h, position_embeddings, kv_mask)
        compiled = fn.lower(*fn_args).compile()
        temp = compiled.memory_analysis().temp_size_in_bytes
        spans = copy_spans(compiled.as_text())
        try:
            step = time_step(fn, fn_args, args.repeats)
        except Exception as e:  # noqa: BLE001 -- report and keep probing
            print(f"{name:8s} failed: {type(e).__name__}: {str(e)[:200]}")
            continue
        rate = layer_bytes / step / 1e9 if streamed else float("nan")
        print(f"{name:8s} {step * 1e3:8.1f} {rate:6.1f} {temp / 1e9:8.2f}  {spans}")


if __name__ == "__main__":
    main()
