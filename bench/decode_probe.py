"""Is GGUF decode launch-bound or bandwidth-bound?

Decode measures ~51% of the memory-bandwidth roofline (see starchart 4kbz1u).
Two explanations fit: the transfers really are that inefficient, or the GPU is
idle between kernels. They call for completely different work -- better kernels
in the first case, CUDA graphs in the second -- so this measures which it is
before anything else gets built.

Three probes, cheapest first:

  wall     decode tok/s under whatever XLA_FLAGS the caller set. Sweep the
           command-buffer setting across processes and compare; if capturing
           the decode loop into a CUDA graph moves the number, launches were
           the cost.
  probe    same, but bracketed in cudaProfilerStart/Stop so
           `nsys profile --capture-range=cudaProfilerApi` sees decode alone
           and not the prefill or the compile.
  batch    decode tok/s at several batch sizes. Weight traffic is identical
           across them and so is the launch count, so a bandwidth-bound decode
           holds tok/s roughly flat per step while total throughput scales.
  roofline measured streaming bandwidth of this card, which is what the decode
           floor should be divided by -- the spec figure is not reachable.
  bytes    weight bytes per decode step, split by quantization type and by
           tensor shape, so profiled kernel times convert into GB/s.

Usage:
    python bench/decode_probe.py wall --variant q4km
    XLA_FLAGS=--xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL,WHILE \
        python bench/decode_probe.py wall --variant q4km
    nsys profile --capture-range=cudaProfilerApi --stats=true -o /tmp/dec \
        python bench/decode_probe.py probe --variant q4km
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import numpy as np

import quantization as q  # noqa: E402  (same directory)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def cuda_profiler():
    """cudaProfilerStart/Stop, for nsys --capture-range=cudaProfilerApi.

    Returns no-ops if libcudart cannot be found, so the script still runs
    outside nsys.
    """
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
        try:
            lib = ctypes.CDLL(name)
            return lib.cudaProfilerStart, lib.cudaProfilerStop
        except OSError:
            continue
    noop = lambda: 0  # noqa: E731
    return noop, noop


# Measured on the 5070 Ti with `decode_probe.py roofline`: read-only streaming
# tops out here, well short of the 896 GB/s on the spec sheet. Dividing by the
# spec figure flatters nothing and understates how close the kernels already are.
ACHIEVABLE_READ_GBS = 790.0


def streamed_bytes(model) -> int:
    """Weight bytes a single cached decode step actually reads.

    Not the model size: the token embedding is gathered one row at a time, so
    the table is never streamed. Its lm_head twin has the same shape and is,
    which is why this goes by tree position rather than by shape.
    """
    from qwen_jax.gguf import GGUFParam
    from qwen_jax.param import path_to_key

    total = 0
    for path, leaf in jax.tree.leaves_with_path(
            model, is_leaf=lambda x: isinstance(x, GGUFParam)):
        if "embed_tokens" in path_to_key(path):
            continue
        if isinstance(leaf, GGUFParam):
            total += leaf.array.data.nbytes
        elif hasattr(leaf, "nbytes"):
            total += leaf.nbytes
    return total


def build(args):
    from transformers import AutoTokenizer

    v = q.VARIANTS[args.variant]
    tokenizer = AutoTokenizer.from_pretrained(q.HF_BF16)
    chunk = q.make_chunks(tokenizer, 1, args.prompt_len)[0]
    prompt = jnp.tile(jnp.asarray(chunk)[None, :], (args.batch, 1))
    model = q.load_variant(v, vision=False)
    return v, model, prompt


def timed_decode(model, prompt, tokens: int, repeats: int):
    """Median wall seconds for `tokens` cached steps, prefill subtracted out."""

    def prefill():
        out = model(input_ids=prompt, attention_mask=jnp.ones_like(prompt),
                    last_logit_only=True)
        return out.last_logits

    def generate(n):
        return model.generate(
            input_ids=prompt, attention_mask=jnp.ones_like(prompt),
            max_new_tokens=n, key=jax.random.key(0), temperature=0.0,
            progress_bar=False,
        ).tokens

    prefill().block_until_ready()
    generate(tokens).block_until_ready()

    def median(fn):
        ts = []
        for _ in range(repeats):
            t = time.perf_counter()
            fn().block_until_ready()
            ts.append(time.perf_counter() - t)
        return float(np.median(ts))

    prefill_s = median(prefill)
    total_s = median(lambda: generate(tokens))
    return max(total_s - prefill_s, 1e-9), prefill_s


def run_wall(args):
    v, model, prompt = build(args)
    gb = q.weight_bytes(model) / 1e9
    decode_s, prefill_s = timed_decode(model, prompt, args.decode_tokens,
                                       args.repeats)
    steps = args.decode_tokens
    tok_s = steps * args.batch / decode_s
    ms = decode_s / steps * 1e3
    # Read once per step whatever the batch is, and less than the model size.
    read_gb = streamed_bytes(model) / 1e9
    floor_s = read_gb / args.bandwidth
    print(f"variant        {v.label}")
    print(f"XLA_FLAGS      {os.environ.get('XLA_FLAGS', '(unset)')}")
    print(f"batch          {args.batch}")
    print(f"weights        {gb:.2f} GB total, {read_gb:.3f} GB streamed per step")
    print(f"prefill        {prefill_s * 1e3:.1f} ms "
          f"({args.prompt_len * args.batch / prefill_s:.0f} tok/s)")
    print(f"decode         {ms:.3f} ms/step   {tok_s:.2f} tok/s")
    print(f"roofline       {floor_s * 1e3:.3f} ms/step at {args.bandwidth:.0f} GB/s "
          f"-> {floor_s / (decode_s / steps) * 100:.1f}% of achievable")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "variant": args.variant, "batch": args.batch, "weights_gb": gb,
            "streamed_gb": read_gb, "bandwidth_gbs": args.bandwidth,
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "decode_ms_per_step": ms, "decode_tok_s": tok_s,
            "prefill_ms": prefill_s * 1e3,
        }, indent=2))


def run_probe(args):
    """Decode under nsys, with the warmup and prefill outside the capture."""
    start, stop = cuda_profiler()
    _, model, prompt = build(args)

    def generate(n):
        return model.generate(
            input_ids=prompt, attention_mask=jnp.ones_like(prompt),
            max_new_tokens=n, key=jax.random.key(0), temperature=0.0,
            progress_bar=False,
        ).tokens

    generate(args.decode_tokens).block_until_ready()  # compile + warm

    start()
    t = time.perf_counter()
    generate(args.decode_tokens).block_until_ready()
    wall = time.perf_counter() - t
    stop()
    print(f"captured {args.decode_tokens} steps in {wall * 1e3:.1f} ms "
          f"({wall / args.decode_tokens * 1e3:.3f} ms/step, includes one "
          f"{args.prompt_len}-token prefill)")


def run_batch(args):
    from transformers import AutoTokenizer

    v = q.VARIANTS[args.variant]
    tokenizer = AutoTokenizer.from_pretrained(q.HF_BF16)
    chunk = q.make_chunks(tokenizer, 1, args.prompt_len)[0]
    model = q.load_variant(v, vision=False)
    gb = q.weight_bytes(model) / 1e9
    print(f"{v.label}, weights {gb:.2f} GB, {args.decode_tokens} steps\n")
    print(f"{'batch':>5s} {'ms/step':>9s} {'tok/s':>9s} {'vs batch 1':>11s}")
    base = None
    for b in args.batches:
        prompt = jnp.tile(jnp.asarray(chunk)[None, :], (b, 1))
        decode_s, _ = timed_decode(model, prompt, args.decode_tokens,
                                   args.repeats)
        ms = decode_s / args.decode_tokens * 1e3
        base = base or ms
        print(f"{b:5d} {ms:9.3f} {args.decode_tokens * b / decode_s:9.2f} "
              f"{ms / base:10.2f}x")


def run_roofline(args):
    """What this card actually streams, as opposed to its spec sheet.

    A decode GEMV is read-dominated, so the read-only figure is the one to
    divide weight bytes by.
    """
    n = args.buffer_mb * 1024 * 1024 // 2
    x = jnp.ones((n,), jnp.bfloat16)

    def best(fn):
        fn(x).block_until_ready()
        return min(_time(lambda: fn(x)) for _ in range(args.repeats * 3))

    read = best(jax.jit(lambda a: a.astype(jnp.float32).sum()))
    copy = best(jax.jit(lambda a: a + jnp.bfloat16(1)))
    print(f"read-only  {n * 2 / 1e9:.2f} GB in {read * 1e3:7.3f} ms  "
          f"-> {n * 2 / read / 1e9:6.1f} GB/s")
    print(f"read+write {n * 4 / 1e9:.2f} GB in {copy * 1e3:7.3f} ms  "
          f"-> {n * 4 / copy / 1e9:6.1f} GB/s")


def run_bytes(args):
    """Weight bytes per decode step, by qtype and by shape.

    The token embedding is excluded: decode gathers one row of it rather than
    streaming the table, so counting it inflates the roofline denominator. Its
    lm_head twin has the same shape and IS streamed, so the two are told apart
    by where they sit in the tree, not by shape.
    """
    import collections

    from qwen_jax.gguf import GGUFParam
    from qwen_jax.param import path_to_key

    model = q.load_variant(q.VARIANTS[args.variant], vision=False)
    agg = collections.defaultdict(lambda: [0, 0])
    for path, leaf in jax.tree.leaves_with_path(
            model, is_leaf=lambda x: isinstance(x, GGUFParam)):
        if not isinstance(leaf, GGUFParam):
            continue
        w = leaf.array
        gathered = "embed_tokens" in path_to_key(path)
        key = (str(w.qtype).split(".")[-1], w.shape, gathered)
        agg[key][0] += 1
        agg[key][1] += w.data.nbytes

    print(f"{'qtype':>6s} {'shape':>18s} {'count':>6s} {'GB':>8s} "
          f"{'blocks':>7s}  streamed per step")
    streamed = 0
    for (qt, shape, gathered), (n, b) in sorted(agg.items(),
                                                key=lambda kv: -kv[1][1]):
        if not gathered:
            streamed += b
        print(f"{qt:>6s} {str(shape):>18s} {n:6d} {b / 1e9:8.3f} "
              f"{shape[0] // 8:7d}  {'no (gathered)' if gathered else 'yes'}")
    print(f"\nstreamed per decode step: {streamed / 1e9:.3f} GB"
          f"   floor {streamed / 1e9 / ACHIEVABLE_READ_GBS * 1e3:.2f} ms at "
          f"{ACHIEVABLE_READ_GBS:.0f} GB/s")


def _time(fn) -> float:
    t = time.perf_counter()
    fn().block_until_ready()
    return time.perf_counter() - t


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("wall", "probe", "batch", "roofline", "bytes"):
        s = sub.add_parser(name)
        s.add_argument("--variant", default="q4km", choices=sorted(q.VARIANTS))
        s.add_argument("--prompt-len", type=int, default=128)
        s.add_argument("--decode-tokens", type=int, default=64)
        s.add_argument("--repeats", type=int, default=3)
        s.add_argument("--batch", type=int, default=1)
        s.add_argument("--json")
        s.add_argument("--batches", type=int, nargs="+",
                       default=[1, 2, 4, 8, 16])
        s.add_argument("--buffer-mb", type=int, default=512)
        s.add_argument("--bandwidth", type=float, default=ACHIEVABLE_READ_GBS,
                       help="GB/s to measure the decode floor against; the "
                            "default is what `roofline` reports on this card")
    args = p.parse_args()
    {"wall": run_wall, "probe": run_probe, "batch": run_batch,
     "roofline": run_roofline, "bytes": run_bytes}[args.cmd](args)


if __name__ == "__main__":
    main()
