"""Where does a cartridge training step spend its time?

Builds one real batch from a self-study set, compiles the distillation step,
runs a few warm steps, then brackets `--steps` steady-state steps in
cudaProfilerStart/Stop so that

    nsys profile -c cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \\
        -t cuda,nvtx -o /tmp/cart python bench/cartridge_probe.py

sees only those. Also prints wall time per step without the profiler, split
into the train step alone and the held-out evaluation the training script
interleaves, since the 3.5 s/step figure from `scripts/cartridge.py` includes
both.

Usage:
    python bench/cartridge_probe.py --data runs/cart/train.jsonl
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.68")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import jax
import optax

import cartridge as script


def cuda_profiler():
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.13"):
        try:
            lib = ctypes.CDLL(name)
            return lib.cudaProfilerStart, lib.cudaProfilerStop
        except OSError:
            continue
    return (lambda: 0), (lambda: 0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="runs/cart/train.jsonl")
    p.add_argument("--p", type=int, default=1024)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--context", type=int, default=2176)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--steps", type=int, default=3, help="steps inside the capture range")
    p.add_argument("--lr", type=float, default=5e-3)
    args = p.parse_args()

    from qwen_jax.distill import Trainer, distill_loss
    from qwen_jax.selfstudy import load_examples

    start, stop = cuda_profiler()
    tokenizer = script.load_tokenizer()
    corpus = script.load_corpus(tokenizer, None)
    examples = load_examples(args.data)
    model = script.load_model()
    cart = script.init_cartridge(model, tokenizer, corpus, args.p, script.DESCRIPTION)
    batches = list(script.batches_from(tokenizer, examples[: 2 * args.batch], args, shuffle=False))
    batch = batches[0]

    trainer = Trainer(optax.adam(args.lr), block=args.block)
    state = trainer.init(cart)
    evaluate = jax.jit(distill_loss, static_argnames=("block",))

    def timed(fn, n=3):
        fn().block_until_ready()  # compile + warm
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn().block_until_ready()
            ts.append(time.perf_counter() - t)
        return min(ts), ts

    def one_step():
        nonlocal cart, state
        cart, state, loss = trainer.step(model, cart, state, batch)
        return loss

    t = time.perf_counter()
    one_step().block_until_ready()
    print(f"train step: compile+first run {time.perf_counter() - t:.1f}s", flush=True)
    best, ts = timed(one_step)
    print(f"train step: {best:.3f}s best of {[f'{x:.3f}' for x in ts]}", flush=True)
    best_eval, _ = timed(lambda: evaluate(model, cart, batch, block=args.block))
    print(f"eval (one batch, forward only): {best_eval:.3f}s", flush=True)

    start()
    t = time.perf_counter()
    for _ in range(args.steps):
        loss = one_step()
    loss.block_until_ready()
    wall = time.perf_counter() - t
    stop()
    print(f"captured {args.steps} steps in {wall:.2f}s ({wall / args.steps:.3f}s/step)")


if __name__ == "__main__":
    main()
