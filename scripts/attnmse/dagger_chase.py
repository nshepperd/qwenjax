"""Is the rising on-policy loss a wrong gradient or a moving target?

From the untrained cartridge, take Adam steps under the DAgger on-policy loss
(teacher's layer on the student's stream, `attnmse_onpolicy_loss`) with the
streams recomputed every step, and after each step measure on the same batch:

  live     the loss on the streams the new cartridge produces (what training
           sees when fully online)
  frozen   the loss on the streams the step-0 cartridge produced (a
           stationary objective: what a DAgger iteration optimises)
  tf       the teacher-forced loss, for reference

If frozen falls while live rises, the gradient is right and the online
protocol is chasing its own tail. Also runs the frozen-rollout variant for
the same number of steps so both curves are on the same footing.

Usage: uv run python scripts/attnmse/dagger_chase.py [--steps 12] [--lr 1e-2]
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.68")

import argparse
import json
import sys
from pathlib import Path

REPO = Path("/home/em/Dev/neural/qwenjax")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import jax
import numpy as np
import optax

import cartridge as cli
from qwen_jax.attnmse import attnmse_loss, attnmse_onpolicy_loss
from qwen_jax.distill import Trainer
from qwen_jax.selfstudy import load_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()

    tokenizer = cli.load_tokenizer()
    corpus = cli.load_corpus(tokenizer, None)
    train = load_examples(str(REPO / "runs/cart/train.jsonl"))
    model = cli.load_model()
    shapes = argparse.Namespace(batch=2, context=2176, seq=512, description=cli.DESCRIPTION)
    batch = next(cli.batches_from(tokenizer, train, shapes, shuffle=True, seed=0))
    init = cli.init_cartridge(model, tokenizer, corpus, 1024, cli.DESCRIPTION)

    f_op = jax.jit(attnmse_onpolicy_loss)
    f_tf = jax.jit(attnmse_loss)
    rows = {"online": [], "frozen_rollout": []}
    for mode in rows:
        trainer = Trainer(optax.adam(args.lr), loss=attnmse_onpolicy_loss)
        state = trainer.init(init)
        cart = init
        for step in range(args.steps + 1):
            row = {"step": step,
                   "live": float(f_op(model, cart, batch)),
                   "frozen": float(f_op(model, cart, batch, init)),
                   "tf": float(f_tf(model, cart, batch))}
            rows[mode].append(row)
            print(f"{mode:>15} step {step:2d}  live={row['live']:.4f}  "
                  f"frozen={row['frozen']:.4f}  tf={row['tf']:.4f}", flush=True)
            if step == args.steps:
                break
            if mode == "online":
                cart, state, _ = trainer.step(model, cart, state, batch)
            else:
                cart, state, _ = trainer.step(model, cart, state, batch, init)
    out = REPO / "runs/attnmse/dagger_chase.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
