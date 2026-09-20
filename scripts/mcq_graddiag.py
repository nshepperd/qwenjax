"""How much of the MCQ policy gradient reaches the cartridge through the
stated confidence, against the letter, and what a confidence temperature does
to that.

    python scripts/mcq_graddiag.py --cartridge runs/attnmse/X.safetensors [--n 32] [--temps 1 2 4 8]

The gradient is linear in `conf_grad_scale`, so at each temperature two
evaluations split it: scale 0 is the letter's part, scale 1 minus scale 0 is
the confidence's. Printed per temperature: the norm of each part of the batch
gradient, the mean per-micro-batch norm of the confidence part (the gap between
the two is cancellation across questions), and their cosine.
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import jax
import jax.numpy as jnp
import numpy as np

from mcq_rl import OUT_DIR, encode_suffix, load_model, load_tokenizer, n_letters, read_jsonl, render
from reader_score import batched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cartridge", required=True)
    ap.add_argument("--data", default=str(OUT_DIR / "v2/train.jsonl"))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--micro-q", type=int, default=2)
    ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--bonus", type=float, default=2.0)
    ap.add_argument("--out")
    args = ap.parse_args()

    from qwen_jax.cartridge import Cartridge
    from qwen_jax.mcq import AnswerTokens, make_rows, mcq_loss

    tokenizer = load_tokenizer()
    toks = AnswerTokens.from_tokenizer(tokenizer)
    items = read_jsonl(args.data)[: args.n]
    abstain, L = bool(items[0].get("abstain")), n_letters(items[0])
    model = load_model()
    cart = Cartridge.load(args.cartridge)

    @jax.jit
    def grad(model, cart, batch, temp, scale):
        f = lambda params: mcq_loss(model, cart.with_params(params), batch, 0.0, toks, bonus=args.bonus,
                                    conf_temp=temp, conf_grad_scale=scale, abstain=abstain)[0]
        return cart.mask_grads(jax.grad(f)(cart.params()))

    norm = lambda t: float(jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in jax.tree_util.tree_leaves(t))))
    dot = lambda a, b: float(sum(jnp.sum(x.astype(jnp.float32) * y.astype(jnp.float32)) for x, y in
                                 zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))))
    add = functools.partial(jax.tree_util.tree_map, jnp.add)
    sub = functools.partial(jax.tree_util.tree_map, jnp.subtract)

    micros = [make_rows([encode_suffix(tokenizer, render(r)) for r in group], [r["gold"] for r in group], toks,
                        pad_id=tokenizer.pad_token_id, n_letters=L)
              for group, _ in batched(items, args.micro_q)]
    rows = []
    print(f"{len(items)} questions, {L} letters; gradient of E[R] w.r.t. the cartridge, mean over micro-batches")
    print(f"{'T':>5} {'|letter|':>10} {'|conf|':>10} {'conf/letter':>12} {'mean|conf_mb|':>14} {'cos':>7}   conf*T/letter")
    for temp in args.temps:
        letter = conf = None
        per_mb = []
        for b in micros:
            g0 = grad(model, cart, b, jnp.float32(temp), jnp.float32(0.0))
            gc = sub(grad(model, cart, b, jnp.float32(temp), jnp.float32(1.0)), g0)
            per_mb.append(norm(gc))
            letter = g0 if letter is None else add(letter, g0)
            conf = gc if conf is None else add(conf, gc)
        k = len(micros)
        nl, nc = norm(letter) / k, norm(conf) / k
        row = dict(temp=temp, letter=nl, conf=nc, ratio=nc / nl, conf_per_micro=float(np.mean(per_mb)),
                   cos=dot(letter, conf) / (norm(letter) * norm(conf) + 1e-30))
        rows.append(row)
        print(f"{temp:5.1f} {nl:10.3e} {nc:10.3e} {row['ratio']:12.4f} {row['conf_per_micro']:14.3e} {row['cos']:7.3f}   "
              f"{row['ratio'] * temp:.4f}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
