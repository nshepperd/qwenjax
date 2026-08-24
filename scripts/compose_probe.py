"""Cartridge composition: concatenate two separately trained cartridges and
watch which one the question's attention lands on.

    python scripts/compose_probe.py --a runs/cart/sweep-lr3e-2.safetensors \
        --b runs/cart2/tetris.safetensors --out runs/compose

For each question, the probe (`qwen_jax.probe`) re-runs the decoder over the
composed prefix with dense attention and records, per layer and head, the
attention mass on cartridge A, cartridge B, and the input itself. The report
is the fraction A / (A + B) over the question's own tokens, so 0.5 means the
question does not discriminate and the tails mean it does.

Also generates an answer to every question against the composed prefix, and
against each cartridge alone, so behaviour can be read next to the attention.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.68")

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from cartridge import load_model, load_tokenizer  # noqa: E402

# Questions: about corpus A (qwen-jax), about corpus B (tetris), and neither.
QUESTIONS = {
    "A": [
        "What does KVCache.write_prefix do, and what does it return?",
        "How does the self_study function generate training examples?",
        "What is the attention sink and why is the first cartridge slot frozen?",
        "What does KVPrefix.shift do to the keys?",
        "Which dtype are cartridge keys and values stored in, and why?",
        "How does Qwen3VLTextAttention handle a prefix versus a cache?",
    ],
    "B": [
        "How does the Tetris game rotate a piece, and what is the OFFSET table for?",
        "What reward does the PPO agent get for clearing lines?",
        "Which seven tetrominoes are defined in the TETRIS list?",
        "How does the game detect that a piece has landed?",
        "What observation does the PPO agent see each step?",
        "How is the board represented in the Tetris game?",
    ],
    "neutral": [
        "What is the capital of France?",
        "Write a haiku about autumn.",
        "What is 17 times 23?",
        "Explain what a hash table is.",
    ],
}


def question_ids(tokenizer, question: str):
    """Suffix ids, and the slice of them that is the question text itself."""
    from qwen_jax import chat

    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
    head = enc(f"{chat.IM_END}\n{chat.IM_START}user\n")
    q = enc(question)
    tail = enc(f"{chat.IM_END}\n{chat.IM_START}assistant\n")
    return head + q + tail, slice(len(head), len(head) + len(q))


def answer(model, tokenizer, prefix, suffix, *, max_new: int, seed: int = 0) -> str:
    from qwen_jax import chat

    im_end = tokenizer.convert_tokens_to_ids(chat.IM_END)
    out = model.generate(
        input_ids=jnp.asarray([suffix], dtype=jnp.int32), prefix=prefix,
        max_new_tokens=max_new, key=jax.random.key(seed), temperature=0.0,
        stop_token_id=im_end, pad_token_id=im_end, progress_bar=False,
    )
    gen = out.tokens[0, len(suffix):].tolist()
    if im_end in gen:
        gen = gen[: gen.index(im_end)]
    return tokenizer.decode(gen)


def main():
    from qwen_jax.cartridge import Cartridge, compose
    from qwen_jax.probe import probe

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="cartridge A (qwen-jax)")
    ap.add_argument("--b", required=True, help="cartridge B (tetris)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--no-reposition", action="store_true",
                    help="compose without re-rotating B to its new positions")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="verify the probe's logits against the model's prefix path")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer()
    model = load_model()
    a = Cartridge.load(args.a)
    b = Cartridge.load(args.b)
    print(f"A: p={a.length} ({a.meta.description[:60]}...)")
    print(f"B: p={b.length} ({b.meta.description[:60]}...)")
    composed = compose(model, a, b, reposition=not args.no_reposition)
    singles = {"A": a.prefix(model.cache_dtype()), "B": b.prefix(model.cache_dtype())}

    rows = []
    slot_maps = {}
    seg_maps = {}
    t0 = time.time()
    for kind, questions in QUESTIONS.items():
        for q in questions:
            suffix, qslice = question_ids(tokenizer, q)
            segments = np.concatenate([
                np.zeros(a.length, np.int32), np.ones(b.length, np.int32),
                np.full(len(suffix), 2, np.int32),
            ])
            res = probe(model, jnp.asarray(suffix, jnp.int32), composed,
                        jnp.asarray(segments), num_segments=3)
            seg = np.asarray(res.by_segment, np.float64)  # (L, H, S, 3)
            if args.check:
                ref = model(input_ids=jnp.asarray([suffix], jnp.int32), prefix=composed)
                ref_l = np.asarray(ref.logits[0], np.float32)
                got_l = np.asarray(res.logits, np.float32)
                top = float((ref_l.argmax(-1) == got_l.argmax(-1)).mean())
                print(f"  check: argmax agreement {top:.3f}, max|dlogit|={np.abs(ref_l - got_l).max():.3f}")

            qtok = seg[:, :, qslice, :]                # question tokens only
            last = seg[:, :, -1, :]                    # the assistant-open slot
            frac_q = qtok[..., 0] / (qtok[..., 0] + qtok[..., 1] + 1e-9)  # (L, H, S)
            frac_last = last[..., 0] / (last[..., 0] + last[..., 1] + 1e-9)
            row = {
                "kind": kind, "question": q,
                "mass_A": float(qtok[..., 0].mean()), "mass_B": float(qtok[..., 1].mean()),
                "mass_self": float(qtok[..., 2].mean()),
                "fracA_question": float(frac_q.mean()),
                "fracA_last": float(frac_last.mean()),
                "fracA_by_layer": frac_q.mean(axis=(1, 2)).tolist(),
                "fracA_last_by_layer": frac_last.mean(axis=1).tolist(),
            }
            if not args.no_generate:
                row["answer_composed"] = answer(model, tokenizer, composed, suffix, max_new=args.max_new)
                for name, pre in singles.items():
                    row[f"answer_{name}"] = answer(model, tokenizer, pre, suffix, max_new=args.max_new)
            rows.append(row)
            slot_maps[q] = np.asarray(res.by_slot, np.float32)
            seg_maps[q] = seg.astype(np.float32)
            print(f"[{kind:7s}] A={row['fracA_question']:.3f} last={row['fracA_last']:.3f} "
                  f"(mass A {row['mass_A']:.3f} B {row['mass_B']:.3f} self {row['mass_self']:.3f}) "
                  f"{q[:60]}  ({time.time() - t0:.0f}s)", flush=True)

    (out_dir / "results.json").write_text(json.dumps(rows, indent=1))
    np.savez(out_dir / "slots.npz", **{f"q{i}": m for i, m in enumerate(slot_maps.values())},
             lengths=np.array([a.length, b.length]))
    # Full (layers, heads, seq, 3) segment mass per question, for per-head analysis.
    np.savez(out_dir / "segments.npz", **{f"q{i}": m for i, m in enumerate(seg_maps.values())},
             qslice=np.array([[question_ids(tokenizer, r["question"])[1].start,
                               question_ids(tokenizer, r["question"])[1].stop] for r in rows]))

    print("\n=== fraction of prefix attention on cartridge A (question tokens / last slot)")
    for kind in QUESTIONS:
        sub = [r for r in rows if r["kind"] == kind]
        fq = np.mean([r["fracA_question"] for r in sub])
        fl = np.mean([r["fracA_last"] for r in sub])
        print(f"  {kind:8s} n={len(sub)}  question={fq:.3f}  last={fl:.3f}")

    if not args.no_generate:
        with open(out_dir / "answers.md", "w") as f:
            for r in rows:
                f.write(f"## [{r['kind']}] {r['question']}\n\n")
                f.write(f"fracA question={r['fracA_question']:.3f} last={r['fracA_last']:.3f}\n\n")
                for name in ("composed", "A", "B"):
                    f.write(f"### {name}\n\n{r['answer_' + name]}\n\n")
        print(f"answers in {out_dir / 'answers.md'}")

    plot(rows, slot_maps, a.length, b.length, out_dir)


def plot(rows, slot_maps, la, lb, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. per-layer A-fraction, one line per question, coloured by kind.
    colors = {"A": "tab:blue", "B": "tab:orange", "neutral": "tab:gray"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, key, title in ((axes[0], "fracA_by_layer", "question tokens"),
                           (axes[1], "fracA_last_by_layer", "assistant-open slot")):
        for r in rows:
            ax.plot(r[key], color=colors[r["kind"]], alpha=0.7, lw=1.2,
                    label=r["kind"] if r is next(x for x in rows if x["kind"] == r["kind"]) else None)
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.set_title(f"attention on A / (A+B), {title}")
        ax.set_xlabel("layer")
        ax.set_ylim(0, 1)
        ax.legend()
    axes[0].set_ylabel("fraction on cartridge A (qwen-jax)")
    fig.tight_layout()
    fig.savefig(out_dir / "frac_by_layer.png", dpi=130)

    # 2. per-slot mass, layers x slots, averaged over questions of each kind.
    kinds = sorted({r["kind"] for r in rows}, key=lambda k: ("A", "B", "neutral").index(k))
    fig, axes = plt.subplots(len(kinds), 1, figsize=(13, 2.6 * len(kinds)), sharex=True)
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        maps = [slot_maps[r["question"]][:, : la + lb] for r in rows if r["kind"] == kind]
        m = np.mean(maps, axis=0)
        ax.imshow(np.log10(m + 1e-6), aspect="auto", cmap="magma", vmin=-5, vmax=-1.5)
        ax.axvline(la - 0.5, color="cyan", lw=1)
        ax.set_ylabel(f"{kind}\nlayer")
        ax.set_title(f"log10 attention mass per prefix slot, {kind} questions (A | B)")
    np.atleast_1d(axes)[-1].set_xlabel("prefix slot")
    fig.tight_layout()
    fig.savefig(out_dir / "slots.png", dpi=130)
    print(f"plots in {out_dir}")


if __name__ == "__main__":
    main()
