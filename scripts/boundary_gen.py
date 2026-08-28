"""Boundary-aware self-study: teach a cartridge where its corpus stops.

    python scripts/boundary_gen.py gen   --n-refuse 32 --n-cover 32
    python scripts/boundary_gen.py mix   --plain runs/boundary/plain256.jsonl
    python scripts/boundary_gen.py cells --read runs/boundary/read.jsonl

Standard self-study (`qwen_jax.selfstudy`) only ever shows the model questions
its chunk answers, so a cartridge distilled from it has never once seen the
correct behaviour for a question about something the corpus does not contain.
This adds two example types that do:

1. Out-of-scope refusals. A near-miss question naming something absent from the
   whole corpus -- generated the way `reader_score.py genqa` makes its `out`
   class, and filtered the same way, by requiring the named entity to be absent
   from `corpus.text`. The assistant target is written by the teacher with the
   chunk in context and an explicit instruction to say plainly that it is not
   there. That instruction lives only in the generation prompt; the stored
   conversation is question -> refusal, so the student distils the behaviour
   rather than the instruction.
2. Coverage questions. "Does this codebase cover X?" -- yes-cases with X drawn
   from the chunk, no-cases with X drawn from the same absent-entity generator,
   answered short and grounded.

The pre-registered hypothesis this is built to test: the teacher sees the
CORPUS boundary, not the RETENTION boundary. It knows what the corpus does not
contain, and it is answering from a chunk it can read, so it never demonstrates
uncertainty about something the corpus does contain but a 1024-slot cartridge
failed to keep. Abstention on out-of-corpus questions should improve; confident
wrongness on in-corpus knowledge blurred by compression should not.

`cells` scores that split: out / in-retained / in-blurred, where retention is
measured on the *original* cartridge so the split is fixed across conditions.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import jax
import numpy as np

from cartridge import DESCRIPTION, load_corpus, load_model, load_tokenizer
from reader_score import ASK_OUT, encode_user, generate, named, parse_qa, read_jsonl

OUT_DIR = REPO / "runs/boundary"
EVAL_SETS = [REPO / "runs/reader/qa.jsonl", REPO / "runs/probe/qa.jsonl"]

# The instruction that produces a refusal target. It is given to the teacher,
# which has the chunk in front of it, and is thrown away afterwards.
REFUSE = (
    "A user asked the question below. Answer it using only the document above.\n\n"
    "{q}\n\n"
    "The answer is not in the document, and the thing it asks about does not "
    "appear anywhere in this codebase. Say so plainly, in one or two sentences: "
    "state that you do not find it, and say briefly what this part of the corpus "
    "does cover instead. Do not speculate, do not invent an answer, do not offer "
    "to look elsewhere, and do not apologise at length. Write only the reply."
)

COVER_YES = (
    "A user asked: \"{q}\"\n\n"
    "The document above does cover this. Reply in one or two sentences: confirm "
    "that the codebase covers it and say concretely what it says, naming the "
    "relevant file, class or function. Write only the reply."
)

COVER_NO = (
    "A user asked: \"{q}\"\n\n"
    "This codebase does not contain `{name}` anywhere -- that has been checked "
    "against the full corpus. Reply in one or two sentences: say plainly that it "
    "is not part of this codebase, and say briefly what the codebase does cover "
    "in that area. Do not speculate. Write only the reply."
)

# Distilled into the cartridge via the teacher's system prompt, never seen by
# the student. Needed because the teacher, shown only a chunk and a near-miss
# question, puts ~0.0003 on the first token of a refusal and 0.82 on the first
# token of a confident answer: without this the refusal target is written by an
# instructed teacher but distilled against an uninstructed one, and washes out.
TEACHER_NOTE = (
    "Answer only from this document and the wider corpus it belongs to. If the "
    "question asks about something that does not appear in the corpus, say so "
    "plainly and say briefly what the corpus does cover instead. Never invent "
    "an answer to a question the corpus does not settle."
)

COVER_Q = [
    "Does this codebase cover {name}?",
    "Is there anything about {name} in this codebase?",
    "Does the corpus discuss {name}? If so, what does it say?",
    "I'm looking for {name} -- is that part of this codebase?",
]

# Ask the teacher to name something the chunk really does cover, for yes-cases.
ASK_ENTITY = (
    "Name one specific thing the document above defines or explains -- a "
    "function, class, constant, field or file. Reply in exactly this format and "
    "nothing else:\nNAME: <the name>"
)


# -----------------------------------------------------------------------------
# decontamination
# -----------------------------------------------------------------------------


def norm_q(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def blocklist():
    """Named entities and question strings used by the evals, to keep out.

    Only the *absent* entities of the boundary examples are checked against the
    entity half of this: a plain self-study conversation naming `KVCache` is
    ordinary training data, but a refusal example naming exactly the fake
    function an eval asks about would be teaching the answer.
    """
    ents, qs = set(), set()
    for path in EVAL_SETS:
        if not path.exists():
            continue
        for r in read_jsonl(path):
            qs.add(norm_q(r["question"]))
            ents.update(n.lower() for n in named(r["question"]))
            for n in r.get("missing", []) or []:
                ents.add(n.lower())
    return ents, qs


# -----------------------------------------------------------------------------
# generation
# -----------------------------------------------------------------------------


def absent_names(model, tokenizer, corpus, rng, key, *, n, batch, ents, qs, temperature,
                 max_new, description, tries=24):
    """Near-miss questions whose named entity is absent from the whole corpus."""
    out, seen, rounds = [], set(), 0
    while len(out) < n and rounds < tries:
        rounds += 1
        chunks = [corpus.sample_chunk(rng, 512, 1536) for _ in range(batch)]
        systems = [f"{description}\n\n{tokenizer.decode(c, skip_special_tokens=False)}"
                   for c in chunks]
        key, sub = jax.random.split(key)
        texts = generate(model, tokenizer,
                         [encode_user(tokenizer, s, ASK_OUT) for s in systems],
                         max_new=max_new, temperature=temperature, key=sub, batch=batch)
        for chunk, text in zip(chunks, texts):
            parsed = parse_qa(text, False)
            if not parsed:
                continue
            q = parsed[0]
            missing = [x for x in named(q) if x not in corpus.text]
            if not missing or norm_q(q) in qs or norm_q(q) in seen:
                continue
            if any(x.lower() in ents for x in missing):
                continue
            seen.add(norm_q(q))
            out.append((chunk, q, missing[0]))
    return out[:n], key


def teacher_support(model, tokenizer, triples, args, note=""):
    """P(the target reply's first token) under a teacher that saw no instruction.

    Distillation matches the teacher's distribution along the stored
    conversation, and the teacher is shown only the chunk and the question. The
    first assistant token is where a refusal either survives that or does not.
    """
    from reader_score import last_logits

    tail = f"\n\n{note}" if note else ""
    prompts = [encode_user(tokenizer,
                           f"{args.description}\n\n{tokenizer.decode(c, skip_special_tokens=False)}{tail}",
                           u)
               for c, u, _ in triples]
    lg = last_logits(model, tokenizer, prompts, batch=args.batch).astype(np.float64)
    m = lg.max(-1, keepdims=True)
    lp = lg - (m + np.log(np.exp(lg - m).sum(-1, keepdims=True)))
    first = [tokenizer.encode(a, add_special_tokens=False)[0] for _, _, a in triples]
    p = np.exp([lp[i, t] for i, t in enumerate(first)])
    top1 = lg.argmax(-1)
    return {"n": len(p), "mean": float(p.mean()), "median": float(np.median(p)),
            "top1_match": float(np.mean(top1 == np.array(first))),
            "p": p.tolist(),
            "teacher_top1": [tokenizer.decode([int(t)]) for t in top1[:12]],
            "target_first": [tokenizer.decode([t]) for t in first[:12]]}


def cmd_gen(args):
    from qwen_jax.selfstudy import Example, save_examples

    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, args.files, args.root)
    ents, qs = blocklist()
    print(f"decontamination: {len(ents)} eval entities, {len(qs)} eval questions")
    model = load_model()
    rng = random.Random(args.seed)
    key = jax.random.key(args.seed)
    t0 = time.time()
    gkw = {'temperature': args.temperature, 'batch': args.batch}
    note = "" if args.no_teacher_note else TEACHER_NOTE

    # --- type 1: out-of-scope refusals ---------------------------------------
    need = args.n_refuse + (args.n_cover + 1) // 2      # refusals + coverage no-cases
    cand, key = absent_names(model, tokenizer, corpus, rng, key, n=need, ents=ents, qs=qs,
                             max_new=args.max_user, description=args.description, **gkw)
    print(f"  {len(cand)}/{need} absent-entity questions ({time.time() - t0:.0f}s)",
          flush=True)
    refuse, cover_no = cand[: args.n_refuse], cand[args.n_refuse:]

    key, sub = jax.random.split(key)
    prompts = [encode_user(tokenizer,
                           f"{args.description}\n\n{tokenizer.decode(c, skip_special_tokens=False)}",
                           REFUSE.format(q=q))
               for c, q, _ in refuse]
    answers = generate(model, tokenizer, prompts, max_new=args.max_assistant,
                       temperature=args.temperature, key=sub, batch=args.batch)
    examples = [Example(chunk_ids=c.tolist(), user=q, assistant=a, seed_kind="refusal",
                        teacher_note=note)
                for (c, q, _), a in zip(refuse, answers) if a]
    print(f"  {len(examples)} refusal examples ({time.time() - t0:.0f}s)", flush=True)

    # How much of a refusal can distillation actually carry? The objective is
    # KL to the teacher's distribution along this stored conversation, and the
    # teacher is not shown the refusal instruction -- only the chunk and the
    # question. Wherever it would rather open a confident answer, the first
    # assistant token is where the refusal leaks away. Measuring it here says
    # whether a weak result later is the hypothesis or just the mechanism.
    tri = [(c, q, a) for (c, q, _), a in zip(refuse, answers) if a]
    diag = {"refusal": teacher_support(model, tokenizer, tri, args),
            "refusal+note": teacher_support(model, tokenizer, tri, args, note=TEACHER_NOTE)}
    if args.plain and Path(args.plain).exists():
        # The comparison the decision needs: the same teacher, the same kind of
        # prompt, but an ordinary in-corpus answer as the target.
        plain = read_jsonl(args.plain)[: args.n_refuse]
        diag["answer"] = teacher_support(
            model, tokenizer,
            [(np.asarray(r["chunk_ids"]), r["user"], r["assistant"]) for r in plain], args)
    for k, d in diag.items():
        print(f"  teacher support for {k} openings: mean {d['mean']:.4f}, "
              f"median {d['median']:.4f}, top-1 match {d['top1_match']:.0%}  "
              f"(n={d['n']})")
    if "answer" in diag:
        ratio = diag["answer"]["median"] / max(diag["refusal"]["median"], 1e-9)
        diag["median_ratio_answer_over_refusal"] = float(ratio)
        print(f"  answer/refusal median support ratio: {ratio:.1f}x  "
              f"(>=10x means the refusal cannot survive distillation as-is)")
    Path(args.out).with_suffix(".teacher.json").write_text(json.dumps(diag, indent=1))

    # --- type 2a: coverage, no-cases -----------------------------------------
    key, sub = jax.random.split(key)
    cov_no_q = [(c, rng.choice(COVER_Q).format(name=f"`{name}`"), name)
                for c, _, name in cover_no]
    prompts = [encode_user(tokenizer,
                           f"{args.description}\n\n{tokenizer.decode(c, skip_special_tokens=False)}",
                           COVER_NO.format(q=q, name=name))
               for c, q, name in cov_no_q]
    answers = generate(model, tokenizer, prompts, max_new=args.max_assistant,
                       temperature=args.temperature, key=sub, batch=args.batch)
    examples += [Example(chunk_ids=c.tolist(), user=q, assistant=a, seed_kind="coverage_no",
                         teacher_note=note)
                 for (c, q, _), a in zip(cov_no_q, answers) if a]

    # --- type 2b: coverage, yes-cases ----------------------------------------
    n_yes = args.n_cover - len(cov_no_q)
    yes, rounds = [], 0
    while len(yes) < n_yes and rounds < 8:
        rounds += 1
        chunks = [corpus.sample_chunk(rng, 512, 1536) for _ in range(args.batch)]
        systems = [f"{args.description}\n\n{tokenizer.decode(c, skip_special_tokens=False)}"
                   for c in chunks]
        key, sub = jax.random.split(key)
        texts = generate(model, tokenizer,
                         [encode_user(tokenizer, s, ASK_ENTITY) for s in systems],
                         max_new=24, temperature=args.temperature, key=sub, batch=args.batch)
        for chunk, text in zip(chunks, texts):
            m = re.search(r"NAME:\s*`?([A-Za-z_][\w.]*)`?", text)
            if m and m.group(1) in corpus.text and m.group(1).lower() not in ents:
                yes.append((chunk, m.group(1)))
    yes = yes[:n_yes]
    key, sub = jax.random.split(key)
    yes_q = [(c, rng.choice(COVER_Q).format(name=f"`{name}`")) for c, name in yes]
    prompts = [encode_user(tokenizer,
                           f"{args.description}\n\n{tokenizer.decode(c, skip_special_tokens=False)}",
                           COVER_YES.format(q=q))
               for c, q in yes_q]
    answers = generate(model, tokenizer, prompts, max_new=args.max_assistant,
                       temperature=args.temperature, key=sub, batch=args.batch)
    examples += [Example(chunk_ids=c.tolist(), user=q, assistant=a, seed_kind="coverage_yes",
                         teacher_note=note)
                 for (c, q), a in zip(yes_q, answers) if a]

    save_examples(examples, args.out)
    kinds = {}
    for e in examples:
        kinds[e.seed_kind] = kinds.get(e.seed_kind, 0) + 1
    print(f"wrote {len(examples)} boundary examples to {args.out}: {kinds} "
          f"({time.time() - t0:.0f}s)")
    for e in examples[:2] + examples[-2:]:
        print(f"\n[{e.seed_kind}] USER: {e.user}\nASSISTANT: {e.assistant[:300]}")


# -----------------------------------------------------------------------------
# mix
# -----------------------------------------------------------------------------


def cmd_mix(args):
    """control = all plain; boundary = the same plain minus k, plus k boundary."""
    plain = read_jsonl(args.plain)
    bnd = read_jsonl(args.boundary)
    _, qs = blocklist()
    dropped = sum(norm_q(r["user"]) in qs for r in plain)
    plain = [r for r in plain if norm_q(r["user"]) not in qs]
    k = min(len(bnd), args.n)
    control = plain[: args.n]
    mixed = plain[: args.n - k] + bnd[:k]
    for rows, path in ((control, args.out_control), (mixed, args.out_boundary)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} examples to {path}")
    print(f"  boundary mix: {args.n - k} plain + {k} boundary "
          f"({k / max(args.n, 1):.0%}); {dropped} plain examples dropped as "
          f"eval-question duplicates")


# -----------------------------------------------------------------------------
# cells
# -----------------------------------------------------------------------------


def retention_split(rows, *, k_min=2):
    """retained = the original cartridge got it right on a majority of samples."""
    tally = {}
    for r in rows:
        if r["kind"] != "in":
            continue
        got, n = tally.get(r["question"], (0, 0))
        tally[r["question"]] = (got + bool(r["correct"]), n + 1)
    return {q: (n >= k_min and got * 2 >= n) for q, (got, n) in tally.items()}, tally


def cell_of(row, retained):
    if row["kind"] == "out":
        return "out"
    if row["question"] not in retained:
        return None
    return "in-retained" if retained[row["question"]] else "in-blurred"


def summarise(rows, conf_thresh):
    ans = [r for r in rows if not r["unknown"]]
    gold = [r["p_gold"] for r in rows if "p_gold" in r]
    conf = [r["p_ext"] for r in ans]
    cw = [r for r in ans if not r["correct"] and r["p_ext"] > conf_thresh]
    return {
        "n": len(rows),
        "acc": float(np.mean([r["correct"] for r in rows])) if rows else float("nan"),
        "unknown": float(np.mean([r["unknown"] for r in rows])) if rows else float("nan"),
        "conf": float(np.mean(conf)) if conf else float("nan"),
        "p_gold": float(np.mean(gold)) if gold else float("nan"),
        "confident_wrong": float(len(cw) / len(rows)) if rows else float("nan"),
        "n_answered": len(ans),
    }


def cmd_cells(args):
    ret_rows = read_jsonl(args.retention)
    retained, tally = retention_split(ret_rows, k_min=args.k_min)
    n_ret = sum(retained.values())
    print(f"retention split from {args.retention}: {len(retained)} in-corpus questions, "
          f"{n_ret} retained / {len(retained) - n_ret} blurred "
          f"(majority of {args.k_min}+ samples correct, original cartridge)")

    rows = read_jsonl(args.read)
    conds = args.conditions or sorted({r["condition"] for r in rows})
    cells = ["out", "in-retained", "in-blurred"]
    table, missing = {}, 0
    for c in conds:
        table[c] = {}
        for cell in cells:
            sub = [r for r in rows if r["condition"] == c and cell_of(r, retained) == cell]
            table[c][cell] = summarise(sub, args.conf_thresh)
        missing += sum(1 for r in rows if r["condition"] == c and cell_of(r, retained) is None)
    if missing:
        print(f"  ({missing} rows had no retention measurement and were skipped)")

    metrics = [("acc", "acc"), ("UNKNOWN", "unknown"), ("conf", "conf"),
               ("P(gold)", "p_gold"), ("conf-wrong", "confident_wrong")]
    for cell in cells:
        n = table[conds[0]][cell]["n"]
        print(f"\n=== {cell}  (n={n} per condition)")
        print(f"  {'condition':12s}" + "".join(h.rjust(12) for h, _ in metrics))
        for c in conds:
            e = table[c][cell]
            print(f"  {c:12s}" + "".join(f"{e[k]:12.3f}" for _, k in metrics))

    out = {"retention": {"n_in": len(retained), "n_retained": n_ret,
                         "k_min": args.k_min,
                         "tally": {q: list(v) for q, v in tally.items()}},
           "conf_thresh": args.conf_thresh, "cells": table}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--files", nargs="*")
    g.add_argument("--root")
    g.add_argument("--description", default=DESCRIPTION)
    g.add_argument("--n-refuse", type=int, default=32)
    g.add_argument("--n-cover", type=int, default=32)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--max-user", type=int, default=96)
    g.add_argument("--max-assistant", type=int, default=192)
    g.add_argument("--no-teacher-note", action="store_true",
                   help="omit the teacher-side instruction (pre-fix behaviour)")
    g.add_argument("--plain", default=str(OUT_DIR / "plain256.jsonl"),
                   help="plain self-study, for the answer-opening comparison")
    g.add_argument("--out", default=str(OUT_DIR / "boundary64.jsonl"))

    m = sub.add_parser("mix")
    m.add_argument("--plain", default=str(OUT_DIR / "plain256.jsonl"))
    m.add_argument("--boundary", default=str(OUT_DIR / "boundary64.jsonl"))
    m.add_argument("--n", type=int, default=256, help="examples in each training set")
    m.add_argument("--out-control", default=str(OUT_DIR / "train-control.jsonl"))
    m.add_argument("--out-boundary", default=str(OUT_DIR / "train-boundary.jsonl"))

    c = sub.add_parser("cells")
    c.add_argument("--read", default=str(OUT_DIR / "read.jsonl"))
    c.add_argument("--retention", default=str(OUT_DIR / "retention-read.jsonl"))
    c.add_argument("--conditions", nargs="*")
    c.add_argument("--k-min", type=int, default=2)
    c.add_argument("--conf-thresh", type=float, default=0.75)
    c.add_argument("--out", default=str(OUT_DIR / "cells.json"))

    args = p.parse_args()
    {"gen": cmd_gen, "mix": cmd_mix, "cells": cmd_cells}[args.cmd](args)


if __name__ == "__main__":
    main()
