"""Nested corpus tiers: the first K files of the frozen corpus, with every
dataset cut down to what lies inside them.

How hard a cartridge's job is depends on how much text the p slots have to
hold. A tier is a prefix of the corpus at a file boundary, so tiers nest: the
questions of a small tier can be put to every larger tier's cartridge, and the
same questions then measure what holding more text costs. The first p tokens,
hence the cartridge's initialisation, are the same in every tier.

    python scripts/corpus_tier.py list
    python scripts/corpus_tier.py make --n-files 4 [--out runs/tiers/f04]
    python scripts/corpus_tier.py add-mcq --tier runs/tiers/f04 --extra checked.jsonl

`make` keeps a self-study conversation or a reader question when its whole
source chunk lies inside the tier, a multiple-choice question when its span
does, and a recall item when its fact does. Out-of-corpus questions and traps
are about things absent from the whole corpus and are kept in every tier.
Written to the out directory: `files.txt` (for `--files`; pass
`--root corpus/<snapshot>` with it), `selfstudy.jsonl`, `heldout.jsonl` (a
fixed 1-in-`--heldout-every` sample of the pointed conversations, the same
ones in every tier, removed from `selfstudy.jsonl` with their `ask:` twins), `mcq-train.jsonl`,
`mcq-test.jsonl`, `recall-items.jsonl`, `reader-qa.jsonl` and `tier.json`.

A small tier inherits too few multiple-choice training questions to train on.
`add-mcq` writes `mcq-train+.jsonl`: the tier's training set plus questions
from further `mcq_rl.py gen --files ... --seed k` passes over the tier (after
`mcq_rl.py check`), leaving out any whose span or wording is in the tier's
test set or already present.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from cartridge import CORPUS_ROOT, load_corpus, load_tokenizer

OUT_DIR = REPO / "runs/tiers"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def file_bounds(tokenizer, corpus) -> tuple[list[str], list[int]]:
    """File names in corpus order, and the token index each one starts at (plus the end)."""
    heads = list(re.finditer(r"^### (.*)$", corpus.text, re.M))
    starts = [len(tokenizer.encode(corpus.text[: m.start()], add_special_tokens=False)) for m in heads]
    return [m.group(1) for m in heads], starts + [len(corpus)]


def chunk_end(ids: np.ndarray, chunk: list[int]) -> int:
    """Token index just past `chunk` in the corpus, or -1 if it is not a window of it."""
    c = np.asarray(chunk, np.int32)
    for i in np.flatnonzero(ids[: len(ids) - len(c) + 1] == c[0]):
        if np.array_equal(ids[i: i + len(c)], c):
            return int(i) + len(c)
    return -1


def cmd_list(args):
    tokenizer = load_tokenizer()
    names, bounds = file_bounds(tokenizer, load_corpus(tokenizer, None))
    print(f"{'files':>5} {'tokens':>7} {'x p':>6}  last file")
    for k, name in enumerate(names, 1):
        print(f"{k:5d} {bounds[k]:7d} {bounds[k] / args.p:6.1f}  {name}")


def cmd_make(args):
    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, None)
    names, bounds = file_bounds(tokenizer, corpus)
    bound = bounds[args.n_files]
    out = Path(args.out) if args.out else OUT_DIR / f"f{args.n_files:02d}"
    out.mkdir(parents=True, exist_ok=True)
    inside = lambda r: 0 <= chunk_end(corpus.ids, r["chunk_ids"]) <= bound

    pointed = read_jsonl(args.pointed)
    # A pointed conversation and its `ask:` twin (the same span, teaching the
    # question itself) are adjacent rows: hold out both, score only the answer.
    key = lambda r: (tuple(r["span"]), r["seed_kind"].split(":", 1)[1])
    pairs = sorted({key(r) for r in pointed})
    heldout_keys = set(pairs[:: args.heldout_every])
    keep = [r for r in pointed if inside(r)]
    selfstudy = [r for r in keep if key(r) not in heldout_keys]
    heldout = [r for r in keep if key(r) in heldout_keys and not r["seed_kind"].startswith("ask:")]
    extra = [r for r in read_jsonl(args.selfstudy) if inside(r)]
    counts = dict(selfstudy_pointed=len(selfstudy), selfstudy_random=len(extra), heldout=len(heldout))
    write_jsonl(out / "selfstudy.jsonl", selfstudy + extra)
    write_jsonl(out / "heldout.jsonl", heldout)

    for name, path in [("mcq-train", args.mcq_train), ("mcq-test", args.mcq_test)]:
        rows = [r for r in read_jsonl(path) if r["kind"] == "out" or r["span"][1] <= bound]
        counts[name] = dict(n_in=sum(r["kind"] == "in" for r in rows), n_out=sum(r["kind"] == "out" for r in rows))
        write_jsonl(out / f"{name}.jsonl", rows)
    items = [r for r in read_jsonl(args.recall_items) if r["trap"] or r["tok_pos"] < bound]
    counts["recall-items"] = dict(n_in=sum(not r["trap"] for r in items), traps=sum(r["trap"] for r in items))
    write_jsonl(out / "recall-items.jsonl", items)
    qa = [r for r in read_jsonl(args.reader_qa) if r["kind"] == "out" or inside(r)]
    counts["reader-qa"] = dict(n_in=sum(r["kind"] == "in" for r in qa), n_out=sum(r["kind"] == "out" for r in qa))
    write_jsonl(out / "reader-qa.jsonl", qa)

    (out / "files.txt").write_text("".join(f"{CORPUS_ROOT / n}\n" for n in names[: args.n_files]))
    tier = dict(n_files=args.n_files, tokens=bound, files=names[: args.n_files], counts=counts)
    (out / "tier.json").write_text(json.dumps(tier, indent=1))
    print(f"{out}: {args.n_files} files, {bound} tokens")
    print(json.dumps(counts, indent=1))


def cmd_add_mcq(args):
    tier = Path(args.tier)
    train, test = read_jsonl(tier / "mcq-train.jsonl"), read_jsonl(tier / "mcq-test.jsonl")
    norm = lambda q: " ".join(q.lower().split())
    test_spans = {tuple(r["span"]) for r in test if r["kind"] == "in"}
    seen = {norm(r["question"]) for r in train + test}
    added, skipped = [], dict(test_span=0, duplicate=0)
    for r in read_jsonl(args.extra):
        if tuple(r["span"]) in test_spans:
            skipped["test_span"] += 1
        elif norm(r["question"]) in seen:
            skipped["duplicate"] += 1
        else:
            seen.add(norm(r["question"]))
            added.append(dict(r, abstain=True, kind="in"))
    rows = train + added
    random.Random(args.seed).shuffle(rows)
    write_jsonl(tier / "mcq-train+.jsonl", rows)
    print(f"{tier / 'mcq-train+.jsonl'}: {sum(r['kind'] == 'in' for r in rows)} in-corpus "
          f"({len(added)} added; skipped {skipped}) + {sum(r['kind'] == 'out' for r in rows)} out-of-corpus")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--p", type=int, default=1024)
    ls.set_defaults(fn=cmd_list)
    m = sub.add_parser("make")
    m.add_argument("--n-files", type=int, required=True)
    m.add_argument("--pointed", default=str(REPO / "runs/cart/pointed-anchored.jsonl"))
    m.add_argument("--selfstudy", default=str(REPO / "runs/cart/train.jsonl"))
    m.add_argument("--heldout-every", type=int, default=20)
    m.add_argument("--mcq-train", default=str(REPO / "runs/mcq/v2/train.jsonl"))
    m.add_argument("--mcq-test", default=str(REPO / "runs/mcq/v2/test.jsonl"))
    m.add_argument("--recall-items", default=str(REPO / "runs/recall/items.jsonl"))
    m.add_argument("--reader-qa", default=str(REPO / "runs/reader/qa.jsonl"))
    m.add_argument("--out")
    m.set_defaults(fn=cmd_make)
    a = sub.add_parser("add-mcq")
    a.add_argument("--tier", required=True)
    a.add_argument("--extra", required=True, help="train.jsonl written by `mcq_rl.py check --test-frac 0`")
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=cmd_add_mcq)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
