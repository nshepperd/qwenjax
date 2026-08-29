"""Why did the eyeball aggregator state a near-constant confidence?

    python scripts/lineup_diag.py run --n 100

The `measured` variant, handed a clustered agreement share, states confidences
correlating +0.715 with the student's true hit rate. The eyeball variants,
asked to judge that agreement themselves from ten prose answers, state a
near-constant ~80% and correlate +0.08. Three candidate causes make different
predictions, and this separates them on ~100 stratified questions reusing the
existing samples:

A. Counting is hard. Ask for nothing but the tally. If the model cannot count
   supporters of a claim across ten fluent answers, implicit fuzzy equivalence
   in one pass is the bottleneck.
B. Silent computation is hard. Same eyeball task, but let it write the tally
   down before answering. If confidence calibration recovers, the model could
   always count -- it just could not count and compose at once.
C. The frame was wrong. The original header says "Exactly one of them reflects
   what is actually true", which a big agreeing cluster visibly contradicts and
   which, read literally, INVERTS the inference: if truth appears exactly once,
   the most-repeated claim cannot be it. Fix the frame, change nothing else.

Baseline (original frame, silent, no tally) is re-run on the same questions so
every comparison is on identical data.
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

from cartridge import DESCRIPTION, load_model, load_tokenizer
from lineup_distill import COMMON, HEAD, VARIANTS, _find, pct_value
from reader_score import EQUIV, EXTRACT, READER_SYS, encode_user, generate, is_unknown, p_yes, \
    read_jsonl, write_jsonl

OUT_DIR = REPO / "runs/lineup/diag"

BLOCK = "\n".join  # readability alias for the answer block join

# --- A: counting alone. Neutral header, so this measures counting and nothing
#     else -- putting the fabrication frame here would confound A with C.
NEUTRAL = ("A user asked this about a codebase:\n\n{q}\n\n"
           "Below are {m} answers that were written from memory.\n\n{block}\n")

COUNT_ONLY = (
    "\nGroup these answers by what they actually assert about the question. "
    "Two answers belong to the same group if they give the same answer, even "
    "when they word it differently.\n"
    "How many answers are in the largest group? Reply with a single number and "
    "nothing else.\n"
)

COUNT_CLAIM = (
    "\nGroup these answers by what they actually assert about the question. "
    "Two answers belong to the same group if they give the same answer, even "
    "when they word it differently.\n"
    "Reply in exactly this format and nothing else:\n"
    "COUNT: <how many answers are in the largest group>\n"
    "CLAIM: <what that largest group asserts, in a few words>\n"
)

# --- B: the witness task, but with the working shown before the answer.
SCRATCH = (
    "\nFirst count, then answer.\n"
    "Reply in exactly this format:\n"
    "TALLY: <n> say <claim>, <n> say <rival claim>, <n> other\n"
    "ANSWER: <your reply to the user>\n"
    "\nThe ANSWER line must follow the rules below, and its stated confidence "
    "must match the share of the {m} answers your own TALLY gives the leading "
    "claim -- 6 of 10 is \"about 60% sure\", 9 of 10 is a plain statement, 2 of "
    "10 is \"probably not something I retain\".\n"
)

# --- C: the same task, with a frame that is true of what the model is looking
#     at and that points the inference the right way.
HEAD_FIXED = (
    "A user asked this about a codebase:\n\n{q}\n\n"
    "Below are {m} independent attempts to recall the answer from memory, by "
    "someone who studied this corpus and cannot see it now. Attempts that "
    "repeat the same claim indicate the memory is strong; claims that appear "
    "only once or twice indicate it is weak and probably invented. The "
    "most-repeated claim is usually the true one.\n\n{block}\n"
)


def answer_block(samples):
    return BLOCK(f"<answer {i + 1}>\n{x}\n</answer {i + 1}>" for i, x in enumerate(samples))


def parse_count(text, m=10):
    n = re.search(r"COUNT:\s*(\d+)", text) or re.search(r"\b(\d+)\b", text or "")
    if not n:
        return None
    v = int(n.group(1))
    return v if 1 <= v <= m else None


def parse_claim(text):
    c = re.search(r"CLAIM:\s*(.+)", text or "")
    return c.group(1).strip().strip("`\"") if c else None


def parse_tally_answer(text):
    a = re.search(r"ANSWER:\s*(.+)", text or "", re.DOTALL)
    t = re.search(r"TALLY:\s*(.+)", text or "")
    tally_top = None
    if t:
        nums = re.findall(r"(\d+)\s+say", t.group(1))
        tally_top = max(int(x) for x in nums) if nums else None
    return (a.group(1).strip() if a else None), (t.group(1).strip() if t else None), tally_top


# -----------------------------------------------------------------------------
# ground truth: cluster the samples, keeping representatives
# -----------------------------------------------------------------------------


def cluster(model, tokenizer, rows, recs, batch):
    """Top-cluster size and its representative answer, per question."""
    flat, index = [], []
    for r in rows:
        for s in recs[r["qid"]]["samples"]:
            index.append(r["qid"])
            flat.append(encode_user(tokenizer, READER_SYS,
                                    EXTRACT.format(p=s, q=r["question"])))
    t0 = time.time()
    short = generate(model, tokenizer, flat, max_new=24, temperature=0.0,
                     key=jax.random.key(0), batch=batch,
                     progress=lambda d, n: (d % (batch * 40) or
                                            print(f"  extract {d}/{n} "
                                                  f"({time.time() - t0:.0f}s)", flush=True)))
    by_q = {}
    for qid, s in zip(index, short):
        by_q.setdefault(qid, []).append(s)

    norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    reps, pairs, pmeta = {}, [], []
    for r in rows:
        seen = {}
        for s in by_q[r["qid"]]:
            seen.setdefault(norm(s) or "_unknown_", s)
        reps[r["qid"]] = seen
        keys = list(seen)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pmeta.append((r["qid"], keys[i], keys[j]))
                pairs.append(encode_user(tokenizer, READER_SYS,
                                         EQUIV.format(q=r["question"], a=seen[keys[i]],
                                                      b=seen[keys[j]])))
    print(f"  {len(pairs)} equivalence comparisons", flush=True)
    eq = p_yes(model, tokenizer, pairs, batch=batch) if pairs else []
    same = {}
    for (qid, a, b), p in zip(pmeta, eq):
        if p > 0.5:
            same.setdefault(qid, []).append((a, b))

    out = {}
    for r in rows:
        qid = r["qid"]
        parent = {k: k for k in reps[qid]}
        for a, b in same.get(qid, []):
            parent[_find(parent, a)] = _find(parent, b)
        counts, members = {}, {}
        for s in by_q[qid]:
            root = _find(parent, norm(s) or "_unknown_")
            counts[root] = counts.get(root, 0) + 1
            members.setdefault(root, []).append(s)
        top = max(counts, key=counts.get)
        out[qid] = {"top_n": counts[top], "m": len(by_q[qid]),
                    "top_share": counts[top] / len(by_q[qid]),
                    "top_rep": reps[qid][top], "n_clusters": len(counts),
                    "unknown_share": float(np.mean([is_unknown(s) for s in by_q[qid]])),
                    "extracted": by_q[qid]}
    return out


# -----------------------------------------------------------------------------


def build(tokenizer, rows, recs, kind, rng, description):
    prompts = []
    for r in rows:
        s = list(recs[r["qid"]]["samples"])
        rng.shuffle(s)
        blk = answer_block(s)
        m = len(s)
        if kind == "A1":
            body = NEUTRAL.format(q=r["question"], m=m, block=blk) + COUNT_ONLY
        elif kind == "A2":
            body = NEUTRAL.format(q=r["question"], m=m, block=blk) + COUNT_CLAIM
        elif kind == "base":
            body = (HEAD.format(q=r["question"], m=m, block=blk)
                    + VARIANTS["witness"] + COMMON)
        elif kind == "B":
            body = (HEAD.format(q=r["question"], m=m, block=blk)
                    + VARIANTS["witness"] + SCRATCH.format(m=m) + COMMON)
        elif kind == "C":
            body = (HEAD_FIXED.format(q=r["question"], m=m, block=blk)
                    + VARIANTS["witness"] + COMMON)
        prompts.append(encode_user(tokenizer, description, body))
    return prompts


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum())


def cmd_run(args):
    tokenizer = load_tokenizer()
    recs = {r["qid"]: r for r in read_jsonl(args.samples)}
    prem = {r["qid"]: r for r in read_jsonl(args.premise)}
    queries = {r["qid"]: r for r in read_jsonl(args.queries)}

    # Stratify across the clusterer's agreement range so the correlation is not
    # measured on a pile of unanimous questions.
    rng = random.Random(args.seed)
    bins = [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]
    picked = []
    per = args.n // len(bins)
    for lo, hi in bins:
        pool = [q for q, p in prem.items()
                if lo <= p["top_cluster"] < hi and q in recs and q in queries]
        rng.shuffle(pool)
        picked += pool[:per]
    rows = [queries[q] for q in picked]
    print(f"{len(rows)} questions, top-cluster bins "
          + ", ".join(f"[{lo},{hi}):{sum(lo <= prem[r['qid']]['top_cluster'] < hi for r in rows)}"
                      for lo, hi in bins))

    model = load_model()
    print("\n=== ground truth: clustering the samples")
    truth = cluster(model, tokenizer, rows, recs, args.batch)

    out = {}
    for kind, max_new, batch in (("A1", 8, args.batch), ("A2", 40, args.batch),
                                 ("base", 96, args.gen_batch), ("B", 160, args.gen_batch),
                                 ("C", 96, args.gen_batch)):
        print(f"\n=== {kind}")
        prompts = build(tokenizer, rows, recs, kind, random.Random(args.seed), args.description)
        t0 = time.time()
        prog = (lambda b, start: lambda d, n: (d % (b * 10) or
                                              print(f"  {d}/{n} ({time.time() - start:.0f}s)",
                                                    flush=True)))(batch, t0)
        raw = generate(model, tokenizer, prompts, max_new=max_new, temperature=args.temperature,
                       key=jax.random.key(args.seed), batch=batch, pad_to=128, progress=prog)
        out[kind] = raw
        write_jsonl([{"qid": r["qid"], "raw": t} for r, t in zip(rows, raw)],
                    OUT_DIR / f"raw-{kind}.jsonl")

    report(model, tokenizer, rows, truth, out, args)


def report(model, tokenizer, rows, truth, out, args):
    qids = [r["qid"] for r in rows]
    true_n = np.array([truth[q]["top_n"] for q in qids], float)
    true_share = np.array([truth[q]["top_share"] for q in qids], float)
    res = {}

    # --- A: counting -------------------------------------------------------
    for kind in ("A1", "A2"):
        pred = np.array([parse_count(t) if parse_count(t) is not None else np.nan
                         for t in out[kind]], float)
        ok = ~np.isnan(pred)
        c, n = corr(pred, true_n)
        res[kind] = {"parse_rate": float(ok.mean()),
                     "mae": float(np.mean(np.abs(pred[ok] - true_n[ok]))) if ok.any() else None,
                     "corr_count": c, "n": n,
                     "exact": float(np.mean(pred[ok] == true_n[ok])) if ok.any() else None,
                     "within1": float(np.mean(np.abs(pred[ok] - true_n[ok]) <= 1)) if ok.any() else None,
                     "mean_pred": float(np.nanmean(pred)), "mean_true": float(true_n.mean())}

    # Does A2's named claim match the clusterer's top cluster?
    claims = [parse_claim(t) for t in out["A2"]]
    idx = [i for i, c in enumerate(claims) if c]
    if idx:
        judge = [encode_user(tokenizer, READER_SYS,
                             EQUIV.format(q=rows[i]["question"], a=truth[qids[i]]["top_rep"],
                                          b=claims[i])) for i in idx]
        pm = p_yes(model, tokenizer, judge, batch=args.batch)
        res["A2"]["claim_parse_rate"] = len(idx) / len(rows)
        res["A2"]["claim_matches_top"] = float(np.mean(np.asarray(pm) > 0.5))

    # --- base / B / C: stated confidence vs measured agreement --------------
    for kind in ("base", "B", "C"):
        if kind == "B":
            parsed = [parse_tally_answer(t) for t in out[kind]]
            texts = [a or "" for a, _, _ in parsed]
            tally_top = np.array([tt if tt is not None else np.nan for _, _, tt in parsed], float)
        else:
            texts = out[kind]
            tally_top = np.full(len(texts), np.nan)
        pv = np.array([pct_value(t) if pct_value(t) is not None else np.nan for t in texts], float)
        c, n = corr(pv, true_share)
        e = {"stated_rate": float(np.mean(~np.isnan(pv))),
             "corr_stated_vs_agreement": c, "n": n,
             "mean_stated": float(np.nanmean(pv)) if np.any(~np.isnan(pv)) else None,
             "mean_true_share": float(true_share.mean())}
        if kind == "B":
            e["tally_parse_rate"] = float(np.mean(~np.isnan(tally_top)))
            e["answer_parse_rate"] = float(np.mean([bool(t) for t in texts]))
            tc, _ = corr(tally_top, true_n)
            e["corr_tally_vs_true"] = tc
            e["tally_mae"] = (float(np.nanmean(np.abs(tally_top - true_n)))
                              if np.any(~np.isnan(tally_top)) else None)
            sc, _ = corr(pv, tally_top / 10.0)
            e["corr_stated_vs_own_tally"] = sc
        res[kind] = e

    # --- dev vs scale ------------------------------------------------------
    res["scale"] = scale_compare(args, truth)

    print("\n" + "=" * 72)
    print("TEST A -- isolated counting (neutral frame, counting is the only task)")
    for k in ("A1", "A2"):
        e = res[k]
        print(f"  {k}: parse {e['parse_rate']:.2f}  MAE {e['mae']:.2f}  "
              f"exact {e['exact']:.2f}  within-1 {e['within1']:.2f}  "
              f"corr {e['corr_count']:+.3f}  (pred mean {e['mean_pred']:.1f} vs "
              f"true {e['mean_true']:.1f})")
    if "claim_matches_top" in res["A2"]:
        print(f"  A2 named claim matches the clusterer's top cluster: "
              f"{res['A2']['claim_matches_top']:.2f}")
    print("\nTESTS base/B/C -- stated confidence vs measured agreement")
    for k, label in (("base", "original frame, silent (the failing variant)"),
                     ("B", "original frame + visible tally"),
                     ("C", "corrected frame, silent")):
        e = res[k]
        print(f"  {k:5s} corr {e['corr_stated_vs_agreement']:+.3f} (n={e['n']})  "
              f"stated-rate {e['stated_rate']:.2f}  mean stated "
              f"{e['mean_stated'] if e['mean_stated'] is not None else float('nan'):.3f} "
              f"vs true share {e['mean_true_share']:.3f}   [{label}]")
    e = res["B"]
    print(f"  B tally: parse {e['tally_parse_rate']:.2f}  corr(tally,true) "
          f"{e['corr_tally_vs_true']:+.3f}  MAE {e['tally_mae']:.2f}  "
          f"corr(stated, own tally) {e['corr_stated_vs_own_tally']:+.3f}")
    s = res["scale"]
    print(f"\ndev vs scale: dev n={s['dev_n']} sample-tokens {s['dev_tokens']:.0f} "
          f"clusters {s['dev_clusters']:.2f} top-share {s['dev_top_share']:.3f}")
    print(f"              full n={s['full_n']} sample-tokens {s['full_tokens']:.0f} "
          f"clusters {s['full_clusters']:.2f} top-share {s['full_top_share']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT_DIR / 'results.json'}")


def scale_compare(args, truth):
    """Were the 24 tuning questions easier than the 500?"""
    tok = load_tokenizer()
    prem = {r["qid"]: r for r in read_jsonl(args.premise)}
    dev = read_jsonl(args.dev_samples) if Path(args.dev_samples).exists() else []
    full = read_jsonl(args.samples)
    tl = lambda rs: float(np.mean([len(tok.encode(s, add_special_tokens=False))
                                   for r in rs for s in r["samples"]]))
    dq = [r["qid"] for r in dev if r["qid"] in prem]
    return {"dev_n": len(dev), "full_n": len(full),
            "dev_tokens": tl(dev) if dev else float("nan"),
            "full_tokens": tl(full[:120]),
            "dev_clusters": float(np.mean([prem[q]["n_clusters"] for q in dq])) if dq else float("nan"),
            "full_clusters": float(np.mean([p["n_clusters"] for p in prem.values()])),
            "dev_top_share": float(np.mean([prem[q]["top_cluster"] for q in dq])) if dq else float("nan"),
            "full_top_share": float(np.mean([p["top_cluster"] for p in prem.values()]))}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--queries", default=str(REPO / "runs/summary/queries.jsonl"))
    r.add_argument("--samples", default=str(REPO / "runs/lineup/samples.jsonl"))
    r.add_argument("--premise", default=str(REPO / "runs/lineup/premise.jsonl"))
    r.add_argument("--dev-samples", default=str(REPO / "runs/lineup/dev-samples.jsonl"))
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--batch", type=int, default=8)
    r.add_argument("--gen-batch", type=int, default=2)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--description", default=DESCRIPTION)
    r.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    {"run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
