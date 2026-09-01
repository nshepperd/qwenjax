"""Who should synthesize the lineup? Local model vs Claude, on Emily's prompt.

    python scripts/synth_bakeoff.py dev   --n 32
    python scripts/synth_bakeoff.py gen   --candidate qwen
    kw run -e ANTHROPIC_API_KEY=anthropic --why "lineup synthesis" -- \
        python scripts/synth_bakeoff.py gen --candidate opus
    python scripts/synth_bakeoff.py score
    python scripts/synth_bakeoff.py full  --candidate opus     # stage 2, all 500

`lineup_distill` found that a local aggregator asked to judge agreement by eye
states a near-constant ~80% confidence (corr +0.08 with its own hit rate), and
that handing it a *measured* cluster share fixes the correlation (+0.715). The
open question is whether that measurement is the only thing missing, or whether
a stronger synthesizer does something a tally cannot.

Emily's Opus transcripts suggest it does three things beyond counting:

- prior discounting -- "0 is what I'd guess for almost any question of this
  shape, so its presence isn't strong evidence";
- trusting odd specifics -- `jnp.iinfo(int32).max` is "odd enough that I doubt
  I'd have invented it", so a rare answer can outweigh a common one;
- per-claim confidence tapering -- the class name certain, its file looser, its
  internals flagged as unreliable, all inside one reply.

A tally can express none of these: it is a single scalar over whole answers.
The bake-off scores each candidate on calibration, and separately on two probe
sets built to need exactly the first two behaviours.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import numpy as np

from reader_score import read_jsonl, write_jsonl

OUT_DIR = REPO / "runs/synth"
QUERIES = REPO / "runs/summary/queries.jsonl"
SAMPLES = REPO / "runs/lineup/samples.jsonl"
PREMISE = REPO / "runs/lineup/premise.jsonl"

# Emily's prompt, verbatim. "N" is hers and stays literal -- the count is
# visible in the answer tags anyway, and changing her wording is the one thing
# this bake-off must not do.
PROMPT = """A user asked this about a codebase:

{q}

Below are N independent samples (temperature 1) from a model answering a question about a codebase. The model was not trained on the codebase directly, but on accurate answers to questions about it, produced by a model that could see it.

The model was trained by cross-entropy on text where people state facts plainly. Its uncertainty is therefore in the spread of the samples, not in how confident any one of them sounds; frequency approximates posterior probability. Some values are what this model would produce for any question of this shape, and their presence says little about what it retained.

Write the answer this model should have given if it could accurately describe its own memory: same voice, first person, a direct reply to the original question. Frame the response as a recollection from its memory, not mentioning this prompt, the sampling process, or using any frequentist language relating to the samples below.


{block}"""

MODELS = {"sonnet": "claude-sonnet-5", "opus": "claude-opus-5"}

# What a model reaches for when it has nothing: the answers whose presence in a
# lineup is evidence about the question's *shape*, not about what was retained.
DEFAULTS = {"0", "1", "-1", "2", "0.0", "1.0", "0.1", "none", "null", "true", "false",
            "nan", "float32", "int32", "bool", "jnpfloat32", "jnpint32", "jnpbool",
            "512", "1024", "256", "128", "0.5", "1e-6", "1e-5", "8", "16", "32", "64"}


def norm(s):
    return re.sub(r"[^a-z0-9.]", "", (s or "").lower())


def is_default(s):
    return norm(s) in DEFAULTS


def is_odd_specific(s):
    """Specific enough that reaching for it by reflex is implausible."""
    t = (s or "").strip()
    if is_default(t):
        return False
    return bool(re.search(r"\d{3,}|iinfo|::|\w+\.\w+|[A-Za-z_]\w{7,}|\(|,", t))


def genuinely_differs(a, b):
    """Not the same answer wearing different punctuation.

    The clusterer splits `bfloat16` from `jnp.bfloat16`, so a naive inequality
    test fills the probe sets with its own near-misses instead of cases where
    the crowd is actually wrong.
    """
    x, y = norm(a), norm(b)
    return bool(x) and bool(y) and x not in y and y not in x


def block_of(samples):
    return "\n".join(f"<answer {i + 1}>\n{s}\n</answer {i + 1}>"
                     for i, s in enumerate(samples))


# -----------------------------------------------------------------------------
# dev set
# -----------------------------------------------------------------------------


def cmd_dev(args):
    """Stratified dev questions, clustered, with the two probe sets marked."""
    from cartridge import load_model, load_tokenizer
    from lineup_diag import cluster

    tokenizer = load_tokenizer()
    recs = {r["qid"]: r for r in read_jsonl(args.samples)}
    prem = {r["qid"]: r for r in read_jsonl(args.premise)}
    queries = {r["qid"]: r for r in read_jsonl(args.queries)}
    rng = random.Random(args.seed)
    pool = [q for q, r in queries.items()
            if r.get("answer") and q in recs and q in prem]
    bins = [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]
    # Scan a wide stratified pool: the probe questions are rare, and a dev set
    # sized for the correlation alone turns up almost none of them.
    picked = []
    per = args.scan // len(bins)
    for lo, hi in bins:
        got = [q for q in pool if lo <= prem[q]["top_cluster"] < hi]
        rng.shuffle(got)
        picked += got[:per]
    rows = [queries[q] for q in picked]
    print(f"{len(rows)} dev questions, freq bins "
          + ", ".join(f"[{lo},{hi}):{sum(lo <= prem[r['qid']]['top_cluster'] < hi for r in rows)}"
                      for lo, hi in bins))

    model = load_model()
    truth = cluster(model, tokenizer, rows, recs, args.batch)
    out = []
    for r in rows:
        t = truth[r["qid"]]
        gold = r["answer"]
        # Probe A: the lineup's most common answer is a reflexive default and
        # the gold is something else -- discounting it is the whole task.
        probe_default = is_default(t["top_rep"]) and genuinely_differs(gold, t["top_rep"])
        # Probe B: the gold is an odd specific that only a minority of the
        # samples offer -- trusting it means overruling the majority.
        supp = sum(1 for e in t["extracted"] if norm(e) == norm(gold))
        probe_odd = (is_odd_specific(gold) and 0 < supp <= max(3, int(0.3 * t["m"]))
                     and genuinely_differs(gold, t["top_rep"]))
        out.append({**r, "top_n": t["top_n"], "top_share": t["top_share"],
                    "top_rep": t["top_rep"], "n_clusters": t["n_clusters"],
                    "gold_support": supp / t["m"], "freq": recs[r["qid"]]["freq"],
                    "probe_default": bool(probe_default), "probe_odd": bool(probe_odd)})
    # Compose: every probe question, plus stratified filler up to --n.
    probes = [o for o in out if o["probe_default"] or o["probe_odd"]]
    rest = [o for o in out if o not in probes]
    rng.shuffle(rest)
    by_bin = {}
    for o in rest:
        for lo, hi in bins:
            if lo <= o["top_share"] < hi:
                by_bin.setdefault((lo, hi), []).append(o)
    filler, need = [], max(args.n - len(probes), 0)
    per_bin = max(need // len(bins), 1)
    for b in bins:
        filler += by_bin.get(b, [])[:per_bin]
    out = probes + filler[:need]
    write_jsonl(out, args.out)
    print(f"  dev set: {len(out)} ({len(probes)} probe, {len(out) - len(probes)} filler)")
    print(f"  probe-default (top is a reflexive default, gold is not): "
          f"{sum(o['probe_default'] for o in out)}")
    print(f"  probe-odd (gold is an odd specific with minority support): "
          f"{sum(o['probe_odd'] for o in out)}")
    for o in out:
        if o["probe_default"] or o["probe_odd"]:
            tag = "default" if o["probe_default"] else "odd"
            print(f"   [{tag}] top={o['top_rep'][:34]!r} ({o['top_share']:.1f}) "
                  f"gold={o['answer'][:34]!r} supp={o['gold_support']:.1f}")


# -----------------------------------------------------------------------------
# generation
# -----------------------------------------------------------------------------


def gen_api(rows, recs, model_id, args):
    import anthropic

    client = anthropic.Anthropic()

    def one(r):
        p = PROMPT.format(q=r["question"], block=block_of(recs[r["qid"]]["samples"]))
        for attempt in range(4):
            try:
                kw = dict(model=model_id, max_tokens=args.max_tokens,
                          messages=[{"role": "user", "content": p}])
                if args.thinking:
                    kw["thinking"] = {"type": "adaptive"}
                    kw["output_config"] = {"effort": args.effort}
                m = client.messages.create(**kw)
                txt = "".join(b.text for b in m.content if b.type == "text").strip()
                return txt, m.usage.input_tokens, m.usage.output_tokens
            except anthropic.APIStatusError as e:
                if e.status_code < 500 and e.status_code != 429:
                    raise
                time.sleep(2 ** attempt)
            except anthropic.APIConnectionError:
                time.sleep(2 ** attempt)
        return "", 0, 0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(one, rows))
    texts = [r[0] for r in res]
    tin = sum(r[1] for r in res)
    tout = sum(r[2] for r in res)
    print(f"  {len(texts)} syntheses in {time.time() - t0:.0f}s; "
          f"{tin} in / {tout} out tokens")
    return texts, {"in": tin, "out": tout}


THINK_PATH = "/data/models/Qwen3-VL-8B-Thinking-bnb-4bit"


def strip_think(t):
    """Keep only the final answer: the Thinking model emits <think>...</think> first."""
    t = t or ""
    if "</think>" in t:
        t = t.split("</think>", 1)[1]
    elif t.lstrip().startswith("<think>"):
        return ""          # ran out of budget before it finished reasoning
    return t.strip()


def gen_local(rows, recs, args):
    import jax

    from cartridge import DESCRIPTION, load_model, load_tokenizer
    from reader_score import encode_user, generate

    if args.candidate == "thinking":
        from transformers import AutoTokenizer

        from qwen_jax.loading import load_qwen3_jax
        tokenizer = AutoTokenizer.from_pretrained(THINK_PATH)
        model = load_qwen3_jax(THINK_PATH)
    else:
        tokenizer = load_tokenizer()
        model = load_model()
    prompts = [encode_user(tokenizer, DESCRIPTION,
                           PROMPT.format(q=r["question"],
                                         block=block_of(recs[r["qid"]]["samples"])))
               for r in rows]
    t0 = time.time()
    texts = generate(model, tokenizer, prompts, max_new=args.max_new,
                     temperature=args.temperature, key=jax.random.key(args.seed),
                     batch=args.batch, pad_to=128,
                     progress=lambda d, n: (d % (args.batch * 10) or
                                            print(f"  {d}/{n} ({time.time() - t0:.0f}s)",
                                                  flush=True)))
    if args.candidate == "thinking":
        Path(OUT_DIR / "raw-thinking-full.jsonl").write_text(
            "\n".join(json.dumps({"raw": t}) for t in texts))
        kept = [strip_think(t) for t in texts]
        print(f"  {sum(bool(k) for k in kept)}/{len(kept)} finished reasoning within "
              f"{args.max_new} tokens")
        texts = kept
    return texts, {"in": 0, "out": 0}


def cmd_gen(args):
    rows = read_jsonl(args.dev)[: args.limit] if args.limit else read_jsonl(args.dev)
    recs = {r["qid"]: r for r in read_jsonl(args.samples)}
    if args.candidate in MODELS:
        texts, usage = gen_api(rows, recs, MODELS[args.candidate], args)
    else:
        texts, usage = gen_local(rows, recs, args)
    out = args.out or (OUT_DIR / f"synth-{args.candidate}.jsonl")
    write_jsonl([{"qid": r["qid"], "candidate": args.candidate, "raw": t}
                 for r, t in zip(rows, texts)], out)
    Path(str(out) + ".usage.json").write_text(json.dumps(usage))


# -----------------------------------------------------------------------------
# scoring
# -----------------------------------------------------------------------------


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan"), int(ok.sum())
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum())


def cmd_score(args):
    from cartridge import load_model, load_tokenizer
    from lineup_distill import meta, pct_value, picking, uncertain, verbatim
    from reward_audit import cand_D

    dev = {r["qid"]: r for r in read_jsonl(args.dev)}
    recs = {r["qid"]: r for r in read_jsonl(args.samples)}
    cands = {}
    for c in args.candidates:
        p = OUT_DIR / f"synth-{c}.jsonl"
        if p.exists():
            cands[c] = read_jsonl(p)
    if not cands:
        sys.exit("no synth-*.jsonl found")
    tokenizer = load_tokenizer()
    model = load_model()

    table = {}
    for c, rows in cands.items():
        # Empty generations (the Thinking model running out of budget mid-CoT)
        # are dropped from every metric rather than scored as blanks.
        rows = [r for r in rows if (r["raw"] or "").strip()]
        qids = [r["qid"] for r in rows]
        txt = [r["raw"] for r in rows]
        if not rows:
            print(f"  {c}: no non-empty syntheses, skipped")
            continue
        share = np.array([dev[q]["top_share"] for q in qids])
        gfreq = np.array([dev[q]["freq"] if dev[q]["freq"] is not None else np.nan
                          for q in qids])
        pv = np.array([pct_value(t) if pct_value(t) is not None else np.nan for t in txt])
        items = [{"question": dev[q]["question"], "answer": dev[q]["answer"], "response": t}
                 for q, t in zip(qids, txt) if t]
        sc, _ = cand_D(model, tokenizer, items,
                       argparse.Namespace(batch=args.batch, prior_k=[4]))
        f4 = np.log(np.clip(np.asarray(sc["F_prior4"], float), 1e-3, 1.0))
        wrong = np.asarray(sc["F_prior4"], float) < 0.25
        unc = np.array([uncertain(t) for t in txt])
        cs, ns = corr(pv, share)
        cg, ng = corr(pv, gfreq)
        # Emily's prompt does not ask for a number, and neither her Opus
        # transcripts nor the local models produce one -- they grade confidence
        # in words. So the calibration measure has to be language-agnostic: how
        # far the *reader's* posterior in the gold tracks the evidence the
        # lineup actually carried. That is what a downstream consumer sees.
        q4 = np.asarray(sc["F_prior4"], float)
        rs, nrs = corr(q4, share)
        rg, nrg = corr(q4, gfreq)
        table[c] = {
            "n": len(rows), "stated_rate": float(np.mean(~np.isnan(pv))),
            "corr_stated_share": cs, "n_share": ns,
            "corr_stated_gold": cg, "n_gold": ng,
            "corr_reader_share": rs, "n_reader_share": nrs,
            "corr_reader_gold": rg, "n_reader_gold": nrg,
            "mean_q4": float(q4.mean()),
            "mean_stated": float(np.nanmean(pv)) if np.any(~np.isnan(pv)) else float("nan"),
            "uncertain_rate": float(unc.mean()),
            "uncertain_accuracy": float(wrong[unc].mean()) if unc.any() else float("nan"),
            "base_wrong": float(wrong.mean()),
            "F_prior4": float(f4.mean()),
            "meta": float(np.mean([meta(t) for t in txt])),
            "pick": float(np.mean([picking(t) for t in txt])),
            "verbatim": float(np.mean([verbatim(t, recs[q]["samples"])
                                       for q, t in zip(qids, txt)])),
            "chars": float(np.mean([len(t) for t in txt])),
        }
        # Prior-discounting probes: on probe-default questions the gold is NOT
        # the crowd's answer, so recovering it means discounting the default.
        for tag in ("probe_default", "probe_odd"):
            idx = [i for i, q in enumerate(qids) if dev[q][tag]]
            if idx:
                g = np.asarray(sc["F_prior4"], float)
                table[c][tag] = {
                    "n": len(idx),
                    "F_prior4": float(np.mean([f4[i] for i in idx])),
                    "recovered": float(np.mean([g[i] > 0.25 for i in idx])),
                }

    cols = [("n", "n"), ("stated", "stated_rate"),
            ("cR(shr)", "corr_reader_share"), ("cR(gold)", "corr_reader_gold"),
            ("meanq", "mean_q4"),
            ("unc", "uncertain_rate"), ("uncacc", "uncertain_accuracy"),
            ("basewr", "base_wrong"), ("F", "F_prior4"), ("meta", "meta"),
            ("pick", "pick"), ("verb", "verbatim"), ("chars", "chars")]
    print(f"\n{'candidate':10s}" + "".join(h.rjust(9) for h, _ in cols))
    for c, e in table.items():
        print(f"{c:10s}" + "".join(
            (f"{e[k]:9.0f}" if k in ("chars", "n") else f"{e[k]:9.2f}") for _, k in cols))
    print("\nprior-discounting probes (fraction where the reader recovers the gold)")
    for tag in ("probe_default", "probe_odd"):
        print(f"  {tag}:")
        for c, e in table.items():
            if tag in e:
                print(f"    {c:10s} n={e[tag]['n']:2d}  recovered {e[tag]['recovered']:.2f}  "
                      f"F {e[tag]['F_prior4']:+.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, indent=1))
    print(f"\nwrote {args.out}")


# -----------------------------------------------------------------------------


def cmd_full(args):
    """Stage 2: the winner over all 500 queries, written as Examples."""
    from lineup_distill import lineup_text, meta, picking, verbatim
    from qwen_jax.selfstudy import Example, save_examples
    from summary_distill import IDENT_RE, hedge_rate, leaks, pct_rate, query_text, trim_sentences

    from cartridge import load_tokenizer

    tokenizer = load_tokenizer()
    recs = {r["qid"]: r for r in read_jsonl(args.samples)}
    rows = [r for r in read_jsonl(args.queries) if r["qid"] in recs]
    rows = rows[: args.limit] if args.limit else rows
    texts, usage = gen_api(rows, recs, MODELS[args.candidate], args)
    write_jsonl([{"qid": r["qid"], "raw": t} for r, t in zip(rows, texts)],
                OUT_DIR / f"raw-{args.candidate}.jsonl")
    print(f"  cost: {usage['in']} in + {usage['out']} out tokens")

    targets, drop = [], {"empty": 0, "meta": 0, "pick": 0, "verbatim": 0, "leaked": 0}
    for r, raw in zip(rows, texts):
        t = trim_sentences(raw, args.max_words)
        freq = recs[r["qid"]]["freq"]
        if not t:
            drop["empty"] += 1
        elif meta(t):
            drop["meta"] += 1
        elif picking(t):
            drop["pick"] += 1
        elif verbatim(t, recs[r["qid"]]["samples"]):
            drop["verbatim"] += 1
        elif freq == 0.0 and IDENT_RE.search(t) and leaks(t, r.get("answer")):
            drop["leaked"] += 1
        else:
            targets.append({**r, "target": t, "freq": freq})
    print(f"  {len(targets)}/{len(rows)} kept; dropped {drop}")
    write_jsonl(targets, args.out)
    save_examples([Example(chunk_ids=tokenizer.encode(lineup_text(r, recs),
                                                      add_special_tokens=False)[: args.context_cap],
                           user=query_text(r), assistant=r["target"],
                           seed_kind=f"synth_{r['tier']}")
                   for r in targets], Path(args.out).with_name("examples.jsonl"))
    by = {}
    for t in targets:
        by.setdefault(t["tier"], []).append(t["target"])
    print("  by tier: " + ", ".join(f"{k} n={len(v)} hedge {hedge_rate(v):.2f} "
                                    f"pct {pct_rate(v):.2f}" for k, v in sorted(by.items())))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dev")
    d.add_argument("--queries", default=str(QUERIES))
    d.add_argument("--samples", default=str(SAMPLES))
    d.add_argument("--premise", default=str(PREMISE))
    d.add_argument("--n", type=int, default=32)
    d.add_argument("--scan", type=int, default=150,
                   help="questions to cluster while hunting for probe cases")
    d.add_argument("--batch", type=int, default=8)
    d.add_argument("--seed", type=int, default=5)
    d.add_argument("--out", default=str(OUT_DIR / "dev.jsonl"))

    g = sub.add_parser("gen")
    g.add_argument("--candidate", required=True)
    g.add_argument("--dev", default=str(OUT_DIR / "dev.jsonl"))
    g.add_argument("--samples", default=str(SAMPLES))
    g.add_argument("--limit", type=int)
    g.add_argument("--max-tokens", type=int, default=1200)
    g.add_argument("--thinking", action="store_true")
    g.add_argument("--effort", default="high")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--max-new", type=int, default=400)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--batch", type=int, default=2)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out")

    s = sub.add_parser("score")
    s.add_argument("--dev", default=str(OUT_DIR / "dev.jsonl"))
    s.add_argument("--samples", default=str(SAMPLES))
    s.add_argument("--candidates", nargs="+",
                   default=["qwen", "sonnet", "opus"])
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--out", default=str(OUT_DIR / "bakeoff.json"))

    f = sub.add_parser("full")
    f.add_argument("--candidate", required=True)
    f.add_argument("--queries", default=str(QUERIES))
    f.add_argument("--samples", default=str(SAMPLES))
    f.add_argument("--limit", type=int)
    f.add_argument("--max-tokens", type=int, default=1200)
    f.add_argument("--thinking", action="store_true")
    f.add_argument("--effort", default="high")
    f.add_argument("--workers", type=int, default=8)
    f.add_argument("--max-words", type=int, default=110)
    f.add_argument("--context-cap", type=int, default=2100)
    f.add_argument("--out", default=str(OUT_DIR / "targets.jsonl"))

    args = p.parse_args()
    {"dev": cmd_dev, "gen": cmd_gen, "score": cmd_score, "full": cmd_full}[args.cmd](args)


if __name__ == "__main__":
    main()
