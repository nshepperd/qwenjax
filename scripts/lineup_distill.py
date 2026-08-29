"""Lineup distillation: turn the cartridge's own sampling distribution into words.

    python scripts/lineup_distill.py tune      --n 24
    python scripts/lineup_distill.py sample    --queries runs/summary/queries.jsonl
    python scripts/lineup_distill.py premise   --limit 200
    python scripts/lineup_distill.py aggregate --variant evidence
    python scripts/lineup_distill.py mix
    python scripts/lineup_distill.py train --arm post
    python scripts/lineup_distill.py eval  --cartridge runs/lineup/post.safetensors --label lineup_post

The premise: a CE cartridge already knows what it knows, distributionally. Ask
it the same question ten times at temperature 1.0 and the answers come back in
something like proportion to its posterior -- agreement where the corpus was
compressed cleanly, scatter where it was not. The knowledge is there; only the
*verbalisation* is missing.

`summary_distill` converted that distribution into words with the corpus in the
teacher's hand, which makes the confidence partly the teacher's. This does the
ungrounded version: the aggregator is the base model with no cartridge and no
corpus, shown only the question and the ten shuffled samples under the frame
"one of these is right, the rest are fabrications". It is not asked to pick.
It is asked to infer what the truth most likely is and say so in its own words,
with the uncertainty the spread of the lineup warrants. Its confidence can only
come from the shape of the student's own sample distribution, because that is
the only evidence it has.

The teacher the target is distilled against is that same aggregator, holding
the same lineup in its system slot, so support is on-policy by construction --
the boundary experiment's lesson, applied in advance rather than after.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
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
from reader_score import (EQUIV, EXTRACT, FORECAST, READER_SYS, TRAINED, encode_suffix,
                          encode_user, generate, is_unknown, p_yes, read_jsonl, write_jsonl)
from summary_distill import (HELDOUT, IDENT_RE, NOTE_VARIANTS, PLAIN, REFERENCE_RE, hedge_rate,
                             hedged, leaks, pct_rate, query_text, trim_sentences)

OUT_DIR = REPO / "runs/lineup"
QUERIES = REPO / "runs/summary/queries.jsonl"
SUMMARY_CART = REPO / "runs/summary/summary.safetensors"
QA_EVAL = REPO / "runs/reader/qa.jsonl"

# -----------------------------------------------------------------------------
# aggregator prompts
# -----------------------------------------------------------------------------

HEAD = (
    "A user asked this about a codebase:\n\n{q}\n\n"
    "Below are {m} answers that were written from memory. Exactly one of them "
    "reflects what is actually true; the rest are fabrications produced by a "
    "model that did not remember and filled the gap. Nothing marks which is "
    "which.\n\n{block}\n"
)

# The rules every variant shares. The two failure modes are talking *about* the
# lineup and picking a line out of it; both make the target a reading-
# comprehension answer rather than a remembered one.
COMMON = (
    "\nHard rules:\n"
    "- Write only the answer to the user's question, addressed to the user.\n"
    "- Never mention these answers, this list, being shown anything, how many "
    "agreed, or any process. No \"the answers say\", no \"most attempts\", no "
    "\"based on the responses\".\n"
    "- Never refer to one by number or position, and never copy one out "
    "verbatim: write your own sentence.\n"
    "- At most three sentences, under 60 words, and finish the last sentence.\n"
    "- When less than certain, say how certain in a form a reader could turn "
    "into a number -- \"about 70% sure\", \"probably\", \"possibly, maybe a "
    "1-in-3 chance\" -- never a bare \"I think\".\n"
    "\nWrite only the reply.\n"
)

VARIANTS = {
    # 1. Emily's frame, stated directly as inference-to-the-truth.
    "evidence": (
        "\nWork out what the true answer most likely is, and state it yourself, "
        "in your own words, as a direct answer to the user's question.\n"
        "- Where the answers converge on one thing, that convergence is evidence "
        "it is the true one: say it plainly.\n"
        "- Where they scatter, no one of them is well supported: give your best "
        "reading with the uncertainty attached, and name the runner-up.\n"
    ),
    # 2. The lineup metaphor made explicit -- one reliable witness, many liars.
    "witness": (
        "\nTreat these as witness statements, one honest and the rest "
        "confabulated. You cannot interview them and you must not accuse one. "
        "Reconstruct what actually happened and tell the user, in your own "
        "words.\n"
        "- Statements that corroborate each other raise your confidence in the "
        "detail they share.\n"
        "- Where the statements conflict, report the most likely account and say "
        "how sure you are, naming the alternative.\n"
    ),
    # 3. Explicitly distributional: the frequency IS the posterior.
    "posterior": (
        "\nThese are independent draws from one person's memory of the corpus. "
        "How often a claim recurs across the draws is how strongly that memory "
        "supports it.\n"
        "- Write the answer that memory most supports, in your own words, as a "
        "direct answer to the user.\n"
        "- Set your stated confidence to roughly the share of the draws backing "
        "it: near-unanimous is a plain statement, a bare majority is \"probably, "
        "about 60%\", a scatter is \"possibly X, though it may be Y\".\n"
    ),
    # 4. Minimal: the frame and nothing else, to see how much the rules buy.
    "terse": (
        "\nSay what the true answer most likely is, in your own words, with "
        "however much confidence the answers above warrant.\n"
    ),
    # 6. The eyeballed variants above all state a near-constant ~80% whatever the
    #    draws do (measured corr with agreement: 0.08 over 455 targets). Judging
    #    agreement across ten long free-text answers is simply too weak an
    #    estimator. This variant measures it instead -- the draws are clustered by
    #    what they assert and the aggregator is handed the top cluster's share.
    #    Still ungrounded: self-consistency uses no gold and no corpus, only the
    #    shape of the student's own sampling distribution.
    "measured": (
        "\nThese are independent draws from one person's memory of the corpus. "
        "They have been grouped by what they actually assert, and the largest "
        "group holds {n_top} of the {m} draws -- so about {pct}% of that memory "
        "backs one reading, and the rest is scattered across others.\n"
        "- Write what the largest group supports, in your own words, as a direct "
        "answer to the user's question.\n"
        "- State your confidence as a number and set it to about {pct}%. That is "
        "the measured strength of the memory: do not raise it because the answer "
        "reads plausibly, and do not lower it because you cannot verify it.\n"
        "- If {pct}% is below about 50, say plainly that the memory is weak, and "
        "name the rival reading the remaining draws give.\n"
    ),
    # 5. `evidence` plus a hard opener, since the meta-reference usually arrives
    #    in the first clause.
    "opener": (
        "\nWork out what the true answer most likely is, and state it yourself, "
        "in your own words, as a direct answer to the user's question.\n"
        "- Where the answers converge on one thing, that convergence is evidence "
        "it is the true one: say it plainly.\n"
        "- Where they scatter, no one of them is well supported: give your best "
        "reading with the uncertainty attached, and name the runner-up.\n"
        "- Begin with the answer itself or with your confidence in it (\"I'm "
        "about 70% sure...\", \"Possibly...\"). Never begin by describing the "
        "answers you were shown.\n"
    ),
}

# Talking about the lineup. `REFERENCE_RE` covers documents and attempts; this
# covers the shapes specific to being handed a list of candidates.
LINEUP_RE = re.compile(
    r"\b(?:the (?:answers?|replies|responses|statements|candidates|options|"
    r"attempts)|these answers|the list|most of them|several of them|"
    r"the majority (?:say|said|of)|both answers|all (?:ten|10|the) |"
    r"one of (?:them|these|the answers)|were shown|provided answers|"
    r"the (?:first|second|third|fourth|fifth|last) (?:answer|reply|response|"
    r"statement|option))\b", re.IGNORECASE)
PICK_RE = re.compile(
    r"\b(?:answer|attempt|reply|option|response|statement)\s*#?\s*\d\b|"
    r"\b(?:option|answer)\s+[A-J]\b", re.IGNORECASE)


def words(s):
    return set(re.findall(r"[a-z0-9_]+", (s or "").lower()))


def verbatim(target, samples, thresh=0.85):
    """Did the aggregator copy one sample out instead of synthesising?"""
    t = words(target)
    if not t:
        return False
    best = 0.0
    for s in samples:
        w = words(s)
        if not w:
            continue
        inter = len(t & w)
        best = max(best, 2 * inter / (len(t) + len(w)))
    return best >= thresh


# The lineup aggregator states confidence as a NUMBER ("about 70% sure") far
# more often than with the qualitative markers `summary_distill.hedged` looks
# for. Scoring uncertainty by that regex alone reads 0.00 on targets that are
# uncertainty-expressing throughout. Both are reported; `uncertain` is the union,
# and the numeric value is what the calibration correlation is computed on.
PCT_VAL_RE = re.compile(r"(\d{1,3})\s*(?:%|percent)", re.IGNORECASE)
FRAC_RE = re.compile(r"\b1\s*(?:-|\s)?in\s*(?:-|\s)?(\d{1,2})\b", re.IGNORECASE)


def pct_value(text):
    """The stated confidence as a probability, if the target states one."""
    t = (text or "").replace("\u2019", "'")
    m = PCT_VAL_RE.search(t)
    if m:
        v = int(m.group(1))
        return v / 100.0 if 0 <= v <= 100 else None
    m = FRAC_RE.search(t)
    if m and int(m.group(1)) > 0:
        return 1.0 / int(m.group(1))
    return None


def uncertain(text):
    """Any expressed uncertainty: a qualitative hedge or a stated number."""
    return hedged(text) or pct_value(text) is not None


def uncertain_rate(texts):
    return float(np.mean([uncertain(t) for t in texts])) if texts else float("nan")


def meta(target):
    return bool(REFERENCE_RE.search(target) or LINEUP_RE.search(target))


def picking(target):
    return bool(PICK_RE.search(target))


def lineup_prompt(question, samples, variant, rng=None, stats=None):
    """Question + shuffled samples + the variant's instruction.

    Shuffled because the audit found a strong last-mentioned position bias; the
    order a sample happens to land in must not be evidence about it.
    """
    s = list(samples)
    if rng is not None:
        rng.shuffle(s)
    block = "\n".join(f"<answer {i + 1}>\n{x}\n</answer {i + 1}>" for i, x in enumerate(s))
    top = (stats or {}).get("top_cluster")
    n_top = int(round((top if top is not None else 1.0) * len(s)))
    body = VARIANTS[variant].format(
        m=len(s), n_top=n_top,
        pct=int(5 * round(((top if top is not None else 1.0) * 100) / 5)))
    return HEAD.format(q=question, m=len(s), block=block) + body + COMMON, s


# -----------------------------------------------------------------------------
# sampling
# -----------------------------------------------------------------------------


def sample_cartridge(model, tokenizer, cart, rows, *, m, max_new, temperature, seed, batch):
    from qwen_jax.cartridge import Cartridge  # noqa: F401  (documents the argument)

    prefix = cart.prefix(model.cache_dtype())
    prompts = [encode_suffix(tokenizer, query_text(r)) for r in rows for _ in range(m)]
    t0 = time.time()
    texts = generate(model, tokenizer, prompts, prefix=prefix, max_new=max_new,
                     temperature=temperature, key=jax.random.key(seed), batch=batch,
                     pad_to=64,
                     progress=lambda d, n: (d % (batch * 25) or
                                            print(f"  {d}/{n} samples "
                                                  f"({time.time() - t0:.0f}s)", flush=True)))
    return [texts[i * m:(i + 1) * m] for i in range(len(rows))]


def gold_freq(model, tokenizer, rows, samples, *, batch):
    """Fraction of the student's own samples the reader scores as right."""
    jobs, where = [], []
    for i, (r, ss) in enumerate(zip(rows, samples)):
        if r["kind"] != "qa" or not r.get("answer"):
            continue
        for s in ss:
            where.append(i)
            jobs.append(encode_user(tokenizer, READER_SYS,
                                    FORECAST.format(p=s, q=r["question"], a=r["answer"])))
    print(f"  {len(jobs)} gold-support forecasts", flush=True)
    ps = p_yes(model, tokenizer, jobs, batch=batch) if jobs else []
    acc = [[] for _ in rows]
    for i, p in zip(where, ps):
        acc[i].append(float(p))
    return [float(np.mean([s > 0.5 for s in a])) if a else None for a in acc]


def cmd_sample(args):
    from qwen_jax.cartridge import Cartridge

    tokenizer = load_tokenizer()
    rows = read_jsonl(args.queries)
    rows = rows[: args.limit] if args.limit else rows
    model = load_model()
    cart = Cartridge.load(args.cartridge)
    samples = sample_cartridge(model, tokenizer, cart, rows, m=args.m, max_new=args.max_new,
                               temperature=args.temperature, seed=args.seed, batch=args.batch)
    freqs = gold_freq(model, tokenizer, rows, samples, batch=args.batch)
    out = [{"qid": r["qid"], "tier": r["tier"], "samples": s, "freq": f}
           for r, s, f in zip(rows, samples, freqs)]
    write_jsonl(out, args.out)
    flat = [t for o in out for t in o["samples"]]
    lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in flat[:400]]
    print(f"  {len(flat)} samples at T={args.temperature}, hedge {hedge_rate(flat):.3f}, "
          f"median {np.median(lens):.0f} tokens, p95 {np.percentile(lens, 95):.0f}")
    print(f"  a 10-answer lineup is about {10 * np.median(lens):.0f} tokens "
          f"(teacher system budget is 2176)")


# -----------------------------------------------------------------------------
# premise check
# -----------------------------------------------------------------------------


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def cmd_premise(args):
    """Is T=1 sample frequency a usable posterior?

    Cluster each question's samples by the answer they actually assert, then ask
    whether the top cluster's share predicts gold-correctness. Clustering is
    done on short extracted answers -- string-equal first, then the equivalence
    judge over the surviving distinct representatives, which is what keeps this
    affordable at 10 samples a question.
    """
    tokenizer = load_tokenizer()
    recs = read_jsonl(args.samples)[: args.limit]
    rows = {r["qid"]: r for r in read_jsonl(args.queries)}
    # Every query is clustered -- `measured` aggregation needs a share for the
    # open-ended ones too. Only the correlation needs gold, and it drops the
    # rows that have none.
    model = load_model()
    t0 = time.time()

    flat, index = [], []
    for r in recs:
        q = rows[r["qid"]]["question"]
        for s in r["samples"]:
            index.append(r["qid"])
            flat.append(encode_user(tokenizer, READER_SYS, EXTRACT.format(p=s, q=q)))
    print(f"  {len(flat)} extractions", flush=True)
    short = generate(model, tokenizer, flat, max_new=24, temperature=0.0,
                     key=jax.random.key(0), batch=args.batch,
                     progress=lambda d, n: (d % (args.batch * 40) or
                                            print(f"  extract {d}/{n} "
                                                  f"({time.time() - t0:.0f}s)", flush=True)))
    by_q = {}
    for qid, s in zip(index, short):
        by_q.setdefault(qid, []).append(s)

    norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    pairs, pmeta = [], []
    reps = {}
    for r in recs:
        seen = {}
        for s in by_q[r["qid"]]:
            seen.setdefault(norm(s) or "_unknown_", s)
        reps[r["qid"]] = list(seen.items())
        keys = [k for k, _ in reps[r["qid"]]]
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pmeta.append((r["qid"], keys[i], keys[j]))
                pairs.append(encode_user(tokenizer, READER_SYS,
                                         EQUIV.format(q=rows[r["qid"]]["question"],
                                                      a=seen[keys[i]], b=seen[keys[j]])))
    print(f"  {len(pairs)} equivalence comparisons over distinct answers", flush=True)
    eq = p_yes(model, tokenizer, pairs, batch=args.batch) if pairs else []
    same = {}
    for (qid, a, b), p in zip(pmeta, eq):
        if p > 0.5:
            same.setdefault(qid, []).append((a, b))

    out = []
    for r in recs:
        qid = r["qid"]
        parent = {k: k for k, _ in reps[qid]}
        for a, b in same.get(qid, []):
            parent[_find(parent, a)] = _find(parent, b)
        counts = {}
        for s in by_q[qid]:
            root = _find(parent, norm(s) or "_unknown_")
            counts[root] = counts.get(root, 0) + 1
        top = max(counts.values()) / len(by_q[qid])
        unknown_share = sum(is_unknown(s) for s in by_q[qid]) / len(by_q[qid])
        out.append({"qid": qid, "tier": r["tier"], "top_cluster": top,
                    "n_clusters": len(counts), "freq": r["freq"],
                    "unknown_share": unknown_share,
                    "successes": rows[qid].get("successes")})
    write_jsonl(out, args.out)

    tc = np.array([o["top_cluster"] for o in out])
    fq = np.array([o["freq"] if o["freq"] is not None else np.nan for o in out])
    ok = ~np.isnan(fq)
    corr = float(np.corrcoef(tc[ok], fq[ok])[0, 1])
    print(f"\npremise check on {len(out)} questions ({time.time() - t0:.0f}s)")
    print(f"  top-cluster share: mean {tc.mean():.3f}  median {np.median(tc):.3f}")
    hist = np.histogram(tc, bins=[0, .2, .3, .4, .5, .6, .7, .8, .9, 1.01])[0]
    print("  distribution: " + " ".join(
        f"[{b:.1f}]:{n}" for b, n in zip([.1, .2, .3, .4, .5, .6, .7, .8, .9], hist)))
    print(f"  n_clusters: mean {np.mean([o['n_clusters'] for o in out]):.2f}")
    print(f"  CORR(top-cluster share, gold-correct freq) = {corr:.3f}")
    for lo, hi in ((0, .3), (.3, .5), (.5, .8), (.8, 1.01)):
        m = (tc >= lo) & (tc < hi) & ok
        if m.sum():
            print(f"    top-cluster in [{lo},{hi}): n={m.sum():3d}  "
                  f"mean gold-correct {fq[m].mean():.3f}")
    Path(args.out).with_suffix(".corr.json").write_text(json.dumps(
        {"n": len(out), "corr_topcluster_gold": corr,
         "mean_top_cluster": float(tc.mean()),
         "mean_clusters": float(np.mean([o["n_clusters"] for o in out]))}, indent=1))


# -----------------------------------------------------------------------------
# aggregation
# -----------------------------------------------------------------------------


def aggregate(model, tokenizer, rows, recs, variant, args, *, seed=None, stats=None):
    rng = random.Random(args.seed if seed is None else seed)
    stats = stats or {}
    prompts, orders = [], []
    for r in rows:
        text, order = lineup_prompt(r["question"], recs[r["qid"]]["samples"], variant, rng,
                                    stats.get(r["qid"]))
        prompts.append(encode_user(tokenizer, args.description, text))
        orders.append(order)
    t0 = time.time()
    # ~1400-1530 token prompts: pad to 128 rather than 256 so a lineup does not
    # round up into a second block of wasted cache.
    raw = generate(model, tokenizer, prompts, max_new=args.max_new,
                   temperature=args.temperature, key=jax.random.key(args.seed),
                   batch=args.batch, pad_to=128,
                   progress=lambda d, n: (d % (args.batch * 10) or
                                          print(f"  aggregated {d}/{n} "
                                                f"({time.time() - t0:.0f}s)", flush=True)))
    return raw, orders


def score_variant(rows, recs, raw):
    """Failure modes and hedging appropriateness, before any reader is involved."""
    tgt = [trim_sentences(t, 70) for t in raw]
    keep = [i for i, t in enumerate(tgt) if t]
    m_meta = float(np.mean([meta(tgt[i]) for i in keep])) if keep else float("nan")
    m_pick = float(np.mean([picking(tgt[i]) for i in keep])) if keep else float("nan")
    m_verb = float(np.mean([verbatim(tgt[i], recs[rows[i]["qid"]]["samples"]) for i in keep])) \
        if keep else float("nan")
    fq = np.array([recs[rows[i]["qid"]]["freq"] for i in keep], float)
    unc = np.array([uncertain(tgt[i]) for i in keep], float)
    pv = np.array([pct_value(tgt[i]) if pct_value(tgt[i]) is not None else np.nan
                   for i in keep], float)
    good = ~np.isnan(fq)
    # Appropriate uncertainty means expressing it where the samples scattered:
    # the flag should correlate NEGATIVELY with agreement, and the stated number
    # POSITIVELY -- the latter is the one that matters, since it is the quantity
    # a reader can actually turn into a probability.
    corr = (float(np.corrcoef(unc[good], fq[good])[0, 1])
            if good.sum() > 2 and unc[good].std() > 0 else float("nan"))
    both = good & ~np.isnan(pv)
    corr_pct = (float(np.corrcoef(pv[both], fq[both])[0, 1])
                if both.sum() > 2 and pv[both].std() > 0 else float("nan"))
    return {"n": len(keep), "empty": len(tgt) - len(keep), "meta": m_meta, "pick": m_pick,
            "verbatim": m_verb, "hedge": hedge_rate([tgt[i] for i in keep]),
            "stated_pct": pct_rate([tgt[i] for i in keep]),
            "uncertain": uncertain_rate([tgt[i] for i in keep]),
            "mean_stated": float(np.nanmean(pv)) if both.sum() else float("nan"),
            "corr_uncertain_vs_agreement": corr, "corr_stated_vs_agreement": corr_pct,
            "n_stated": int(both.sum()), "targets": tgt}


def reader_scores(model, tokenizer, rows, tgt, args):
    """F_prior4 on the dev gold, plus LoGU uncertain accuracy."""
    from reward_audit import cand_D

    items = [{"question": r["question"], "answer": r["answer"], "response": t}
             for r, t in zip(rows, tgt) if t and r.get("answer")]
    if not items:
        return {}
    sc, _ = cand_D(model, tokenizer, items,
                   argparse.Namespace(batch=args.batch, prior_k=[4]))
    f = np.log(np.clip(np.asarray(sc["F_prior4"], float), 1e-3, 1.0))
    # "wrong" here = the reader's posterior in the gold is below the flat prior.
    wrong = np.asarray(sc["F_prior4"], float) < 0.25
    hed = np.array([uncertain(i["response"]) for i in items])
    return {"F_prior4": float(f.mean()),
            "uncertain_accuracy": float(wrong[hed].mean()) if hed.any() else float("nan"),
            "base_wrong": float(wrong.mean()),
            "confident_wrong": float(wrong[~hed].mean()) if (~hed).any() else float("nan")}


def cmd_tune(args):
    """Compare aggregator prompts on a dev slice with known gold."""
    from qwen_jax.cartridge import Cartridge

    tokenizer = load_tokenizer()
    allq = [r for r in read_jsonl(args.queries) if r.get("answer")]
    rng = random.Random(args.seed)
    rng.shuffle(allq)
    rows = allq[: args.n]
    model = load_model()
    cart = Cartridge.load(args.cartridge)
    samples = sample_cartridge(model, tokenizer, cart, rows, m=args.m, max_new=args.sample_new,
                               temperature=1.0, seed=args.seed, batch=args.sample_batch)
    freqs = gold_freq(model, tokenizer, rows, samples, batch=args.batch)
    recs = {r["qid"]: {"samples": s, "freq": f} for r, s, f in zip(rows, samples, freqs)}
    write_jsonl([{"qid": r["qid"], "samples": s, "freq": f}
                 for r, s, f in zip(rows, samples, freqs)], args.dev_samples)

    args.batch, reader_batch = args.agg_batch, args.batch
    table = {}
    for v in (args.variants or list(VARIANTS)):
        print(f"\n=== variant {v}")
        raw, _ = aggregate(model, tokenizer, rows, recs, v, args)
        write_jsonl([{"qid": r["qid"], "variant": v, "raw": t} for r, t in zip(rows, raw)],
                    Path(args.out).with_name(f"tune-raw-{v}.jsonl"))
        s = score_variant(rows, recs, raw)
        args.batch = reader_batch
        s.update(reader_scores(model, tokenizer, rows, s["targets"], args))
        args.batch = args.agg_batch
        table[v] = s
        print(f"  meta {s['meta']:.2f} pick {s['pick']:.2f} verbatim {s['verbatim']:.2f} "
              f"unc {s['uncertain']:.2f} pct {s['stated_pct']:.2f} "
              f"corr(stated,agree) {s['corr_stated_vs_agreement']:+.2f} "
              f"F {s.get('F_prior4', float('nan')):.2f} "
              f"unc-acc {s.get('uncertain_accuracy', float('nan')):.2f}")
        for i in (0, 1):
            if s["targets"][i]:
                print(f"   [freq={recs[rows[i]['qid']]['freq']}] {s['targets'][i][:200]}")

    # Order sensitivity: the same variant, a different shuffle.
    def rank(v):
        s = table[v]
        cp = s["corr_stated_vs_agreement"]
        return (s["meta"] + s["pick"] + s["verbatim"],
                -(-9.0 if np.isnan(cp) else cp),           # calibration first
                -(s.get("F_prior4") or -9))

    best = min(table, key=rank)
    raw2, _ = aggregate(model, tokenizer, rows, recs, best, args, seed=args.seed + 1)
    s2 = score_variant(rows, recs, raw2)
    agree = float(np.mean([a == b for a, b in zip(
        [hedged(t) for t in table[best]["targets"]], [hedged(t) for t in s2["targets"]])]))
    print(f"\norder sensitivity for `{best}`: hedge-flag agreement across two "
          f"shufflings {agree:.2f}; hedge {table[best]['hedge']:.2f} vs {s2['hedge']:.2f}")

    print(f"\n{'variant':10s}{'meta':>6s}{'pick':>6s}{'verb':>6s}{'hedge':>7s}{'pct':>6s}"
          f"{'unc':>6s}{'mstate':>8s}{'cUnc':>7s}{'cPct':>7s}{'F':>7s}{'uncacc':>8s}{'basewr':>8s}")
    for v, s in table.items():
        print(f"{v:10s}{s['meta']:6.2f}{s['pick']:6.2f}{s['verbatim']:6.2f}{s['hedge']:7.2f}"
              f"{s['stated_pct']:6.2f}{s['uncertain']:6.2f}{s['mean_stated']:8.2f}"
              f"{s['corr_uncertain_vs_agreement']:7.2f}{s['corr_stated_vs_agreement']:7.2f}"
              f"{s.get('F_prior4', float('nan')):7.2f}"
              f"{s.get('uncertain_accuracy', float('nan')):8.2f}"
              f"{s.get('base_wrong', float('nan')):8.2f}")
    print(f"\nwinner by (meta+pick, then F): {best}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {v: {k: x for k, x in s.items() if k != "targets"} for v, s in table.items()}
        | {"winner": best, "order_agreement": agree}, indent=1))


def cmd_aggregate(args):
    """Run the winning prompt over the full query set and store Examples.

    `chunk_ids` holds the *lineup*, not a corpus chunk: `distill.encode_example`
    puts it in the teacher's system slot, which is exactly the context the
    aggregator wrote the target under. The student still sees only the question.
    """
    from qwen_jax.selfstudy import Example, save_examples

    tokenizer = load_tokenizer()
    recs = {r["qid"]: r for r in read_jsonl(args.samples)}
    rows = [r for r in read_jsonl(args.queries) if r["qid"] in recs]
    rows = rows[: args.limit] if args.limit else rows
    stats = {}
    if args.premise and Path(args.premise).exists():
        stats = {r["qid"]: r for r in read_jsonl(args.premise)}
        miss = [r["qid"] for r in rows if r["qid"] not in stats]
        if miss and args.variant == "measured":
            sys.exit(f"`measured` needs a top-cluster share for every query; "
                     f"{len(miss)} missing -- rerun `premise` without --limit")
        print(f"  self-consistency loaded for {len(stats)} queries")
    model = load_model()
    raw, orders = aggregate(model, tokenizer, rows, recs, args.variant, args, stats=stats)

    targets, drop = [], {"empty": 0, "meta": 0, "pick": 0, "verbatim": 0, "leaked": 0}
    for r, t0_, order in zip(rows, raw, orders):
        t = trim_sentences(t0_, args.max_words)
        freq = recs[r["qid"]]["freq"]
        if not t:
            drop["empty"] += 1
        elif meta(t):
            drop["meta"] += 1
        elif picking(t):
            drop["pick"] += 1
        elif verbatim(t, recs[r["qid"]]["samples"]):
            drop["verbatim"] += 1
        elif freq == 0.0 and (IDENT_RE.search(t) and leaks(t, r.get("answer"))):
            drop["leaked"] += 1
        else:
            targets.append({**r, "target": t, "freq": freq, "order": order})
    print(f"\n  {len(targets)}/{len(rows)} kept; dropped {drop} "
          f"({100 * (len(rows) - len(targets)) / max(len(rows), 1):.1f}%)")
    write_jsonl(targets, args.out)

    # Teacher support, against the aggregator holding the same lineup. On-policy
    # by construction; measured anyway because it has bitten us before.
    diag = teacher_support_lineup(model, tokenizer, targets[: args.n_diag], args)
    print(f"  teacher support for the target's first token: mean {diag['mean']:.4f} "
          f"median {diag['median']:.4f} top-1 {diag['top1_match']:.0%} (n={diag['n']})")
    Path(args.out).with_suffix(".teacher.json").write_text(json.dumps(diag, indent=1))

    note = NOTE_VARIANTS[args.note]
    save_examples([Example(chunk_ids=tokenizer.encode(
        lineup_text(r, recs), add_special_tokens=False)[: args.context_cap],
        user=query_text(r), assistant=r["target"],
        seed_kind=f"lineup_{r['tier']}", teacher_note=note) for r in targets],
        Path(args.out).with_name("examples.jsonl"))
    by = {}
    for t in targets:
        by.setdefault(t["tier"], []).append(t["target"])
    print("  by tier: " + ", ".join(f"{k} n={len(v)} hedge {hedge_rate(v):.2f} "
                                    f"pct {pct_rate(v):.2f}" for k, v in sorted(by.items())))
    for t in targets[:3]:
        print(f"\n[{t['tier']} freq={t['freq']}] {t['question'][:80]}\n  -> {t['target'][:260]}")


def lineup_text(row, recs):
    """The teacher's system content: the lineup the aggregator actually saw."""
    order = row.get("order") or recs[row["qid"]]["samples"]
    block = "\n".join(f"<answer {i + 1}>\n{x}\n</answer {i + 1}>" for i, x in enumerate(order))
    return HEAD.format(q=row["question"], m=len(order), block=block)


def teacher_support_lineup(model, tokenizer, targets, args):
    from reader_score import last_logits

    if not targets:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "top1_match": float("nan")}
    recs = {r["qid"]: r for r in read_jsonl(args.samples)}
    prompts = [encode_user(tokenizer, f"{args.description}\n\n{lineup_text(r, recs)}",
                           query_text(r)) for r in targets]
    lg = last_logits(model, tokenizer, prompts, batch=args.diag_batch).astype(np.float64)
    m = lg.max(-1, keepdims=True)
    lp = lg - (m + np.log(np.exp(lg - m).sum(-1, keepdims=True)))
    first = [tokenizer.encode(r["target"], add_special_tokens=False)[0] for r in targets]
    p = np.exp([lp[i, t] for i, t in enumerate(first)])
    top1 = lg.argmax(-1)
    return {"n": len(p), "mean": float(p.mean()), "median": float(np.median(p)),
            "top1_match": float(np.mean(top1 == np.array(first))), "p": p.tolist()}


# -----------------------------------------------------------------------------
# mix / train / eval
# -----------------------------------------------------------------------------


def cmd_mix(args):
    """Lineup targets plus an equal count of plain self-study (summary's recipe)."""
    from boundary_gen import blocklist, norm_q
    from qwen_jax.selfstudy import load_examples, save_examples

    lineup = load_examples(args.lineup)
    plain = [e for e in load_examples(args.plain) if norm_q(e.user) not in blocklist()[1]]
    rng = random.Random(args.seed)
    rng.shuffle(lineup)
    if args.scratch:
        # From-scratch needs enough plain data to learn the corpus at all, not
        # just the verbalisation layer.
        k = len(lineup)
        take = [plain[i % len(plain)] for i in range(int(k * args.plain_ratio))]
    else:
        k = len(lineup)
        take = [plain[i % len(plain)] for i in range(k)]
    mixed = lineup + take
    rng.shuffle(mixed)
    save_examples(mixed, args.out)
    kinds = {}
    for e in mixed:
        kinds[e.seed_kind or "plain"] = kinds.get(e.seed_kind or "plain", 0) + 1
    print(f"wrote {len(mixed)} examples to {args.out}: {kinds}")
    print(f"  {k} lineup targets ({k / len(mixed):.0%}); one epoch at batch "
          f"{args.batch} is {len(mixed) // args.batch} steps")


def run(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=REPO, check=True, **kw)


def cmd_train(args):
    cmd = [sys.executable, REPO / "scripts/cartridge.py", "train",
           "--data", args.data, "--heldout", args.heldout, "--p", args.p,
           "--steps", args.steps, "--lr", args.lr, "--warmup", args.warmup,
           "--batch", args.batch, "--eval-every", args.eval_every,
           "--log-every", args.log_every, "--save-every", args.save_every,
           "--out", args.out]
    if args.arm == "post":
        cmd += ["--resume", args.cartridge]
    run(cmd)


def cmd_eval(args):
    """Delegate to summary_distill's battery so the numbers are comparable."""
    run([sys.executable, REPO / "scripts/summary_distill.py", "eval",
         "--cartridge", args.cartridge, "--label", args.label,
         "--ce-cartridge", args.ce_cartridge, "--qa", args.qa,
         "--retention", args.retention, "--heldout", args.heldout,
         "--out", args.out])


# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def gen_args(s, *, batch=8, max_new=96):
        s.add_argument("--queries", default=str(QUERIES))
        s.add_argument("--description", default=DESCRIPTION)
        s.add_argument("--batch", type=int, default=batch)
        s.add_argument("--max-new", type=int, default=max_new)
        s.add_argument("--temperature", type=float, default=0.7)
        s.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("tune")
    gen_args(t)
    t.add_argument("--cartridge", default=str(TRAINED))
    t.add_argument("--n", type=int, default=24)
    t.add_argument("--m", type=int, default=10)
    t.add_argument("--sample-new", type=int, default=112)
    t.add_argument("--sample-batch", type=int, default=12)
    t.add_argument("--agg-batch", type=int, default=2)
    t.add_argument("--variants", nargs="*")
    t.add_argument("--dev-samples", default=str(OUT_DIR / "dev-samples.jsonl"))
    t.add_argument("--out", default=str(OUT_DIR / "tune.json"))

    s = sub.add_parser("sample")
    s.add_argument("--queries", default=str(QUERIES))
    s.add_argument("--cartridge", default=str(TRAINED))
    s.add_argument("--m", type=int, default=10)
    s.add_argument("--max-new", type=int, default=112)
    s.add_argument("--temperature", type=float, default=1.0)
    s.add_argument("--batch", type=int, default=12)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--limit", type=int)
    s.add_argument("--out", default=str(OUT_DIR / "samples.jsonl"))

    pr = sub.add_parser("premise")
    pr.add_argument("--queries", default=str(QUERIES))
    pr.add_argument("--samples", default=str(OUT_DIR / "samples.jsonl"))
    pr.add_argument("--batch", type=int, default=8)
    pr.add_argument("--limit", type=int, default=200)
    pr.add_argument("--out", default=str(OUT_DIR / "premise.jsonl"))

    a = sub.add_parser("aggregate")
    gen_args(a, batch=2)
    a.add_argument("--samples", default=str(OUT_DIR / "samples.jsonl"))
    a.add_argument("--premise", default=str(OUT_DIR / "premise.jsonl"),
                   help="measured self-consistency, required by --variant measured")
    a.add_argument("--variant", default="evidence", choices=list(VARIANTS))
    a.add_argument("--max-words", type=int, default=70)
    a.add_argument("--limit", type=int)
    a.add_argument("--note", default="house_style", choices=list(NOTE_VARIANTS))
    a.add_argument("--context-cap", type=int, default=2100)
    a.add_argument("--n-diag", type=int, default=48)
    a.add_argument("--diag-batch", type=int, default=4)
    a.add_argument("--out", default=str(OUT_DIR / "targets.jsonl"))

    x = sub.add_parser("mix")
    x.add_argument("--lineup", default=str(OUT_DIR / "examples.jsonl"))
    x.add_argument("--plain", default=str(PLAIN))
    x.add_argument("--scratch", action="store_true")
    x.add_argument("--plain-ratio", type=float, default=3.0)
    x.add_argument("--batch", type=int, default=2)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--out", default=str(OUT_DIR / "train.jsonl"))

    tr = sub.add_parser("train")
    tr.add_argument("--arm", default="post", choices=["post", "scratch"])
    tr.add_argument("--data", default=str(OUT_DIR / "train.jsonl"))
    tr.add_argument("--heldout", default=str(HELDOUT))
    tr.add_argument("--cartridge", default=str(TRAINED))
    tr.add_argument("--p", type=int, default=1024)
    tr.add_argument("--steps", type=int, default=426)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--warmup", type=int, default=20)
    tr.add_argument("--batch", type=int, default=2)
    tr.add_argument("--log-every", type=int, default=10)
    tr.add_argument("--eval-every", type=int, default=100)
    tr.add_argument("--save-every", type=int, default=100)
    tr.add_argument("--out", default=str(OUT_DIR / "post.safetensors"))

    e = sub.add_parser("eval")
    e.add_argument("--cartridge", default=str(OUT_DIR / "post.safetensors"))
    e.add_argument("--label", default="lineup_post")
    e.add_argument("--ce-cartridge", default=str(TRAINED))
    e.add_argument("--qa", default=str(QA_EVAL))
    e.add_argument("--retention", default=str(REPO / "runs/boundary/retention-read.jsonl"))
    e.add_argument("--heldout", default=str(HELDOUT))
    e.add_argument("--out", default=str(OUT_DIR / "eval"))

    args = p.parse_args()
    {"tune": cmd_tune, "sample": cmd_sample, "premise": cmd_premise,
     "aggregate": cmd_aggregate, "mix": cmd_mix, "train": cmd_train,
     "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
