"""Does the reader reward price calibrated uncertainty, or only entailment?

    python scripts/reward_audit.py suite  --n-per-cell 16 --out runs/reward-audit/suite.jsonl
    python scripts/reward_audit.py score  --suite runs/reward-audit/suite.jsonl
    python scripts/reward_audit.py report --items runs/reward-audit/items.jsonl

The RL run's rewards came back exactly bimodal -- every rollout scored either 0
or the clip floor, nothing between. The hypothesis this audit tests is that
`p_yes("does the passage support {gold}?")` measures *entailment* of the gold
answer rather than the reader's posterior *belief* in it. Entailment is
near-binary, so a calibrated hedge, an honest abstention and a confident
fabrication all collapse to the floor together, and RL sees no reason to prefer
any of them.

The measurement is a controlled suite: for each question, eight response
variants spanning correct/wrong x confident/hedged, plus abstention and a
deliberately vague non-answer. A reward that prices uncertainty has to order
them the way a proper score would, and in particular has to put abstention
strictly above confident-wrong -- that single inequality is what RL needs in
order to ever learn to say "I don't recall".

Candidates: the current reward (R0), a temperature sweep on its logit gap (A),
a phrasing ensemble (B), verbalized probability in three readings (C), Band's
two-stage normalized forecast (D), and a Monte-Carlo forecast over the reader's
own answers (E).
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
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax
import numpy as np

from cartridge import load_model, load_tokenizer
from reader_score import (EQUIV, FORECAST, READER_SYS, encode_user, generate,
                          is_unknown, last_logits, read_jsonl, write_jsonl, yes_no_ids)

OUT_DIR = REPO / "runs/reward-audit"
RETENTION = REPO / "runs/boundary/retention-read.jsonl"
QA = REPO / "runs/reader/qa.jsonl"

# The eight response variants. `a`..`h` follow the audit spec; every template is
# written in the same register and at roughly the same length, so that what
# moves the reward is what is claimed and how certainly, not style or length.
VARIANTS = {
    "a_confident_correct":
        "The answer is {gold}. That is what this part of the codebase uses, and "
        "the surrounding module refers to it that way throughout.",
    "b_confident_wrong":
        "The answer is {d1}. That is what this part of the codebase uses, and "
        "the surrounding module refers to it that way throughout.",
    "c_hedged_correct":
        "I'm not completely certain, but I believe the answer is {gold}. That is "
        "my recollection of how this part of the codebase is written.",
    "d_hedged_multi_with_gold":
        "I'm not sure about this one. It is possibly {gold}, or possibly {d1} -- "
        "I can't tell which of the two it is from memory.",
    "e_hedged_multi_without_gold":
        "I'm not sure about this one. It is possibly {d1}, or possibly {d2} -- "
        "I can't tell which of the two it is from memory.",
    "f_abstention":
        "The corpus does cover this area, but I don't recall this particular "
        "detail and I would rather not guess at what it is.",
    "g_hedged_wrong":
        "I'm not completely certain, but I believe the answer is {d1}. That is "
        "my recollection of how this part of the codebase is written.",
    "h_vague_nonanswer":
        "This is about how that part of the codebase is set up. The relevant "
        "module handles it in the usual way, consistent with the code around it.",
    # i..k: stated confidence. Band's post-SFT policy always names its
    # candidates *with* a number, so their reader reads a probability off the
    # page instead of inferring one. These test whether that is what our
    # zero-shot reader needs in order to price uncertainty at all.
    "i_stated90_correct":
        "I'm about 90% sure the answer is {gold}. That is my recollection of "
        "how this part of the codebase is written.",
    "j_stated_split":
        "It's either {gold} -- about 60% -- or {d1}, about 40%. I can't pin it "
        "down more tightly than that from memory.",
    "k_stated90_wrong":
        "I'm about 90% sure the answer is {d1}. That is my recollection of "
        "how this part of the codebase is written.",
}

STATED = ["i_stated90_correct", "j_stated_split", "k_stated90_wrong"]

ORDER = list(VARIANTS)

ASK_DISTRACTORS = (
    "Question about a codebase: {q}\n"
    "The correct answer is: {gold}\n\n"
    "Give two *incorrect* alternative answers of exactly the same kind and "
    "format as the correct one -- the sort of thing someone might plausibly "
    "misremember. They must be wrong, and must not be paraphrases of the "
    "correct answer. Reply in exactly this format and nothing else:\n"
    "D1: <first wrong answer>\nD2: <second wrong answer>"
)

FORECAST_HEAD = ("Below is a passage someone wrote about a codebase.\n\n"
                 "<passage>\n{p}\n</passage>\n\n"
                 "Question: {q}\nProposed answer: {a}\n\n")

# Candidate B's ensemble. The first is exactly the phrasing the RL run uses.
PHRASINGS = [
    "Does the passage support the proposed answer? Answer Yes or No.",
    "Based only on the passage, is the proposed answer correct? Answer Yes or No.",
    ("Would someone reading only this passage conclude that the answer is the "
     "proposed one? Answer Yes or No."),
    ("Judging only by this passage, is the proposed answer more likely right than "
     "wrong? Answer Yes or No."),
]
assert FORECAST_HEAD + PHRASINGS[0] == FORECAST, "R0 must reproduce reader_rl's prompt"

# Candidate C, three readings of the same question. The free-form percentage is
# what the spec asks for; the digit and letter forms are single forwards, which
# is what an RL loop can actually afford.
VERB_PCT = (
    FORECAST_HEAD +
    "Based only on the passage, what is the probability (0-100) that the answer "
    "to the question is the proposed answer?\n"
    "Reply with a single number between 0 and 100 and nothing else."
)
VERB_DIGIT = (
    FORECAST_HEAD +
    "Based only on the passage, out of ten readers of this passage, how many "
    "would say the answer to the question is the proposed answer?\n"
    "Reply with a single digit 0-9 and nothing else."
)
SCALE = [("A", 0.05), ("B", 0.25), ("C", 0.50), ("D", 0.75), ("E", 0.95)]
VERB_SCALE = (
    FORECAST_HEAD +
    "Based only on the passage, how likely is it that the proposed answer is "
    "correct?\nA. Very unlikely\nB. Unlikely\nC. About even\nD. Likely\n"
    "E. Very likely\nAnswer with a single letter."
)

# Candidate G: don't infer the author's belief, read the confidence they state.
# This is what Band's trained ForecastProbs regressor gets for free from its
# graded-probability labels; the Wallsten variant hands the zero-shot reader the
# phrase-to-probability table (Wallsten et al. 1986) instead of training it.
READOFF = (
    FORECAST_HEAD +
    "Read only the confidence the author of the passage expresses. According to "
    "their own stated confidence, what probability do they assign to the "
    "proposed answer being the correct one?\n"
    "{table}Reply with a single number between 0 and 100 and nothing else."
)
WALLSTEN = (
    "If they state a number, use it. If they use words, translate them: "
    "\"certain\"/\"definitely\" = 95, \"very likely\"/\"almost certain\" = 90, "
    "\"probably\"/\"likely\" = 75, \"possibly\"/\"may be\"/\"might be\" = 40, "
    "\"unlikely\"/\"doubtful\" = 20, \"very unlikely\" = 5. If they propose no "
    "answer at all, or say they do not recall, reply 0.\n"
)

# Candidate D: what answers does the passage actually put on the table?
EXTRACT_MANY = (
    "Below is a passage someone wrote about a codebase.\n\n"
    "<passage>\n{p}\n</passage>\n\n"
    "Question: {q}\n\n"
    "List every distinct answer to that question that the passage puts forward, "
    "one per line, at most three, each a few words. If the passage puts forward "
    "none, write NONE. List only the answers themselves."
)
UNDETERMINED = (
    "Below is a passage someone wrote about a codebase.\n\n"
    "<passage>\n{p}\n</passage>\n\n"
    "Question: {q}\n\n"
    "Based only on the passage, does the passage leave this question "
    "undetermined? Answer Yes or No."
)

# Candidate E: let the reader answer in its own words, several times over.
MC_ANSWER = (
    "Below is a passage someone wrote about a codebase.\n\n"
    "<passage>\n{p}\n</passage>\n\n"
    "Based ONLY on the passage, answer this question in a few words:\n\n{q}\n\n"
    "If the passage does not determine the answer, give your best guess anyway. "
    "Only if the passage says nothing at all about it, reply exactly UNKNOWN. "
    "Give the answer alone."
)

NO_PASSAGE = "(The writer produced no passage.)"


# -----------------------------------------------------------------------------
# suite
# -----------------------------------------------------------------------------


def parse_distractors(text, gold):
    d1 = re.search(r"D1:\s*(.+)", text)
    d2 = re.search(r"D2:\s*(.+)", text)
    if not d1 or not d2:
        return None
    a, b = (m.group(1).strip().strip('"`. ') for m in (d1, d2))
    low = {a.lower(), b.lower(), gold.lower()}
    if not a or not b or len(low) < 3:
        return None
    if gold.lower() in a.lower() or gold.lower() in b.lower():
        return None
    return a, b


def shape(a):
    """Coarse type of an answer, for mining same-shape distractors."""
    return (min(len(a.split()), 4),
            bool(re.search(r"[_.]|[a-z][A-Z]", a)),
            bool(re.fullmatch(r"[-\d.]+", a.strip())))


def mine_distractors(gold, pool, rng):
    """Fallback: other questions' gold answers of the same shape."""
    same = [a for a in pool if shape(a) == shape(gold) and a.lower() != gold.lower()]
    if len(same) < 2:
        same = [a for a in pool if a.lower() != gold.lower()]
    return tuple(rng.sample(same, 2))


def cmd_suite(args):
    from boundary_gen import retention_split

    variants = {k: VARIANTS[k] for k in (args.variants or VARIANTS)}
    if args.from_suite:
        # Same questions, same distractors, new variants: an addendum has to sit
        # in the same cells as the run it extends or the means are not comparable.
        seen, rows = set(), []
        for r in read_jsonl(args.from_suite):
            if r["question"] in seen:
                continue
            seen.add(r["question"])
            for name, tmpl in variants.items():
                rows.append({k: v for k, v in r.items() if k != "response"}
                            | {"variant": name,
                               "response": tmpl.format(gold=r["answer"], d1=r["d1"], d2=r["d2"])})
        write_jsonl(rows, args.out)
        print(f"  {len(seen)} questions x {len(variants)} variants reused from "
              f"{args.from_suite}")
        for r in rows[: len(variants)]:
            print(f"\n[{r['variant']}] {r['response'][:160]}")
        return

    tokenizer = load_tokenizer()
    retained, tally = retention_split(read_jsonl(args.retention), k_min=args.k_min)
    gold = {}
    for r in read_jsonl(args.qa):
        if r["kind"] == "in" and r.get("answer") and r["question"] not in gold:
            gold[r["question"]] = r["answer"]

    rng = random.Random(args.seed)
    pools = {"retained": sorted(q for q, v in retained.items() if v and q in gold),
             "blurred": sorted(q for q, v in retained.items() if not v and q in gold)}
    picked = []
    for cell, qs in pools.items():
        qs = list(qs)
        rng.shuffle(qs)
        picked += [{"question": q, "answer": gold[q], "cell": cell,
                    "retention_correct": tally[q][0], "retention_n": tally[q][1]}
                   for q in qs[: args.n_per_cell]]
    print(f"{len(picked)} questions of {len(retained)} scored "
          f"({len(pools['retained'])} retained / {len(pools['blurred'])} blurred available)")

    model = load_model()
    prompts = [encode_user(tokenizer, READER_SYS,
                           ASK_DISTRACTORS.format(q=p["question"], gold=p["answer"]))
               for p in picked]
    texts = generate(model, tokenizer, prompts, max_new=64, temperature=0.7,
                     key=jax.random.key(args.seed), batch=args.batch)

    all_gold = sorted(set(gold.values()))
    rows, mined = [], 0
    for p, t in zip(picked, texts):
        d = parse_distractors(t, p["answer"])
        src = "model"
        if not d:
            d, src, mined = mine_distractors(p["answer"], all_gold, rng), "mined", mined + 1
        for name, tmpl in variants.items():
            rows.append({**p, "variant": name, "d1": d[0], "d2": d[1], "distractor_src": src,
                         "response": tmpl.format(gold=p["answer"], d1=d[0], d2=d[1])})
    write_jsonl(rows, args.out)
    meta = {"qa": str(args.qa), "retention": str(args.retention), "k_min": args.k_min,
            "seed": args.seed, "n_questions": len(picked), "n_variants": len(variants),
            "mined_distractors": mined, "variants": variants}
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {mined} questions fell back to mined distractors")
    for r in rows[: len(variants)]:
        print(f"\n[{r['variant']}] {r['response'][:160]}")


# -----------------------------------------------------------------------------
# reward candidates
# -----------------------------------------------------------------------------


def logit_gap(model, tokenizer, prompts, *, batch):
    """(yes_logsumexp - no_logsumexp) at the first assistant token."""
    if not prompts:
        return np.zeros(0)
    yes, no = yes_no_ids(tokenizer)
    lg = last_logits(model, tokenizer, prompts, batch=batch).astype(np.float64)

    def lse(ids):
        z = lg[:, ids]
        m = z.max(-1)
        return m + np.log(np.exp(z - m[:, None]).sum(-1))

    return lse(yes) - lse(no)


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def forecast_prompts(tokenizer, rows, phrasing, *, passage=None, answer=None):
    return [encode_user(tokenizer, READER_SYS,
                        FORECAST_HEAD.format(p=passage or r["response"], q=r["question"],
                                             a=answer(r) if answer else r["answer"])
                        + PHRASINGS[phrasing])
            for r in rows]


def cand_R0_A(model, tokenizer, rows, args):
    """The current reward, and the same logit gap read at higher temperatures."""
    gap = logit_gap(model, tokenizer, forecast_prompts(tokenizer, rows, 0), batch=args.batch)
    out = {"R0": sig(gap)}
    for T in (2, 4, 8):
        out[f"A_T{T}"] = sig(gap / T)
    return out, gap


def cand_B(model, tokenizer, rows, args, p0):
    """Average the forecast over several phrasings, in probability space."""
    ps = [p0]
    for i in range(1, len(PHRASINGS)):
        ps.append(sig(logit_gap(model, tokenizer, forecast_prompts(tokenizer, rows, i),
                                batch=args.batch)))
    return {"B_ensemble": np.mean(ps, axis=0)}, np.stack(ps)


def cand_C(model, tokenizer, rows, args):
    """Verbalized probability: free-form percentage, single digit, letter scale."""
    n, sub = max(len(rows), 1), {}
    t = time.time()
    # free-form 0-100, greedily generated and parsed
    gen = generate(model, tokenizer,
                   [encode_user(tokenizer, READER_SYS,
                                VERB_PCT.format(p=r["response"], q=r["question"], a=r["answer"]))
                    for r in rows],
                   max_new=8, temperature=0.0, key=jax.random.key(0), batch=args.batch)
    pct, parsed = [], 0
    for text in gen:
        m = re.search(r"\d+(?:\.\d+)?", text)
        if m:
            parsed += 1
            pct.append(min(max(float(m.group()), 0.0), 100.0) / 100.0)
        else:
            pct.append(0.5)
    sub["C_pct"], t = (time.time() - t) / n, time.time()

    # one forward, expectation over the digit tokens 0..9 -> (d + 0.5)/10
    lg = last_logits(model, tokenizer,
                     [encode_user(tokenizer, READER_SYS,
                                  VERB_DIGIT.format(p=r["response"], q=r["question"],
                                                    a=r["answer"])) for r in rows],
                     batch=args.batch).astype(np.float64)
    ids = np.array([tokenizer.encode(str(d), add_special_tokens=False)[0] for d in range(10)])
    w = np.exp(lg[:, ids] - lg[:, ids].max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    digit = w @ ((np.arange(10) + 0.5) / 10.0)
    sub["C_digit"], t = (time.time() - t) / n, time.time()

    # one forward, expectation over a five-point verbal scale
    lg = last_logits(model, tokenizer,
                     [encode_user(tokenizer, READER_SYS,
                                  VERB_SCALE.format(p=r["response"], q=r["question"],
                                                    a=r["answer"])) for r in rows],
                     batch=args.batch).astype(np.float64)
    ids = np.array([tokenizer.encode(l, add_special_tokens=False)[0] for l, _ in SCALE])
    w = np.exp(lg[:, ids] - lg[:, ids].max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    scale = w @ np.array([v for _, v in SCALE])
    sub["C_scale"] = (time.time() - t) / n

    return ({"C_pct": np.asarray(pct), "C_digit": digit, "C_scale": scale},
            {"parse_rate": parsed / n, "raw": gen, "cost": sub})


def parse_pct(texts, default=0.5):
    out, parsed = [], 0
    for t in texts:
        m = re.search(r"\d+(?:\.\d+)?", t)
        if m:
            parsed += 1
            out.append(min(max(float(m.group()), 0.0), 100.0) / 100.0)
        else:
            out.append(default)
    return np.asarray(out), parsed / max(len(texts), 1)


def cand_G(model, tokenizer, rows, args):
    """Read the confidence the author states, rather than inferring a belief."""
    out, extra = {}, {}
    for name, table in (("G_readoff", ""), ("G_wallsten", WALLSTEN)):
        t = time.time()
        gen = generate(model, tokenizer,
                       [encode_user(tokenizer, READER_SYS,
                                    READOFF.format(p=r["response"], q=r["question"],
                                                   a=r["answer"], table=table))
                        for r in rows],
                       max_new=8, temperature=0.0, key=jax.random.key(0), batch=args.batch)
        out[name], rate = parse_pct(gen, default=0.0)
        extra[name] = {"parse_rate": rate, "raw": gen,
                       "cost": (time.time() - t) / max(len(rows), 1)}
    return out, extra


def cand_D(model, tokenizer, rows, args):
    """Band-style normalisation (D), and the same quantities read as a posterior (F).

    D asks what share of the passage's own support the gold answer holds, with an
    "undetermined" bucket competing alongside it. That still scores an honest
    abstention at zero: no support for the gold is no support, however honestly
    arrived at. F keeps the same two forwards but reads the undetermined mass as
    *ignorance* rather than *evidence against*: a reader who learns nothing from
    the passage falls back on a flat prior over K plausible answers, so silence
    is worth log(1/K) and confident fabrication is still worth the floor. That
    single change is what makes abstention beat a wrong assertion.
    """
    listed = generate(model, tokenizer,
                      [encode_user(tokenizer, READER_SYS,
                                   EXTRACT_MANY.format(p=r["response"], q=r["question"]))
                       for r in rows],
                      max_new=48, temperature=0.0, key=jax.random.key(0), batch=args.batch)
    cands, flat, index, empty = [], [], [], 0
    for r, text in zip(rows, listed):
        got = [ln.strip(" -*\t0123456789.") for ln in text.splitlines() if ln.strip()][:3]
        got = [g for g in got if g and not is_unknown(g)
               and re.sub(r"[^A-Za-z]", "", g).upper() not in ("NONE", "NOANSWER")]
        empty += not got
        # The gold always competes, so a passage that never mentions it still gets
        # a well-defined (small) share instead of a divide-by-zero.
        opts = [r["answer"]] + [g for g in got if g.lower() != r["answer"].lower()]
        cands.append(opts)
        index.append((len(flat), len(opts)))
        flat += [encode_user(tokenizer, READER_SYS,
                             FORECAST_HEAD.format(p=r["response"], q=r["question"], a=o)
                             + PHRASINGS[0]) for o in opts]
    p_all = sig(logit_gap(model, tokenizer, flat, batch=args.batch))
    p_none = sig(logit_gap(model, tokenizer,
                           [encode_user(tokenizer, READER_SYS,
                                        UNDETERMINED.format(p=r["response"], q=r["question"]))
                            for r in rows], batch=args.batch))
    q = np.zeros(len(rows))
    s_gold, s_sum = np.zeros(len(rows)), np.zeros(len(rows))
    for i, (start, n) in enumerate(index):
        share = p_all[start:start + n]
        s_gold[i], s_sum[i] = share[0], share.sum()
        q[i] = share[0] / max(share.sum() + p_none[i], 1e-9)
    out = {"D_normalized": q}
    for K in args.prior_k:
        out[f"F_prior{K}"] = (s_gold + p_none / K) / np.maximum(s_sum + p_none, 1e-9)
    k0 = args.prior_k[len(args.prior_k) // 2]
    out[f"F_mix{k0}"] = ((1 - p_none) * s_gold / np.maximum(s_sum, 1e-9) + p_none / k0)
    return out, {"extracted": cands, "p_none": p_none.tolist(),
                 "support_gold": s_gold.tolist(), "support_sum": s_sum.tolist(),
                 "empty_extraction_rate": empty / max(len(rows), 1)}


def cand_E(model, tokenizer, rows, args):
    """Monte-Carlo: how often does the reader itself answer with the gold?"""
    k = args.mc_k
    prompts = [encode_user(tokenizer, READER_SYS,
                           MC_ANSWER.format(p=r["response"], q=r["question"]))
               for r in rows for _ in range(k)]
    out, extra = {}, {}
    for T in args.mc_temp:
        t0 = time.time()
        ans = generate(model, tokenizer, prompts, max_new=24, temperature=T,
                       key=jax.random.key(1), batch=args.batch,
                       progress=lambda d, n: (d % (args.batch * 20) or
                                              print(f"    mc T={T} {d}/{n} "
                                                    f"({time.time() - t0:.0f}s)", flush=True)))
        judge, index = [], []
        for i, r in enumerate(rows):
            for j in range(k):
                a = ans[i * k + j]
                if is_unknown(a):
                    continue
                index.append((i, len(judge)))
                judge.append(encode_user(tokenizer, READER_SYS,
                                         EQUIV.format(q=r["question"], a=r["answer"], b=a)))
        eq = sig(logit_gap(model, tokenizer, judge, batch=args.batch))
        hard, soft = np.zeros(len(rows)), np.zeros(len(rows))
        for i, j in index:
            hard[i] += float(eq[j] > 0.5) / k
            soft[i] += float(eq[j]) / k
        tag = f"T{T:g}".replace(".", "")
        out[f"E_mc_hard_{tag}"], out[f"E_mc_soft_{tag}"] = hard, soft
        extra[tag] = {
            "unknown_rate": [sum(is_unknown(ans[i * k + j]) for j in range(k)) / k
                             for i in range(len(rows))],
            "distinct": [len(set(ans[i * k:(i + 1) * k])) for i in range(len(rows))],
            "samples": [ans[i * k:(i + 1) * k] for i in range(len(rows))]}
    return out, extra


def cmd_score(args):
    tokenizer = load_tokenizer()
    rows = read_jsonl(args.suite)
    model = load_model()
    t0 = time.time()
    scores, cost, extra = {}, {}, {}

    def timed(name, fn):
        t = time.time()
        out = fn()
        dt = time.time() - t
        cost[name] = dt / len(rows)
        print(f"  {name}: {dt:.0f}s ({dt / len(rows) * 1000:.0f} ms/response)", flush=True)
        return out

    (r0a, gap) = timed("R0", lambda: cand_R0_A(model, tokenizer, rows, args))
    scores.update(r0a)
    for T in (2, 4, 8):
        cost[f"A_T{T}"] = cost["R0"]          # same forward, different read
    b, ens = timed("B_ensemble", lambda: cand_B(model, tokenizer, rows, args, r0a["R0"]))
    scores.update(b)
    cost["B_ensemble"] += cost["R0"]
    c, extra["C"] = timed("C", lambda: cand_C(model, tokenizer, rows, args))
    scores.update(c)
    cost.update(extra["C"]["cost"])           # three independent reads, timed apart
    g, extra["G"] = timed("G", lambda: cand_G(model, tokenizer, rows, args))
    scores.update(g)
    cost.update({k: v["cost"] for k, v in extra["G"].items()})
    d, extra["D"] = timed("D_normalized", lambda: cand_D(model, tokenizer, rows, args))
    scores.update(d)
    for k in d:
        cost[k] = cost["D_normalized"]        # D and F share the same two passes
    e, extra["E"] = timed("E", lambda: cand_E(model, tokenizer, rows, args))
    scores.update(e)
    for k in e:
        cost[k] = cost["E"] / len(args.mc_temp)

    # No-passage control: the same forecast with the passage removed. If a
    # retained question still scores high, the reader is answering from its own
    # priors rather than from the passage.
    uniq = {r["question"]: r for r in rows}.values()
    ctrl = timed("no_passage_control", lambda: {
        "gold": sig(logit_gap(model, tokenizer,
                              forecast_prompts(tokenizer, list(uniq), 0, passage=NO_PASSAGE),
                              batch=args.batch)).tolist(),
        "d1": sig(logit_gap(model, tokenizer,
                            forecast_prompts(tokenizer, list(uniq), 0, passage=NO_PASSAGE,
                                             answer=lambda r: r["d1"]),
                            batch=args.batch)).tolist(),
        "question": [r["question"] for r in uniq], "cell": [r["cell"] for r in uniq]})

    items = []
    for i, r in enumerate(rows):
        items.append({**{k: v for k, v in r.items()},
                      "logit_gap": float(gap[i]),
                      "phrasing_p": [float(p) for p in ens[:, i]],
                      "d_extracted": extra["D"]["extracted"][i],
                      "d_p_none": extra["D"]["p_none"][i],
                      "d_support_gold": extra["D"]["support_gold"][i],
                      "d_support_sum": extra["D"]["support_sum"][i],
                      "e_unknown_rate": {t: x["unknown_rate"][i] for t, x in extra["E"].items()},
                      "e_distinct": {t: x["distinct"][i] for t, x in extra["E"].items()},
                      "e_samples": {t: x["samples"][i] for t, x in extra["E"].items()},
                      "c_pct_raw": extra["C"]["raw"][i],
                      "g_raw": {k: v["raw"][i] for k, v in extra["G"].items()},
                      "p": {k: float(v[i]) for k, v in scores.items()}})
    write_jsonl(items, args.out)
    meta = {"clip_lo": args.clip_lo, "candidates": list(scores),
            "cost_s_per_response": cost, "elapsed_s": time.time() - t0,
            "mc_k": args.mc_k, "mc_temp": args.mc_temp, "batch": args.batch,
            "c_parse_rate": extra["C"]["parse_rate"],
            "g_parse_rate": {k: v["parse_rate"] for k, v in extra["G"].items()},
            "d_empty_extraction_rate": extra["D"]["empty_extraction_rate"],
            "no_passage_control": ctrl}
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


# -----------------------------------------------------------------------------
# report
# -----------------------------------------------------------------------------

# Strict chain the spec asks for, plus the two informational placements.
CHAIN = ["a_confident_correct", "c_hedged_correct", "d_hedged_multi_with_gold",
         "f_abstention", "g_hedged_wrong"]


def cmd_report(args):
    items = read_jsonl(args.items)
    meta = json.loads(Path(args.items).with_suffix(".meta.json").read_text())
    clip = meta["clip_lo"]
    floor = float(np.log(clip))
    names = meta["candidates"]
    var = [r["variant"] for r in items]
    cell = [r["cell"] for r in items]
    logr = {k: np.log(np.clip(np.array([r["p"][k] for r in items]), clip, 1.0)) for k in names}

    def mean(k, v, c=None):
        m = [i for i in range(len(items)) if var[i] == v and (c is None or cell[i] == c)]
        return float(np.mean(logr[k][m])) if m else float("nan")

    matrix = {c or "all": {k: {v: mean(k, v, c) for v in ORDER} for k in names}
              for c in (None, "retained", "blurred")}
    mean_p = {k: {v: float(np.mean([items[i]["p"][k] for i in range(len(items))
                                    if var[i] == v])) for v in ORDER} for k in names}

    print(f"\nmean log reward per variant (clip floor {floor:.2f}), "
          f"{len(items) // len(ORDER)} questions x {len(ORDER)} variants\n")
    for c in ("all", "retained", "blurred"):
        print(f"--- {c}")
        print(f"  {'variant':28s}" + "".join(n.rjust(12) for n in names))
        for v in ORDER:
            print(f"  {v:28s}" + "".join(f"{matrix[c][k][v]:12.2f}" for k in names))
        print()

    verdict = {}
    for k in names:
        m = matrix["all"][k]
        ok_chain = all(m[CHAIN[i]] > m[CHAIN[i + 1]] + args.eps for i in range(len(CHAIN) - 1))
        ok_fb = m["f_abstention"] > m["b_confident_wrong"] + args.eps
        d_half = m["d_hedged_multi_with_gold"] - float(np.log(0.5))
        ok_d = abs(d_half) < args.d_tol
        ok_h = m["h_vague_nonanswer"] <= m["f_abstention"] + args.h_tol
        ok_e = (abs(m["e_hedged_multi_without_gold"] - m["g_hedged_wrong"]) < args.e_tol
                and m["e_hedged_multi_without_gold"] < m["d_hedged_multi_with_gold"] - args.eps)
        interior = float(np.mean((logr[k] > floor + 0.1) & (logr[k] < -0.1)))
        verdict[k] = {
            "order_a_c_d_f_g": ok_chain, "f_gt_b": ok_fb, "d_near_log_half": ok_d,
            "d_minus_log_half": d_half, "h_le_f": ok_h, "e_near_g_below_d": ok_e,
            "h_minus_f": m["h_vague_nonanswer"] - m["f_abstention"],
            "f_minus_b": m["f_abstention"] - m["b_confident_wrong"],
            "margins": {f"{CHAIN[i]}>{CHAIN[i + 1]}": m[CHAIN[i]] - m[CHAIN[i + 1]]
                        for i in range(len(CHAIN) - 1)},
            "interior_mass": interior,
            "hedge_delta_c_minus_a": m["c_hedged_correct"] - m["a_confident_correct"],
            "hedge_delta_g_minus_b": m["g_hedged_wrong"] - m["b_confident_wrong"],
            "cost_s_per_response": meta["cost_s_per_response"].get(k),
            "pass": bool(ok_chain and ok_fb and ok_d and ok_h),
        }

    hdr = [("a>c>d>f>g", "order_a_c_d_f_g"), ("f>b", "f_gt_b"), ("d~log.5", "d_near_log_half"),
           ("h<=f", "h_le_f"), ("e~g<d", "e_near_g_below_d")]
    print("acceptance criteria")
    print(f"  {'candidate':14s}" + "".join(h.rjust(10) for h, _ in hdr)
          + "  interior   d-log.5   c-a    g-b    ms/resp  PASS")
    for k in names:
        v = verdict[k]
        cells = "".join(("yes" if v[key] else "NO").rjust(10) for _, key in hdr)
        print(f"  {k:14s}{cells}{v['interior_mass']:10.2f}{v['d_minus_log_half']:10.2f}"
              f"{v['hedge_delta_c_minus_a']:7.2f}{v['hedge_delta_g_minus_b']:7.2f}"
              f"{(v['cost_s_per_response'] or 0) * 1000:9.0f}"
              f"   {'PASS' if v['pass'] else 'fail'}")

    stated = {}
    if all(any(v == s for v in var) for s in STATED):
        print("\nstated-confidence (Band parity): does a passage that names its own "
              "numbers get priced?")
        print(f"  {'candidate':14s}{'i (90 right)':>14s}{'j (60/40)':>12s}"
              f"{'k (90 wrong)':>14s}{'i>j>k':>8s}{'j~log.6':>9s}{'k>b':>6s}")
        for k in names:
            m = matrix["all"][k]
            i_, j_, k_ = (m[s] for s in STATED)
            ok_ord = i_ > j_ + args.eps and j_ > k_ + args.eps
            j_off = j_ - float(np.log(0.6))
            ok_j = abs(j_off) < args.d_tol
            ok_kb = k_ > m["b_confident_wrong"] + args.eps
            stated[k] = {"i": i_, "j": j_, "k": k_, "i_gt_j_gt_k": ok_ord,
                         "j_minus_log_0.6": j_off, "j_near_log_0.6": ok_j, "k_gt_b": ok_kb,
                         "band_parity": bool(ok_ord and ok_j and ok_kb)}
            print(f"  {k:14s}{i_:14.2f}{j_:12.2f}{k_:14.2f}"
                  f"{('yes' if ok_ord else 'NO'):>8s}{j_off:9.2f}"
                  f"{('yes' if ok_kb else 'NO'):>6s}")

    print("\nR0 diagnosis: raw Yes/No logit gap by variant (entailment reads as +-large)")
    gaps = {}
    for v in ORDER:
        g = np.array([items[i]["logit_gap"] for i in range(len(items)) if var[i] == v])
        gaps[v] = {"mean": float(g.mean()), "median": float(np.median(g)),
                   "p10": float(np.quantile(g, 0.1)), "p90": float(np.quantile(g, 0.9)),
                   "frac_abs_lt_2": float(np.mean(np.abs(g) < 2)),
                   "frac_lt_minus7": float(np.mean(g < -7)), "frac_gt_7": float(np.mean(g > 7))}
        print(f"  {v:28s} mean {g.mean():8.2f}  median {np.median(g):8.2f}  "
              f"[p10 {np.quantile(g, 0.1):7.2f}, p90 {np.quantile(g, 0.9):7.2f}]  "
              f"|gap|<2: {np.mean(np.abs(g) < 2):.2f}")

    mc = {}
    for t in items[0]["e_distinct"]:
        mc[t] = {"mean_distinct_of_k": float(np.mean([r["e_distinct"][t] for r in items])),
                 "frac_rows_degenerate": float(np.mean([r["e_distinct"][t] == 1 for r in items])),
                 "unknown_rate": float(np.mean([r["e_unknown_rate"][t] for r in items]))}
    print("\nE diagnosis: is the reader's own answer distribution non-degenerate?")
    for t, v in mc.items():
        print(f"  {t:6s} mean distinct answers of k={meta['mc_k']}: "
              f"{v['mean_distinct_of_k']:.2f}   rows with a single answer: "
              f"{v['frac_rows_degenerate']:.2f}   UNKNOWN rate {v['unknown_rate']:.2f}")

    ctrl = meta["no_passage_control"]
    print("\nno-passage control (R0 phrasing, passage removed)")
    for c in ("retained", "blurred"):
        sel = [i for i, x in enumerate(ctrl["cell"]) if x == c]
        print(f"  {c:10s} n={len(sel):3d}  p(gold) {np.mean([ctrl['gold'][i] for i in sel]):.3f}"
              f"   p(distractor) {np.mean([ctrl['d1'][i] for i in sel]):.3f}")

    out = {"meta": {k: meta[k] for k in ("clip_lo", "cost_s_per_response", "elapsed_s",
                                         "mc_k", "mc_temp", "c_parse_rate",
                                         "d_empty_extraction_rate")},
           "n_questions": len(items) // len(ORDER),
           "matrix_log_reward": matrix, "mean_probability": mean_p, "verdict": verdict,
           "r0_logit_gap": gaps, "mc_degeneracy": mc, "stated_confidence": stated,
           "no_passage_control": {
               c: {"n": sum(x == c for x in ctrl["cell"]),
                   "p_gold": float(np.mean([ctrl["gold"][i] for i, x in enumerate(ctrl["cell"])
                                            if x == c])),
                   "p_distractor": float(np.mean([ctrl["d1"][i] for i, x in enumerate(ctrl["cell"])
                                                  if x == c]))}
               for c in ("retained", "blurred")}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("suite")
    s.add_argument("--qa", default=str(QA))
    s.add_argument("--retention", default=str(RETENTION))
    s.add_argument("--k-min", type=int, default=2)
    s.add_argument("--n-per-cell", type=int, default=16)
    s.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    s.add_argument("--from-suite", help="reuse the questions and distractors of an "
                                        "existing suite instead of sampling fresh ones")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--out", default=str(OUT_DIR / "suite.jsonl"))

    c = sub.add_parser("score")
    c.add_argument("--suite", default=str(OUT_DIR / "suite.jsonl"))
    c.add_argument("--batch", type=int, default=16)
    c.add_argument("--mc-k", type=int, default=8)
    c.add_argument("--mc-temp", type=float, nargs="+", default=[1.0, 1.5])
    c.add_argument("--clip-lo", type=float, default=1e-3)
    c.add_argument("--prior-k", type=int, nargs="+", default=[3, 4, 8],
                   help="F's flat prior: a passage that determines nothing is worth log(1/K)")
    c.add_argument("--out", default=str(OUT_DIR / "items.jsonl"))

    r = sub.add_parser("report")
    r.add_argument("--items", default=str(OUT_DIR / "items.jsonl"))
    r.add_argument("--eps", type=float, default=1e-6)
    r.add_argument("--d-tol", type=float, default=0.7,
                   help="how far d may sit from log(1/2) and still count as partial credit")
    r.add_argument("--e-tol", type=float, default=0.7)
    r.add_argument("--h-tol", type=float, default=0.1,
                   help="how far above f the vague non-answer may sit and still count as tied")
    r.add_argument("--out", default=str(OUT_DIR / "matrix.json"))

    args = p.parse_args()
    {"suite": cmd_suite, "score": cmd_score, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
