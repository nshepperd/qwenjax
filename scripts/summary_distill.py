"""Summary distillation: install retention-aware hedging before the RL relaunch.

    python scripts/summary_distill.py queries --n-qa 400 --n-open 100
    python scripts/summary_distill.py sample  --m 6
    python scripts/summary_distill.py merge
    python scripts/summary_distill.py mix     --steps 250
    python scripts/summary_distill.py train   --steps 250 --lr 1e-3
    python scripts/summary_distill.py eval    --cartridge runs/summary/summary.safetensors

Stage 1 of Band et al. (arXiv:2404.00474), adapted to cartridges. The CE
cartridge hedges 3% of the time where the base model hedges 29%: distillation
against a teacher that can see the chunk deletes hedging, because a teacher
reading the answer off the page is never uncertain. RL then has no strategy
space to shape -- it cannot select for calibrated language that the policy never
emits.

The fix is to build targets the teacher could not have written alone. For each
query we sample the *student* M times, and the teacher -- with the source chunk
in front of it -- merges those samples into one reply whose confidence language
reflects (a) how much the samples agree and (b) what the chunk says is true.
The result is a target whose hedging tracks the student's own retention, which
is the thing teacher-grounded data provably cannot supply (the boundary
experiment: the teacher sees the CORPUS boundary, never the RETENTION one).

Three tiers, gated on the measured retention of the RL learnability filter --
`successes`, the number of 8 rollouts the reader scored right:

    successes >= 6   the student knows this      -> state it plainly
    1..5             the student half-knows it   -> hedge, naming the candidates
    0                the student does not know   -> calibrated non-answer

The chunk grounds all three: a claim it contradicts is corrected or dropped,
and in the third tier the teacher is forbidden to supply the answer it can see.
That last rule is what makes this a *retention* target rather than a knowledge
target, and `merge` measures how often it leaks anyway.

Training continues from the CE cartridge (a post-train, not a fresh
distillation) on the merged targets mixed 50/50 with the plain self-study the
CE cartridge already learned, so the hedging is installed without unlearning
the corpus.
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
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax
import numpy as np

from cartridge import DESCRIPTION, load_corpus, load_model, load_tokenizer
from reader_score import (FORECAST, QUERY, READER_SYS, TRAINED, encode_suffix, encode_user,
                          generate, p_yes, read_jsonl, write_jsonl)

OUT_DIR = REPO / "runs/summary"
POOL = REPO / "runs/rl/learnable-pool2.all.jsonl"      # 735 questions with `successes`
PLAIN = REPO / "runs/cart/train.jsonl"                 # what the CE cartridge learned
HELDOUT = REPO / "runs/cart/heldout.jsonl"
QA_EVAL = REPO / "runs/reader/qa.jsonl"
CE_READ = REPO / "runs/reader/read.jsonl"              # CE baseline, same eval set
CE_RESP = REPO / "runs/reader/responses.jsonl"
RETENTION = REPO / "runs/boundary/retention-read.jsonl"

# Open-ended queries. Hedging has to survive in long-form prose, not just in
# one-line answers, so a fraction of the targets are "tell me about X".
OPEN_Q = [
    "Tell me about `{name}` in this codebase -- what is it and how does it work?",
    "Walk me through `{name}`: what it does, and how it fits with the rest of the code.",
    "I keep running into `{name}`. Explain what it is for and how it is implemented.",
    "Give me an overview of `{name}` and the part of the codebase it belongs to.",
]

# -----------------------------------------------------------------------------
# merge prompts
# -----------------------------------------------------------------------------

MERGE_HEAD = (
    "A user asked this about the codebase:\n\n{q}\n\n"
    "{m} replies to it were written from memory by someone who studied this "
    "corpus and cannot see it now. Here they are:\n\n{samples}\n"
)

MERGE_TAIL = (
    "Write the single reply that person should have given -- in their own "
    "voice, from memory.\n\n"
    "Hard rules:\n"
    "- At most three sentences, under 60 words, and finish the last sentence.\n"
    "- Never mention a document, the code, the corpus text, the attempts, or "
    "anything you were shown. No \"the document says\", no \"the code actually "
    "uses\", no \"the surrounding code shows\". Only what the person remembers.\n"
    "- Whenever less than certain, say how certain in a form a reader could turn "
    "into a number -- \"about 70% sure\", \"probably\", \"possibly, maybe a "
    "1-in-3 chance\" -- never a bare \"I think\".\n"
    "\nHow certain to sound here:\n{rule}\n"
    "\nWrite only the reply.\n"
)

# The tiers, gated on the *measured* frequency with which the student's own
# samples got this right -- not on the teacher's view of the chunk. This is the
# whole point: the teacher can see the answer, so left to itself it states every
# one of them plainly.
#
# The stated number is load-bearing, and it is Band's trick rather than ours:
# their post-SFT policy names its candidates with numeric confidence, so the
# reader reads a probability off the page instead of inferring one from tone.
# Our own audit reproduced that -- qualitative hedges collapse to the reward's
# clip floor, stated percentages land in the interior.
RULES = {
    "confident": (
        "- All {m} attempts agree and they are right: state the answer plainly "
        "and directly, as a fact. No hedging and no percentage.\n"
        "- Add at most one clause of surrounding detail that the attempts agree "
        "on and that is true."
    ),
    "probable": (
        "- {n_ok} of the {m} attempts got this right: the person half-remembers "
        "it. Give the answer WITH that confidence attached, in words and a "
        "number -- \"I'm about {pct}% sure it's X\", or \"probably X, around "
        "{pct}%\".\n"
        "- If the attempts offer a rival answer, name it with the remaining "
        "share: \"...or possibly Y\".\n"
        "- Drop any claim that is wrong rather than correcting it into a "
        "confident statement."
    ),
    "possible": (
        "- Only {n_ok} of the {m} attempts got this right: the memory is weak. "
        "Open with the confidence and the number, in this shape: \"I'm only "
        "about {pct}% sure it's X, and I may be confusing it with Y\" -- naming "
        "the rival the attempts offer. Begin the reply with \"I'm\".\n"
        "- Do not state it as a fact, and do not repair it into a confident "
        "answer: a weak memory stays weak. Drop wrong claims, do not correct "
        "them."
    ),
    "unknown": (
        "- Every attempt got this wrong. You can see the right answer; the "
        "person writing cannot remember it. Writing it down -- or any part of "
        "it, or a near-miss of it -- would be dishonest, so do NOT.\n"
        "- Name no identifier at all. Say plainly that the corpus covers this "
        "area but that you do not retain this particular detail, and say in one "
        "clause what kind of thing the area is about. Do not guess and do not "
        "apologise."
    ),
    "open": (
        "- State plainly whatever all {m} attempts agree on and that is true.\n"
        "- Where the attempts disagree, or claim specifics that are not right, "
        "give that part with an explicit confidence -- \"probably X, about "
        "70%\", \"possibly X, though I may be misremembering\" -- naming the "
        "alternatives in proportion to how many attempts back each one.\n"
        "- Drop any claim that is wrong."
    ),
}


# The teacher writing *about the page* rather than from memory: the tell that a
# target is a reading-comprehension answer wearing a memory's clothes.
REFERENCE_RE = re.compile(
    r"\b(?:the document|the passage|the attempts?|the (?:surrounding |above )?code "
    r"(?:above|shows|says|actually|confirms)|the corpus text|as shown above|"
    r"according to the (?:document|code|passage)|in the document|shown above)\b",
    re.IGNORECASE)

# Any code-like identifier. An honest non-answer names nothing.
IDENT_RE = re.compile(r"`[^`\n]+`|\b[A-Za-z_]\w*_\w+\b|\b[a-z]+[A-Z]\w*\b|\b\w+\(\)")


def leaks(text, gold):
    """Did the withheld answer, or its distinctive identifier, come back out?"""
    if not gold:
        return False
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
    if norm(gold) and norm(gold) in norm(text):
        return True
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", gold) if len(p) >= 4]
    return any(p.lower() in text.lower() for p in parts)


def trim_sentences(text, max_words=70):
    """Cut back to the last complete sentence; drop what ran past the budget.

    A target truncated mid-clause teaches the student to stop mid-clause, and
    the teacher overruns the word budget often enough to matter.
    """
    t = " ".join((text or "").split())
    words = t.split(" ")
    if len(words) > max_words:
        t = " ".join(words[:max_words])
    cut = max(t.rfind(". "), t.rfind("! "), t.rfind("? "))
    end = len(t) - 1 if t[-1:] in ".!?" else (cut + 1 if cut > 40 else -1)
    return t[:end + 1].strip() if end > 0 else ""

# Sample frequency -> target style. The percentage handed to the teacher is the
# student's own empirical hit rate, so the number in the target is calibrated by
# construction rather than by the teacher's taste.
def style_of(freq):
    if freq is None:
        return "open"
    if freq >= 1.0:
        return "confident"
    if freq >= 0.5:
        return "probable"
    return "possible" if freq > 0 else "unknown"

# Context distillation (arXiv:2209.15189), the fix the boundary experiment
# needed: the teacher that the merged target is distilled against is shown only
# the chunk and the question, and on a fact it can read it opens a plain answer.
# This note goes in the teacher's system prompt only, never the student's.
TEACHER_NOTE = (
    "The user is asking you to answer from memory of this corpus, not by reading "
    "it back. Where your memory of a detail would be uncertain, support their "
    "calibrated expression of that uncertainty -- hedged wording and honest "
    "non-answers are correct behaviour here, not failures."
)

# Candidates for the note, searched by `notes` before the real run. The boundary
# experiment's lesson: the target is written by an instructed teacher and then
# distilled against an uninstructed one, and whatever the uninstructed teacher
# will not open, the student never learns. The note is the only lever on that.
NOTE_VARIANTS = {
    "none": "",
    "support": TEACHER_NOTE,
    "memory": (
        "Answer from memory of this corpus, not by reading it back, and say how "
        "confident that memory is. Open with the confidence: \"I'm about 70% "
        "sure...\", \"Possibly...\", \"I don't retain that detail...\" unless "
        "you are certain, in which case answer plainly. Never refer to the "
        "document."
    ),
    "house_style": (
        "House style for replies here: every reply begins by signalling how well "
        "the answer is remembered. Certain answers are stated plainly; "
        "half-remembered ones open with a stated probability (\"I'm about 60% "
        "sure it's ...\"); weak ones open with \"Possibly ...\"; unremembered "
        "ones open with \"I don't recall ...\". Replies never mention the "
        "document and never quote it."
    ),
}

# -----------------------------------------------------------------------------
# hedging
# -----------------------------------------------------------------------------

# Applied identically to every condition, so the comparison is internally
# consistent whatever the absolute level. Deliberately conservative: markers of
# *speaker* uncertainty only, not "may" in the sense of permission.
HEDGE_RE = re.compile(
    r"\b(?:I think|I believe|I suspect|I'm not (?:entirely |completely |totally )?"
    r"(?:sure|certain)|I am not (?:entirely |completely )?(?:sure|certain)|"
    r"not (?:entirely|completely|totally) (?:sure|certain)|possibly|perhaps|"
    r"might be|may be|may well be|I don't (?:recall|remember)|I do not "
    r"(?:recall|remember)|don't fully recall|if I recall|as far as I "
    r"(?:recall|remember)|from what I (?:recall|remember)|I would guess|I'd "
    r"guess|my recollection|misremember\w*|confusing it with|I cannot recall|"
    r"I can't recall|do not retain|don't retain|uncertain|not sure)\b",
    re.IGNORECASE)


def ascii_quotes(s):
    """The model writes curly apostrophes; "I don't" and "I don’t" must match."""
    return (s or "").replace("’", "'").replace("‘", "'").replace("“", '"')


def hedged(text):
    return bool(HEDGE_RE.search(ascii_quotes(text)))


def hedge_rate(texts):
    return float(np.mean([hedged(t) for t in texts])) if texts else float("nan")


# Band's load-bearing detail: a stated number the reader can read off, rather
# than a tone it has to infer. Tracked separately from qualitative hedging.
PCT_RE = re.compile(r"\b\d{1,3}\s?%|\b\d{1,3} percent|\bone[ -]in[ -]\w+|\b\d\s?/\s?\d\b",
                    re.IGNORECASE)


def stated_pct(text):
    return bool(PCT_RE.search(ascii_quotes(text)))


def pct_rate(texts):
    return float(np.mean([stated_pct(t) for t in texts])) if texts else float("nan")


# -----------------------------------------------------------------------------
# queries
# -----------------------------------------------------------------------------


def tier_of(successes, group=8):
    if successes is None:
        return "open"
    if successes >= group - 2:
        return "confident"
    return "hedged" if successes > 0 else "unknown"


def cmd_queries(args):
    """Stratified questions from the RL pool, plus open-ended corpus queries."""
    from boundary_gen import ASK_ENTITY, blocklist, norm_q

    tokenizer = load_tokenizer()
    ents, qs = blocklist()
    pool = [r for r in read_jsonl(args.pool) if norm_q(r["question"]) not in qs]
    rng = random.Random(args.seed)

    # Quotas, not the pool's own distribution: half the pool is 0/8, and a
    # target set that is half non-answers trains a blanket refuser. Over-
    # conservatism is as easy to fall into as hallucination (arXiv:2312.07000).
    by_tier = {}
    for r in pool:
        by_tier.setdefault(tier_of(r["successes"], args.group), []).append(r)
    quotas = {"confident": args.n_confident, "hedged": args.n_hedged, "unknown": args.n_unknown}
    rows = []
    for tier, want in quotas.items():
        got = by_tier.get(tier, [])
        rng.shuffle(got)
        rows += [{"qid": f"qa{len(rows) + i:04d}", "kind": "qa", "tier": tier,
                  "question": r["question"], "answer": r["answer"],
                  "successes": r["successes"], "chunk_ids": r["chunk_ids"]}
                 for i, r in enumerate(got[:want])]
        print(f"  {tier:10s} {min(want, len(got))}/{want} available {len(got)}")

    # Open-ended: name something the chunk really defines, then ask about it.
    if args.n_open:
        corpus = load_corpus(tokenizer, args.files, args.root)
        model = load_model()
        key = jax.random.key(args.seed)
        seen, open_rows = set(), []
        while len(open_rows) < args.n_open:
            chunks = [corpus.sample_chunk(rng, args.chunk_min, args.chunk_max)
                      for _ in range(args.batch)]
            systems = [f"{args.description}\n\n{tokenizer.decode(c, skip_special_tokens=False)}"
                       for c in chunks]
            key, sub = jax.random.split(key)
            texts = generate(model, tokenizer,
                             [encode_user(tokenizer, s, ASK_ENTITY) for s in systems],
                             max_new=24, temperature=args.temperature, key=sub, batch=args.batch)
            for c, t in zip(chunks, texts):
                m = re.search(r"NAME:\s*`?([A-Za-z_][\w.]*)`?", t)
                if not m or len(open_rows) >= args.n_open:
                    continue
                name = m.group(1)
                if name in seen or name not in corpus.text or name.lower() in ents:
                    continue
                seen.add(name)
                open_rows.append({"qid": f"open{len(open_rows):04d}", "kind": "open",
                                  "tier": "open", "question": rng.choice(OPEN_Q).format(name=name),
                                  "answer": None, "successes": None, "chunk_ids": c.tolist(),
                                  "entity": name})
            print(f"  open {len(open_rows)}/{args.n_open}", flush=True)
        rows += open_rows

    write_jsonl(rows, args.out)
    print(f"  {len(rows)} queries: "
          + ", ".join(f"{t} {sum(r['tier'] == t for r in rows)}" for t in
                      ("confident", "hedged", "unknown", "open")))


# -----------------------------------------------------------------------------
# sample
# -----------------------------------------------------------------------------


def query_text(r):
    return QUERY.format(q=r["question"]) if r["kind"] == "qa" else r["question"]


def cmd_sample(args):
    """M responses per query from the CE cartridge -- the student's own memory."""
    from qwen_jax.cartridge import Cartridge

    tokenizer = load_tokenizer()
    rows = read_jsonl(args.queries)[: args.limit] if args.limit else read_jsonl(args.queries)
    model = load_model()
    cart = Cartridge.load(args.cartridge)
    prefix = cart.prefix(model.cache_dtype())
    prompts = [encode_suffix(tokenizer, query_text(r)) for r in rows for _ in range(args.m)]
    t0 = time.time()
    texts = generate(model, tokenizer, prompts, prefix=prefix, max_new=args.max_new,
                     temperature=args.temperature, key=jax.random.key(args.seed),
                     batch=args.batch, pad_to=64,
                     progress=lambda d, n: (d % (args.batch * 25) or
                                            print(f"  {d}/{n} samples "
                                                  f"({time.time() - t0:.0f}s)", flush=True)))
    out = [{"qid": r["qid"], "tier": r["tier"], "samples": texts[i * args.m:(i + 1) * args.m]}
           for i, r in enumerate(rows)]

    # How often did the student's own memory actually land on the gold answer?
    # The same forecast the RL reward uses, so the number that goes into the
    # target is the student's measured hit rate, not the teacher's impression.
    jobs, where = [], []
    for i, (r, o) in enumerate(zip(rows, out)):
        if r["kind"] != "qa" or not r.get("answer"):
            continue
        for s in o["samples"]:
            where.append(i)
            jobs.append(encode_user(tokenizer, READER_SYS,
                                    FORECAST.format(p=s, q=r["question"], a=r["answer"])))
    print(f"  {len(jobs)} gold-support forecasts", flush=True)
    ps = p_yes(model, tokenizer, jobs, batch=args.batch) if jobs else []
    for i, p in zip(where, ps):
        out[i].setdefault("support", []).append(float(p))
    for o in out:
        sup = o.get("support")
        o["freq"] = (float(np.mean([s > 0.5 for s in sup])) if sup else None)
        o["style"] = style_of(o["freq"])
    write_jsonl(out, args.out)
    flat = [t for o in out for t in o["samples"]]
    styles = {}
    for o in out:
        styles[o["style"]] = styles.get(o["style"], 0) + 1
    print(f"  {len(flat)} samples, hedge-marker rate {hedge_rate(flat):.3f} "
          f"({time.time() - t0:.0f}s)")
    print(f"  target styles from sample frequency: {styles}")


# -----------------------------------------------------------------------------
# merge
# -----------------------------------------------------------------------------


def merge_prompt(row, rec):
    samples = rec["samples"]
    m = len(samples)
    freq = rec.get("freq")
    n_ok = int(round((freq or 0) * m))
    # The stated percentage is the student's own hit rate, rounded to the
    # nearest 5 -- verbalized confidences collapse onto round anchors anyway
    # (arXiv:2604.23333), so anchoring them deliberately costs nothing.
    pct = int(5 * round((freq if freq is not None else 0.5) * 100 / 5))
    rule = RULES[rec["style"]].format(m=m, n_ok=n_ok, pct=pct)
    block = "\n".join(f"<attempt {i + 1}>\n{s}\n</attempt {i + 1}>"
                      for i, s in enumerate(samples))
    return (MERGE_HEAD.format(q=row["question"], m=m, samples=block)
            + MERGE_TAIL.format(rule=rule))


def cmd_merge(args):
    """The teacher, holding the chunk, writes one consensus reply per query."""
    from qwen_jax.selfstudy import Example, save_examples

    tokenizer = load_tokenizer()
    recs = {r["qid"]: r for r in read_jsonl(args.samples)}
    rows = [{**r, "style": recs[r["qid"]]["style"], "freq": recs[r["qid"]]["freq"]}
            for r in read_jsonl(args.queries) if r["qid"] in recs]
    # `--styles` re-merges one tier in place: a rule the teacher would not open
    # is worth rewriting without paying for the tiers that already distil.
    keep = ([] if not args.into else
            [t for t in read_jsonl(args.into) if t["style"] not in (args.styles or [])])
    if args.styles:
        rows = [r for r in rows if r["style"] in args.styles]
        print(f"  re-merging {len(rows)} rows of style {args.styles}; "
              f"keeping {len(keep)} existing targets")
    model = load_model()
    t0 = time.time()

    def run(rs, grounded):
        prompts = []
        for r in rs:
            chunk = tokenizer.decode(r["chunk_ids"], skip_special_tokens=False)
            system = f"{args.description}\n\n{chunk}" if grounded else args.description
            user = merge_prompt(r, recs[r["qid"]])
            if not grounded:      # Band-faithful: consensus with nothing to check it against
                user = user.replace("the document above contradicts what the attempts say, "
                                    "correct it or leave it out; never repeat a claim the "
                                    "document shows is wrong",
                                    "the attempts contradict each other, prefer what most of "
                                    "them say")
                user = user.replace(" and the document confirms", "")
            prompts.append(encode_user(tokenizer, system, user))
        return generate(model, tokenizer, prompts, max_new=args.max_new,
                        temperature=args.temperature, key=jax.random.key(args.seed),
                        batch=args.batch, pad_to=256,
                        progress=lambda d, n: (d % (args.batch * 10) or
                                               print(f"  merged {d}/{n} "
                                                     f"({time.time() - t0:.0f}s)", flush=True)))

    merged = run(rows, True)
    targets, dropped = [], {"empty": 0, "reference": 0, "leaked": 0}
    for r, raw in zip(rows, merged):
        t = trim_sentences(raw, args.max_words)
        if not t:
            dropped["empty"] += 1
        elif REFERENCE_RE.search(t):
            dropped["reference"] += 1        # the teacher talking about the page
        elif r["style"] == "unknown" and (IDENT_RE.search(t) or leaks(t, r.get("answer"))):
            dropped["leaked"] += 1           # a non-answer that names something
        else:
            targets.append({**r, "target": t, "raw_target": raw})
    print(f"  {len(targets)}/{len(rows)} grounded merges kept ({time.time() - t0:.0f}s); "
          f"dropped {dropped}")
    targets += keep

    # Did the teacher hand back the answer it was told to withhold? The one way
    # this method can quietly become knowledge distillation again.
    leaked = [t for t in targets
              if t["style"] == "unknown" and leaks(t["target"], t.get("answer"))]
    by_style = {}
    for t in targets:
        by_style.setdefault(t["style"], []).append(t["target"])
    print("  by style: " + ", ".join(
        f"{k} n={len(v)} hedge {hedge_rate(v):.2f} stated-pct {pct_rate(v):.2f}"
        for k, v in sorted(by_style.items())))
    print(f"  gold leaked into {len(leaked)}/{sum(t['style'] == 'unknown' for t in targets)} "
          f"`unknown`-style targets")

    # The Band-faithful ablation: same merge, no chunk to check the samples against.
    abl_rows = rows[: args.n_ungrounded]
    abl = run(abl_rows, False) if args.n_ungrounded else []
    ungrounded = [{**r, "target": t} for r, t in zip(abl_rows, abl) if t]

    write_jsonl(targets, args.out)
    if ungrounded:
        write_jsonl(ungrounded, Path(args.out).with_name("targets-ungrounded.jsonl"))
        print(f"  ungrounded ablation: {len(ungrounded)} merges, hedge rate "
              f"{hedge_rate([u['target'] for u in ungrounded]):.2f} vs grounded "
              f"{hedge_rate([t['target'] for t in targets[:len(ungrounded)]]):.2f}")

    # Teacher-support diagnostic. Distillation matches the teacher's next-token
    # distribution along the stored conversation, and the teacher is shown only
    # the chunk and the question. If it will not open a hedge, the hedge cannot
    # survive being distilled, whatever the target says (the boundary lesson).
    from boundary_gen import teacher_support

    ns = argparse.Namespace(description=args.description, batch=args.diag_batch)
    tri = lambda rs: [(np.asarray(r["chunk_ids"]), query_text(r), r["target"]) for r in rs]
    diag, per_tier = {}, {}
    sub = targets[: args.n_diag]
    diag["target"] = teacher_support(model, tokenizer, tri(sub), ns)
    diag["target+note"] = teacher_support(model, tokenizer, tri(sub), ns, note=TEACHER_NOTE)
    for tier in ("confident", "probable", "possible", "unknown", "open"):
        rs = [t for t in targets if t["style"] == tier][: args.n_diag]
        if rs:
            per_tier[tier] = {
                "plain": teacher_support(model, tokenizer, tri(rs), ns),
                "note": teacher_support(model, tokenizer, tri(rs), ns, note=TEACHER_NOTE)}
    if Path(args.plain).exists():
        from qwen_jax.selfstudy import load_examples as _le

        plain = _le(args.plain)[: args.n_diag]
        diag["plain_answer"] = teacher_support(
            model, tokenizer,
            [(np.asarray(e.chunk_ids), e.user, e.assistant) for e in plain], ns)
    print("\nteacher support for the first target token (P and top-1 match):")
    for k, d in diag.items():
        print(f"  {k:16s} mean {d['mean']:.4f}  median {d['median']:.4f}  "
              f"top-1 {d['top1_match']:.0%}  (n={d['n']})")
    for tier, d in per_tier.items():
        print(f"  {tier:10s} plain median {d['plain']['median']:.4f} -> "
              f"with note {d['note']['median']:.4f}  "
              f"(top-1 {d['plain']['top1_match']:.0%} -> {d['note']['top1_match']:.0%})")
    if "plain_answer" in diag:
        for k in ("target", "target+note"):
            ratio = diag["plain_answer"]["median"] / max(diag[k]["median"], 1e-9)
            diag[f"ratio_plain_over_{k}"] = float(ratio)
            print(f"  plain-answer / {k} median support ratio: {ratio:.1f}x "
                  f"(>=10x means the target cannot survive distillation as-is)")
    diag["per_tier"] = {k: {n: {m: v[n][m] for m in ("mean", "median", "top1_match", "n")}
                            for n in v} for k, v in per_tier.items()}
    Path(args.out).with_suffix(".teacher.json").write_text(json.dumps(diag, indent=1))

    # Stored as Examples: user -> target. The merge instruction and the samples
    # are thrown away, exactly as boundary_gen throws away its refusal prompt.
    note = "" if args.no_teacher_note else NOTE_VARIANTS[args.note]
    save_examples([Example(chunk_ids=r["chunk_ids"], user=query_text(r), assistant=r["target"],
                           seed_kind=f"summary_{r['style']}", teacher_note=note)
                   for r in targets],
                  Path(args.out).with_name("examples.jsonl"))
    for style in ("confident", "probable", "possible", "unknown", "open"):
        for t in [x for x in targets if x["style"] == style][:2]:
            print(f"\n[{style} freq={t['freq']}] {t['question'][:90]}\n  -> {t['target'][:320]}")


# -----------------------------------------------------------------------------
# mix
# -----------------------------------------------------------------------------


# A note that tells the teacher to hedge lifts the hedged targets and sinks the
# plain ones, so the choice is per example, not per run -- `teacher_note` is an
# Example field precisely so it can vary. `mix` overrides these from a measured
# notes.json when one exists.
NOTE_BY_STYLE = {"confident": "support", "probable": "house_style",
                 "possible": "house_style", "unknown": "house_style", "open": "support"}


def cmd_notes(args):
    """Which teacher-side note, if any, lets the hedged targets be distilled?

    Distillation matches the teacher's distribution along the stored
    conversation, and the teacher sees only the chunk and the question. If it
    will not open a hedge, no amount of hedging in the target survives. This
    searches the note that goes in the teacher's system prompt -- and only the
    teacher's -- for the one that gives the targets the most support.
    """
    from boundary_gen import teacher_support
    from qwen_jax.selfstudy import load_examples

    tokenizer = load_tokenizer()
    targets = read_jsonl(args.targets)
    model = load_model()
    ns = argparse.Namespace(description=args.description, batch=args.batch)
    styles = ["confident", "probable", "possible", "unknown", "open"]
    pick = {s: [t for t in targets if t["style"] == s][: args.n] for s in styles}
    tri = lambda rs: [(np.asarray(r["chunk_ids"]), query_text(r), r["target"]) for r in rs]

    ref = None
    if Path(args.plain).exists():
        plain = load_examples(args.plain)[: args.n]
        ref = teacher_support(model, tokenizer,
                              [(np.asarray(e.chunk_ids), e.user, e.assistant) for e in plain], ns)
        print(f"reference: plain self-study answers, median support {ref['median']:.4f}, "
              f"top-1 {ref['top1_match']:.0%}")

    out = {}
    print(f"\n{'note':14s}" + "".join(f"{s[:9]:>11s}" for s in styles) + f"{'all':>11s}"
          + f"{'top-1':>8s}{'ratio':>8s}")
    for name, note in NOTE_VARIANTS.items():
        row, allp = {}, []
        for s in styles:
            if not pick[s]:
                row[s] = float("nan")
                continue
            d = teacher_support(model, tokenizer, tri(pick[s]), ns, note=note)
            row[s] = d["median"]
            allp += d["p"]
        med = float(np.median(allp)) if allp else float("nan")
        top1 = teacher_support(model, tokenizer,
                               tri([t for s in styles for t in pick[s]]), ns, note=note)
        ratio = (ref["median"] / max(med, 1e-12)) if ref else float("nan")
        out[name] = {"by_style": row, "median": med, "top1_match": top1["top1_match"],
                     "mean": float(np.mean(allp)) if allp else float("nan"),
                     "ratio_plain_over_target": ratio}
        print(f"  {name:12s}" + "".join(f"{row[s]:11.4f}" for s in styles)
              + f"{med:11.4f}{top1['top1_match']:8.0%}{ratio:8.0f}")
    print("\n  median = P(first target token) under a teacher shown only the chunk and "
          "the question.\n  ratio = plain-answer support / this; >=10x is the "
          "materiality bar from the boundary run.")
    Path(args.out).write_text(json.dumps(
        {"reference_plain": {k: ref[k] for k in ("mean", "median", "top1_match")} if ref else None,
         "notes": out, "n_per_style": {s: len(v) for s, v in pick.items()}}, indent=1))
    best = max(out, key=lambda k: out[k]["median"])
    print(f"\nbest note by median support: {best}\nwrote {args.out}")


def cmd_mix(args):
    """Merged targets plus the plain self-study the CE cartridge already learned."""
    from boundary_gen import blocklist, norm_q
    from qwen_jax.selfstudy import load_examples, save_examples

    summary = load_examples(args.summary)
    plain = [e for e in load_examples(args.plain) if norm_q(e.user) not in blocklist()[1]]

    # Per-style teacher notes, measured if `notes` has run, defaulted if not.
    chosen = dict(NOTE_BY_STYLE)
    if args.notes and Path(args.notes).exists():
        d = json.loads(Path(args.notes).read_text())["notes"]
        for style in chosen:
            best = max(d, key=lambda n: (d[n]["by_style"].get(style) or -1.0))
            if (d[best]["by_style"].get(style) or 0) > 0:
                chosen[style] = best
        print(f"teacher notes from {args.notes}: {chosen}")
    for e in summary:
        style = e.seed_kind.replace("summary_", "")
        e.teacher_note = NOTE_VARIANTS[chosen.get(style, "support")]
    rng = random.Random(args.seed)
    n = args.n or 2 * len(summary)                       # every target, matched 50/50
    k = min(len(summary), n // 2)
    rng.shuffle(summary)
    take_plain = [plain[i % len(plain)] for i in range(n - k)]
    mixed = summary[:k] + take_plain
    rng.shuffle(mixed)
    save_examples(mixed, args.out)
    kinds = {}
    for e in mixed:
        kinds[e.seed_kind or "plain"] = kinds.get(e.seed_kind or "plain", 0) + 1
    print(f"wrote {len(mixed)} examples to {args.out}: {kinds}")
    print(f"  {k} summary targets ({k / len(mixed):.0%}), {len(mixed) - k} plain "
          f"({len(set(id(e) for e in take_plain))} distinct)")
    print(f"  one epoch at --batch {args.batch} is {len(mixed) // args.batch} steps")


# -----------------------------------------------------------------------------
# train / eval
# -----------------------------------------------------------------------------


def run(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=REPO, check=True, **kw)


def cmd_train(args):
    """A post-train: continue distillation from the CE cartridge, lower LR."""
    run([sys.executable, REPO / "scripts/cartridge.py", "train",
         "--data", args.data, "--heldout", args.heldout, "--resume", args.cartridge,
         "--steps", args.steps, "--lr", args.lr, "--warmup", args.warmup,
         "--batch", args.batch, "--eval-every", args.eval_every,
         "--log-every", args.log_every, "--save-every", args.save_every,
         "--out", args.out])


def read_rows(path, condition=None):
    rows = read_jsonl(path)
    return [r for r in rows if condition is None or r["condition"] == condition]


def reward_delta(model, tokenizer, rows, batch):
    """What the RL relaunch will see: R0 vs the audited F_prior4 reward."""
    from reward_audit import PHRASINGS, FORECAST_HEAD, cand_D, logit_gap, sig

    items = [{"question": r["question"], "answer": r["answer"], "response": r["response"]}
             for r in rows]
    p0 = sig(logit_gap(model, tokenizer,
                       [encode_user(tokenizer, READER_SYS,
                                    FORECAST_HEAD.format(p=i["response"], q=i["question"],
                                                         a=i["answer"]) + PHRASINGS[0])
                        for i in items], batch=batch))
    scores, _ = cand_D(model, tokenizer, items, argparse.Namespace(batch=batch, prior_k=[4]))
    lg = lambda p: np.log(np.clip(np.asarray(p, np.float64), 1e-3, 1.0))
    return {"n": len(items), "R0": lg(p0), "F_prior4": lg(scores["F_prior4"])}


def cmd_eval(args):
    """Reader score, three cells, hedging, uncertain accuracy, KL, reward delta."""
    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    rs = [sys.executable, REPO / "scripts/reader_score.py"]
    resp, read = d / f"resp-{args.label}.jsonl", d / f"read-{args.label}.jsonl"
    if not args.skip_respond:
        run(rs + ["respond", "--qa", args.qa, "--conditions", "trained",
                  "--cartridge", args.cartridge, "--label", args.label, "--out", resp])
        run(rs + ["read", "--responses", resp, "--out", read])
        run(rs + ["score", "--read", read, "--out", d / f"scores-{args.label}.json"])

    # One read file holding both conditions, so `cells` tabulates them together.
    ce_read = [{**r, "condition": "ce"} for r in read_rows(args.ce_read, "trained")]
    both = ce_read + read_jsonl(read)
    write_jsonl(both, d / "read-both.jsonl")
    run([sys.executable, REPO / "scripts/boundary_gen.py", "cells",
         "--read", d / "read-both.jsonl", "--retention", args.retention,
         "--conditions", "ce", args.label, "--out", d / "cells.json"])

    # Hedging, and LoGU uncertain accuracy: of the hedged answers, how many were
    # actually wrong. Cosmetic hedging hedges everywhere and scores near the base
    # rate; retention-aware hedging concentrates on what it gets wrong.
    from boundary_gen import cell_of, retention_split

    retained, _ = retention_split(read_jsonl(args.retention), k_min=2)
    base_resp = [{**r, "condition": "base"} for r in read_rows(args.ce_resp, "none")]
    summary = {}
    for name, rows in (("base", base_resp), ("ce", ce_read), (args.label, read_jsonl(read))):
        ins = [r for r in rows if r["kind"] == "in"]
        graded = "correct" in (rows[0] if rows else {})
        hed = [r for r in ins if hedged(r["response"])] if graded else []
        wrong_hed = [r for r in hed if not r["correct"]]
        conf = [r for r in ins if not hedged(r["response"])] if graded else []
        summary[name] = {
            "n_in": len(ins), "hedge_rate": hedge_rate([r["response"] for r in ins]),
            "hedge_rate_all": hedge_rate([r["response"] for r in rows]),
            "stated_pct_rate": pct_rate([r["response"] for r in ins]),
            "uncertain_accuracy": (len(wrong_hed) / len(hed)) if hed else float("nan"),
            "confident_wrong_rate": (float(np.mean([not r["correct"] for r in conf]))
                                     if conf else float("nan")),
            "hedge_by_cell": {
                c: hedge_rate([r["response"] for r in rows if cell_of(r, retained) == c])
                for c in ("in-retained", "in-blurred", "out")},
        }
        if graded:
            summary[name]["acc_in"] = float(np.mean([r["correct"] for r in ins]))
    print("\nhedging (marker rate over in-corpus responses)")
    print(f"  {'condition':10s}{'hedge':>8s}{'stated%':>9s}{'hedge-all':>11s}{'unc-acc':>9s}"
          f"{'conf-wrong':>12s}{'retained':>10s}{'blurred':>9s}{'out':>7s}")
    for k, v in summary.items():
        c = v["hedge_by_cell"]
        print(f"  {k:10s}{v['hedge_rate']:8.3f}{v['stated_pct_rate']:9.3f}"
              f"{v['hedge_rate_all']:11.3f}"
              f"{v['uncertain_accuracy']:9.3f}{v['confident_wrong_rate']:12.3f}"
              f"{c['in-retained']:10.3f}{c['in-blurred']:9.3f}{c['out']:7.3f}")

    # Held-out KL before the model is resident here, so the subprocess has the
    # card to itself.
    for name, cart in (("ce", args.ce_cartridge), (args.label, args.cartridge)):
        print(f"\n=== held-out KL, {name}")
        run([sys.executable, REPO / "scripts/cartridge.py", "eval",
             "--cartridge", cart, "--heldout", args.heldout])

    # Reward delta on a sample of in-corpus responses, both conditions.
    tokenizer = load_tokenizer()
    model = load_model()
    rng = random.Random(0)
    rewards = {}
    for name, rows in (("ce", ce_read), (args.label, read_jsonl(read))):
        ins = [r for r in rows if r["kind"] == "in" and r.get("answer")]
        pick = rng.sample(ins, min(args.n_reward, len(ins)))
        rw = reward_delta(model, tokenizer, pick, args.batch)
        rewards[name] = {"n": rw["n"],
                         "R0_mean": float(rw["R0"].mean()),
                         "F_prior4_mean": float(rw["F_prior4"].mean()),
                         "R0_floor_frac": float(np.mean(rw["R0"] <= np.log(1e-3) + 1e-9)),
                         "F_interior_frac": float(np.mean((rw["F_prior4"] > np.log(1e-3) + 0.1)
                                                          & (rw["F_prior4"] < -0.1))),
                         "questions": [r["question"] for r in pick]}
    print("\nreward on held-out responses (what the RL relaunch will see)")
    print(f"  {'condition':10s}{'R0':>9s}{'R0 floor':>10s}{'F_prior4':>10s}{'F interior':>12s}")
    for k, v in rewards.items():
        print(f"  {k:10s}{v['R0_mean']:9.2f}{v['R0_floor_frac']:10.2f}"
              f"{v['F_prior4_mean']:10.2f}{v['F_interior_frac']:12.2f}")

    out = {"hedging": summary, "rewards": rewards,
           "cells": json.loads((d / "cells.json").read_text())["cells"]}
    (d / f"summary-{args.label}.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {d / f'summary-{args.label}.json'}")


# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--files", nargs="*")
        s.add_argument("--root")
        s.add_argument("--description", default=DESCRIPTION)

    q = sub.add_parser("queries")
    common(q)
    q.add_argument("--pool", default=str(POOL))
    q.add_argument("--group", type=int, default=8)
    q.add_argument("--n-confident", type=int, default=160)
    q.add_argument("--n-hedged", type=int, default=160)
    q.add_argument("--n-unknown", type=int, default=80)
    q.add_argument("--n-open", type=int, default=100)
    q.add_argument("--chunk-min", type=int, default=512)
    q.add_argument("--chunk-max", type=int, default=1536)
    q.add_argument("--temperature", type=float, default=0.9)
    q.add_argument("--batch", type=int, default=8)
    q.add_argument("--seed", type=int, default=11)
    q.add_argument("--out", default=str(OUT_DIR / "queries.jsonl"))

    s = sub.add_parser("sample")
    s.add_argument("--queries", default=str(OUT_DIR / "queries.jsonl"))
    s.add_argument("--cartridge", default=str(TRAINED))
    s.add_argument("--m", type=int, default=6)
    s.add_argument("--max-new", type=int, default=192)
    s.add_argument("--temperature", type=float, default=0.9)
    s.add_argument("--batch", type=int, default=16)
    s.add_argument("--limit", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default=str(OUT_DIR / "samples.jsonl"))

    m = sub.add_parser("merge")
    common(m)
    m.add_argument("--queries", default=str(OUT_DIR / "queries.jsonl"))
    m.add_argument("--samples", default=str(OUT_DIR / "samples.jsonl"))
    m.add_argument("--max-new", type=int, default=160)
    m.add_argument("--max-words", type=int, default=70)
    m.add_argument("--temperature", type=float, default=0.7)
    m.add_argument("--batch", type=int, default=8)
    m.add_argument("--note", default="support", choices=list(NOTE_VARIANTS),
                   help="teacher-side note stored on each Example")
    m.add_argument("--diag-batch", type=int, default=4)
    m.add_argument("--n-diag", type=int, default=48)
    m.add_argument("--n-ungrounded", type=int, default=30)
    m.add_argument("--styles", nargs="+", choices=list(RULES),
                   help="re-merge only these styles")
    m.add_argument("--into", help="existing targets file to merge the result into")
    m.add_argument("--no-teacher-note", action="store_true")
    m.add_argument("--plain", default=str(PLAIN))
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default=str(OUT_DIR / "targets.jsonl"))

    n = sub.add_parser("notes")
    common(n)
    n.add_argument("--targets", default=str(OUT_DIR / "targets.jsonl"))
    n.add_argument("--plain", default=str(PLAIN))
    n.add_argument("--n", type=int, default=16, help="targets per style")
    n.add_argument("--batch", type=int, default=8)
    n.add_argument("--out", default=str(OUT_DIR / "notes.json"))

    x = sub.add_parser("mix")
    x.add_argument("--summary", default=str(OUT_DIR / "examples.jsonl"))
    x.add_argument("--plain", default=str(PLAIN))
    x.add_argument("--notes", default=str(OUT_DIR / "notes.json"),
                   help="measured teacher-note support, to pick the note per style")
    x.add_argument("--steps", type=int, default=250)
    x.add_argument("--batch", type=int, default=2)
    x.add_argument("--n", type=int, help="examples in the mix (default: 2 epochs of --steps)")
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--out", default=str(OUT_DIR / "train.jsonl"))

    t = sub.add_parser("train")
    t.add_argument("--data", default=str(OUT_DIR / "train.jsonl"))
    t.add_argument("--heldout", default=str(HELDOUT))
    t.add_argument("--cartridge", default=str(TRAINED), help="the cartridge to continue from")
    t.add_argument("--steps", type=int, default=250)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--warmup", type=int, default=20)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--eval-every", type=int, default=50)
    t.add_argument("--save-every", type=int, default=50)
    t.add_argument("--out", default=str(OUT_DIR / "summary.safetensors"))

    e = sub.add_parser("eval")
    e.add_argument("--cartridge", default=str(OUT_DIR / "summary.safetensors"))
    e.add_argument("--ce-cartridge", default=str(TRAINED))
    e.add_argument("--label", default="summary")
    e.add_argument("--qa", default=str(QA_EVAL))
    e.add_argument("--ce-read", default=str(CE_READ))
    e.add_argument("--ce-resp", default=str(CE_RESP))
    e.add_argument("--retention", default=str(RETENTION))
    e.add_argument("--heldout", default=str(HELDOUT))
    e.add_argument("--n-reward", type=int, default=30)
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--skip-respond", action="store_true")
    e.add_argument("--out", default=str(OUT_DIR))

    args = p.parse_args()
    {"queries": cmd_queries, "sample": cmd_sample, "merge": cmd_merge, "notes": cmd_notes,
     "mix": cmd_mix, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
