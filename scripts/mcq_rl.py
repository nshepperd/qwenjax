"""Calibration RL on a cartridge from multiple-choice questions, with an exact gradient.

    python scripts/mcq_rl.py gen     --n 0 --out runs/mcq/raw.jsonl
    python scripts/mcq_rl.py check   --raw runs/mcq/raw.jsonl --out runs/mcq/
    python scripts/mcq_rl.py gen-out --n 300 --out runs/mcq/out-raw.jsonl
    python scripts/mcq_rl.py mix     --out runs/mcq/v2
    python scripts/mcq_rl.py eval    base=runs/attnmse/x.safetensors [--init] [--none] [--icl]
    python scripts/mcq_rl.py train   --cartridge runs/attnmse/x.safetensors --warm-steps 40 --steps 60 --out runs/mcq/run4/

The policy -- frozen model behind a trainable cartridge -- answers a
multiple-choice question about the corpus with a letter and a confidence,
`B, 70% confidence`, and is rewarded

    R = 2 [letter right] - (confidence - [letter right])^2 - penalty [wrong answer letter]

The flat bonus is what makes the right letter worth picking (Brier alone is
maximised by naming an option you are sure is wrong, at 0%); the Brier term is
proper, so the best stated confidence is the true chance of being right.

There is no sampling. The possible responses are enumerated and the expected
reward is differentiated directly (`qwen_jax.mcq`), which is the gradient GRPO
estimates from a group of rollouts, without the variance. Each step costs one
short teacher-forced row per letter per question.

Questions are written by the model itself with a marked 40-token span of the
corpus in front of it (as in pointed self-study), then kept only if the model
with the source chunk in context picks the intended answer: an ambiguous
question or a wrong key would train noise. What the model gets right with no
context at all is recorded (`p_none`), not removed -- being sure of what you
know from priors is part of being calibrated.

v2 (`mix`, and `train --warm-steps`): the first runs (runs/mcq/run1..3) taught
the cartridge to guess. With four options and nothing lost on a wrong letter
said at 0%, free-form answers abstained less at unchanged precision, and the
stated confidence barely moved because the model never says anything below
80%. So: a fifth letter that is simply the key of out-of-corpus questions
(`gen-out`), under the same reward; and a supervised warm start that writes
the policy's own letter certainty, recalibrated, into the stated number before
any RL. `--wrong-penalty` (a cost on a wrong answer letter, so that abstaining
beats a weak guess) was tried against the plain reward and made no difference;
it defaults to 0.

v3: the fifth letter is worded as a factual claim (see ABSTAIN_OPTION), and the
warm-start targets are built on-policy for the letter the current cartridge
picks, floored at 1/L, with the abstain letter left to the reward. What is not
solved: the warm start pulls the letter choice toward abstaining through the
shared cartridge even though its loss never touches the letter softmax, and the
whole procedure costs ~0.10 of held-out distillation KL. Board: mtx3o4.

Without the warm start (run7) the letter improves and the stated number never
leaves 98-100%. Measured with scripts/mcq_graddiag.py: the part of the policy
gradient that reaches the cartridge through the confidence logits is ~1.6% of
the letter's part, and a confidence temperature, held or annealed
(`--conf-temp-anneal`, run8), leaves that ratio where it was.
`--conf-grad-match R` rescales the confidence part to R times the letter's.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import jax
import jax.numpy as jnp
import numpy as np

from cartridge import DESCRIPTION, batches_from, init_cartridge, load_corpus, load_model, load_tokenizer
from reader_score import batched, encode_suffix, encode_user, generate, last_logits, named, read_jsonl, write_jsonl

OUT_DIR = REPO / "runs/mcq"

# -----------------------------------------------------------------------------
# prompts
# -----------------------------------------------------------------------------

ANSWER = (
    "{q}\n\n{options}\n\n"
    "Reply with the letter of the correct option and your confidence that it is "
    "correct, as a multiple of ten percent, in exactly this form and nothing "
    "else: <letter>, <confidence>% confidence"
)
# A factual claim, so that "confidence that this option is correct" has a truth
# value. The first wording, "I don't know, or it is not in the codebase", made
# the base model answer "E, 0% confidence" while certain of E and right: it read
# the number as how much it knew (Emily, 2026-09-20; runs/mcq/v2/calibration.png).
ABSTAIN_OPTION = "None of these: what the question asks about is not in the codebase"


def n_letters(item) -> int:
    return 5 if item.get("abstain") else 4


def render(item) -> str:
    options = list(item["options"]) + ([ABSTAIN_OPTION] if item.get("abstain") else [])
    opts = "\n".join(f"{l}) {o}" for l, o in zip("ABCDE", options))
    return ANSWER.format(q=item["question"], options=opts)


def write_prompt(open_mark, close_mark):
    return (
        "Write one multiple-choice question about the part of the document above "
        f"between {open_mark} and {close_mark}, for someone who knows this codebase "
        "well but does not have the document in front of them.\n"
        "- The question must name the file, function, class, constant or field it is "
        "about. Never mention that anything is marked, and never refer to 'the "
        "document' or 'the code above'.\n"
        "- It must have exactly one correct answer, which the marked part states, and "
        "three wrong answers that are the same kind of thing as the correct one -- "
        "another real name from this codebase, a nearby number, a similar type -- "
        "plausible to someone who half remembers, and definitely wrong.\n"
        "- Each answer is at most twelve words. Do not make the correct answer stand "
        "out by length or detail.\n"
        "Reply in exactly this format and nothing else:\n"
        "QUESTION: <question>\nCORRECT: <correct answer>\nWRONG: <wrong answer>\n"
        "WRONG: <wrong answer>\nWRONG: <wrong answer>"
    )


OPTIONS_FOR = (
    "Here is a question someone asked about this codebase:\n\n{q}\n\n"
    "Write four different short answers someone might plausibly give, of the kind "
    "a real answer would be -- names in the style of this codebase, plausible "
    "numbers or types -- each at most twelve words. Do not comment on whether the "
    "question can be answered.\n"
    "Reply in exactly this format and nothing else:\n"
    "OPTION: <answer>\nOPTION: <answer>\nOPTION: <answer>\nOPTION: <answer>"
)


DENIAL = re.compile(r"\b(no such|does not|doesn't|not exist|not defined|not present|not implemented|no method|"
                    r"no function|not supported|isn't|is not|none of|lacks?|without)\b|^no\b", re.I)


def _clean(s):
    return s.strip().strip('"').strip("`").strip()


def parse_mcq(text):
    q = re.search(r"QUESTION:\s*(.+)", text)
    c = re.search(r"CORRECT:\s*(.+)", text)
    w = re.findall(r"WRONG:\s*(.+)", text)
    if not q or not c or len(w) < 3:
        return None
    question, correct, wrong = _clean(q.group(1)), _clean(c.group(1)), [_clean(x) for x in w[:3]]
    answers = [correct] + wrong
    if len(question) < 15 or any(not a or len(a.split()) > 14 for a in answers):
        return None
    if len({a.lower() for a in answers}) < 4:
        return None
    return question, correct, wrong


def parse_options(text):
    o = [_clean(x) for x in re.findall(r"OPTION:\s*(.+)", text)[:4]]
    if len(o) < 4 or any(not a or len(a.split()) > 14 for a in o) or len({a.lower() for a in o}) < 4:
        return None
    return o


# -----------------------------------------------------------------------------
# gen
# -----------------------------------------------------------------------------


def cmd_gen(args):
    from qwen_jax.selfstudy import MARK_CLOSE, MARK_OPEN, corpus_spans

    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, args.files, args.root)
    spans = corpus_spans(len(corpus), args.span_tokens)
    rng = random.Random(args.seed)
    rng.shuffle(spans)
    model = load_model()
    key = jax.random.key(args.seed)
    ids, n = corpus.ids, len(corpus.ids)
    dec = lambda x: tokenizer.decode(x, skip_special_tokens=False)
    ask = write_prompt(MARK_OPEN, MARK_CLOSE)
    rows, stats, t0 = [], dict(tried=0, unparsed=0, unanchored=0, leaked=0), time.time()
    todo = spans[: args.n * 2] if args.n else spans
    for group, k in batched(todo, args.batch):
        if args.n and len(rows) >= args.n:
            break
        jobs = []
        for s, e in group[:k]:
            length = min(rng.randint(args.chunk_min, args.chunk_max), n)
            lo, hi = max(0, e - length), min(s, n - length)
            start = rng.randint(lo, hi) if hi >= lo else max(0, min(s, n - length))
            chunk = ids[start:start + length]
            text = (dec(chunk[: s - start]) + MARK_OPEN + dec(chunk[s - start: e - start])
                    + MARK_CLOSE + dec(chunk[e - start:]))
            jobs.append(dict(span=[int(s), int(e)], chunk=chunk,
                             prompt=encode_user(tokenizer, f"{args.description}\n\n{text}", ask)))
        key, sub = jax.random.split(key)
        texts = generate(model, tokenizer, [j["prompt"] for j in jobs], max_new=args.max_new,
                         temperature=args.temperature, key=sub, batch=args.batch)
        for job, text in zip(jobs, texts):
            stats["tried"] += 1
            parsed = parse_mcq(text)
            if not parsed:
                stats["unparsed"] += 1
                continue
            question, correct, wrong = parsed
            blob = " ".join([question, correct] + wrong).lower()
            if "marked" in blob or "the document" in blob or "code above" in blob:
                stats["leaked"] += 1
                continue
            if not any(nm in corpus.text for nm in named(question)):
                stats["unanchored"] += 1
                continue
            options = [correct] + wrong
            order = list(range(4))
            rng.shuffle(order)
            rows.append(dict(question=question, options=[options[i] for i in order],
                             gold=order.index(0), span=job["span"], chunk_ids=job["chunk"].tolist()))
        print(f"  {len(rows)} kept of {stats['tried']}  {stats}  ({time.time() - t0:.0f}s)", flush=True)
    write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} questions to {args.out}")
    for r in rows[:4]:
        print(f"\nQ: {r['question']}\n" + "\n".join(
            f"  {'*' if i == r['gold'] else ' '} {l}) {o}" for i, (l, o) in enumerate(zip("ABCD", r["options"]))))


def cmd_gen_out(args):
    """Questions about things the corpus does not contain. The key is the abstain letter.

    Two stages, because asked for a multiple-choice question outright the model
    writes "which function does X?" with real names as options -- an answerable
    question that names nothing. So the question comes from the reader
    benchmark's near-miss prompt, which earns its label by naming something
    absent from the corpus text, and the options are written for it afterwards.
    """
    from reader_score import ASK_OUT, parse_qa

    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, args.files, args.root)
    model = load_model()
    rng = random.Random(args.seed)
    key = jax.random.key(args.seed)
    dec = lambda c: tokenizer.decode(c, skip_special_tokens=False)
    held = set()
    for path in args.decontam:
        if path and Path(path).exists():
            for r in read_jsonl(path):
                held.update(named(r.get("question", "")))
                held.update(str(x).rsplit(".", 1)[-1] for x in [r.get("entity"), r.get("qual")] if x)
    rows, seen, t0 = [], set(), time.time()
    stats = dict(tried=0, unparsed=0, real=0, held=0, dup=0, no_options=0)
    while len(rows) < args.n and stats["tried"] < args.n * 8:
        chunks = [corpus.sample_chunk(rng, args.chunk_min, args.chunk_max) for _ in range(args.batch)]
        systems = [f"{args.description}\n\n{dec(c)}" for c in chunks]
        key, k1, k2 = jax.random.split(key, 3)
        texts = generate(model, tokenizer, [encode_user(tokenizer, s, ASK_OUT) for s in systems],
                         max_new=96, temperature=0.9, key=k1, batch=args.batch)
        cands = []
        for system, text in zip(systems, texts):
            stats["tried"] += 1
            parsed = parse_qa(text, False)
            if not parsed:
                stats["unparsed"] += 1
                continue
            question = parsed[0]
            missing = [nm for nm in named(question) if nm not in corpus.text]
            if not missing:
                stats["real"] += 1
            elif set(missing) & held:
                stats["held"] += 1
            elif any(m in seen for m in missing):
                stats["dup"] += 1
            else:
                seen.update(missing)
                cands.append((system, question, missing))
        if cands:
            texts = generate(model, tokenizer,
                             [encode_user(tokenizer, s, OPTIONS_FOR.format(q=q)) for s, q, _ in cands],
                             max_new=args.max_new, temperature=args.temperature, key=k2, batch=args.batch)
            for (_, question, missing), text in zip(cands, texts):
                options = parse_options(text)
                if not options:
                    stats["no_options"] += 1
                    continue
                rng.shuffle(options)
                rows.append(dict(question=question, options=options, gold=4, kind="out", missing=missing))
        print(f"  {len(rows)} kept of {stats['tried']}  {stats}  ({time.time() - t0:.0f}s)", flush=True)
    write_jsonl(rows[: args.n], args.out)
    print(f"wrote {min(len(rows), args.n)} out-of-corpus questions to {args.out}")
    for r in rows[:4]:
        print(f"\nQ: {r['question']}  (missing: {r['missing']})\n" + "\n".join(f"    {l}) {o}" for l, o in zip("ABCD", r["options"])))


def cmd_mix(args):
    """v2 sets: every question gains the abstain letter; out-of-corpus questions are keyed to it."""
    rng = random.Random(args.seed)
    outs = read_jsonl(args.outs)
    # An option that denies the premise ("No such function exists") is as right
    # as the abstain letter and would be scored wrong: drop those questions.
    clean = [r for r in outs if not any(DENIAL.search(o) for o in r["options"])]
    print(f"out-of-corpus: {len(clean)} of {len(outs)} kept ({len(outs) - len(clean)} had a denial among the options)")
    outs = clean
    rng.shuffle(outs)
    train_in, test_in = read_jsonl(args.train), read_jsonl(args.test)
    n_train_out = int(round(len(train_in) * args.out_frac / (1 - args.out_frac)))
    n_train_out = min(n_train_out, len(outs) // 2)
    sets = {"train": train_in + outs[:n_train_out], "test": test_in + outs[n_train_out:]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        rows = [dict(r, abstain=True, kind=r.get("kind", "in")) for r in rows]
        rng.shuffle(rows)
        write_jsonl(rows, out / f"{name}.jsonl")
        print(f"{name}: {sum(r['kind'] == 'in' for r in rows)} in-corpus + {sum(r['kind'] == 'out' for r in rows)} "
              f"out-of-corpus -> {out / (name + '.jsonl')}")


# -----------------------------------------------------------------------------
# check: is the key right, and what does the model know with no context?
# -----------------------------------------------------------------------------


def letter_probs(model, tokenizer, prompts, toks, *, prefix=None, batch=4):
    logits = last_logits(model, tokenizer, prompts, prefix=prefix, batch=batch)
    l = logits[:, list(toks.letters[:4])]
    l = l - l.max(-1, keepdims=True)
    p = np.exp(l)
    return p / p.sum(-1, keepdims=True)


def cmd_check(args):
    from qwen_jax.mcq import AnswerTokens

    tokenizer = load_tokenizer()
    toks = AnswerTokens.from_tokenizer(tokenizer)
    items = read_jsonl(args.raw)
    model = load_model()
    dec = lambda x: tokenizer.decode(x, skip_special_tokens=False)
    t0 = time.time()
    icl = letter_probs(model, tokenizer, [
        encode_user(tokenizer, f"{args.description}\n\n{dec(r['chunk_ids'])}", render(r)) for r in items],
        toks, batch=args.batch)
    print(f"  icl done ({time.time() - t0:.0f}s)", flush=True)
    none = letter_probs(model, tokenizer, [encode_user(tokenizer, args.description, render(r)) for r in items],
                        toks, batch=args.batch * 2)
    for r, pi, pn in zip(items, icl, none):
        r["p_icl"], r["p_none"] = float(pi[r["gold"]]), float(pn[r["gold"]])
        r["icl_pick"], r["none_pick"] = int(pi.argmax()), int(pn.argmax())
    valid = [r for r in items if r["icl_pick"] == r["gold"] and r["p_icl"] >= args.min_icl]
    print(f"{len(valid)}/{len(items)} have a key the model confirms with the source in context "
          f"(p_icl >= {args.min_icl})")

    dropped = 0
    if args.decontam and Path(args.decontam).exists():
        held = set()
        for r in read_jsonl(args.decontam):
            held.update(named(r["question"]))
        keep = [r for r in valid if not (set(named(r["question"])) & held)]
        dropped, valid = len(valid) - len(keep), keep
        print(f"dropped {dropped} sharing a named entity with {args.decontam}; {len(valid)} left")

    none_acc = float(np.mean([r["none_pick"] == r["gold"] for r in valid]))
    print(f"with no context the model picks the key on {none_acc:.3f} of them "
          f"(mean p_none {np.mean([r['p_none'] for r in valid]):.3f}); gold letters "
          f"{np.bincount([r['gold'] for r in valid], minlength=4).tolist()}")
    rng = random.Random(args.seed)
    rng.shuffle(valid)
    n_test = int(round(len(valid) * args.test_frac))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(valid[:n_test], out / "test.jsonl")
    write_jsonl(valid[n_test:], out / "train.jsonl")
    print(f"wrote {len(valid) - n_test} train / {n_test} test to {out}")


# -----------------------------------------------------------------------------
# exact evaluation
# -----------------------------------------------------------------------------


def exact_policy(model, tokenizer, toks, items, *, prefix, system=None, micro_q=2, fn=None):
    """log pi over the L x 11 responses for every item, and the format mass. No gradient."""
    from qwen_jax.mcq import make_rows, policy

    fn = fn or jax.jit(functools.partial(policy, toks=toks))
    n = n_letters(items[0])
    logps, masses = [], []
    for group, k in batched(items, micro_q):  # the last group is padded by repetition
        if system is None:
            prompts = [encode_suffix(tokenizer, render(r)) for r in group]
        else:
            prompts = [encode_user(tokenizer, system(r), render(r)) for r in group]
        b = make_rows(prompts, [r["gold"] for r in group], toks, pad_id=tokenizer.pad_token_id, n_letters=n)
        lp, m = fn(model, prefix, b)
        logps.append(np.asarray(lp)[:k])
        masses.append(np.asarray(m)[:k])
    return np.concatenate(logps), np.concatenate(masses)


COLS = [("n", "n"), ("acc", "acc"), ("P(gold)", "p_gold"), ("E[R]", "exp_reward"), ("E[Brier]", "exp_brier"),
        ("conf", "modal_conf"), ("ECE", "ece"), ("AUROC", "auroc"), ("conf|right", "conf_when_right"),
        ("conf|wrong", "conf_when_wrong")]
COLS_ABSTAIN = [("caught", "caught"), ("abst-in", "abstain_in"), ("acc-in", "acc_in"), ("sel-acc", "selective_acc"),
                ("conf-ans", "conf_answered"), ("AUROC-ans", "auroc_answered")]


def print_table(table):
    w = max(len(k) for k in table) + 2
    cols = COLS + (COLS_ABSTAIN if "caught" in next(iter(table.values())) else [])
    print(f"\n{'':{w}}" + "".join(f"{c:>11}" for c, _ in cols))
    for name, s in table.items():
        print(f"{name:{w}}" + "".join(f"{s[k]:11.3f}" if isinstance(s[k], float) else f"{s[k]:11d}" for _, k in cols))
    print("\n  acc/conf/ECE/AUROC describe the greedy response; E[.] are expectations over every response.")
    if "caught" in next(iter(table.values())):
        print("  caught: out-of-corpus questions answered with the abstain letter. abst-in: real questions abstained on.\n"
              "  sel-acc / conf-ans / AUROC-ans: real questions the policy chose to answer.")
    for name, s in table.items():
        if "format_mass" in s:
            print(f"  format mass {name}: " + "  ".join(f"{x:.3f}" for x in s["format_mass"]))


def cmd_eval(args):
    from qwen_jax.cartridge import Cartridge
    from qwen_jax.mcq import AnswerTokens, summarise

    tokenizer = load_tokenizer()
    toks = AnswerTokens.from_tokenizer(tokenizer)
    items = read_jsonl(args.data)[: args.limit]
    gold = np.asarray([r["gold"] for r in items])
    abstain = bool(items[0].get("abstain"))
    summ = functools.partial(summarise, abstain=abstain, wrong_penalty=args.wrong_penalty)
    model = load_model()
    dec = lambda x: tokenizer.decode(x, skip_special_tokens=False)
    table, dump = {}, {}
    conds = [(spec.split("=", 1)[0], spec.split("=", 1)[1]) for spec in args.cartridges]
    for name, path in conds:
        prefix = Cartridge.load(path).prefix(model.cache_dtype())
        lp, m = exact_policy(model, tokenizer, toks, items, prefix=prefix, micro_q=args.micro_q)
        table[name], dump[name] = summ(lp, gold, m), lp
        print(f"  {name} done", flush=True)
    if args.init:
        corpus = load_corpus(tokenizer, args.files, args.root)
        prefix = init_cartridge(model, tokenizer, corpus, args.p, args.description).prefix(model.cache_dtype())
        lp, m = exact_policy(model, tokenizer, toks, items, prefix=prefix, micro_q=args.micro_q)
        table["init"], dump["init"] = summ(lp, gold, m), lp
    if args.none:
        lp, m = exact_policy(model, tokenizer, toks, items, prefix=None, system=lambda r: args.description,
                             micro_q=args.micro_q)
        table["none"], dump["none"] = summ(lp, gold, m), lp
    if args.icl:
        if any("chunk_ids" not in r for r in items):
            print("  (icl skipped: out-of-corpus questions have no source chunk)")
        else:
            lp, m = exact_policy(model, tokenizer, toks, items, prefix=None, micro_q=1,
                                 system=lambda r: f"{args.description}\n\n{dec(r['chunk_ids'])}")
            table["icl"], dump["icl"] = summ(lp, gold, m), lp
    print_table(table)
    if args.out:
        Path(args.out).write_text(json.dumps(table, indent=1))
        np.savez(Path(args.out).with_suffix(".npz"), gold=gold, **dump)


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------


def anchor_loss(model, cart, ref_prefix, batch, *, block=128):
    """KL(reference cartridge || cartridge) on self-study conversations: a leash on
    everything the multiple-choice questions do not look at."""
    from qwen_jax.distill import blockwise_kl

    ref_hidden, _, _ = model.model(input_ids=batch.student_ids, attention_mask=batch.student_mask, prefix=ref_prefix)
    hidden, _, _ = model.model(input_ids=batch.student_ids, attention_mask=batch.student_mask,
                               prefix=cart.prefix(model.cache_dtype()))
    return blockwise_kl(model.get_lm_head(), hidden, jax.lax.stop_gradient(ref_hidden), batch.loss_mask, block=block)


def conf_temp_at(rl_step: int, start: float, anneal: int) -> float:
    """Exploration temperature at the `rl_step`-th (0-based) policy-gradient
    step: `start`, decayed geometrically to 1 over `anneal` steps."""
    if anneal <= 0 or rl_step < 0:
        return start
    return start ** max(1.0 - rl_step / anneal, 0.0)


def cmd_train(args):
    import optax

    from qwen_jax.cartridge import Cartridge
    from qwen_jax.distill import evaluate
    from qwen_jax.mcq import (AnswerTokens, conf_sft_loss, conf_targets, isotonic_fit, make_rows, mcq_loss,
                              policy, summarise)
    from qwen_jax.selfstudy import load_examples

    tokenizer = load_tokenizer()
    toks = AnswerTokens.from_tokenizer(tokenizer)
    train, test = read_jsonl(args.data), read_jsonl(args.test)
    abstain = bool(train[0].get("abstain"))
    L = n_letters(train[0])
    penalty = args.wrong_penalty
    print(f"{len(train)} train / {len(test)} test questions, {L} letters, wrong-answer penalty {penalty}")
    model = load_model()
    cart = Cartridge.load(args.cartridge)
    ref_prefix = Cartridge.load(args.cartridge).prefix(model.cache_dtype())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shape = SimpleNamespace(batch=2, context=2176, seq=512, description=args.description)
    held = list(batches_from(tokenizer, load_examples(args.heldout), shape, shuffle=False)) if args.heldout else []
    anchors = load_examples(args.anchor_data) if args.anchor > 0 else []

    policy_fn = jax.jit(functools.partial(policy, toks=toks))
    summ = functools.partial(summarise, bonus=args.bonus, wrong_penalty=penalty, abstain=abstain)

    def rl_grad(model, cart, batch, beta, temp, scale):
        f = lambda params: mcq_loss(model, cart.with_params(params), batch, beta, toks, normalize=args.normalize,
                                    bonus=args.bonus, conf_temp=temp, conf_grad_scale=scale,
                                    wrong_penalty=penalty, abstain=abstain)
        (loss, aux), grads = jax.value_and_grad(f, has_aux=True)(cart.params())
        return grads, loss, aux

    def sft_grad(model, cart, batch, beta, temp, scale):
        f = lambda params: conf_sft_loss(model, cart.with_params(params), batch, toks)
        (loss, aux), grads = jax.value_and_grad(f, has_aux=True)(cart.params())
        return grads, loss, aux

    def anchor_grad(model, cart, ref_prefix, batch):
        f = lambda params: anchor_loss(model, cart.with_params(params), ref_prefix, batch)
        return jax.value_and_grad(f)(cart.params())

    rl_fn, sft_fn, anchor_fn = jax.jit(rl_grad), jax.jit(sft_grad), jax.jit(anchor_grad)

    # ---- the starting policy on every training question: the KL reference, and
    # ---- (for the warm start) the certainty the stated confidence should carry.
    t0 = time.time()
    ref_lp, _ = exact_policy(model, tokenizer, toks, train, prefix=ref_prefix, micro_q=args.micro_q, fn=policy_fn)
    letter_p = np.exp(ref_lp).sum(-1)  # (N, L)
    knots, floor = None, int(np.ceil(10 / L))  # the chosen one of L letters is worth at least 1/L
    if args.warm_steps:
        # Recalibration of the CHOSEN letter's probability: how often is the letter
        # the policy picks right, as a function of how sure it was of the letter?
        # Targets are then built on-policy, for the letter the current cartridge
        # picks. (First version: a target for every letter from the reference
        # policy. The letter choice drifted onto E, whose target had been written
        # for a letter assumed not chosen, and the greedy response became "E, 10%"
        # -- below the 20% floor of a five-way choice.)
        gold = np.asarray([r["gold"] for r in train])
        pick = letter_p.argmax(-1)
        knots = isotonic_fit(letter_p[np.arange(len(train)), pick], (pick == gold).astype(float))
        t_ref = np.maximum(conf_targets(letter_p[np.arange(len(train)), pick], knots), floor)
        print(f"warm-start targets for the chosen letter at the start, count by 0..100%: "
              f"{np.bincount(t_ref, minlength=11).tolist()}  ({time.time() - t0:.0f}s)")
        np.savez(out / "recalibration.npz", kx=knots[0], ky=knots[1])
    for i, r in enumerate(train):
        r["_i"] = i

    schedule = optax.linear_schedule(0.0, args.lr, args.warmup) if args.warmup else optax.constant_schedule(args.lr)
    optimizer = optax.adam(schedule)
    state = optimizer.init(cart.params())
    test_gold = np.asarray([r["gold"] for r in test])
    rng = random.Random(args.seed)

    def evaluate_now(step, phase):
        lp, m = exact_policy(model, tokenizer, toks, test, prefix=cart.prefix(model.cache_dtype()),
                             micro_q=args.micro_q, fn=policy_fn)
        s = summ(lp, test_gold, m)
        if held:
            s["heldout_kl"] = evaluate(model, cart, held)
        s["step"], s["phase"] = step, phase
        with open(out / "eval.jsonl", "a") as f:
            f.write(json.dumps(s) + "\n")
        line = (f"    TEST {phase} {step:4d}  acc {s['acc']:.3f}  E[R] {s['exp_reward']:.3f}  E[Brier] {s['exp_brier']:.3f}  "
                f"conf {s['modal_conf']:.3f}  ECE {s['ece']:.3f}  AUROC {s['auroc']:.3f}  "
                f"conf|right {s['conf_when_right']:.2f}  conf|wrong {s['conf_when_wrong']:.2f}")
        if abstain:
            line += (f"  | caught {s['caught']:.3f}  abst-in {s['abstain_in']:.3f}  acc-in {s['acc_in']:.3f}  "
                     f"sel-acc {s['selective_acc']:.3f}")
        print(line + (f"  heldoutKL {s['heldout_kl']:.4f}" if held else ""), flush=True)

    if args.batch_q % args.micro_q:
        raise SystemExit("--batch-q must be a multiple of --micro-q (padded micro-batches would double count)")
    evaluate_now(0, "start")
    order, t_start = [], time.time()
    for step in range(1, args.warm_steps + args.steps + 1):
        t0 = time.time()
        warm = step <= args.warm_steps
        grad_fn = sft_fn if warm else rl_fn
        temp = conf_temp_at(step - args.warm_steps - 1, args.conf_temp, args.conf_temp_anneal)
        if len(order) < args.batch_q:
            order = list(range(len(train)))
            rng.shuffle(order)
        items, order = [train[i] for i in order[: args.batch_q]], order[args.batch_q:]
        scale = args.conf_grad_scale * (temp if args.conf_temp_rescale else 1.0)
        match = args.conf_grad_match if not warm else 0.0
        total, letter_part, stats, mass = None, None, np.zeros(3), np.zeros(3)
        micros = list(batched(items, args.micro_q))
        for group, _ in micros:
            idx = [r["_i"] for r in group]
            prompts, golds = [encode_suffix(tokenizer, render(r)) for r in group], [r["gold"] for r in group]
            target = weight = None
            if warm:
                cur, _ = policy_fn(model, cart.prefix(model.cache_dtype()),
                                   make_rows(prompts, golds, toks, pad_id=tokenizer.pad_token_id, n_letters=L))
                pl = np.exp(np.asarray(cur)).sum(-1)
                pick, rows_i = pl.argmax(-1), np.arange(len(group))
                target = np.zeros((len(group), L), np.int32)
                weight = np.zeros((len(group), L), np.float32)
                target[rows_i, pick] = np.maximum(conf_targets(pl[rows_i, pick], knots), floor)
                weight[rows_i, pick] = 1.0
                if abstain and not args.sft_abstain:
                    weight[pick == L - 1] = 0.0  # leave the abstain letter's number to the reward
            b = make_rows(prompts, golds, toks, pad_id=tokenizer.pad_token_id, n_letters=L, ref_logp=ref_lp[idx],
                          target=target, weight=weight)
            g, loss, (j, kl, m) = grad_fn(model, cart, b, jnp.float32(args.beta), jnp.float32(temp),
                                          jnp.float32(1.0 if match else scale))
            total = g if total is None else jax.tree_util.tree_map(jnp.add, total, g)
            if match:  # the gradient is linear in the scale: scale 0 is the letter's part
                g0 = grad_fn(model, cart, b, jnp.float32(args.beta), jnp.float32(temp), jnp.float32(0.0))[0]
                letter_part = g0 if letter_part is None else jax.tree_util.tree_map(jnp.add, letter_part, g0)
            stats += np.array([float(loss), float(j), float(kl)])
            mass += np.asarray(m)
        conf_factor = scale
        if match:
            # Rescale the confidence's part of the batch gradient to `match` times the letter's norm.
            conf_part = cart.mask_grads(jax.tree_util.tree_map(jnp.subtract, total, letter_part))
            letter_part = cart.mask_grads(letter_part)
            norm = lambda t: float(optax.global_norm(jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), t)))
            conf_factor = match * norm(letter_part) / max(norm(conf_part), 1e-12)
            total = jax.tree_util.tree_map(lambda a, c: a + conf_factor * c, letter_part, conf_part)
        total = jax.tree_util.tree_map(lambda x: x / len(micros), total)
        a_kl = 0.0
        if anchors:
            ab = next(batches_from(tokenizer, rng.sample(anchors, 2), shape, shuffle=False))
            a_val, a_g = anchor_fn(model, cart, ref_prefix, ab)
            a_kl = float(a_val)
            total = jax.tree_util.tree_map(lambda x, y: x + args.anchor * y, total, a_g)
        updates, state = optimizer.update(cart.mask_grads(total), state, cart.params())
        cart = cart.with_params(optax.apply_updates(cart.params(), updates)).advance(1)
        loss, j, kl = stats / len(micros)
        row = dict(step=step, phase="warm" if warm else "rl", loss=float(loss), objective=float(j), kl=float(kl),
                   anchor_kl=a_kl, conf_temp=temp, conf_factor=float(conf_factor), format_mass=(mass / len(micros)).tolist(), t=time.time() - t0)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  {row['phase']:4s} {step:4d}  {'CE' if warm else 'E[R]'} {j:6.3f}  KL {kl:.4f}  anchorKL {a_kl:.4f}  T {temp:.2f} x{conf_factor:.3g}  "
              f"mass {row['format_mass'][0]:.3f}/{row['format_mass'][1]:.3f}/{row['format_mass'][2]:.3f}  "
              f"[{row['t']:.1f}s]", flush=True)
        last = step == args.warm_steps + args.steps
        if step % args.eval_every == 0 or last or step == args.warm_steps:
            evaluate_now(step, "warm" if warm else "rl")
        if step % args.save_every == 0 or last or step == args.warm_steps:
            cart.save(out / f"step{step:04d}.safetensors")
            print(f"    saved step{step:04d} ({time.time() - t_start:.0f}s total)", flush=True)


# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--files", nargs="*", help="corpus files (default: src/qwen_jax/**/*.py)")
        s.add_argument("--root", help="directory the corpus file headers are relative to")
        s.add_argument("--description", default=DESCRIPTION)

    def sampling(s, max_new):
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--batch", type=int, default=8)
        s.add_argument("--chunk-min", type=int, default=512)
        s.add_argument("--chunk-max", type=int, default=2048)
        s.add_argument("--max-new", type=int, default=max_new)
        s.add_argument("--temperature", type=float, default=0.7)

    g = sub.add_parser("gen")
    common(g)
    sampling(g, 160)
    g.add_argument("--n", type=int, default=800, help="stop after this many kept (0 = every span)")
    g.add_argument("--span-tokens", type=int, default=40)
    g.add_argument("--out", default=str(OUT_DIR / "raw.jsonl"))

    go = sub.add_parser("gen-out")
    common(go)
    sampling(go, 160)
    go.add_argument("--n", type=int, default=300)
    go.add_argument("--decontam", nargs="*", default=[str(REPO / "runs/reader/qa.jsonl"),
                                                       str(REPO / "runs/recall/items.jsonl")],
                    help="eval sets whose invented names must not be reused")
    go.add_argument("--out", default=str(OUT_DIR / "out-raw.jsonl"))

    m = sub.add_parser("mix")
    m.add_argument("--train", default=str(OUT_DIR / "train.jsonl"))
    m.add_argument("--test", default=str(OUT_DIR / "test-big.jsonl"))
    m.add_argument("--outs", default=str(OUT_DIR / "out-raw.jsonl"))
    m.add_argument("--out-frac", type=float, default=0.3, help="share of out-of-corpus questions in the train set")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default=str(OUT_DIR / "v2"))

    c = sub.add_parser("check")
    common(c)
    c.add_argument("--raw", default=str(OUT_DIR / "raw.jsonl"))
    c.add_argument("--min-icl", type=float, default=0.6)
    c.add_argument("--decontam", default=str(REPO / "runs/reader/qa.jsonl"),
                   help="drop questions sharing a named entity with this eval set ('' to skip)")
    c.add_argument("--test-frac", type=float, default=0.25)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--batch", type=int, default=4)
    c.add_argument("--out", default=str(OUT_DIR))

    e = sub.add_parser("eval")
    common(e)
    e.add_argument("cartridges", nargs="*", help="name=path.safetensors")
    e.add_argument("--data", default=str(OUT_DIR / "test.jsonl"))
    e.add_argument("--init", action="store_true")
    e.add_argument("--none", action="store_true")
    e.add_argument("--icl", action="store_true")
    e.add_argument("--p", type=int, default=1024)
    e.add_argument("--limit", type=int)
    e.add_argument("--micro-q", type=int, default=2)
    e.add_argument("--wrong-penalty", type=float, default=0.0)
    e.add_argument("--out")

    t = sub.add_parser("train")
    common(t)
    t.add_argument("--cartridge", required=True)
    t.add_argument("--data", default=str(OUT_DIR / "train.jsonl"))
    t.add_argument("--test", default=str(OUT_DIR / "test.jsonl"))
    t.add_argument("--heldout", default=str(REPO / "runs/cart/heldout.jsonl"),
                   help="self-study conversations for the drift check ('' to skip)")
    t.add_argument("--warm-steps", type=int, default=0,
                   help="supervised steps first: stated confidence of the letter the policy picks <- the "
                        "recalibrated probability of that letter")
    t.add_argument("--sft-abstain", action="store_true",
                   help="also give the abstain letter a warm-start target (default: leave its number to the reward)")
    t.add_argument("--steps", type=int, default=100, help="exact policy-gradient steps after the warm start")
    t.add_argument("--batch-q", type=int, default=8)
    t.add_argument("--micro-q", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--warmup", type=int, default=5)
    t.add_argument("--beta", type=float, default=0.0, help="exact KL leash to the starting policy, over the responses")
    t.add_argument("--anchor", type=float, default=0.0,
                   help="weight of KL(starting cartridge || cartridge) on self-study conversations")
    t.add_argument("--anchor-data", default=str(REPO / "runs/cart/train.jsonl"))
    t.add_argument("--bonus", type=float, default=2.0)
    t.add_argument("--wrong-penalty", type=float, default=0.0,
                   help="cost of a wrong answer letter other than the abstain letter (tried at 1: no difference)")
    t.add_argument("--conf-temp", type=float, default=1.0,
                   help="exploration temperature on the confidence softmaxes, inside the objective only")
    t.add_argument("--conf-temp-anneal", type=int, default=0,
                   help="decay --conf-temp geometrically to 1 over this many RL steps (0 = hold it constant)")
    t.add_argument("--conf-temp-rescale", action="store_true",
                   help="multiply the confidence gradient by the temperature (undo the 1/T of the tempered softmax)")
    t.add_argument("--conf-grad-scale", type=float, default=1.0,
                   help="weight of the gradient through the confidence logits relative to the letter's")
    t.add_argument("--conf-grad-match", type=float, default=0.0,
                   help="rescale the confidence's part of each batch gradient to this multiple of the letter's "
                        "norm (0 = off; costs a second backward pass)")
    t.add_argument("--normalize", action="store_true", help="GRPO-style per-question advantage standardisation")
    t.add_argument("--eval-every", type=int, default=10)
    t.add_argument("--save-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)

    args = ap.parse_args()
    {"gen": cmd_gen, "gen-out": cmd_gen_out, "mix": cmd_mix, "check": cmd_check, "eval": cmd_eval,
     "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
