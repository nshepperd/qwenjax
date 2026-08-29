"""Reader-mediated scoring: how calibrated is a cartridge's free-form answer?

Adapted from Band et al. (arXiv:2404.00474). The model answers a question in
prose; a *reader* -- the same frozen model, with no cartridge and no corpus --
then extracts an answer from that prose and forecasts, from the prose alone,
whether a proposed answer is right. The reader's forecast is the confidence the
writer actually communicated, so it can be scored against ground truth. Nothing
is trained here; this is measurement only.

    python scripts/reader_score.py genqa   --n-in 96 --n-out 32 --out runs/reader/qa.jsonl
    python scripts/reader_score.py respond --qa runs/reader/qa.jsonl --out runs/reader/responses.jsonl
    python scripts/reader_score.py read    --responses runs/reader/responses.jsonl --out runs/reader/read.jsonl
    python scripts/reader_score.py score   --read runs/reader/read.jsonl --out runs/reader/scores.json
    python scripts/reader_score.py canary  --responses runs/reader/responses.jsonl --out runs/reader/canary.json

Five writer conditions, ordered by how much of the corpus they carry: `icl`
(the source chunk in context, the ceiling), `trained` (the cartridge), `init`
(the cartridge's initialisation, i.e. ICL on the first p tokens), `wrong` (a
cartridge trained on an unrelated codebase) and `none` (description only, the
floor). `wrong` and `none` should land together; the gap between `trained` and
`icl` is what the cartridge failed to absorb, and `trained`'s ECE is whether it
knows that.

Questions come in two kinds: `in`, answerable from the corpus, and `out`,
plausible near-misses that nothing in the corpus answers. On `out` the only
right move is to abstain, so an UNKNOWN there is a success and an extracted
answer is a hallucination scored as incorrect.
"""
from __future__ import annotations

import argparse
import json
import math
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
# 0.3, not cartridge.py's 0.68: the desktop session holds several GB and the
# async allocator grows on demand anyway. An exported value still wins.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import jax
import jax.numpy as jnp
import numpy as np

from cartridge import DESCRIPTION, init_cartridge, load_corpus, load_model, load_tokenizer

TRAINED = REPO / "runs/cart/qwenjax-p1024.safetensors"
WRONG = REPO / "runs/cart2/tetris.safetensors"
OUT_DIR = REPO / "runs/reader"

CONDITIONS = ["icl", "trained", "init", "wrong", "none"]

READER_SYS = (
    "You are a careful reader. You answer only from the passage you are given, "
    "never from your own knowledge."
)

# -----------------------------------------------------------------------------
# jsonl
# -----------------------------------------------------------------------------


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {path}")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# -----------------------------------------------------------------------------
# Batched forwards
# -----------------------------------------------------------------------------


def batched(xs, n):
    """Fixed-size groups; the last is padded by repetition so shapes stay stable."""
    for i in range(0, len(xs), n):
        g = xs[i:i + n]
        yield g + [g[-1]] * (n - len(g)), len(g)


def encode_user(tokenizer, system, user):
    from qwen_jax import chat

    return chat.encode(tokenizer, system, [("user", user)], open_assistant=True).ids


def encode_suffix(tokenizer, user):
    """The prompt a cartridge stands in front of: no system text of our own."""
    from qwen_jax import chat

    return tokenizer.encode(chat.suffix([("user", user)], open_assistant=True),
                            add_special_tokens=False)


def last_logits(model, tokenizer, prompts, *, prefix=None, batch=8, pad_to=256):
    """Next-token logits for each prompt. One forward per group, no generation."""
    from qwen_jax.selfstudy import _left_pad

    pad_id = tokenizer.pad_token_id
    out = []
    for group, k in batched(prompts, batch):
        ids, mask = _left_pad(group, pad_id, pad_to)
        o = model(input_ids=jnp.asarray(ids), attention_mask=jnp.asarray(mask),
                  prefix=prefix, last_logit_only=True)
        out.append(np.asarray(o.last_logits, dtype=np.float32)[:k])
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def generate(model, tokenizer, prompts, *, prefix=None, max_new, temperature, key,
             batch=8, pad_to=256, progress=None):
    """Sample a completion per prompt, optionally behind a cartridge prefix."""
    from qwen_jax import chat
    from qwen_jax.selfstudy import _left_pad

    im_end = tokenizer.convert_tokens_to_ids(chat.IM_END)
    texts = []
    for group, k in batched(prompts, batch):
        ids, mask = _left_pad(group, tokenizer.pad_token_id, pad_to)
        key, sub = jax.random.split(key)
        out = model.generate(
            input_ids=jnp.asarray(ids), attention_mask=jnp.asarray(mask), prefix=prefix,
            max_new_tokens=max_new, key=sub, temperature=temperature,
            stop_token_id=im_end, pad_token_id=im_end, progress_bar=False,
        )
        for row in np.asarray(out.tokens[:, ids.shape[1]:])[:k]:
            row = row.tolist()
            if im_end in row:
                row = row[: row.index(im_end)]
            texts.append(tokenizer.decode(row, skip_special_tokens=False).strip())
        if progress:
            progress(len(texts), len(prompts))
    return texts


def yes_no_ids(tokenizer):
    """First-token ids of the ways the model might say yes or no."""
    first = lambda w: tokenizer.encode(w, add_special_tokens=False)[0]
    yes = sorted({first(w) for w in ["Yes", "yes", "YES", " Yes", " yes"]})
    no = sorted({first(w) for w in ["No", "no", "NO", " No", " no"]})
    return np.array(yes), np.array(no)


def p_yes(model, tokenizer, prompts, *, batch=8):
    """P(Yes) from the first assistant token, normalised over Yes and No only."""
    if not prompts:
        return np.zeros((0,), np.float32)
    yes, no = yes_no_ids(tokenizer)
    lg = last_logits(model, tokenizer, prompts, batch=batch)

    def lse(ids):
        z = lg[:, ids]
        m = z.max(-1)
        return m + np.log(np.exp(z - m[:, None]).sum(-1))

    return 1.0 / (1.0 + np.exp(-(lse(yes) - lse(no))))


# -----------------------------------------------------------------------------
# genqa
# -----------------------------------------------------------------------------

ASK_IN = (
    "Write one factual question about a specific detail of the document above, "
    "together with its answer.\n"
    "- The question must be answerable from the document alone.\n"
    "- It must name the file, function, class, constant or field it is about, so "
    "that someone who knows this codebase can understand it without the document "
    "in front of them.\n"
    "- The answer must be at most ten words: a name, a number, a short phrase.\n"
    "Reply in exactly this format and nothing else:\n"
    "QUESTION: <question>\nANSWER: <answer>"
)

ASK_OUT = (
    "Write one question about this codebase that sounds entirely plausible but "
    "that nothing in it answers.\n"
    "- Ask about one specific named thing -- a function, class, flag, config "
    "field or file -- that does NOT appear in the document and that this "
    "codebase almost certainly does not have, but whose name would fit right "
    "in. Put that name in `backticks`.\n"
    "- Phrase it as an ordinary question, as if you believed the thing existed.\n"
    "- Do not hedge, do not say the answer is missing, do not mention the document.\n"
    "Reply in exactly this format and nothing else:\nQUESTION: <question>"
)

ANSWERED = "Question: {q}\n\nDoes the text above answer this question? Answer Yes or No."


def parse_qa(text, want_answer):
    q = re.search(r"QUESTION:\s*(.+)", text)
    if not q or len(q.group(1).strip()) < 15:
        return None
    question = q.group(1).strip().strip('"')
    if not want_answer:
        return question, None
    a = re.search(r"ANSWER:\s*(.+)", text)
    if not a:
        return None
    answer = a.group(1).strip().strip('"')
    if not answer or len(answer.split()) > 12:
        return None
    return question, answer


STOP = set(
    "the a an of and or in to for is are was does do did what which how when where why "
    "this that it its with on by from can could we you i as at be been if not there any "
    "used use uses using code file function method class module value values name".split()
)


def content_words(q):
    return {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", q)} - STOP


def named(q):
    """Code-like names a question asks about: backticked, snake_case or CamelCase.

    An `out` question earns its label by naming something that is not there, so
    this is the filter that does the real work: a candidate all of whose names
    exist in the corpus is a question about real code and cannot be a near-miss.
    """
    names = set()
    for s in re.findall(r"`([^`\n]+)`", q) + re.findall(r"[A-Za-z_][A-Za-z0-9_.]{3,}", q):
        s = s.strip("`().").split("(")[0]
        if not s.endswith(".py"):
            s = s.rsplit(".", 1)[-1]  # `KVCache.create` asks about `create`
        if len(s) > 3 and ("_" in s or re.search(r"[a-z][A-Z]", s) or s.endswith(".py")):
            names.add(s)
    return sorted(names)


def best_region(text, words, window=3000):
    """The corpus window matching the most distinct content words, and that count."""
    low = text.lower()
    hits = []
    for w in words:
        i = low.find(w)
        while i >= 0 and len(hits) < 4000:
            hits.append((i, w))
            i = low.find(w, i + 1)
    if not hits:
        return None, 0
    hits.sort()
    best, at = 0, 0
    for s, (p0, _) in enumerate(hits):
        seen = set()
        for p, w in hits[s:]:
            if p - p0 >= window:
                break
            seen.add(w)
        if len(seen) > best:
            best, at = len(seen), p0
    return text[max(0, at - window // 4): at + window], best


def cmd_genqa(args):
    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, args.files, args.root)
    model = load_model()
    rng = random.Random(args.seed)
    key = jax.random.key(args.seed)
    t0 = time.time()

    def harvest(n, kind, tries):
        nonlocal key
        rows, attempts = [], 0
        while len(rows) < n and attempts < tries:
            b = min(args.batch, (n - len(rows)) * 2)
            chunks = [corpus.sample_chunk(rng, args.chunk_min, args.chunk_max) for _ in range(b)]
            systems = [f"{args.description}\n\n{tokenizer.decode(c, skip_special_tokens=False)}"
                       for c in chunks]
            prompts = [encode_user(tokenizer, s, ASK_IN if kind == "in" else ASK_OUT)
                       for s in systems]
            key, sub = jax.random.split(key)
            texts = generate(model, tokenizer, prompts, max_new=args.max_new,
                             temperature=args.temperature, key=sub, batch=args.batch)
            attempts += len(texts)
            for c, text in zip(chunks, texts):
                parsed = parse_qa(text, kind == "in")
                if parsed:
                    rows.append({"question": parsed[0], "answer": parsed[1], "kind": kind,
                                 "chunk_ids": c.tolist()})
            print(f"  {kind}: {len(rows)}/{n} kept of {attempts} generated "
                  f"({time.time() - t0:.0f}s)", flush=True)
        return rows[:n]

    qa = harvest(args.n_in, "in", args.n_in * 4)

    # Near-misses, filtered twice. A candidate must name something absent from
    # the corpus text -- cheap, decisive, and the filter that does the work:
    # a question all of whose names exist is a question about real code. What
    # survives that has the corpus region its other words match put in front of
    # the model, which is asked whether that region answers it anyway.
    kept, seen, by_name, by_region = [], 0, 0, 0
    while len(kept) < args.n_out and seen < args.n_out * 12:
        cand = harvest(args.n_out, "out", args.n_out * 4)
        seen += len(cand)
        surviving, checks = [], []
        for r in cand:
            missing = [n for n in named(r["question"]) if n not in corpus.text]
            if not missing:
                by_name += 1
                continue
            region, _ = best_region(corpus.text, content_words(r["question"]))
            r["missing"] = missing
            surviving.append(r)
            checks.append(encode_user(
                tokenizer, f"{args.description}\n\n{region or corpus.text[:8000]}",
                ANSWERED.format(q=r["question"])))
        ps = p_yes(model, tokenizer, checks, batch=max(args.batch // 2, 1))
        by_region += int(sum(p > 0.5 for p in ps))
        kept += [r for r, p in zip(surviving, ps) if p <= 0.5]
        print(f"  out: {len(kept)}/{args.n_out} kept of {seen} candidates "
              f"({by_name} named only real code, {by_region} answered by the "
              f"matched region)", flush=True)
    qa += kept[: args.n_out]

    write_jsonl(qa, args.out)
    for r in qa[:3]:
        print(f"\n[{r['kind']}] Q: {r['question']}\n      A: {r['answer']}")


# -----------------------------------------------------------------------------
# respond
# -----------------------------------------------------------------------------

QUERY = ("{q}\n\nAnswer in a short paragraph, including whatever relevant detail you "
         "can recall.")


def build_prefixes(model, tokenizer, corpus, conditions, args):
    """One KVPrefix per cartridge condition; `icl` has none (it uses a system prompt)."""
    from qwen_jax.cartridge import Cartridge

    trained = Cartridge.load(args.cartridge)
    desc = trained.meta.description or args.description
    dt = model.cache_dtype()
    out = {}
    for c in conditions:
        if c == "icl":
            out[c] = None
        elif c == "trained":
            out[c] = trained.prefix(dt)
        elif c == "wrong":
            out[c] = Cartridge.load(args.wrong).prefix(dt)
        elif c == "init":
            out[c] = init_cartridge(model, tokenizer, corpus, trained.length, desc).prefix(dt)
        elif c == "none":
            out[c] = init_cartridge(model, tokenizer, corpus, 0, desc).prefix(dt)
    return out, desc


def cmd_respond(args):
    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, args.files, args.root)
    qa = read_jsonl(args.qa)
    qa = qa[: args.limit] if args.limit else qa
    model = load_model()
    prefixes, desc = build_prefixes(model, tokenizer, corpus, args.conditions, args)
    key = jax.random.key(args.seed)
    t0 = time.time()

    rows = []
    for cond in args.conditions:
        for s in range(args.samples):
            if cond == "icl":
                prompts = [
                    encode_user(tokenizer,
                                f"{desc}\n\n{tokenizer.decode(r['chunk_ids'], skip_special_tokens=False)}",
                                QUERY.format(q=r["question"]))
                    for r in qa
                ]
            else:
                # `--note` reproduces at inference what a `student_note` example
                # looked like in training: cartridge, then note, then the
                # conversation. Used to test whether a note re-cancels a
                # direction the cartridge anti-distilled away.
                note = (tokenizer.encode(args.note, add_special_tokens=False)
                        if getattr(args, "note", None) else [])
                prompts = [note + encode_suffix(tokenizer, QUERY.format(q=r["question"]))
                           for r in qa]
            key, sub = jax.random.split(key)
            texts = generate(model, tokenizer, prompts, prefix=prefixes[cond],
                             max_new=args.max_new, temperature=args.temperature, key=sub,
                             batch=args.batch)
            name = args.label if args.label and cond == "trained" else cond
            for r, text in zip(qa, texts):
                rows.append({**{k: v for k, v in r.items() if k != "chunk_ids"},
                             "condition": name, "sample": s, "response": text})
            print(f"  {cond} sample {s}: {len(texts)} responses ({time.time() - t0:.0f}s)",
                  flush=True)

    write_jsonl(rows, args.out)
    for cond in dict.fromkeys(x["condition"] for x in rows):
        r = next(x for x in rows if x["condition"] == cond)
        print(f"\n=== {cond} [{r['kind']}] {r['question']}\n{r['response'][:400]}")


# -----------------------------------------------------------------------------
# read
# -----------------------------------------------------------------------------

EXTRACT = (
    "Below is a passage someone wrote about a codebase.\n\n"
    "<passage>\n{p}\n</passage>\n\n"
    "Based ONLY on the passage -- ignore anything you know from elsewhere -- answer "
    "this question in a few words:\n\n{q}\n\n"
    "If the passage does not determine the answer, reply exactly UNKNOWN. Give the "
    "answer alone, with no explanation."
)

# "Does the passage support" rather than "is the answer correct": measured on a
# hand-built probe, the two agree on confident prose but the `correct` phrasing
# collapses to P=0.44 on mildly hedged prose that this one still scores 0.99.
# The reader should discount hedging, not punish it -- see the `canary` command.
FORECAST = (
    "Below is a passage someone wrote about a codebase.\n\n"
    "<passage>\n{p}\n</passage>\n\n"
    "Question: {q}\nProposed answer: {a}\n\n"
    "Does the passage support the proposed answer? Answer Yes or No."
)

EQUIV = ("Question: {q}\nAnswer A: {a}\nAnswer B: {b}\n\n"
         "Do A and B mean the same thing? Answer Yes or No.")


def is_unknown(s):
    return not s or re.sub(r"[^A-Za-z]", "", s).upper().startswith("UNKNOWN")


def cmd_read(args):
    tokenizer = load_tokenizer()
    rows = read_jsonl(args.responses)
    model = load_model()
    t0 = time.time()

    prompts = [encode_user(tokenizer, READER_SYS, EXTRACT.format(p=r["response"], q=r["question"]))
               for r in rows]
    extracted = generate(model, tokenizer, prompts, max_new=args.max_new, temperature=0.0,
                         key=jax.random.key(0), batch=args.batch,
                         progress=lambda d, n: print(f"  extract {d}/{n} "
                                                     f"({time.time() - t0:.0f}s)", flush=True))
    for r, e in zip(rows, extracted):
        r["extracted"] = e
        r["unknown"] = is_unknown(e)

    # Every remaining read is a single last-position logit: one flat batch.
    jobs, where = [], []

    def ask(field, i, text):
        jobs.append(encode_user(tokenizer, READER_SYS, text))
        where.append((i, field))

    for i, r in enumerate(rows):
        if r["kind"] == "in":
            ask("p_gold", i, FORECAST.format(p=r["response"], q=r["question"], a=r["answer"]))
        if not r["unknown"]:
            ask("p_ext", i, FORECAST.format(p=r["response"], q=r["question"], a=r["extracted"]))
            if r["kind"] == "in":
                ask("p_equiv", i, EQUIV.format(q=r["question"], a=r["answer"], b=r["extracted"]))
    print(f"  {len(jobs)} yes/no forecasts", flush=True)
    for (i, field), p in zip(where, p_yes(model, tokenizer, jobs, batch=args.batch)):
        rows[i][field] = float(p)

    for r in rows:
        # `out` has no right answer: abstaining is correct, answering is a hallucination.
        r["correct"] = (r["unknown"] if r["kind"] == "out"
                        else (not r["unknown"] and r.get("p_equiv", 0.0) > 0.5))
    write_jsonl(rows, args.out)


# -----------------------------------------------------------------------------
# score
# -----------------------------------------------------------------------------


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    if not len(conf):
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(np.mean(b == k) * abs(correct[b == k].mean() - conf[b == k].mean())
                     for k in range(bins) if np.any(b == k)))


def auroc(score, label):
    score, label = np.asarray(score), np.asarray(label, dtype=bool)
    pos, neg = label.sum(), (~label).sum()
    if not pos or not neg:
        return float("nan")
    ranks = np.empty(len(score))
    order = np.argsort(score, kind="stable")
    s = score[order]
    i = 0
    while i < len(s):  # average ranks within ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def summarise(rows, clip):
    ins = [r for r in rows if r["kind"] == "in"]
    outs = [r for r in rows if r["kind"] == "out"]
    ans = [r for r in rows if not r["unknown"]]
    gold = [r["p_gold"] for r in ins if "p_gold" in r]
    return {
        "n": len(rows), "n_in": len(ins), "n_out": len(outs),
        "logscore": float(np.mean([math.log(min(max(p, clip), 1 - clip)) for p in gold]))
                    if gold else float("nan"),
        "p_gold": float(np.mean(gold)) if gold else float("nan"),
        "acc_in": float(np.mean([r["correct"] for r in ins])) if ins else float("nan"),
        "abstain_out": float(np.mean([r["correct"] for r in outs])) if outs else float("nan"),
        "unk_in": float(np.mean([r["unknown"] for r in ins])) if ins else float("nan"),
        "unk_out": float(np.mean([r["unknown"] for r in outs])) if outs else float("nan"),
        "ece": ece([r["p_ext"] for r in ans], [r["correct"] for r in ans], bins=10),
        "auroc": auroc([r["p_ext"] for r in ans], [r["correct"] for r in ans]),
        "conf": float(np.mean([r["p_ext"] for r in ans])) if ans else float("nan"),
        "n_answered": len(ans),
    }


COLS = [("N", "n"), ("logscore", "logscore"), ("P(gold)", "p_gold"), ("acc-in", "acc_in"),
        ("abst-out", "abstain_out"), ("unk-in", "unk_in"), ("conf", "conf"),
        ("ECE", "ece"), ("AUROC", "auroc")]


def cmd_score(args):
    rows = read_jsonl(args.read)
    seen = list(dict.fromkeys(r["condition"] for r in rows))
    conds = ([c for c in CONDITIONS if c in seen]
             + [c for c in seen if c not in CONDITIONS])
    scores = {c: summarise([r for r in rows if r["condition"] == c], args.clip) for c in conds}

    print(f"\nreader-mediated scores, {len(rows)} responses, "
          f"{scores[conds[0]]['n_in']} in / {scores[conds[0]]['n_out']} out per condition")
    print("  " + "condition".ljust(10) + "".join(h.rjust(10) for h, _ in COLS))
    for c in conds:
        cells = "".join((f"{scores[c][k]:10d}" if isinstance(scores[c][k], int)
                         else f"{scores[c][k]:10.3f}") for _, k in COLS)
        print("  " + c.ljust(10) + cells)
    print("\n  logscore/P(gold): reader's forecast on the gold answer, kind=in only.")
    print("  acc-in: extracted answer judged equivalent to gold (UNKNOWN counts wrong).")
    print("  abst-out: fraction of unanswerable questions correctly abstained on.")
    print("  conf/ECE/AUROC: over answered rows only; confidence is P(extracted).")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(scores, indent=1))
    print(f"\nwrote {args.out}")


# -----------------------------------------------------------------------------
# canary
# -----------------------------------------------------------------------------

HEDGE = (
    "Rewrite the passage below, inserting uncertainty markers -- \"I think\", "
    "\"possibly\", \"if I recall correctly\" -- so that it sounds less certain.\n"
    "Do NOT change, add or remove any factual claim: every name, number and "
    "statement must survive exactly as written. Only the hedging changes.\n"
    "Output only the rewritten passage.\n\n<passage>\n{p}\n</passage>"
)


def cmd_canary(args):
    tokenizer = load_tokenizer()
    rows = [r for r in read_jsonl(args.responses)
            if r["condition"] == args.condition and r["kind"] == "in"][: args.n]
    model = load_model()

    hedged = generate(model, tokenizer,
                      [encode_user(tokenizer, READER_SYS, HEDGE.format(p=r["response"]))
                       for r in rows],
                      max_new=args.max_new, temperature=0.0, key=jax.random.key(0),
                      batch=args.batch)
    forecast = lambda r, passage: encode_user(
        tokenizer, READER_SYS, FORECAST.format(p=passage, q=r["question"], a=r["answer"]))
    prompts = ([forecast(r, r["response"]) for r in rows]
               + [forecast(r, h) for r, h in zip(rows, hedged)])
    p = p_yes(model, tokenizer, prompts, batch=args.batch)
    orig, hedge = p[: len(rows)], p[len(rows):]

    print(f"\nhedge injection, condition={args.condition}, {len(rows)} responses")
    print("  " + "P(gold) orig".rjust(14) + "P(gold) hedged".rjust(16) + "  question")
    for r, a, b in zip(rows, orig, hedge):
        print(f"  {a:14.3f}{b:16.3f}  {r['question'][:60]}")
    print(f"  {'-' * 30}\n  {orig.mean():14.3f}{hedge.mean():16.3f}  mean")
    print(f"\n  mean shift {hedge.mean() - orig.mean():+.3f}; toward 0.5 is spread "
          f"(trustworthy), toward 0 is penalty (the reader punishes hedging)")
    print(f"  |p-0.5| mean: orig {np.abs(orig - 0.5).mean():.3f} -> "
          f"hedged {np.abs(hedge - 0.5).mean():.3f}")

    out = {"condition": args.condition, "n": len(rows),
           "orig": orig.tolist(), "hedged": hedge.tolist(),
           "mean_orig": float(orig.mean()), "mean_hedged": float(hedge.mean()),
           "rows": [{"question": r["question"], "answer": r["answer"],
                     "response": r["response"], "hedged": h}
                    for r, h in zip(rows, hedged)]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--files", nargs="*", help="corpus files (default: src/qwen_jax/**/*.py)")
        s.add_argument("--root", help="directory the corpus file headers are relative to")
        s.add_argument("--description", default=DESCRIPTION)

    g = sub.add_parser("genqa")
    common(g)
    g.add_argument("--n-in", type=int, default=96)
    g.add_argument("--n-out", type=int, default=32)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--chunk-min", type=int, default=512)
    g.add_argument("--chunk-max", type=int, default=1536)
    g.add_argument("--max-new", type=int, default=96)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--out", default=str(OUT_DIR / "qa.jsonl"))

    r = sub.add_parser("respond")
    common(r)
    r.add_argument("--qa", default=str(OUT_DIR / "qa.jsonl"))
    r.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    r.add_argument("--cartridge", default=str(TRAINED))
    r.add_argument("--wrong", default=str(WRONG))
    r.add_argument("--samples", type=int, default=1)
    r.add_argument("--note", help="text placed between the cartridge and the "
                                  "conversation, as `student_note` does in training")
    r.add_argument("--label", help="rename the `trained` condition in the output, "
                                  "so several cartridges can be compared")
    r.add_argument("--limit", type=int, help="only the first N questions")
    r.add_argument("--batch", type=int, default=4)
    r.add_argument("--max-new", type=int, default=256)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default=str(OUT_DIR / "responses.jsonl"))

    d = sub.add_parser("read")
    d.add_argument("--responses", default=str(OUT_DIR / "responses.jsonl"))
    d.add_argument("--batch", type=int, default=8)
    d.add_argument("--max-new", type=int, default=24)
    d.add_argument("--out", default=str(OUT_DIR / "read.jsonl"))

    s = sub.add_parser("score")
    s.add_argument("--read", default=str(OUT_DIR / "read.jsonl"))
    s.add_argument("--clip", type=float, default=1e-3)
    s.add_argument("--out", default=str(OUT_DIR / "scores.json"))

    c = sub.add_parser("canary")
    c.add_argument("--responses", default=str(OUT_DIR / "responses.jsonl"))
    c.add_argument("--condition", default="trained")
    c.add_argument("--n", type=int, default=16)
    c.add_argument("--batch", type=int, default=4)
    c.add_argument("--max-new", type=int, default=384)
    c.add_argument("--out", default=str(OUT_DIR / "canary.json"))

    args = p.parse_args()
    {"genqa": cmd_genqa, "respond": cmd_respond, "read": cmd_read,
     "score": cmd_score, "canary": cmd_canary}[args.cmd](args)


if __name__ == "__main__":
    main()
