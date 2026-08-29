"""Reader-reward RL on a cartridge: GRPO against a frozen reader's forecast.

    python scripts/reader_rl.py data  --pool runs/rl/pool-raw.jsonl
    python scripts/reader_rl.py train --steps 300 --out runs/rl/
    python scripts/reader_rl.py eval  --cartridge runs/rl/step0010.safetensors --label rl10

Stage 2 of Band et al. (arXiv:2404.00474), with a cartridge as the entire
trainable surface. The policy is the frozen base model reading a trainable KV
prefix; the reward is what a *reader* -- the same frozen model, no cartridge,
no corpus -- forecasts for the gold answer after reading the policy's prose.
Nothing else moves: 302 MB of K/V against a reward signal that only ever sees
free-form text.

    q = (support(gold) + u/K) / (support(all answers offered) + u)
    r = log clip(q, 1e-3)

with `u` the mass the reader puts on "this passage leaves the question
undetermined" and K = 4. The first two RL runs used an entailment reward,
`P_reader(Yes | supports gold)`, and it could not tell an honest hedge from a
confident lie: both scored the clip floor, 93% of rewards sat at one of two
extremes, and GRPO duly drove the hedge rate to 2.5%. Reading the undetermined
mass as *ignorance* rather than as evidence against the gold is what puts an
abstention above a fabrication, at log(1/K). See `reward_audit.py`, which
measured this ordering before it was trained on.

Training starts from the summary-distilled cartridge
(`runs/summary/summary.safetensors`), which already hedges at ~19%, so the
policy has hedged rollouts to compare against rather than needing to discover
them; the KL leash references that same cartridge.

What the reward deliberately does NOT contain: any abstention bonus or
out-of-corpus penalty. The goal is not a model that refuses everything outside
`src/qwen_jax` -- it should answer freely from general pretraining knowledge
and only stop fabricating corpus-specific detail. The reader reward already has
that shape: general-knowledge content is never penalised, because the reader is
only ever asked about the gold answer to an in-corpus question. v0 trains on
in-corpus questions alone, which is Band's setting. If an out-of-corpus term is
ever added it has to punish fabricated corpus-specifics, never answering from
general knowledge.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
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
import jax.numpy as jnp
import numpy as np

from cartridge import load_model, load_tokenizer
from reader_score import (QUERY, READER_SYS, TRAINED, encode_suffix, encode_user, generate,
                          is_unknown, named, p_yes, read_jsonl, write_jsonl)
# Reward prompts and hedge metrics are imported, never re-typed: the reward RL
# optimises has to be exactly the one the audit measured, and the hedge rate has
# to be exactly the one the summary-distill run reported.
from reward_audit import EXTRACT_MANY, FORECAST_HEAD, PHRASINGS, UNDETERMINED
from summary_distill import hedge_rate, hedged, pct_rate

OUT_DIR = REPO / "runs/rl"
EVAL_SETS = [REPO / "runs/reader/qa.jsonl", REPO / "runs/probe/qa.jsonl"]
HELDOUT = REPO / "runs/cart/heldout.jsonl"
SUMMARY = REPO / "runs/summary/summary.safetensors"


# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------


def cmd_data(args):
    """Decontaminate fresh genqa pools against the held-out eval sets.

    The eval sets are the whole basis of every downstream claim, so nothing
    that shares a question or a named entity with them may be trained on.
    """
    import re

    ents, qs = set(), set()
    for path in EVAL_SETS:
        for r in read_jsonl(path):
            qs.add(re.sub(r"[^a-z0-9 ]", "", r["question"].lower()).strip())
            ents.update(n.lower() for n in named(r["question"]))
    pool = [r for p in args.pool for r in read_jsonl(p)]
    kept, seen = [], set()
    drop_q = drop_e = drop_dup = 0
    for r in pool:
        if r["kind"] != "in" or not r.get("answer"):
            continue
        q = re.sub(r"[^a-z0-9 ]", "", r["question"].lower()).strip()
        if q in qs:
            drop_q += 1
        elif q in seen:
            drop_dup += 1
        elif any(n.lower() in ents for n in named(r["question"])):
            drop_e += 1
        else:
            seen.add(q)
            kept.append(r)
    write_jsonl(kept, args.out)
    print(f"  {len(pool)} -> {len(kept)}: dropped {drop_q} eval questions, "
          f"{drop_e} sharing a named entity with an eval question, {drop_dup} duplicates")


def cmd_filter(args):
    """Keep only questions the starting policy sometimes gets right.

    Group-relative advantages are zero whenever all `G` rollouts of a question
    score the same, and reader forecasts are near-bimodal, so a question the CE
    cartridge always fails (or always passes) contributes no gradient however
    long it is trained on. DAPO calls this dynamic sampling; done once up front
    it is the same saving without the per-step resampling. `successes` is the
    number of rollouts whose reader forecast on the gold answer exceeds 0.5.
    """
    from qwen_jax.cartridge import Cartridge

    tokenizer = load_tokenizer()
    pool = read_jsonl(args.data)
    cached = {}
    for path in args.cache or []:
        if Path(path).exists():
            cached.update({r["question"]: r["successes"] for r in read_jsonl(path)
                           if "successes" in r})
    rows = [{**r, "successes": cached[r["question"]]} for r in pool if r["question"] in cached]
    todo = [r for r in pool if r["question"] not in cached]
    print(f"{len(pool)} questions: {len(rows)} success counts reused, {len(todo)} to sample")
    model = load_model()
    cart = Cartridge.load(args.cartridge)
    key = jax.random.key(args.seed)
    t0 = time.time()

    for i in range(0, len(todo), args.chunk):
        items = todo[i:i + args.chunk]
        ids, mask, tmask, texts, key = sample_rollouts(
            model, tokenizer, cart, items, group=args.group, max_new=args.max_new,
            temperature=args.temperature, key=key, gen_batch=args.gen_batch)
        _, p, _ = rewards(model, tokenizer, items, texts, group=args.group,
                          clip_lo=args.clip_lo, batch=args.reader_batch)
        succ = (p.reshape(len(items), args.group) > 0.5).sum(-1)
        rows += [{**it, "successes": int(s)} for it, s in zip(items, succ)]
        print(f"  {len(rows)}/{len(cached) + len(todo)} questions ({time.time() - t0:.0f}s)", flush=True)

    hist = np.bincount([r["successes"] for r in rows], minlength=args.group + 1)
    kept = [r for r in rows if 0 < r["successes"] < args.group]
    write_jsonl(rows, Path(args.out).with_suffix(".all.jsonl"))
    write_jsonl(kept, args.out)
    print(f"\npass-rate histogram over {len(rows)} questions (G={args.group}):")
    for s, n in enumerate(hist):
        print(f"  {s}/{args.group} correct: {n:4d}  {'#' * int(60 * n / max(hist.max(), 1))}")
    print(f"learnable (1..{args.group - 1}): {len(kept)}  "
          f"({len(kept) / max(len(rows), 1):.0%} of the pool)")


# -----------------------------------------------------------------------------
# rollouts
# -----------------------------------------------------------------------------


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Rollouts:
    """One step's sampled responses, laid out at a fixed shape.

    Rows are `[left pad | prompt | response | right pad]`. `tmask` marks the
    positions holding response tokens (the stop token included -- learning when
    to stop is part of the policy); the loss shifts it by one to line targets
    up with the hidden states that predict them.
    """

    ids: jnp.ndarray        # (n, S) int32
    mask: jnp.ndarray       # (n, S) int32, attention over real tokens
    tmask: jnp.ndarray      # (n, S) float32, 1 where ids[t] is a response token
    adv: jnp.ndarray        # (n,) float32
    ref_lp: jnp.ndarray     # (n, S) float32, per-token logprob under the reference


def sample_rollouts(model, tokenizer, cart, items, *, group, max_new, temperature, key,
                    gen_batch, pad_to=64):
    """`group` responses per item, plus the arrays the gradient step needs."""
    from qwen_jax import chat
    from qwen_jax.selfstudy import _left_pad

    im_end = tokenizer.convert_tokens_to_ids(chat.IM_END)
    prompts = [encode_suffix(tokenizer, QUERY.format(q=it["question"]))
               for it in items for _ in range(group)]
    width = -(-max(len(p) for p in prompts) // pad_to) * pad_to
    prefix = cart.prefix(model.cache_dtype())

    ids_all, mask_all, tmask_all, texts = [], [], [], []
    for i in range(0, len(prompts), gen_batch):
        chunk = prompts[i:i + gen_batch]
        pad = chunk + [chunk[-1]] * (gen_batch - len(chunk))
        pids, pmask = _left_pad(pad, tokenizer.pad_token_id, width)
        key, sub = jax.random.split(key)
        out = model.generate(
            input_ids=jnp.asarray(pids), attention_mask=jnp.asarray(pmask), prefix=prefix,
            max_new_tokens=max_new, key=sub, temperature=temperature,
            stop_token_id=im_end, pad_token_id=im_end, progress_bar=False,
        )
        toks = np.asarray(out.tokens)[: len(chunk)]
        for row, pm in zip(toks, pmask[: len(chunk)]):
            gen = row[width:].tolist()
            n = gen.index(im_end) + 1 if im_end in gen else len(gen)
            m = np.concatenate([pm, np.zeros(max_new, np.int32)])
            t = np.zeros(width + max_new, np.float32)
            m[width:width + n] = 1
            t[width:width + n] = 1.0
            ids_all.append(row)
            mask_all.append(m)
            tmask_all.append(t)
            body = gen[: n - 1] if im_end in gen else gen
            texts.append(tokenizer.decode(body, skip_special_tokens=False).strip())
    return (np.stack(ids_all), np.stack(mask_all), np.stack(tmask_all), texts, key)


def _options(answer, text):
    """The answers the passage puts forward, with the gold always competing."""
    got = [ln.strip(" -*\t0123456789.") for ln in text.splitlines() if ln.strip()][:3]
    got = [g for g in got if g and not is_unknown(g)
           and re.sub(r"[^A-Za-z]", "", g).upper() not in ("NONE", "NOANSWER")]
    return [answer] + [g for g in got if g.lower() != answer.lower()], not got


def rewards(model, tokenizer, items, texts, *, group, clip_lo, batch, prior_k=4,
            extract_new=24):
    """F_prior: the reader's posterior in the gold answer, with a flat-prior fallback.

    Three reader queries per rollout. What the passage puts forward (a), how far
    the reader supports each of those and the gold (b), and how far it thinks the
    passage settles nothing at all (c). The undetermined mass is then read as
    *ignorance* rather than evidence against:

        q = (support(gold) + u/K) / (support(all) + u)

    so a passage that determines nothing is worth log(1/K) -- above a confident
    fabrication, below a correct answer. That is the inequality the old
    entailment reward could not express, and the reason it scored an honest
    hedge and a confident lie identically at the clip floor. K=4 was the
    audited setting. Prompts are lifted verbatim from `reward_audit.py` so the
    reward RL optimises is the one that was measured.
    """
    expanded = [it for it in items for _ in range(group)]   # rollouts are item-major
    assert len(expanded) == len(texts), f"{len(expanded)} items vs {len(texts)} responses"

    listed = generate(model, tokenizer,
                      [encode_user(tokenizer, READER_SYS,
                                   EXTRACT_MANY.format(p=t, q=it["question"]))
                       for it, t in zip(expanded, texts)],
                      max_new=extract_new, temperature=0.0, key=jax.random.key(0),
                      batch=batch)
    flat, index, empty = [], [], []
    for it, t, text in zip(expanded, texts, listed):
        opts, none = _options(it["answer"], text)
        empty.append(none)
        index.append((len(flat), len(opts)))
        flat += [encode_user(tokenizer, READER_SYS,
                             FORECAST_HEAD.format(p=t, q=it["question"], a=o) + PHRASINGS[0])
                 for o in opts]
    p_all = np.asarray(p_yes(model, tokenizer, flat, batch=batch), np.float64)
    p_none = np.asarray(p_yes(model, tokenizer,
                              [encode_user(tokenizer, READER_SYS,
                                           UNDETERMINED.format(p=t, q=it["question"]))
                               for it, t in zip(expanded, texts)], batch=batch), np.float64)
    q = np.zeros(len(texts))
    for i, (start, n) in enumerate(index):
        share = p_all[start:start + n]
        q[i] = (share[0] + p_none[i] / prior_k) / max(share.sum() + p_none[i], 1e-9)
    return np.log(np.clip(q, clip_lo, 1.0)), q, np.asarray(empty, float)


def advantages(r, *, group, normalize, eps=1e-4):
    """Group-relative: centre within each question's `group` rollouts."""
    g = r.reshape(-1, group)
    a = g - g.mean(-1, keepdims=True)
    if normalize:
        a = a / (g.std(-1, keepdims=True) + eps)
    live = float(np.mean(g.std(-1) > 1e-9))
    return a.reshape(-1).astype(np.float32), live


# -----------------------------------------------------------------------------
# policy gradient
# -----------------------------------------------------------------------------


def blockwise_token_logprob(lm_head, hidden, targets, mask, *, block):
    """log p(target_t) at every position, `block` positions at a time.

    Same trick as `distill.blockwise_kl`: the vocabulary projection is far too
    big to materialise for a whole sequence, so the positions are scanned in
    blocks and each block is rematerialised on the backward pass instead of
    being kept.
    """
    b, s, _ = hidden.shape
    if s % block:
        raise ValueError(f"seq {s} must be a multiple of block {block}")
    n = s // block
    blocks = lambda x: x.reshape(b, n, block, *x.shape[2:]).swapaxes(0, 1)

    @jax.checkpoint
    def one(args):
        hs, tg, m = args
        lp = jax.nn.log_softmax(lm_head(hs).astype(jnp.float32), axis=-1)
        return jnp.take_along_axis(lp, tg[..., None], axis=-1)[..., 0] * m

    out = jax.lax.map(one, (blocks(hidden), blocks(targets), blocks(mask)))
    return out.swapaxes(0, 1).reshape(b, s)


def token_logprobs(model, prefix, ids, mask, tmask, *, block):
    """Per-token logprob of the response tokens, under the policy given `prefix`.

    Position `t` predicts token `t+1`, so both targets and their mask are
    shifted left one place and the final column is dropped -- which also keeps
    the sequence length a clean multiple of `block`.
    """
    hidden, _, _ = model.model(input_ids=ids, attention_mask=mask, prefix=prefix)
    z = jnp.zeros((ids.shape[0], 1), ids.dtype)
    targets = jnp.concatenate([ids[:, 1:], z], axis=1)
    tgt_mask = jnp.concatenate([tmask[:, 1:], z.astype(tmask.dtype)], axis=1)
    return blockwise_token_logprob(model.get_lm_head(), hidden, targets, tgt_mask, block=block)


def grpo_loss(model, cart, r: Rollouts, beta, denom, *, block):
    """Policy gradient with a KL leash to the frozen reference cartridge.

    The leash uses the k3 estimator on the sampled tokens (Schulman):
    `exp(d) - d - 1` for `d = logp_ref - logp`. It needs one extra no-grad
    forward per step, against a full-distribution KL that would need the whole
    151936-way distribution from both policies at every rollout token and would
    roughly double the lm_head cost of the backward pass. Rollouts dominate the
    step either way, so the cheap estimator buys real time; it is non-negative
    and low-variance, which is what a leash needs.
    """
    lp = token_logprobs(model, cart.prefix(model.cache_dtype()), r.ids, r.mask, r.tmask,
                        block=block)
    shifted = jnp.concatenate([r.tmask[:, 1:], jnp.zeros((r.ids.shape[0], 1), r.tmask.dtype)],
                              axis=1)
    pg = -jnp.sum(r.adv[:, None] * lp) / denom
    d = (r.ref_lp - lp) * shifted
    kl = jnp.sum(jnp.exp(d) - d - 1.0) / jnp.maximum(jnp.sum(shifted), 1.0)
    return pg + beta * kl, (pg, kl, jnp.sum(lp) / jnp.maximum(jnp.sum(shifted), 1.0))


@dataclasses.dataclass
class Trainer:
    """Jitted microbatch gradient plus an Adam step on the cartridge leaves."""

    optimizer: object
    block: int = 128

    def __post_init__(self):
        self._grad = jax.jit(functools.partial(self._grad_impl), static_argnames=("block",))
        self._ref = jax.jit(token_logprobs, static_argnames=("block",))

    @staticmethod
    def _grad_impl(model, cart, r, beta, denom, *, block):
        def f(params):
            return grpo_loss(model, cart.with_params(params), r, beta, denom, block=block)

        (loss, aux), grads = jax.value_and_grad(f, has_aux=True)(cart.params())
        return grads, loss, aux

    def ref_logprobs(self, model, ref_cart, ids, mask, tmask):
        return self._ref(model, ref_cart.prefix(model.cache_dtype()), ids, mask, tmask,
                         block=self.block)

    def step(self, model, cart, opt_state, micros, beta, denom):
        import optax

        total = None
        stats = np.zeros(4)
        for r in micros:
            g, loss, (pg, kl, mlp) = self._grad(model, cart, r, beta, denom, block=self.block)
            total = g if total is None else jax.tree_util.tree_map(jnp.add, total, g)
            stats += np.array([float(loss), float(pg), float(kl), float(mlp)])
        grads = cart.mask_grads(total)
        updates, opt_state = self.optimizer.update(grads, opt_state, cart.params())
        cart = cart.with_params(optax.apply_updates(cart.params(), updates))
        return cart.advance(1), opt_state, stats / len(micros)


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------


def to_rollouts(ids, mask, tmask, adv, ref_lp, sl):
    return Rollouts(ids=jnp.asarray(ids[sl]), mask=jnp.asarray(mask[sl]),
                    tmask=jnp.asarray(tmask[sl]), adv=jnp.asarray(adv[sl]),
                    ref_lp=ref_lp[sl])


def cmd_train(args):
    import optax

    from qwen_jax.cartridge import Cartridge

    tokenizer = load_tokenizer()
    pool = read_jsonl(args.data)
    print(f"{len(pool)} training questions")
    model = load_model()
    cart = Cartridge.load(args.cartridge)
    ref = Cartridge.load(args.reference or args.cartridge)
    print(f"cartridge p={cart.length}, {cart.keys.size * 2 * 4 / 1e6:.0f} MB float32")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    schedule = (optax.linear_schedule(0.0, args.lr, args.warmup) if args.warmup
                else optax.constant_schedule(args.lr))
    trainer = Trainer(optax.adam(schedule), block=args.block)
    state = trainer.optimizer.init(cart.params())
    rng = random.Random(args.seed)
    key = jax.random.key(args.seed)
    denom = float(args.batch_q * args.group * args.max_new)
    log_path, sample_path = out_dir / "log.jsonl", out_dir / "samples.jsonl"
    t_start = time.time()

    for step in range(1, args.steps + 1):
        t0 = time.time()
        items = rng.sample(pool, min(args.batch_q, len(pool)))
        ids, mask, tmask, texts, key = sample_rollouts(
            model, tokenizer, cart, items, group=args.group, max_new=args.max_new,
            temperature=args.temperature, key=key, gen_batch=args.gen_batch)
        t_roll = time.time() - t0

        t0 = time.time()
        r, p, empty = rewards(model, tokenizer, items, texts, group=args.group,
                              clip_lo=args.clip_lo, batch=args.reader_batch,
                              prior_k=args.prior_k, extract_new=args.extract_new)
        adv, live = advantages(r, group=args.group, normalize=not args.no_norm)
        t_read = time.time() - t0

        t0 = time.time()
        ref_lp = trainer.ref_logprobs(model, ref, jnp.asarray(ids), jnp.asarray(mask),
                                      jnp.asarray(tmask))
        micros = [to_rollouts(ids, mask, tmask, adv, ref_lp, slice(i, i + args.micro))
                  for i in range(0, len(ids), args.micro)]
        cart, state, (loss, pg, kl, mlp) = trainer.step(
            model, cart, state, micros, jnp.float32(args.beta), jnp.float32(denom))
        t_grad = time.time() - t0

        lens = tmask.sum(-1)
        floor = float(np.log(args.clip_lo))
        # The whole point of the new reward is that it has an interior. Track the
        # shape of the distribution, not just its mean, or a collapse back to
        # bimodal would be invisible until the eval.
        hist = np.histogram(r, bins=[floor - 1e-6, -4.0, -2.0, -1.0, -0.4, -0.1, 0.01])[0]
        row = {"step": step, "reward": float(r.mean()), "p_gold": float(p.mean()),
               "live": live, "kl": float(kl), "loss": float(loss), "pg": float(pg),
               "logp_tok": float(mlp), "len": float(lens.mean()),
               "interior": float(np.mean((r > floor + 1e-6) & (r < -0.1))),
               "at_floor": float(np.mean(r <= floor + 1e-6)),
               "hedge": hedge_rate(texts), "stated_pct": pct_rate(texts),
               "abstain": float(empty.mean()), "hist": hist.tolist(),
               "t_roll": t_roll, "t_read": t_read, "t_grad": t_grad,
               "t_total": t_roll + t_read + t_grad}
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  step {step:4d}  r {r.mean():6.2f}  q {p.mean():.3f}  live {live:.2f}  "
              f"int {row['interior']:.2f}  hedge {row['hedge']:.2f}  "
              f"abst {row['abstain']:.2f}  KL {float(kl):.4f}  len {lens.mean():5.1f}  "
              f"hist {hist.tolist()}  "
              f"[{t_roll:4.1f}/{t_read:4.1f}/{t_grad:4.1f}s]", flush=True)

        if step % args.sample_every == 0 or step == 1:
            with open(sample_path, "a") as f:
                for j in range(min(args.n_samples, len(texts))):
                    f.write(json.dumps({"step": step, "question": items[j // args.group]["question"],
                                        "answer": items[j // args.group]["answer"],
                                        "reward": float(r[j]), "p": float(p[j]),
                                        "hedged": hedged(texts[j]),
                                        "abstain": bool(empty[j]),
                                        "response": texts[j]}) + "\n")
        if step % args.save_every == 0 or step == args.steps:
            cart.save(out_dir / f"step{step:04d}.safetensors")
            print(f"    saved step{step:04d} ({time.time() - t_start:.0f}s total)", flush=True)


# -----------------------------------------------------------------------------
# eval
# -----------------------------------------------------------------------------


def cmd_eval(args):
    """Held-out reader score plus held-out KL, for one checkpoint."""
    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    run = lambda c: subprocess.run(c, cwd=REPO, check=True)
    rs = [sys.executable, str(REPO / "scripts/reader_score.py")]
    run(rs + ["respond", "--qa", args.qa, "--conditions", "trained",
              "--cartridge", args.cartridge, "--label", args.label,
              "--out", str(d / f"resp-{args.label}.jsonl")])
    run(rs + ["read", "--responses", str(d / f"resp-{args.label}.jsonl"),
              "--out", str(d / f"read-{args.label}.jsonl")])
    run(rs + ["score", "--read", str(d / f"read-{args.label}.jsonl"),
              "--out", str(d / f"scores-{args.label}.json")])
    run([sys.executable, str(REPO / "scripts/cartridge.py"), "eval",
         "--cartridge", args.cartridge, "--heldout", args.heldout])


# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data")
    d.add_argument("--pool", nargs="+", default=[str(OUT_DIR / "pool-raw.jsonl")])
    d.add_argument("--out", default=str(OUT_DIR / "train-pool.jsonl"))

    f = sub.add_parser("filter")
    f.add_argument("--data", default=str(OUT_DIR / "train-pool.jsonl"))
    f.add_argument("--cartridge", default=str(TRAINED))
    f.add_argument("--group", type=int, default=8)
    f.add_argument("--chunk", type=int, default=4, help="questions per rollout pass")
    f.add_argument("--gen-batch", type=int, default=8)
    f.add_argument("--reader-batch", type=int, default=8)
    f.add_argument("--max-new", type=int, default=192)
    f.add_argument("--temperature", type=float, default=0.9)
    f.add_argument("--clip-lo", type=float, default=1e-3)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--cache", nargs="*", help="prior *.all.jsonl files to reuse counts from")
    f.add_argument("--out", default=str(OUT_DIR / "learnable-pool.jsonl"))

    t = sub.add_parser("train")
    t.add_argument("--data", default=str(OUT_DIR / "train-pool.jsonl"))
    t.add_argument("--cartridge", default=str(SUMMARY))
    t.add_argument("--reference", help="KL reference (default: the starting cartridge)")
    t.add_argument("--steps", type=int, default=300)
    t.add_argument("--batch-q", type=int, default=4, help="questions per step")
    t.add_argument("--group", type=int, default=8, help="rollouts per question (G)")
    t.add_argument("--micro", type=int, default=4, help="sequences per gradient microbatch")
    t.add_argument("--gen-batch", type=int, default=16)
    t.add_argument("--reader-batch", type=int, default=8)
    t.add_argument("--max-new", type=int, default=192)
    t.add_argument("--temperature", type=float, default=0.9)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=10)
    t.add_argument("--beta", type=float, default=0.02, help="KL leash coefficient")
    t.add_argument("--clip-lo", type=float, default=1e-3)
    t.add_argument("--prior-k", type=int, default=4,
                   help="F_prior K: an undetermined passage is worth log(1/K)")
    t.add_argument("--extract-new", type=int, default=24,
                   help="max_new for the reward's extraction query")
    t.add_argument("--no-norm", action="store_true", help="do not divide by group std")
    t.add_argument("--block", type=int, default=128)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save-every", type=int, default=25)
    t.add_argument("--sample-every", type=int, default=5)
    t.add_argument("--n-samples", type=int, default=4)
    t.add_argument("--out", default=str(OUT_DIR))

    e = sub.add_parser("eval")
    e.add_argument("--cartridge", required=True)
    e.add_argument("--label", required=True)
    e.add_argument("--qa", default=str(REPO / "runs/reader/qa.jsonl"))
    e.add_argument("--heldout", default=str(HELDOUT))
    e.add_argument("--out", default=str(OUT_DIR / "eval"))

    args = p.parse_args()
    {"data": cmd_data, "filter": cmd_filter, "train": cmd_train,
     "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
