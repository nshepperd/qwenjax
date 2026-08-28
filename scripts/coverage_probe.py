"""Does the model know whether the cartridge it is holding covers the question?

    python scripts/coverage_probe.py data     --extra runs/probe/qa-out64.jsonl
    python scripts/coverage_probe.py capture  --qa runs/probe/qa.jsonl
    python scripts/coverage_probe.py analyze          # naive in/out, secondary
    python scripts/coverage_probe.py factorial        # the 2x2, primary

`reader_score.py` showed the trained cartridge answering unanswerable questions
at 99% confidence. Either the model already represents "this is outside my
corpus" and merely never says it -- calibration training would then have to
surface a signal that exists -- or it does not represent it at all and the
signal has to be installed. This script asks which, by reading the label off a
single forward pass, before any token is generated.

The obvious experiment does not work. Ask a probe to sort in-corpus from
out-of-corpus questions with one cartridge loaded and it scores ~0.87 -- but it
scores just as well with *no cartridge loaded at all*, because "is this
question about qwen-jax" is a surface property of the question that needs no
cartridge knowledge whatsoever. That measurement is kept as `analyze`, for
sizing the confound.

`factorial` is the real design. Two cartridges on unrelated corpora, A =
qwen-jax and B = tetris, and two question sets, QA and QB, each in-corpus for
its own cartridge. Every question is run under both cartridges, and the label
is whether the loaded cartridge covers it:

        A loaded   B loaded
    QA     1          0
    QB     0          1

The label is the XOR of question topic and loaded-cartridge identity. Each
question appears once with each label, so *any* per-question feature -- topic,
wording, length -- is exactly uninformative; so is cartridge identity, which is
constant down each column. A linear probe cannot build XOR out of two
independent linear features. So if this is linearly decodable, the model itself
has already computed the question-context match, and the probe is reading a
representation rather than reconstructing one.

`none` (no cartridge, description only) is captured as a falsification test:
the factorial probe applied there should call everything out-of-scope, because
nothing is loaded. It does not, quite -- the fitted direction keeps a topic
component that is inert in the 2x2 (it cancels between the columns) but shows
up once the design is broken. So the decisive version is the paired contrast,
x(q, A loaded) - x(q, B loaded), which cancels every per-question term
algebraically instead of controlling for it: whatever separates QA from QB in
that difference can only be the interaction.
"""
from __future__ import annotations

import argparse
import json
import os
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

from reader_score import QUERY, TRAINED, WRONG, auroc, encode_suffix, read_jsonl, write_jsonl

OUT_DIR = REPO / "runs/probe"
COMPOSE = REPO / "runs/compose/head_sep.npy"
CONDITIONS = ["trained", "init", "none", "wrong"]


# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------


def cmd_data(args):
    """One question file for both analyses: qwen-jax in/out, plus tetris in.

    Deduplicated by question text -- the generators repeat themselves, and a
    question appearing twice would put the same activations in two CV folds.
    """
    groups = [(args.base, "qwenjax", args.base_seed)]
    groups += [(p, "qwenjax", s) for p, s in zip(args.extra, args.extra_seeds)]
    groups += [(p, "tetris", s) for p, s in zip(args.tetris, args.tetris_seeds)]
    rows, seen, dups = [], set(), 0
    for path, topic, seed in groups:
        if not Path(path).exists():
            print(f"  (skipping missing {path})")
            continue
        for r in read_jsonl(path):
            if r["question"] in seen:
                dups += 1
                continue
            seen.add(r["question"])
            rows.append({"question": r["question"], "answer": r.get("answer"),
                         "kind": r["kind"], "topic": topic, "seed": seed})

    def cell(topic, kind):
        return [r for r in rows if r["topic"] == topic and r["kind"] == kind]

    a_in, a_out, b_in = cell("qwenjax", "in"), cell("qwenjax", "out"), cell("tetris", "in")
    if args.balance:  # the factorial wants QA and QB the same size
        keep = min(len(a_in), len(b_in))
        a_in, b_in = a_in[:keep], b_in[:keep]
        a_out = a_out[: len(a_in)]
    out = [r for trio in zip(a_in, a_out, b_in) for r in trio]  # interleaved
    longest = max(len(a_in), len(a_out), len(b_in))
    for lst in (a_in, a_out, b_in):
        out += lst[min(len(a_in), len(a_out), len(b_in)):longest]
    out = list({r["question"]: r for r in out}.values())
    write_jsonl(out, args.out)
    print(f"  qwenjax in {len(a_in)} / qwenjax out {len(a_out)} / tetris in {len(b_in)}"
          f"   ({dups} duplicate questions dropped)")


# -----------------------------------------------------------------------------
# capture
# -----------------------------------------------------------------------------


@jax.jit
def capture(model, input_ids, prefix, last):
    """Residual stream and cartridge attention mass at position `last`.

    The decoder loop of `qwen_jax.probe.probe` -- dense float32 attention, so
    the weights exist to be read -- with the hidden state kept at every layer
    and everything reduced to one query position.

    `input_ids` may be right-padded to a bucket width: attention is causal, so
    tokens after `last` cannot reach it and the readout is exact regardless.
    That is what keeps this to a handful of compilations instead of one per
    distinct question length.
    """
    from qwen_jax.probe import _layer_attention

    lm = model.model.language_model
    ids = input_ids[None]
    s = ids.shape[1]
    p = prefix.length

    h = lm.embed_tokens(ids)
    pos = jnp.arange(s) + p
    cos, sin = lm.rotary_emb(jnp.broadcast_to(pos[None, None, :], (3, 1, s)))
    cos, sin = cos.astype(h.dtype), sin.astype(h.dtype)

    resid, mass = [h[0, last]], []
    for i, layer in enumerate(lm.layers):
        x = layer.input_layernorm(h)
        a, w = _layer_attention(
            layer.self_attn, x, cos, sin, prefix.keys[i], prefix.values[i], p,
        )
        h = h + a
        h = h + layer.mlp(layer.post_attention_layernorm(h))
        resid.append(h[0, last])
        mass.append(w[:, last, :p].sum(-1))
    return jnp.stack(resid), jnp.stack(mass), lm.norm(h)[0, last]


def prefixes(model, tokenizer, corpus, conditions, args):
    """The four cartridge conditions as KVPrefixes. Mirrors reader_score.respond."""
    from cartridge import init_cartridge
    from qwen_jax.cartridge import Cartridge

    trained = Cartridge.load(args.cartridge)
    desc = trained.meta.description
    dt = model.cache_dtype()
    build = {
        "trained": lambda: trained.prefix(dt),
        "wrong": lambda: Cartridge.load(args.wrong).prefix(dt),
        "init": lambda: init_cartridge(model, tokenizer, corpus, trained.length, desc).prefix(dt),
        "none": lambda: init_cartridge(model, tokenizer, corpus, 0, desc).prefix(dt),
    }
    return {c: build[c]() for c in conditions}


def cmd_capture(args):
    from cartridge import load_corpus, load_model, load_tokenizer

    tokenizer = load_tokenizer()
    corpus = load_corpus(tokenizer, args.files, args.root)
    qa = read_jsonl(args.qa)
    qa = qa[: args.limit] if args.limit else qa
    model = load_model()
    pres = prefixes(model, tokenizer, corpus, args.conditions, args)
    for c, pre in pres.items():
        print(f"  {c}: prefix p={pre.length}")

    # One width for every question, so each condition compiles exactly once.
    # The decoder loop is unrolled over 36 layers and costs minutes to compile;
    # a shape per question length would dwarf the capture itself.
    prompts = [encode_suffix(tokenizer, QUERY.format(q=r["question"])) for r in qa]
    lasts = [len(ids) - 1 for ids in prompts]
    w = -(-(max(lasts) + 1) // args.bucket) * args.bucket
    padded = [np.pad(np.asarray(ids, np.int32), (0, w - len(ids))) for ids in prompts]
    print(f"  {len(qa)} questions, prompt lengths {min(lasts) + 1}-{max(lasts) + 1}, "
          f"padded to {w}")

    labels = np.array([r["kind"] == "in" for r in qa], np.int32)
    topic_a = np.array([r.get("topic", "qwenjax") == "qwenjax" for r in qa], np.int32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for cond, prefix in pres.items():
        resid = np.zeros((len(qa), model.config.text_config.num_hidden_layers + 1,
                          model.config.text_config.hidden_size), np.float16)
        mass = np.zeros((len(qa), model.config.text_config.num_hidden_layers,
                         model.config.text_config.num_attention_heads), np.float32)
        for i, (ids, last) in enumerate(zip(padded, lasts)):
            r, m, final = capture(model, jnp.asarray(ids), prefix,
                                  jnp.asarray(last, jnp.int32))
            resid[i] = np.asarray(r, np.float16)
            mass[i] = np.asarray(m, np.float32)
            if args.check and i == 0:
                ref = model.model(input_ids=jnp.asarray(ids)[None],
                                  attention_mask=jnp.ones((1, len(ids)), jnp.int32),
                                  prefix=prefix)[0][0, last]
                # The probe path is a dense float32 softmax, the model's is the
                # fused bf16 kernel, so agreement is a direction check, not bitwise.
                u = np.asarray(ref, np.float64)
                v = np.asarray(final, np.float64)
                cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
                print(f"    check {cond}: cos={cos:.6f} "
                      f"rel|d|={np.linalg.norm(u - v) / np.linalg.norm(u):.2e} "
                      f"max|ref|={np.abs(u).max():.1f}")
            if (i + 1) % 32 == 0:
                print(f"  {cond} {i + 1}/{len(qa)} ({time.time() - t0:.0f}s)", flush=True)
        np.savez(OUT_DIR / f"act-{cond}.npz", resid=resid, mass=mass, labels=labels,
                 topic_a=topic_a, lengths=np.array(lasts, np.int32) + 1,
                 prefix_len=np.array(prefix.length))
        print(f"  wrote {OUT_DIR / f'act-{cond}.npz'} "
              f"resid{resid.shape} mass{mass.shape} ({time.time() - t0:.0f}s)", flush=True)
    write_jsonl(qa, OUT_DIR / "captured-qa.jsonl")


# -----------------------------------------------------------------------------
# logistic probe
# -----------------------------------------------------------------------------


def fit_logistic(Z, y, *, l2, iters=25):
    """L2-regularised logistic regression by Newton/IRLS. `Z` is (n, k), k small."""
    n, k = Z.shape
    Z = np.concatenate([Z, np.ones((n, 1))], axis=1)
    w = np.zeros(k + 1)
    pen = np.eye(k + 1) * l2
    pen[-1, -1] = 0.0  # never penalise the intercept
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + pen @ w
        s = np.clip(p * (1 - p), 1e-6, None)
        H = Z.T @ (Z * s[:, None]) + pen
        step = np.linalg.solve(H + 1e-6 * np.eye(k + 1), g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def project(X, stats):
    mu, sd, V, sv = stats
    return ((X - mu) / sd) @ V / sv


def make_projection(X, k):
    """Standardise, then whiten onto the top `k` principal directions.

    With 4096 features and ~190 points the raw problem is degenerate; the row
    space has rank <= n, so this loses nothing that a linear probe could have
    used and leaves the Newton solve well conditioned. Fitted on training rows
    only -- the test fold never touches these statistics.
    """
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Xs = (X - mu) / sd
    _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    k = min(k, (S > 1e-8).sum())
    V = Vt[:k].T
    sv = np.maximum(S[:k], 1e-8) / np.sqrt(len(X))  # unit-variance components
    return mu, sd, V, sv


def residualise(Xtr, Xte, ctr, cte):
    """Project a covariate out of both folds, using the training fit only.

    `out` questions run about ten tokens longer than `in` ones -- they come
    from a more formulaic generator -- so prompt length alone separates the
    classes almost perfectly. Any probe can read that instead of coverage.
    Removing the length component is what makes the residual-stream number
    mean what it claims to mean.
    """
    mu = ctr.mean()
    a, b = ctr - mu, cte - mu
    beta = (a @ Xtr) / max(a @ a, 1e-12)
    return Xtr - np.outer(a, beta), Xte - np.outer(b, beta)


def cv_auroc(X, y, *, folds, k, l2, seed=0, covariate=None):
    """Out-of-fold predicted probabilities, and their AUROC."""
    X = X.astype(np.float64)
    rng = np.random.default_rng(seed)
    order = np.concatenate([rng.permutation(np.where(y == c)[0]) for c in (0, 1)])
    fold = np.zeros(len(y), int)
    for c in (0, 1):  # stratified: deal each class round-robin into folds
        idx = order[y[order] == c]
        fold[idx] = np.arange(len(idx)) % folds
    oof = np.zeros(len(y))
    for f in range(folds):
        tr, te = fold != f, fold == f
        Xtr, Xte = X[tr], X[te]
        if covariate is not None:
            Xtr, Xte = residualise(Xtr, Xte, covariate[tr].astype(np.float64),
                                   covariate[te].astype(np.float64))
        stats = make_projection(Xtr, k)
        w = fit_logistic(project(Xtr, stats), y[tr].astype(np.float64), l2=l2)
        z = np.concatenate([project(Xte, stats), np.ones((te.sum(), 1))], axis=1) @ w
        oof[te] = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return auroc(oof, y.astype(bool)), oof


def fit_full(X, y, *, k, l2, covariate=None):
    """A probe on all the data, plus a predict function, for the transfer test.

    The covariate is removed with one fit shared by both sides, so the transfer
    comparison stays like-for-like: the questions, and therefore their lengths,
    are identical under both cartridges.
    """
    X = X.astype(np.float64)
    beta = cmu = None
    if covariate is not None:
        cmu = covariate.mean()
        a = covariate.astype(np.float64) - cmu
        beta = (a @ X) / max(a @ a, 1e-12)
        X = X - np.outer(a, beta)
    stats = make_projection(X, k)
    w = fit_logistic(project(X, stats), y.astype(np.float64), l2=l2)

    def predict(Z, c=None):
        Z = Z.astype(np.float64)
        if beta is not None:
            Z = Z - np.outer(c.astype(np.float64) - cmu, beta)
        z = project(Z, stats)
        return 1.0 / (1.0 + np.exp(-np.clip(np.concatenate(
            [z, np.ones((len(z), 1))], axis=1) @ w, -30, 30)))

    return predict


# -----------------------------------------------------------------------------
# analyze
# -----------------------------------------------------------------------------


def transfer_stats(predict, Xt, Xw, ins, layer, cov=None):
    """How far a probe fit on one cartridge moves when handed another."""
    pt = predict(Xt, cov) if cov is not None else predict(Xt)
    pw = predict(Xw, cov) if cov is not None else predict(Xw)
    return {
        "layer": layer,
        "mean_p_in_trained": float(pt[ins].mean()), "mean_p_in_wrong": float(pw[ins].mean()),
        "shift_in": float(pw[ins].mean() - pt[ins].mean()),
        "mean_p_out_trained": float(pt[~ins].mean()), "mean_p_out_wrong": float(pw[~ins].mean()),
        "shift_out": float(pw[~ins].mean() - pt[~ins].mean()),
        "shift_all": float(pw.mean() - pt.mean()),
        "flip_rate_in": float(((pt[ins] > 0.5) & (pw[ins] <= 0.5)).mean()),
        "frac_called_in_trained": float((pt[ins] > 0.5).mean()),
        "frac_called_in_wrong": float((pw[ins] > 0.5).mean()),
        "train_auroc": auroc(pt, ins),
    }


def cv_auroc_grouped(X, y, groups, *, folds, k, l2, seed=0, covariate=None):
    """Out-of-fold AUROC with whole groups held out together.

    The factorial pools two rows per question -- the same prompt under each
    cartridge -- and they must never straddle the split, or the probe can learn
    a per-question offset in training and reuse it at test.
    """
    X = X.astype(np.float64)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    assign = {g: i % folds for i, g in enumerate(rng.permutation(uniq))}
    fold = np.array([assign[g] for g in groups])
    oof = np.zeros(len(y))
    for f in range(folds):
        tr, te = fold != f, fold == f
        if not (0 < y[tr].sum() < tr.sum()):
            return float("nan"), oof
        Xtr, Xte = X[tr], X[te]
        if covariate is not None:
            Xtr, Xte = residualise(Xtr, Xte, covariate[tr].astype(np.float64),
                                   covariate[te].astype(np.float64))
        stats = make_projection(Xtr, k)
        w = fit_logistic(project(Xtr, stats), y[tr].astype(np.float64), l2=l2)
        z = np.concatenate([project(Xte, stats), np.ones((te.sum(), 1))], axis=1) @ w
        oof[te] = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return auroc(oof, y.astype(bool)), oof


def cmd_factorial(args):
    """The 2x2: is the loaded cartridge the one that covers this question?

    Label = topic XOR loaded-cartridge. Because every question is captured
    under both cartridges, and each cartridge sees both topics, the two main
    effects are exactly balanced: P(label | topic) = P(label | cartridge) = 1/2.
    A linear probe has no access to the product of two features it must read
    independently, so above-chance AUROC here cannot come from either main
    effect. It has to come from a representation of the match itself, computed
    inside the model.
    """
    acts = load_acts([args.a_cond, args.b_cond, "none"])
    for c in (args.a_cond, args.b_cond):
        if c not in acts:
            sys.exit(f"missing act-{c}.npz; run `capture` first")
    A, B = acts[args.a_cond], acts[args.b_cond]
    kind_in = A["labels"].astype(bool)
    topic_a = A["topic_a"].astype(bool)
    qa, qb = kind_in & topic_a, kind_in & ~topic_a  # in-corpus, by topic
    n_layers = A["resid"].shape[1]
    print(f"factorial: QA (qwen-jax, in) {int(qa.sum())}, QB (tetris, in) {int(qb.sum())}; "
          f"cartridges A={args.a_cond} B={args.b_cond}; {n_layers} readout points")
    if not qb.sum():
        sys.exit("no tetris questions in the capture; rebuild with `data --tetris ...`")

    idx = np.where(qa | qb)[0]
    # Row order: every question under A, then every question under B.
    y = np.concatenate([topic_a[idx], ~topic_a[idx]]).astype(np.int32)
    groups = np.concatenate([idx, idx])
    loaded_b = np.concatenate([np.zeros(len(idx)), np.ones(len(idx))]).astype(bool)
    L = np.concatenate([A["lengths"][idx], B["lengths"][idx]]).astype(np.float64)
    print(f"  {len(y)} rows, {int(y.sum())} in-scope / {len(y) - int(y.sum())} out-of-scope")
    print(f"  main effects (should both be 0.500): topic AUROC "
          f"{auroc(np.concatenate([topic_a[idx]] * 2).astype(float), y.astype(bool)):.3f}, "
          f"loaded-cartridge AUROC {auroc(loaded_b.astype(float), y.astype(bool)):.3f}, "
          f"prompt-length AUROC {auroc(L, y.astype(bool)):.3f}")

    def pooled(arr_a, arr_b):
        return np.concatenate([arr_a[idx], arr_b[idx]])

    # --- 1. per-layer probe on the pooled 2x2 --------------------------------
    # Topic is a per-question feature and so carries no information about the
    # XOR label -- but a fitted probe can still pick up a component along it,
    # which only shows itself when the design is broken (the `none` test
    # below). Projecting topic out costs the factorial nothing and leaves a
    # direction that reads the match and nothing else.
    topic_cov = np.concatenate([topic_a[idx], topic_a[idx]]).astype(np.float64)
    by_layer, by_layer_rt, oofs, oofs_rt = [], [], {}, {}
    for l in range(n_layers):
        X = pooled(A["resid"][:, l], B["resid"][:, l])
        kw = {'folds': args.folds, 'k': args.k, 'l2': args.l2, 'seed': args.seed}
        a, oof = cv_auroc_grouped(X, y, groups, **kw)
        b, oof_rt = cv_auroc_grouped(X, y, groups, covariate=topic_cov, **kw)
        by_layer.append(a)
        by_layer_rt.append(b)
        oofs[l] = oof
        oofs_rt[l] = oof_rt
    best = int(np.nanargmax(by_layer))
    best_rt = int(np.nanargmax(by_layer_rt))
    print(f"\n  best factorial AUROC {by_layer[best]:.3f} at layer {best}"
          f"   (topic projected out: {by_layer_rt[best_rt]:.3f} at layer {best_rt})")

    results = {"n_qa": int(qa.sum()), "n_qb": int(qb.sum()), "rows": len(y),
               "a_cond": args.a_cond, "b_cond": args.b_cond,
               "auroc_by_layer": by_layer, "best_layer": best,
               "best_auroc": float(by_layer[best]),
               "auroc_by_layer_topic_removed": by_layer_rt,
               "best_layer_topic_removed": best_rt,
               "best_auroc_topic_removed": float(by_layer_rt[best_rt])}

    # --- 1b. the paired contrast, the airtight form of the same question ----
    # d_q = x(q, A loaded) - x(q, B loaded) cancels every per-question term
    # exactly -- topic, wording, length, tokenisation -- rather than
    # controlling for it statistically. Cartridge identity survives only as a
    # constant offset shared by all q, which the intercept absorbs. What is
    # left is the interaction alone, so a probe that separates QA from QB here
    # is reading the model's own question-context match and can be reading
    # nothing else.
    paired = []
    for l in range(n_layers):
        D = A["resid"][idx][:, l].astype(np.float64) - B["resid"][idx][:, l].astype(np.float64)
        paired.append(cv_auroc(D, topic_a[idx].astype(np.int32), folds=args.folds,
                               k=args.k, l2=args.l2, seed=args.seed)[0])
    best_p = int(np.nanargmax(paired))
    print(f"  paired contrast (A-loaded minus B-loaded, label = topic): "
          f"{paired[best_p]:.3f} at layer {best_p}")
    results["paired_auroc_by_layer"] = paired
    results["paired_best_layer"] = best_p
    results["paired_best_auroc"] = float(paired[best_p])

    # Per-cell mean out-of-fold score at the best layer: all four should be
    # visible, and the diagonal (in-scope) should sit above the off-diagonal.
    oof = oofs[best]
    cells = {"QA x A": (~loaded_b) & np.concatenate([topic_a[idx]] * 2),
             "QB x A": (~loaded_b) & ~np.concatenate([topic_a[idx]] * 2),
             "QA x B": loaded_b & np.concatenate([topic_a[idx]] * 2),
             "QB x B": loaded_b & ~np.concatenate([topic_a[idx]] * 2)}
    results["cell_means"] = {k: float(oof[m].mean()) for k, m in cells.items()}
    print("  out-of-fold P(in-scope) per cell at that layer:")
    for k, m in cells.items():
        print(f"    {k}: {oof[m].mean():.3f}  (n={int(m.sum())}, "
              f"label {'in' if k[1] == k[-1] else 'out'}-scope)")

    # --- 2. confound size: the naive probe, same layers ----------------------
    naive = None
    if args.naive:
        naive = []
        m = topic_a  # qwen-jax questions only, in vs out, cartridge A loaded
        for l in range(n_layers):
            a, _ = cv_auroc_grouped(A["resid"][m][:, l], kind_in[m].astype(np.int32),
                                    np.arange(int(m.sum())), folds=args.folds,
                                    k=args.k, l2=args.l2, seed=args.seed)
            naive.append(a)
        results["naive_auroc_by_layer"] = naive
        results["naive_best"] = float(np.nanmax(naive))
        results["confound_by_layer"] = [n - f for n, f in zip(naive, by_layer)]
        print(f"\n  naive in/out probe (A loaded, qwen-jax questions only): "
              f"best {np.nanmax(naive):.3f} at layer {int(np.nanargmax(naive))}")
        print(f"  confound size (naive - factorial) at their bests: "
              f"{np.nanmax(naive) - by_layer[best]:+.3f}")

    # --- 3. falsification: apply the probe where nothing is loaded -----------
    if "none" in acts:
        N = acts["none"]
        ta = topic_a[idx]
        print(f"\n  falsification: the factorial probe applied to `none`-loaded "
              f"activations, where nothing is in context")
        for tag, layer, cov in (("as fitted", args.transfer_layer if
                                 args.transfer_layer is not None else best, None),
                                ("topic projected out", best_rt, topic_cov)):
            X = pooled(A["resid"][:, layer], B["resid"][:, layer])
            predict = fit_full(X, y, k=args.k, l2=args.l2, covariate=cov)
            ref = predict(X, cov) if cov is not None else predict(X)
            qcov = ta.astype(np.float64) if cov is not None else None
            pn = (predict(N["resid"][idx][:, layer], qcov) if cov is not None
                  else predict(N["resid"][idx][:, layer]))
            entry = {
                "layer": layer,
                "mean_score": float(pn.mean()),
                "mean_score_qa": float(pn[ta].mean()), "mean_score_qb": float(pn[~ta].mean()),
                "qa_vs_qb_auroc": auroc(pn, ta),
                "mean_score_in_scope_train": float(ref[y == 1].mean()),
                "mean_score_out_scope_train": float(ref[y == 0].mean()),
            }
            results["none_test" + ("" if cov is None else "_topic_removed")] = entry
            print(f"    [{tag}, layer {layer}] in-sample reference: in-scope "
                  f"{entry['mean_score_in_scope_train']:.3f} / out-of-scope "
                  f"{entry['mean_score_out_scope_train']:.3f}")
            print(f"      none-loaded mean {entry['mean_score']:.3f} "
                  f"(QA {entry['mean_score_qa']:.3f}, QB {entry['mean_score_qb']:.3f}), "
                  f"QA-vs-QB AUROC {entry['qa_vs_qb_auroc']:.3f}"
                  f"   <- 0.5 means the probe is not reading topic")

    # --- 4. attention mass, factorial ---------------------------------------
    mA, mB = A["mass"], B["mass"]
    m = np.concatenate([mA[idx], mB[idx]])           # (rows, layers, heads)
    per_layer = m.sum(-1)
    total = per_layer.sum(-1)
    yb = y.astype(bool)
    layer_auc = [auroc(per_layer[:, l], yb) for l in range(m.shape[1])]
    head_auc = np.array([[auroc(m[:, l, h], yb) for h in range(m.shape[2])]
                         for l in range(m.shape[1])])
    far = lambda v: float(max(v, key=lambda x: abs(x - 0.5)))
    mass = {"total_auroc": auroc(total, yb),
            "mean_total_in_scope": float(total[y == 1].mean()),
            "mean_total_out_scope": float(total[y == 0].mean()),
            "best_layer_auroc": far(layer_auc),
            "best_layer": int(np.argmax(np.abs(np.array(layer_auc) - 0.5))),
            "auroc_by_layer": layer_auc}
    i = int(np.nanargmax(np.abs(head_auc - 0.5)))
    mass["best_head"] = [i // m.shape[2], i % m.shape[2]]
    mass["best_head_auroc"] = float(head_auc.ravel()[i])
    if COMPOSE.exists():
        hs = np.load(COMPOSE)
        top = np.argsort(-np.abs(hs).ravel())[: args.routing_heads]
        rows_, cols_ = top // hs.shape[1], top % hs.shape[1]
        mass["routing_heads_mean_absdev"] = float(np.mean(np.abs(head_auc[rows_, cols_] - 0.5)))
        mass["all_heads_mean_absdev"] = float(np.mean(np.abs(head_auc - 0.5)))
        mass["routing_heads_summed_auroc"] = auroc(m[:, rows_, cols_].sum(-1), yb)
        mass["routing_heads"] = [[int(r), int(c)] for r, c in zip(rows_, cols_)]
    results["mass"] = mass
    print("\n  attention mass on cartridge slots, in-scope vs out-of-scope:")
    print(f"    layer-summed total AUROC {mass['total_auroc']:.3f} "
          f"(mean {mass['mean_total_in_scope']:.1f} in-scope / "
          f"{mass['mean_total_out_scope']:.1f} out)")
    print(f"    best single layer {mass['best_layer_auroc']:.3f} at layer {mass['best_layer']}")
    print(f"    best single head {mass['best_head_auroc']:.3f} at "
          f"layer {mass['best_head'][0]} head {mass['best_head'][1]}")
    if "routing_heads_summed_auroc" in mass:
        print(f"    top-{args.routing_heads} composition routing heads: summed AUROC "
              f"{mass['routing_heads_summed_auroc']:.3f}, mean |AUROC-0.5| "
              f"{mass['routing_heads_mean_absdev']:.3f} vs {mass['all_heads_mean_absdev']:.3f} "
              f"over all heads")

    # --- 5. does the score track membership or retention? -------------------
    if args.read and Path(args.read).exists():
        qa_rows = read_jsonl(OUT_DIR / "captured-qa.jsonl")
        correct = {r["question"]: r["correct"] for r in read_jsonl(args.read)
                   if r["condition"] == "trained"}
        oof = oofs[best]
        got, missed = [], []
        for pos, qi in enumerate(idx):                # rows under A come first
            q = qa_rows[qi]["question"]
            if topic_a[qi] and q in correct:
                (got if correct[q] else missed).append(oof[pos])
        if got and missed:
            results["retention"] = {
                "layer": best, "n_correct": len(got), "n_wrong": len(missed),
                "mean_score_correct": float(np.mean(got)),
                "mean_score_wrong": float(np.mean(missed)),
                "delta": float(np.mean(got) - np.mean(missed)),
            }
            r = results["retention"]
            print(f"\n  QA x A rows split by whether the reader judged the trained "
                  f"cartridge's answer correct:")
            print(f"    answered correctly (n={r['n_correct']}): "
                  f"P(in-scope) {r['mean_score_correct']:.3f}")
            print(f"    answered wrongly  (n={r['n_wrong']}): "
                  f"P(in-scope) {r['mean_score_wrong']:.3f}   "
                  f"delta {r['delta']:+.3f}")
            print("    a delta near zero means the probe reads corpus membership, "
                  "not retention")

    print("\nper-layer AUROC (cols: layer 0..N)")
    print("  factorial " + " ".join(f"{v:4.2f}" for v in by_layer))
    print("  fact-noT  " + " ".join(f"{v:4.2f}" for v in by_layer_rt))
    print("  paired    " + " ".join(f"{v:4.2f}" for v in paired))
    if naive:
        print("  naive     " + " ".join(f"{v:4.2f}" for v in naive))
    print("  mass      " + " ".join(f"{v:4.2f}" for v in layer_auc))

    (OUT_DIR / "factorial.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT_DIR / 'factorial.json'}")
    plot_factorial(by_layer, by_layer_rt, paired, naive, layer_auc, results, OUT_DIR)


def plot_factorial(by_layer, by_layer_rt, paired, naive, mass_layer, results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot(by_layer, color="tab:blue", lw=2, marker="o", ms=3,
                 label="factorial (2x2, topic XOR cartridge)")
    axes[0].plot(by_layer_rt, color="tab:cyan", lw=1.3, marker="o", ms=2,
                 label="factorial, topic direction projected out")
    axes[0].plot(paired, color="tab:green", lw=2, marker="s", ms=3,
                 label="paired contrast (per-question terms cancelled)")
    if naive:
        axes[0].plot(naive, color="tab:orange", lw=1.6, ls="--", marker="o", ms=2.5,
                     label="naive in/out (one cartridge) -- confounded")
    axes[0].axhline(0.5, color="k", ls=":", lw=0.8)
    axes[0].set(xlabel="layer (0 = embeddings)", ylabel="out-of-fold AUROC",
                title="is the loaded cartridge the right one for this question?",
                ylim=(0.3, 1.02))
    axes[0].legend(fontsize=8)

    axes[1].plot(mass_layer, color="tab:purple", lw=2, marker="o", ms=3,
                 label="attention mass on cartridge slots")
    axes[1].axhline(0.5, color="k", ls=":", lw=0.8)
    axes[1].set(xlabel="layer", ylabel="AUROC, in-scope vs out-of-scope",
                title="cartridge attention mass, factorial label", ylim=(0.3, 1.02))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "factorial.png", dpi=130)
    print(f"plot in {out_dir / 'factorial.png'}")


def load_acts(conditions):
    out = {}
    for c in conditions:
        path = OUT_DIR / f"act-{c}.npz"
        if path.exists():
            out[c] = np.load(path)
    return out


def cmd_analyze(args):
    acts = load_acts(args.conditions)
    if not acts:
        sys.exit(f"no act-*.npz in {OUT_DIR}; run `capture` first")
    first = next(iter(acts.values()))
    # The capture also holds cartridge B's questions, which have no place in
    # the naive in/out comparison -- it is about one corpus at a time.
    sel = first["topic_a"].astype(bool) if "topic_a" in first else slice(None)
    acts = {c: {"resid": a["resid"][sel], "mass": a["mass"][sel],
                "labels": a["labels"][sel], "lengths": a["lengths"][sel]}
            for c, a in acts.items()}
    first = next(iter(acts.values()))
    y = first["labels"]
    n_layers = first["resid"].shape[1]
    L = first["lengths"].astype(np.float64)
    print(f"{len(y)} questions, {int(y.sum())} in / {int((1 - y).sum())} out, "
          f"{n_layers} readout points, conditions {list(acts)}")

    # The confound. `out` questions are longer, so length alone nearly sorts
    # the classes; every raw number below has to be read against this.
    len_auc = auroc(L, y.astype(bool))
    print(f"  prompt length: in {L[y == 1].mean():.1f} / out {L[y == 0].mean():.1f} "
          f"tokens, AUROC of length alone {len_auc:.3f}")

    # A length-matched subset: greedy caliper pairing of in and out questions
    # of equal length. Small, but free of the confound by construction.
    keep, pool = [], {}
    for i in np.where(y == 0)[0]:
        pool.setdefault(int(L[i]), []).append(i)
    for i in np.where(y == 1)[0]:
        for dl in (0, 1, -1):
            if pool.get(int(L[i]) + dl):
                keep += [i, pool[int(L[i]) + dl].pop()]
                break
    keep = np.array(sorted(keep))
    matched_ok = len(keep) >= args.min_matched and 0 < y[keep].sum() < len(keep)
    print(f"  length-matched subset: {len(keep)} questions "
          f"({int(y[keep].sum()) if len(keep) else 0} in), "
          f"AUROC of length there {auroc(L[keep], y[keep].astype(bool)):.3f}"
          if len(keep) else "  length-matched subset: empty")

    # --- 1. per-layer linear probe -------------------------------------------
    by_layer, by_layer_ctl, by_layer_matched = {}, {}, {}
    for cond, a in acts.items():
        t0 = time.time()
        R = a["resid"]
        kw = {'folds': args.folds, 'k': args.k, 'l2': args.l2, 'seed': args.seed}
        by_layer[cond] = [cv_auroc(R[:, l], y, **kw)[0] for l in range(n_layers)]
        by_layer_ctl[cond] = [cv_auroc(R[:, l], y, covariate=L, **kw)[0]
                              for l in range(n_layers)]
        if matched_ok:
            by_layer_matched[cond] = [cv_auroc(R[keep][:, l], y[keep], **kw)[0]
                                      for l in range(n_layers)]
        print(f"  probe {cond}: raw {max(by_layer[cond]):.3f} @L{int(np.argmax(by_layer[cond]))}"
              f"   length-controlled {max(by_layer_ctl[cond]):.3f} "
              f"@L{int(np.argmax(by_layer_ctl[cond]))}"
              + (f"   matched {max(by_layer_matched[cond]):.3f}" if matched_ok else "")
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    best_layer = {c: int(np.argmax(v)) for c, v in by_layer_ctl.items()}
    results = {"n": len(y), "n_in": int(y.sum()), "layers": n_layers,
               "length_auroc": len_auc, "n_matched": len(keep),
               "mean_len_in": float(L[y == 1].mean()), "mean_len_out": float(L[y == 0].mean()),
               "auroc_by_layer": by_layer,
               "auroc_by_layer_length_controlled": by_layer_ctl,
               "auroc_by_layer_length_matched": by_layer_matched,
               "best_layer": best_layer,
               "best_auroc": {c: float(max(v)) for c, v in by_layer.items()},
               "best_auroc_length_controlled": {c: float(max(v)) for c, v in by_layer_ctl.items()}}

    # --- 2. the control: how much of that is question surface form? ----------
    if "none" in by_layer_ctl and "trained" in by_layer_ctl:
        for tag, src in (("", by_layer), ("_length_controlled", by_layer_ctl)):
            gap = [t - n for t, n in zip(src["trained"], src["none"])]
            results["trained_minus_none_by_layer" + tag] = gap
            results["trained_minus_none_max" + tag] = float(max(gap))
            results["trained_minus_none_argmax" + tag] = int(np.argmax(gap))
        print(f"  trained - none (length-controlled): max {max(results['trained_minus_none_by_layer_length_controlled']):+.3f} "
              f"at layer {results['trained_minus_none_argmax_length_controlled']}, "
              f"raw max {max(results['trained_minus_none_by_layer']):+.3f}")

    # --- 3. transfer: the same probe, a different cartridge -------------------
    if "trained" in acts and "wrong" in acts:
        layer = args.transfer_layer if args.transfer_layer is not None else best_layer["trained"]
        Xt = acts["trained"]["resid"][:, layer]
        Xw = acts["wrong"]["resid"][:, layer]
        ins = y == 1
        results["transfer_raw"] = transfer_stats(
            fit_full(Xt, y, k=args.k, l2=args.l2), Xt, Xw, ins, layer)
        results["transfer"] = transfer_stats(
            fit_full(Xt, y, k=args.k, l2=args.l2, covariate=L), Xt, Xw, ins, layer, cov=L)
        for tag, t in (("length-controlled", results["transfer"]),
                       ("raw", results["transfer_raw"])):
            print(f"\n  transfer, {tag} (probe fit on trained layer {layer}, applied to wrong):")
            print(f"    in-corpus questions:  P(in) {t['mean_p_in_trained']:.3f} -> "
                  f"{t['mean_p_in_wrong']:.3f}  ({t['shift_in']:+.3f})")
            print(f"    out questions:        P(in) {t['mean_p_out_trained']:.3f} -> "
                  f"{t['mean_p_out_wrong']:.3f}  ({t['shift_out']:+.3f})")
            print(f"    called 'in':          {t['frac_called_in_trained']:.1%} -> "
                  f"{t['frac_called_in_wrong']:.1%}   flip rate {t['flip_rate_in']:.1%}")

    # --- 4. attention mass ---------------------------------------------------
    results["mass"] = {}
    head_sep = np.load(COMPOSE) if COMPOSE.exists() else None

    def resid_on_len(v):
        """Linearly remove prompt length from a scalar readout."""
        a = L - L.mean()
        return v - a * float((a @ (v - v.mean())) / max(a @ a, 1e-12))

    for cond, a in acts.items():
        m = a["mass"]                       # (n, layers, heads)
        per_layer = m.sum(-1)               # mass summed over heads
        total = per_layer.sum(-1)           # the single scalar
        layer_auc = [auroc(per_layer[:, l], y.astype(bool)) for l in range(m.shape[1])]
        layer_auc_ctl = [auroc(resid_on_len(per_layer[:, l]), y.astype(bool))
                         for l in range(m.shape[1])]
        head_auc = np.array([[auroc(m[:, l, h], y.astype(bool)) for h in range(m.shape[2])]
                             for l in range(m.shape[1])])
        far = lambda v: float(max(v, key=lambda x: abs(x - 0.5)))
        entry = {
            "total_auroc": auroc(total, y.astype(bool)),
            "total_auroc_length_controlled": auroc(resid_on_len(total), y.astype(bool)),
            "mean_total_in": float(total[y == 1].mean()),
            "mean_total_out": float(total[y == 0].mean()),
            "best_layer_auroc": far(layer_auc),
            "best_layer": int(np.argmax(np.abs(np.array(layer_auc) - 0.5))),
            "best_layer_auroc_length_controlled": far(layer_auc_ctl),
            "auroc_by_layer": layer_auc,
            "auroc_by_layer_length_controlled": layer_auc_ctl,
            "best_head_auroc": float(np.nanmax(np.abs(head_auc - 0.5)) + 0.5),
        }
        i = int(np.nanargmax(np.abs(head_auc - 0.5)))
        entry["best_head"] = [i // m.shape[2], i % m.shape[2]]
        entry["best_head_signed_auroc"] = float(head_auc.ravel()[i])
        if head_sep is not None:
            top = np.argsort(-np.abs(head_sep).ravel())[: args.routing_heads]
            rows, cols = top // head_sep.shape[1], top % head_sep.shape[1]
            entry["routing_heads_mean_auroc"] = float(np.mean(head_auc[rows, cols]))
            entry["routing_heads_mean_absdev"] = float(np.mean(np.abs(head_auc[rows, cols] - 0.5)))
            entry["all_heads_mean_absdev"] = float(np.mean(np.abs(head_auc - 0.5)))
            entry["routing_heads_summed_auroc"] = auroc(
                m[:, rows, cols].sum(-1), y.astype(bool))
        results["mass"][cond] = entry

    # --- report ---------------------------------------------------------------
    print(f"\n{'':10s}{'--- residual-stream probe ---':^40s}{'--- cartridge attention mass ---':^42s}")
    print(f"{'condition':10s}{'raw':>8s}{'len-ctl':>9s}{'@layer':>8s}{'matched':>9s}"
          f"{'total':>9s}{'tot-ctl':>9s}{'best-L':>8s}{'@layer':>8s}{'in/out mass':>20s}")
    for c in acts:
        m = results["mass"][c]
        print(f"{c:10s}{max(by_layer[c]):8.3f}{max(by_layer_ctl[c]):9.3f}"
              f"{best_layer[c]:8d}"
              f"{(max(by_layer_matched[c]) if matched_ok else float('nan')):9.3f}"
              f"{m['total_auroc']:9.3f}{m['total_auroc_length_controlled']:9.3f}"
              f"{m['best_layer_auroc']:8.3f}{m['best_layer']:8d}"
              f"{m['mean_total_in']:10.2f} /{m['mean_total_out']:8.2f}")
    print(f"{'(length)':10s}{len_auc:8.3f}{0.5:9.3f}{'-':>8s}"
          f"{(auroc(L[keep], y[keep].astype(bool)) if matched_ok else float('nan')):9.3f}"
          "   <- the confound baseline")
    if head_sep is not None:
        print(f"\nrouting heads (top {args.routing_heads} from runs/compose), mean |AUROC-0.5|:")
        for c in acts:
            m = results["mass"][c]
            print(f"  {c:10s} routing {m['routing_heads_mean_absdev']:.3f}  "
                  f"all heads {m['all_heads_mean_absdev']:.3f}  "
                  f"summed-routing AUROC {m['routing_heads_summed_auroc']:.3f}")
    print("\nper-layer probe AUROC, length-controlled (cols: layer 0..N)")
    for c in acts:
        print(f"  {c:8s} " + " ".join(f"{v:4.2f}" for v in by_layer_ctl[c]))
    print("\nper-layer cartridge-mass AUROC, length-controlled")
    for c in acts:
        print(f"  {c:8s} " + " ".join(
            f"{v:4.2f}" for v in results["mass"][c]["auroc_by_layer_length_controlled"]))

    # Robustness: does the headline survive a different probe capacity?
    if args.k_sweep:
        results["k_sweep"] = {}
        print("\nbest length-controlled AUROC vs probe capacity k")
        for k in args.k_sweep:
            row = {c: max(cv_auroc(a["resid"][:, l], y, folds=args.folds, k=k, l2=args.l2,
                                   seed=args.seed, covariate=L)[0]
                          for l in range(n_layers))
                   for c, a in acts.items()}
            results["k_sweep"][k] = row
            print(f"  k={k:<4} " + "  ".join(f"{c}={v:.3f}" for c, v in row.items()))

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT_DIR / 'results.json'}")
    plot(by_layer, by_layer_ctl, results, OUT_DIR)


def plot(by_layer, by_layer_ctl, results, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"trained": "tab:blue", "init": "tab:green", "none": "tab:gray",
              "wrong": "tab:red"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    for ax, src, title in (
        (axes[0], by_layer, "residual stream, raw"),
        (axes[1], by_layer_ctl, "residual stream, prompt length removed"),
        (axes[2], {c: e["auroc_by_layer_length_controlled"]
                   for c, e in results["mass"].items()},
         "cartridge attention mass, length removed"),
    ):
        for c, v in src.items():
            ax.plot(v, color=colors.get(c), lw=1.8, label=c, marker="o", ms=2.5)
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.set(xlabel="layer (0 = embeddings)", ylabel="AUROC, in vs out",
               title=title, ylim=(0.3, 1.02))
        ax.legend(fontsize=8)
    axes[0].axhline(results["length_auroc"], color="k", ls="--", lw=0.9)
    axes[0].annotate("prompt length alone", (0.5, results["length_auroc"]),
                     fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "auroc_by_layer.png", dpi=130)
    print(f"plot in {out_dir / 'auroc_by_layer.png'}")


# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data")
    d.add_argument("--base", default=str(REPO / "runs/reader/qa.jsonl"))
    d.add_argument("--base-seed", type=int, default=0)
    d.add_argument("--extra", nargs="*", default=[str(OUT_DIR / "qa-out64.jsonl")])
    d.add_argument("--extra-seeds", nargs="*", type=int, default=[2])
    d.add_argument("--tetris", nargs="*", default=[str(OUT_DIR / "qa-tetris.jsonl")],
                   help="in-corpus questions for cartridge B")
    d.add_argument("--tetris-seeds", nargs="*", type=int, default=[0])
    d.add_argument("--balance", action=argparse.BooleanOptionalAction, default=True)
    d.add_argument("--out", default=str(OUT_DIR / "qa.jsonl"))

    c = sub.add_parser("capture")
    c.add_argument("--qa", default=str(OUT_DIR / "qa.jsonl"))
    c.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    c.add_argument("--cartridge", default=str(TRAINED))
    c.add_argument("--wrong", default=str(WRONG))
    c.add_argument("--files", nargs="*")
    c.add_argument("--root")
    c.add_argument("--limit", type=int)
    c.add_argument("--bucket", type=int, default=32, help="right-pad prompts to a multiple")
    c.add_argument("--check", action="store_true",
                   help="verify the captured hidden state against the model's own path")

    a = sub.add_parser("analyze")
    a.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    a.add_argument("--folds", type=int, default=5)
    a.add_argument("--k", type=int, default=64, help="principal components kept")
    a.add_argument("--k-sweep", nargs="*", type=int, default=[16, 32, 64, 128],
                   help="robustness check: best-layer AUROC at each k")
    a.add_argument("--l2", type=float, default=1.0)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--transfer-layer", type=int)
    a.add_argument("--routing-heads", type=int, default=24)
    a.add_argument("--min-matched", type=int, default=24,
                   help="skip the length-matched subset below this many questions")

    f = sub.add_parser("factorial")
    f.add_argument("--a-cond", default="trained", help="condition holding cartridge A")
    f.add_argument("--b-cond", default="wrong", help="condition holding cartridge B")
    f.add_argument("--folds", type=int, default=5)
    f.add_argument("--k", type=int, default=64)
    f.add_argument("--l2", type=float, default=1.0)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--transfer-layer", type=int)
    f.add_argument("--routing-heads", type=int, default=24)
    f.add_argument("--naive", action=argparse.BooleanOptionalAction, default=True)
    f.add_argument("--read", default=str(REPO / "runs/reader/read.jsonl"),
                   help="reader_score output, for the retention split")

    args = p.parse_args()
    {"data": cmd_data, "capture": cmd_capture, "analyze": cmd_analyze,
     "factorial": cmd_factorial}[args.cmd](args)


if __name__ == "__main__":
    main()
