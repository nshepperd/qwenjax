"""Optimization-landscape study of the per-layer attention-output objective.

Is the teacher-forced attn-MSE hard to optimize with Adam (KVSculpt's claim),
and if so why? Per layer, the subproblem is self-contained: the teacher-forced
queries and suffix K/V are constants, only the cartridge slots (Kc, Vc) move:

    loss(Kc, Vc) = rel-MSE( o_proj(attn(q, [Kc;k], [Vc;v])), target )

The objective is a quadratic (convex) in Vc and nonconvex in Kc only through
the softmax, so K-only / V-only / joint runs under Adam vs L-BFGS dissect
where any difficulty lives. Everything is full-batch on a fixed 16-conversation
set: the landscape is deterministic, no SGD noise.

Usage: uv run python scripts/attnmse/study.py [--layers 1,15,20,29]
Writes results.json and curves-layer*.npz to runs/attnmse/landscape/.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.68")

import argparse
import functools
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/em/Dev/neural/qwenjax")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax

import cartridge as cli
from qwen_jax.attention import apply_rotary_pos_emb
from qwen_jax.cartridge import Cartridge
from qwen_jax.distill import Batch
from qwen_jax.selfstudy import load_examples

OUT = REPO / "runs/attnmse/landscape"
P = 1024
SCALE = 128 ** -0.5


# --------------------------------------------------------------------------
# Per-layer subproblem extraction
# --------------------------------------------------------------------------


def teacher_layer_inputs(model, batches: list[Batch], layer_ids: list[int]):
    """For each requested layer: teacher-forced q (student positions), suffix
    k/v (student positions), post-o_proj target, all at suffix positions."""
    lm = model.model.language_model
    per_layer = {i: {"q": [], "k": [], "v": [], "tgt": []} for i in layer_ids}
    masks = []
    for batch in batches:
        ids, mask = batch.teacher_ids, batch.teacher_mask
        context = batch.context
        pos_t, _, _ = model.model._resolve_position_ids(ids, None, mask, None, None, None, None)
        cos_t, sin_t = lm.rotary_emb(pos_t)
        dt = lm.embed_tokens.weight().dtype
        cos_t, sin_t = cos_t.astype(dt), sin_t.astype(dt)
        s_ids, s_mask = batch.student_ids, batch.student_mask
        pos_s, _, _ = model.model._resolve_position_ids(s_ids, None, s_mask, None, None, None, P)
        cos_s, sin_s = lm.rotary_emb(pos_s)
        cos_s, sin_s = cos_s.astype(dt), sin_s.astype(dt)

        h = lm.embed_tokens(ids)
        for li, layer in enumerate(lm.layers):
            attn_mod = layer.self_attn
            x = layer.input_layernorm(h)
            if li in per_layer:
                b, s, _ = x.shape
                xs = x[:, context:]
                q = attn_mod.q_norm(attn_mod.q_proj(xs).reshape(b, s - context, 32, 128))
                k = attn_mod.k_norm(attn_mod.k_proj(xs).reshape(b, s - context, 8, 128))
                v = attn_mod.v_proj(xs).reshape(b, s - context, 8, 128)
                q, k = apply_rotary_pos_emb(q, k, cos_s, sin_s)
                per_layer[li]["q"].append(np.asarray(q, np.float32))
                per_layer[li]["k"].append(np.asarray(k, np.float32))
                per_layer[li]["v"].append(np.asarray(v, np.float32))
            a, _ = attn_mod(x, position_embeddings=(cos_t, sin_t),
                            kv_mask=mask.astype(jnp.bool))
            if li in per_layer:
                per_layer[li]["tgt"].append(np.asarray(a[:, context:], np.float32))
            h = h + a
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        masks.append(np.asarray(batch.loss_mask))
    out = {}
    for li, d in per_layer.items():
        out[li] = {k2: np.concatenate(v2, axis=0) for k2, v2 in d.items()}
    return out, np.concatenate(masks, axis=0)


# --------------------------------------------------------------------------
# Subproblem loss (dense f32 attention, chunked over rows)
# --------------------------------------------------------------------------


def make_loss(o_proj, q, k, v, lmask):
    """loss(params) with params = {'K': (P,8,128), 'V': (P,8,128)} in f32.

    Dense softmax attention in f32; causal among suffix positions, prefix
    fully visible, invalid suffix slots masked. Chunked with lax.map +
    checkpoint over rows so the (32, 512, P+512) score tensors never all
    materialise at once.
    """
    n, s = q.shape[:2]
    causal = np.tril(np.ones((s, s), bool))
    tnorm_holder = {}

    def row_out(params, qi, ki, vi, mi):
        K = jnp.concatenate([params["K"], ki], axis=0)  # (P+S, 8, 128)
        V = jnp.concatenate([params["V"], vi], axis=0)
        qg = qi.reshape(s, 8, 4, 128)
        scores = jnp.einsum("qhgd,khd->hgqk", qg, K) * SCALE
        valid = jnp.concatenate([jnp.ones((P,), bool), mi], axis=0)[None, None, None, :]
        cmask = jnp.concatenate([jnp.ones((s, P), bool), causal], axis=1)[None, None, :, :]
        scores = jnp.where(valid & cmask, scores, -jnp.inf)
        w = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("hgqk,khd->qhgd", w, V).reshape(s, 4096)
        return o_proj(out[None].astype(jnp.bfloat16))[0].astype(jnp.float32)

    @jax.checkpoint
    def row_sqerr(params, args):
        qi, ki, vi, ti, mi = args
        d = (row_out(params, qi, ki, vi, mi) - ti) * mi.astype(jnp.float32)[:, None]
        return jnp.sum(d * d)

    def loss(params, tgt):
        sq = jax.lax.map(functools.partial(row_sqerr, params),
                         (q, k, v, tgt, lmask.astype(jnp.bool)))
        return jnp.sum(sq) / tnorm_holder["tnorm"]

    return loss, tnorm_holder


# --------------------------------------------------------------------------
# Optimizer runs
# --------------------------------------------------------------------------


def run_opt(loss_fn, params0, opt, steps, *, lbfgs=False, record_every=10):
    curve = []
    if lbfgs:
        value_and_grad = optax.value_and_grad_from_state(loss_fn)

        @jax.jit
        def step(params, state):
            value, grad = value_and_grad(params, state=state)
            updates, state = opt.update(grad, state, params, value=value, grad=grad,
                                        value_fn=loss_fn)
            return optax.apply_updates(params, updates), state, value
    else:
        @jax.jit
        def step(params, state):
            value, grad = jax.value_and_grad(loss_fn)(params)
            updates, state = opt.update(grad, state, params)
            return optax.apply_updates(params, updates), state, value

    params, state = params0, opt.init(params0)
    for i in range(steps):
        params, state, value = step(params, state)
        if i % record_every == 0 or i == steps - 1:
            curve.append(float(value))
            if not np.isfinite(curve[-1]):
                break
    return params, curve


def variants(loss_full, K0, V0):
    """(name, loss_fn, params0) for joint / K-only / V-only."""
    return [
        ("joint", loss_full, {"K": K0, "V": V0}),
        ("Konly", lambda p: loss_full({"K": p["K"], "V": V0}), {"K": K0}),
        ("Vonly", lambda p: loss_full({"K": K0, "V": p["V"]}), {"V": V0}),
    ]


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def slot_diagnostics(o_proj, q, k, v, lmask, params):
    """Attention mass landing on each prefix slot, and per-slot grad norms."""
    n, s = q.shape[:2]
    causal = np.tril(np.ones((s, s), bool))

    @jax.jit
    def mass_row(params, qi, ki, mi):
        K = jnp.concatenate([params["K"], ki], axis=0)
        qg = qi.reshape(s, 8, 4, 128)
        scores = jnp.einsum("qhgd,khd->hgqk", qg, K) * SCALE
        valid = jnp.concatenate([jnp.ones((P,), bool), mi], axis=0)[None, None, None, :]
        cmask = jnp.concatenate([jnp.ones((s, P), bool), causal], axis=1)[None, None, :, :]
        scores = jnp.where(valid & cmask, scores, -jnp.inf)
        w = jax.nn.softmax(scores, axis=-1)  # (8, 4, S, P+S)
        w = w * mi.astype(jnp.float32)[None, None, :, None]
        return jnp.sum(w[..., :P], axis=(1, 2)), jnp.sum(mi)  # (8, P)

    mass = np.zeros((8, P))
    nq = 0.0
    for i in range(n):
        m, c = mass_row(params, q[i], k[i], lmask[i].astype(jnp.bool))
        mass += np.asarray(m)
        nq += float(c)
    return mass, nq


def curvature(loss_fn, params, key, iters=30, probes=8):
    """Top Hessian eigenvalue (power iteration on HVPs) and Hutchinson mean."""
    flat, unravel = jax.flatten_util.ravel_pytree(params)
    g = jax.grad(lambda f: loss_fn(unravel(f)))

    @jax.jit
    def hvp(f, u):
        return jax.jvp(g, (f,), (u,))[1]

    u = jax.random.normal(key, flat.shape)
    u = u / jnp.linalg.norm(u)
    lam = 0.0
    for _ in range(iters):
        hu = hvp(flat, u)
        lam = float(jnp.vdot(u, hu))
        nrm = jnp.linalg.norm(hu)
        u = hu / jnp.maximum(nrm, 1e-30)
    tr = 0.0
    for i in range(probes):
        z = jax.random.rademacher(jax.random.fold_in(key, i), flat.shape, dtype=jnp.float32)
        tr += float(jnp.vdot(z, hvp(flat, z)))
    return lam, tr / probes / flat.size


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,15,20,29")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--adam-steps", type=int, default=2000)
    ap.add_argument("--lbfgs-steps", type=int, default=300)
    args = ap.parse_args()
    layer_ids = [int(x) for x in args.layers.split(",")]

    tokenizer = cli.load_tokenizer()
    corpus = cli.load_corpus(tokenizer, None)
    train = load_examples(str(REPO / "runs/cart/train.jsonl"))
    model = cli.load_model()
    shapes = argparse.Namespace(batch=2, context=2176, seq=512, description=cli.DESCRIPTION)
    batches = []
    for b in cli.batches_from(tokenizer, train, shapes, shuffle=False):
        batches.append(b)
        if len(batches) >= args.batches:
            break

    print("capturing teacher-forced subproblems...", flush=True)
    t0 = time.time()
    sub, lmask = teacher_layer_inputs(model, batches, layer_ids)
    print(f"  captured in {time.time() - t0:.0f}s", flush=True)

    init = cli.init_cartridge(model, tokenizer, corpus, P, cli.DESCRIPTION)
    trained = Cartridge.load(str(REPO / "runs/attnmse/attnmse-lr1e-2.safetensors"))
    lm = model.model.language_model

    results = {}
    for li in layer_ids:
        d = sub[li]
        q, k, v, tgt = (jnp.asarray(d[x]) for x in ("q", "k", "v", "tgt"))
        lmask_j = jnp.asarray(lmask)
        o_proj = lm.layers[li].self_attn.o_proj
        loss_raw, holder = make_loss(o_proj, q, k, v, lmask_j)
        t = jnp.asarray(d["tgt"]) * lmask_j[..., None]
        holder["tnorm"] = float(jnp.sum(t * t))
        loss_fn = jax.jit(functools.partial(loss_raw, tgt=tgt))

        K0 = jnp.asarray(init.physical_keys[li], jnp.float32)
        V0 = jnp.asarray(init.physical_values[li], jnp.float32)
        Kt = jnp.asarray(trained.physical_keys[li], jnp.float32)
        Vt = jnp.asarray(trained.physical_values[li], jnp.float32)

        r = {"init_loss": float(loss_fn({"K": K0, "V": V0})),
             "sgd300_loss": float(loss_fn({"K": Kt, "V": Vt}))}
        print(f"layer {li}: init={r['init_loss']:.4f} sgd300={r['sgd300_loss']:.4f}", flush=True)

        curves = {}
        for vname, vloss, p0 in variants(loss_fn, K0, V0):
            for lr in (1e-3, 1e-2, 1e-1):
                sched = optax.cosine_decay_schedule(lr, args.adam_steps)
                pf, c = run_opt(vloss, p0, optax.adam(sched), args.adam_steps)
                curves[f"adam-{vname}-lr{lr:g}"] = c
                print(f"  adam {vname} lr={lr:g}: {c[0]:.4f} -> {c[-1]:.4f}", flush=True)
            pf, c = run_opt(vloss, p0, optax.lbfgs(), args.lbfgs_steps, lbfgs=True)
            curves[f"lbfgs-{vname}"] = c
            r[f"lbfgs_{vname}_final"] = c[-1]
            print(f"  lbfgs {vname}: {c[0]:.4f} -> {c[-1]:.4f}", flush=True)
            if vname == "joint":
                r["lbfgs_joint_params"] = pf

        # diagnostics at init
        mass, nq = slot_diagnostics(o_proj, q, k, v, np.asarray(lmask), {"K": K0, "V": V0})
        r["slot_mass_at_init"] = mass.sum(axis=0).tolist()
        r["queries"] = nq
        g = jax.grad(loss_fn)({"K": K0, "V": V0})
        r["gradnorm_K_per_slot"] = np.linalg.norm(
            np.asarray(g["K"]).reshape(P, -1), axis=1).tolist()
        r["gradnorm_V_per_slot"] = np.linalg.norm(
            np.asarray(g["V"]).reshape(P, -1), axis=1).tolist()
        kf = jax.random.key(li)
        lamK, meanK = curvature(lambda p: loss_fn({"K": p["K"], "V": V0}), {"K": K0}, kf)
        lamV, meanV = curvature(lambda p: loss_fn({"K": K0, "V": p["V"]}), {"V": V0}, kf)
        r["curv_K"] = {"lam_max": lamK, "lam_mean": meanK}
        r["curv_V"] = {"lam_max": lamV, "lam_mean": meanV}
        print(f"  curvature K: lam_max={lamK:.3e} mean={meanK:.3e}  "
              f"V: lam_max={lamV:.3e} mean={meanV:.3e}", flush=True)

        pf = r.pop("lbfgs_joint_params")
        np.savez(OUT / f"curves-layer{li}.npz",
                 **{k2: np.asarray(v2) for k2, v2 in curves.items()},
                 K_lbfgs=np.asarray(pf["K"]), V_lbfgs=np.asarray(pf["V"]))
        results[str(li)] = r
        (OUT / "results.json").write_text(json.dumps(results, indent=1))

    print("done ->", OUT / "results.json")


if __name__ == "__main__":
    main()
