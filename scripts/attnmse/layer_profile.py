"""Per-layer profile: where the attention-MSE residual lives, and how much each
layer reads from the corpus (teacher) versus the cartridge (student).

Replaces the battery's eff_slots (entropy of prefix attention mass, sink
included), which pins sink-dominated deep layers near 2 and inflates diffuse
early layers whatever the slots contribute. Board: p5tsd4.

Per cartridge, on the held-out set:
  attnmse[l]      relative attention-output error at layer l (mean over batches)
  student[l]      teacher-forced suffix queries against [cartridge | suffix]:
                  mass on the sink slot, on the other slots, on the suffix;
                  and the number of slots carrying > 1e-3 of the non-sink slot mass
Per layer, the teacher (corpus in context):
  teacher[l]      mass on the sink token, on the rest of the corpus context, on
                  the suffix -- what the student is trying to reproduce

All masses are averaged over loss-mask positions and heads.

Usage: uv run python scripts/attnmse/layer_profile.py [name=path ...]
Writes runs/attnmse/profile.json (and prints a table).
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.68")

import argparse
import functools
import json
import sys
from pathlib import Path

REPO = Path("/home/em/Dev/neural/qwenjax")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import jax
import jax.numpy as jnp
import numpy as np

import cartridge as cli
from qwen_jax.attention import apply_rotary_pos_emb
from qwen_jax.attnmse import attnmse_layers
from qwen_jax.cartridge import Cartridge
from qwen_jax.selfstudy import load_examples

P = 1024
SCALE = 128 ** -0.5
DEFAULT_CARTS = {
    "init": None,
    "attnmse2k": REPO / "runs/attnmse/attnmse-2k.safetensors",
    "rms1": REPO / "runs/attnmse/overnight/rms1.safetensors",
    "kl-rms": REPO / "runs/cart/rms-lr1e-2.safetensors",
    "kl-raw": REPO / "runs/cart/raw-lr1e-2.safetensors",
}


@functools.partial(jax.jit, static_argnames=("context",))
def teacher_masses(q, k, kmask, lmask, sink_idx, context):
    """q: (b, S, 32, 128) suffix queries; k: (b, s, 8, 128) full keys.
    Returns summed (sink, corpus, suffix) mass over loss positions and heads,
    and the count of (position, head) terms."""
    b, S = q.shape[:2]
    s = k.shape[1]
    qg = q.reshape(b, S, 8, 4, 128)
    scores = jnp.einsum("bqhgd,bkhd->bhgqk", qg, k) * SCALE
    kidx = jnp.arange(s)[None, None, None, None, :]
    qabs = (context + jnp.arange(S))[None, None, None, :, None]
    ok = kmask[:, None, None, None, :] & (kidx <= qabs)
    w = jax.nn.softmax(jnp.where(ok, scores, -jnp.inf), axis=-1)
    lm = lmask[:, None, None, :, None].astype(jnp.float32)
    w = w * lm
    onehot_sink = (kidx == sink_idx[:, None, None, None, None]).astype(jnp.float32)
    sink = jnp.sum(w * onehot_sink)
    ctx = jnp.sum(w[..., :context]) - sink
    suf = jnp.sum(w[..., context:])
    n = jnp.sum(lmask) * 32
    return jnp.stack([sink, ctx, suf, n])


@jax.jit
def student_masses(q, k, Kc, smask, lmask):
    """q, k: (b, S, ...) suffix at student positions; Kc: (P, 8, 128) physical
    cartridge keys. Returns (sink slot, other slots, suffix, n) sums and the
    per-slot mass vector (P,)."""
    b, S = q.shape[:2]
    causal = jnp.tril(jnp.ones((S, S), bool))
    K = jnp.concatenate([jnp.broadcast_to(Kc[None], (b, *Kc.shape)), k], axis=1)
    qg = q.reshape(b, S, 8, 4, 128)
    scores = jnp.einsum("bqhgd,bkhd->bhgqk", qg, K) * SCALE
    valid = jnp.concatenate([jnp.ones((b, P), bool), smask], axis=1)[:, None, None, None, :]
    cmask = jnp.concatenate([jnp.ones((S, P), bool), causal], axis=1)[None, None, None, :, :]
    w = jax.nn.softmax(jnp.where(valid & cmask, scores, -jnp.inf), axis=-1)
    w = w * lmask[:, None, None, :, None].astype(jnp.float32)
    per_slot = jnp.sum(w[..., :P], axis=(0, 1, 2, 3))
    sink = per_slot[0]
    other = jnp.sum(per_slot[1:])
    suf = jnp.sum(w[..., P:])
    n = jnp.sum(lmask) * 32
    return jnp.stack([sink, other, suf, n]), per_slot


def walk(model, batch, carts):
    """One teacher forward, collecting the per-layer masses for the teacher
    and for every cartridge (teacher-forced queries)."""
    lm = model.model.language_model
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
    kmask = mask.astype(jnp.bool)
    smask = s_mask.astype(jnp.bool)
    lmask = batch.loss_mask.astype(jnp.bool)
    sink_idx = jnp.argmax(mask, axis=1)

    b, s = ids.shape
    S = s - context
    t_out = np.zeros((36, 4))
    s_out = {name: np.zeros((36, 4)) for name in carts}
    s_slots = {name: np.zeros((36, P)) for name in carts}
    h = lm.embed_tokens(ids)
    for li, layer in enumerate(lm.layers):
        attn = layer.self_attn
        x = layer.input_layernorm(h)
        q_full = attn.q_norm(attn.q_proj(x).reshape(b, s, 32, 128))
        k_full = attn.k_norm(attn.k_proj(x).reshape(b, s, 8, 128))
        q_t, k_t = apply_rotary_pos_emb(q_full, k_full, cos_t, sin_t)
        t_out[li] += np.asarray(teacher_masses(q_t[:, context:].astype(jnp.float32),
                                               k_t.astype(jnp.float32), kmask, lmask,
                                               sink_idx, context=int(context)))
        xs = x[:, context:]
        q_s = attn.q_norm(attn.q_proj(xs).reshape(b, S, 32, 128))
        k_s = attn.k_norm(attn.k_proj(xs).reshape(b, S, 8, 128))
        q_s, k_s = apply_rotary_pos_emb(q_s, k_s, cos_s, sin_s)
        q_s, k_s = q_s.astype(jnp.float32), k_s.astype(jnp.float32)
        for name, cart in carts.items():
            Kc = jnp.asarray(cart.physical_keys[li], jnp.float32)
            m, per_slot = student_masses(q_s, k_s, Kc, smask, lmask)
            s_out[name][li] += np.asarray(m)
            s_slots[name][li] += np.asarray(per_slot)
        a, _ = attn(x, position_embeddings=(cos_t, sin_t), kv_mask=kmask)
        h = h + a
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    return t_out, s_out, s_slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carts", nargs="*", help="name=path; default: the study's cartridges")
    args = ap.parse_args()
    specs = dict(DEFAULT_CARTS)
    if args.carts:
        specs = {c.split("=", 1)[0]: Path(c.split("=", 1)[1]) for c in args.carts}

    tokenizer = cli.load_tokenizer()
    corpus = cli.load_corpus(tokenizer, None)
    heldout = load_examples(str(REPO / "runs/cart/heldout.jsonl"))
    model = cli.load_model()
    shapes = argparse.Namespace(batch=2, context=2176, seq=512, description=cli.DESCRIPTION)
    held = list(cli.batches_from(tokenizer, heldout, shapes, shuffle=False))

    carts = {}
    for name, path in specs.items():
        if path is None:
            carts[name] = cli.init_cartridge(model, tokenizer, corpus, P, cli.DESCRIPTION)
        elif path.exists():
            carts[name] = Cartridge.load(path)
        else:
            print(f"{name}: {path} missing, skipped", flush=True)

    f_layers = jax.jit(attnmse_layers)
    attn = {name: np.mean([np.asarray(f_layers(model, c, b)) for b in held], axis=0)
            for name, c in carts.items()}
    print("per-layer held-out attnmse done", flush=True)

    t_tot = np.zeros((36, 4))
    s_tot = {n: np.zeros((36, 4)) for n in carts}
    slots = {n: np.zeros((36, P)) for n in carts}
    for i, batch in enumerate(held):
        t, so, ss = walk(model, batch, carts)
        t_tot += t
        for n in carts:
            s_tot[n] += so[n]
            slots[n] += ss[n]
        print(f"  batch {i + 1}/{len(held)}", flush=True)

    def frac(m):
        return {"sink": (m[:, 0] / m[:, 3]).tolist(), "prefix": (m[:, 1] / m[:, 3]).tolist(),
                "suffix": (m[:, 2] / m[:, 3]).tolist()}

    out = {"teacher": frac(t_tot), "carts": {}}
    for n in carts:
        rest = slots[n][:, 1:]
        share = rest / np.maximum(rest.sum(axis=1, keepdims=True), 1e-12)
        out["carts"][n] = {
            "attnmse": attn[n].tolist(),
            **frac(s_tot[n]),
            "slots_gt_1e-3": (share > 1e-3).sum(axis=1).tolist(),
            "slots_gt_1e-4": (share > 1e-4).sum(axis=1).tolist(),
        }
    out_path = REPO / "runs/attnmse/profile.json"
    out_path.write_text(json.dumps(out, indent=1))

    names = list(carts)
    print(f"\n{'layer':>5} | {'T.sink':>6} {'T.corp':>6} {'T.suf':>6} | " +
          " | ".join(f"{n:>22}" for n in names))
    print(f"{'':>5} | {'':>20} | " + " | ".join(f"{'err':>6} {'sink':>5} {'pfx':>4} {'#>1e-3':>5}"
                                               for _ in names))
    T = out["teacher"]
    for li in range(36):
        row = f"{li:>5} | {T['sink'][li]:6.3f} {T['prefix'][li]:6.3f} {T['suffix'][li]:6.3f} | "
        cells = []
        for n in names:
            c = out["carts"][n]
            cells.append(f"{c['attnmse'][li]:6.3f} {c['sink'][li]:5.2f} {c['prefix'][li]:4.2f} "
                         f"{c['slots_gt_1e-3'][li]:5d}")
        print(row + " | ".join(cells))
    print(f"\nmean attnmse: " + "  ".join(f"{n}={np.mean(attn[n]):.4f}" for n in names))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
