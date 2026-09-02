"""Student-query (on-policy) attention profile: how much of the attention-MSE
objective's residual is compounding through depth.

The teacher-forced objective feeds layer l the TEACHER's residual stream. At
inference layer l sees the STUDENT's stream, carrying the accumulated error of
layers < l. Per layer, on the held-out set, this measures:

  err_tf     teacher-forced error (the training objective): the cartridge's
             attention on the teacher's stream vs the teacher's attention
  err_op     on-policy error: the student's actual attention output vs what
             the teacher's layer would produce from the student's own stream
             over the corpus KV (the DAgger target)
  out_shift  the student's actual attention output vs the teacher-forced
             target -- the end-to-end attention error at that layer
  drift      relative distance between the student's and the teacher's
             residual streams entering the layer (the compounding itself)
  tgt_shift  how far the DAgger target moved from the teacher-forced target

All relative squared errors over loss-mask positions, normalised by the
reference's power. `check` is the teacher's own attention recomputed through
the prefix path (should be bf16 noise; validates the geometry).

Usage: uv run python scripts/attnmse/onpolicy_profile.py [name=path ...]
Writes runs/attnmse/onpolicy_profile.json. Board: k7hbif.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.68")

import argparse
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
from qwen_jax.cache import KVCacheLayer
from qwen_jax.cartridge import Cartridge
from qwen_jax.selfstudy import load_examples

P = 1024
DEFAULT_CARTS = {
    "init": None,
    "attnmse2k": REPO / "runs/attnmse/attnmse-2k.safetensors",
    "rms1": REPO / "runs/attnmse/overnight/rms1.safetensors",
    "kl-rms": REPO / "runs/cart/rms-lr1e-2.safetensors",
    "kl-raw": REPO / "runs/cart/raw-lr1e-2.safetensors",
}
KEYS = ("err_tf", "err_op", "out_shift", "drift", "tgt_shift", "check")


def _rotary(model, position_ids):
    lm = model.model.language_model
    cos, sin = lm.rotary_emb(position_ids)
    dt = lm.embed_tokens.weight().dtype
    return cos.astype(dt), sin.astype(dt)


def profile(model, cartridge, batch, *, eps=1e-6):
    """(layers, len(KEYS)) relative errors for one batch."""
    lm = model.model.language_model
    ids, mask = batch.teacher_ids, batch.teacher_mask
    context = batch.context
    b, s = ids.shape
    S = s - context
    pos_t, _, _ = model.model._resolve_position_ids(ids, None, mask, None, None, None, None)
    cos_t, sin_t = _rotary(model, pos_t)
    kmask_t = mask.astype(jnp.bool)
    s_ids, s_mask = batch.student_ids, batch.student_mask
    pos_s, _, _ = model.model._resolve_position_ids(s_ids, None, s_mask, None, None, None, P)
    pos_s_emb = _rotary(model, pos_s)
    smask = s_mask.astype(jnp.bool)
    prefix = cartridge.prefix(model.cache_dtype())
    kv_mask_s = jnp.concatenate([jnp.ones((b, P), jnp.bool), smask], axis=1)
    # Suffix input at the teacher's positions, attending over the teacher's
    # context KV as a prefix: the teacher's layer, on whatever stream we feed it.
    pos_tc_emb = (cos_t[:, context:], sin_t[:, context:])
    kv_mask_tc = jnp.concatenate([kmask_t[:, :context], smask], axis=1)
    m = batch.loss_mask[..., None]

    def rel(x, ref):
        d = jnp.where(m, (x - ref).astype(jnp.float32), 0.0)
        r = jnp.where(m, ref.astype(jnp.float32), 0.0)
        return jnp.sum(d * d) / (jnp.sum(r * r) + eps)

    ht = lm.embed_tokens(ids)
    hs = lm.embed_tokens(s_ids)
    rows = []
    for i, layer in enumerate(lm.layers):
        attn = layer.self_attn
        xt = layer.input_layernorm(ht)
        k_full = attn.k_norm(attn.k_proj(xt).reshape(b, s, 8, 128))
        v_full = attn.v_proj(xt).reshape(b, s, 8, 128)
        _, k_full = apply_rotary_pos_emb(k_full, k_full, cos_t, sin_t)
        ctx = KVCacheLayer(keys=k_full[:, :context], values=v_full[:, :context])
        a_t, _ = attn(xt, position_embeddings=(cos_t, sin_t), kv_mask=kmask_t)
        tgt_tf = a_t[:, context:]
        xt_suf = xt[:, context:]
        xs = layer.input_layernorm(hs)
        a_s, _ = attn(xs, position_embeddings=pos_s_emb, kv_mask=kv_mask_s,
                      prefix=prefix.layer(i))
        tgt_op, _ = attn(xs, position_embeddings=pos_tc_emb, kv_mask=kv_mask_tc, prefix=ctx)
        a_tf, _ = attn(xt_suf, position_embeddings=pos_s_emb, kv_mask=kv_mask_s,
                       prefix=prefix.layer(i))
        chk, _ = attn(xt_suf, position_embeddings=pos_tc_emb, kv_mask=kv_mask_tc, prefix=ctx)
        rows.append(jnp.stack([
            rel(a_tf, tgt_tf),
            rel(a_s, tgt_op),
            rel(a_s, tgt_tf),
            rel(hs, ht[:, context:]),
            rel(tgt_op, tgt_tf),
            rel(chk, tgt_tf),
        ]))
        ht = ht + a_t
        ht = ht + layer.mlp(layer.post_attention_layernorm(ht))
        hs = hs + a_s
        hs = hs + layer.mlp(layer.post_attention_layernorm(hs))
    return jnp.stack(rows)


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
    for bt in held:
        assert bool(jnp.all(bt.teacher_mask[:, bt.context:] == bt.student_mask))

    carts = {}
    for name, path in specs.items():
        if path is None:
            carts[name] = cli.init_cartridge(model, tokenizer, corpus, P, cli.DESCRIPTION)
        elif path.exists():
            carts[name] = Cartridge.load(path)
        else:
            print(f"{name}: {path} missing, skipped", flush=True)

    f = jax.jit(profile)
    out = {}
    for name, cart in carts.items():
        acc = np.mean([np.asarray(f(model, cart, bt)) for bt in held], axis=0)  # (36, 6)
        out[name] = {k: acc[:, j].tolist() for j, k in enumerate(KEYS)}
        print(f"{name}: " + "  ".join(f"{k}={acc[:, j].mean():.4f}" for j, k in enumerate(KEYS)),
              flush=True)

    path = REPO / "runs/attnmse/onpolicy_profile.json"
    path.write_text(json.dumps(out, indent=1))
    names = list(carts)
    print(f"\n{'layer':>5} | " + " | ".join(f"{n:>34}" for n in names))
    print(f"{'':>5} | " + " | ".join(f"{'err_tf':>6} {'err_op':>6} {'out_sh':>6} {'drift':>6} {'tgt_sh':>6}"
                                     for _ in names))
    for li in range(36):
        cells = []
        for n in names:
            o = out[n]
            cells.append(f"{o['err_tf'][li]:6.3f} {o['err_op'][li]:6.3f} {o['out_shift'][li]:6.3f} "
                         f"{o['drift'][li]:6.3f} {o['tgt_shift'][li]:6.3f}")
        print(f"{li:>5} | " + " | ".join(cells))
    print("\ncheck (max over layers): " +
          "  ".join(f"{n}={max(out[n]['check']):.2e}" for n in names))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
