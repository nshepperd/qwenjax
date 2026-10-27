"""Probe: the mean key at each full-attention layer, and what centering buys.

Runs a qwen35 GGUF (default: Saluki) with float32 activations over real text
and captures the post-RoPE q, k, v entering each full-attention layer. Then,
per layer:

  key stats   rms |k|, |mu| (mu = mean key over positions, per kv head),
              and the share of key energy in the mean, split into the
              rotated (first rotary_dim) and pass-through dims;
  logits      per query, the constant offset q.mu/sqrt(d) next to the
              spread across visible keys of q.(k - mu)/sqrt(d). Subtracting
              a fixed mu from every key shifts each query's logits by a
              constant, which the softmax ignores -- exactly, for the
              pass-through dims;
  rounding    relative error of the attention output when k is stored in
              bf16 / fp8 (q and v in bf16 throughout), plain vs centered
              (k - mu stored, mu from the *other* text, as a fixed
              calibration constant would be), against the float32 output.

    uv run python bench/key_mean_probe.py
"""
from __future__ import annotations

import os

os.environ.pop("TF_FORCE_UNIFIED_MEMORY", None)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.84"
# The float32 reference path: fa4 would round q/k/v to fp16 inside the run.
os.environ["QWEN_JAX_ATTENTION"] = "xla"

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from transformers import AutoTokenizer

import qwen_jax.qwen35.attention as q35attn
from qwen_jax.qwen35 import load_qwen35_gguf

CONFIG = "/data/models/saluki/qwen3.8-27b-config"
GGUF = "/data/models/saluki/Underdog-Saluki-27B-1.0-IQ2-mix.gguf"
SITE = Path(__file__).resolve().parents[1] / ".venv/lib/python3.14/site-packages"
TEXTS = {
    "prose": [
        Path.home() / "Dev/neural/fa4-jax/README.md",
        Path.home() / "Dev/neural/fa4-jax/DEVELOPING.md",
        Path("/data/models/saluki/README.md"),
    ],
    "code": [SITE / "transformers/models/qwen3_5/modeling_qwen3_5.py"],
}

captured: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []


def capture_attention():
    """Wrap the attention call so each layer's q, k, v are appended to `captured`.

    The layer comes from execution order (an ordered callback), not from a
    trace-time counter: the attention __call__ is jax.remat'd, and remat reuses
    one trace for every layer with the same structure, so a counter would see
    only the distinct structures and the reused callbacks would overwrite.
    """
    inner = q35attn.masked_causal_attention

    def store(q, k, v):
        captured.append((np.asarray(q[0]), np.asarray(k[0]), np.asarray(v[0])))

    def wrapped(q, k, v, **kw):
        jax.debug.callback(store, q, k, v, ordered=True)
        return inner(q, k, v, **kw)

    q35attn.masked_causal_attention = wrapped


def run(model, tokenizer, paths, length, n_layers):
    text = "\n\n".join(p.read_text() for p in paths)
    ids = tokenizer(text, return_tensors="np")["input_ids"][:, :length]
    assert ids.shape[1] == length, f"only {ids.shape[1]} tokens of text"
    captured.clear()
    jax.block_until_ready(model(input_ids=jnp.asarray(ids), last_logit_only=True).last_logits)
    jax.effects_barrier()
    assert len(captured) == n_layers, f"captured {len(captured)} layers, expected {n_layers}"
    return dict(enumerate(captured))


def rnd(x, dtype):
    return x.astype(dtype).astype(np.float32)


def attend(q, k, v, causal_from):
    """q (n, d) at positions causal_from (n,), k/v (S, d): softmax(q.k/sqrt d) v."""
    s = q @ k.T / np.sqrt(q.shape[-1])
    s = np.where(np.arange(k.shape[0])[None] <= causal_from[:, None], s, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)) @ v


def analyse(layers, mus_other, rotary_dim, n_queries, skip):
    rows = []
    for i, (q, k, v) in layers.items():
        S, H, d = q.shape
        KH = k.shape[1]
        G = H // KH
        ks = k[skip:]  # the first tokens can be attention sinks; keep them out of mu
        mu = ks.mean(0)  # (KH, d)
        rms_k = np.sqrt((ks**2).sum(-1).mean(0))  # (KH,)
        share = (mu**2).sum(-1) / (ks**2).sum(-1).mean(0)
        share_rot = (mu[:, :rotary_dim] ** 2).sum(-1) / (ks[..., :rotary_dim] ** 2).sum(-1).mean(0)
        share_pass = (mu[:, rotary_dim:] ** 2).sum(-1) / (ks[..., rotary_dim:] ** 2).sum(-1).mean(0)

        pos = np.linspace(S // 4, S - 1, n_queries).astype(int)
        offset, spread, err = [], [], {m: [] for m in ("bf16", "bf16 c", "fp8", "fp8 c")}
        mu_cal = mus_other[i]
        for h in range(H):
            g = h // G
            qh, kh, vh = q[pos, h], k[:, g], v[:, g]
            vis = np.arange(S)[None] <= pos[:, None]
            off = qh @ mu[g] / np.sqrt(d)
            inf = (qh @ (kh - mu[g]).T) / np.sqrt(d)
            sd = np.sqrt((np.where(vis, inf, 0) ** 2).sum(1) / vis.sum(1)
                         - (np.where(vis, inf, 0).sum(1) / vis.sum(1)) ** 2)
            offset.append(np.abs(off).mean())
            spread.append(sd.mean())
            ref = attend(qh, kh, vh, pos)
            qb, vb = rnd(qh, ml_dtypes.bfloat16), rnd(vh, ml_dtypes.bfloat16)
            for name, dt in (("bf16", ml_dtypes.bfloat16), ("fp8", ml_dtypes.float8_e4m3fn)):
                for centred in (False, True):
                    kk = rnd(kh - mu_cal[g], dt) + mu_cal[g] if centred else rnd(kh, dt)
                    out = attend(qb, kk, vb, pos)
                    e = np.linalg.norm(out - ref, axis=-1) / np.linalg.norm(ref, axis=-1)
                    err[name + (" c" if centred else "")].append(e.mean())
        rows.append(dict(
            layer=i, rms_k=rms_k.mean(), mu=np.linalg.norm(mu, axis=-1).mean(),
            share=share.mean(), share_rot=share_rot.mean(), share_pass=share_pass.mean(),
            k0=np.linalg.norm(k[0], axis=-1).mean(),
            offset=np.mean(offset), spread=np.mean(spread),
            **{m: np.mean(e) for m, e in err.items()},
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--gguf", default=GGUF)
    parser.add_argument("--length", type=int, default=4000)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--skip", type=int, default=4, help="leading tokens kept out of mu")
    args = parser.parse_args()

    capture_attention()
    model = load_qwen35_gguf(args.config, args.gguf, dtype=jnp.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.config)
    hf_text = json.loads((Path(args.config) / "config.json").read_text())["text_config"]
    rotary_dim = int(hf_text["head_dim"] * hf_text["rope_parameters"]["partial_rotary_factor"])

    n_layers = hf_text["layer_types"].count("full_attention")
    runs = {name: run(model, tokenizer, paths, args.length, n_layers)
            for name, paths in TEXTS.items()}
    mus = {name: {i: k[args.skip:].mean(0) for i, (_, k, _) in layers.items()}
           for name, layers in runs.items()}
    names = list(runs)
    for name in names:
        other = names[1 - names.index(name)]
        cos = [
            float(np.mean([
                mus[name][i][g] @ mus[other][i][g]
                / (np.linalg.norm(mus[name][i][g]) * np.linalg.norm(mus[other][i][g]))
                for g in range(mus[name][i].shape[0])
            ]))
            for i in mus[name]
        ]
        rows = analyse(runs[name], mus[other], rotary_dim, args.queries, args.skip)
        print(f"\n== {name} ({args.length} tokens); centering uses mu from '{other}' ==")
        print(f"{'layer':>5s} {'rms|k|':>7s} {'|mu|':>6s} {'mu share':>8s} {'rot':>5s} {'pass':>5s}"
              f" {'|k_0|':>6s} {'cos mu':>6s} {'|q.mu|':>7s} {'spread':>7s}"
              f" {'bf16':>8s} {'bf16 c':>8s} {'fp8':>8s} {'fp8 c':>8s}")
        for r, c in zip(rows, cos):
            print(f"{r['layer']:5d} {r['rms_k']:7.2f} {r['mu']:6.2f} {r['share']:8.2f}"
                  f" {r['share_rot']:5.2f} {r['share_pass']:5.2f} {r['k0']:6.2f} {c:6.2f}"
                  f" {r['offset']:7.2f} {r['spread']:7.2f}"
                  f" {r['bf16']:8.1e} {r['bf16 c']:8.1e} {r['fp8']:8.1e} {r['fp8 c']:8.1e}")


if __name__ == "__main__":
    main()
