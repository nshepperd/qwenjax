"""Probe: where a qwen35 GGUF's decode and prefill time goes, per component.

Times, on real layers of the model (default: Saluki), each component on its
own -- the GatedDeltaNet mixer and its delta-rule core, the full-attention
mixer against a cache of --context, the MLP -- at decode (1 token) and at a
prefill chunk, then the whole-model decode step for comparison. Per-token
totals scale the per-layer times by the layer counts.

    env -u TF_FORCE_UNIFIED_MEMORY XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
      XLA_PYTHON_CLIENT_MEM_FRACTION=0.84 uv run python bench/layer_breakdown.py
"""
from __future__ import annotations

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp

from qwen_jax.qwen35 import load_qwen35_gguf
from qwen_jax.qwen35.deltanet import chunk_gated_delta_rule, recurrent_gated_delta_rule_step

CONFIG = "/data/models/saluki/qwen3.8-27b-config"
GGUF = "/data/models/saluki/Underdog-Saluki-27B-1.0-IQ2-mix.gguf"


def timed(fn, *args, n=20):
    """Median-free mean ms per call, after a warm call; outputs discarded."""
    jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n * 1e3


@eqx.filter_jit
def mixer_gdn(m, h, cache, token_mask):
    return m(h, cache=cache, token_mask=token_mask)


@eqx.filter_jit
def mixer_attn(m, h, pe, cache, pos, kv_mask):
    return m(h, pe, cache=cache, cache_position=pos, kv_mask=kv_mask)


@eqx.filter_jit
def mlp(m, h):
    return m(h)


@jax.jit
def core_step(q, k, v, g, beta, state):
    return recurrent_gated_delta_rule_step(q, k, v, g, beta, state)


@jax.jit
def core_chunk(q, k, v, g, beta, state):
    return chunk_gated_delta_rule(q, k, v, g, beta, initial_state=state)


@eqx.filter_jit(donate="all-except-first")
def model_step(model, ids, cache):
    out = model(input_ids=ids, cache=cache, last_logit_only=True)
    return out.last_logits, out.cache


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--gguf", default=GGUF)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()

    model = load_qwen35_gguf(args.config, args.gguf)
    tc = model.config.text_config
    lm = model.model.language_model
    types = tc.resolved_layer_types()
    gi = types.index("linear_attention")
    ai = types.index("full_attention")
    n_gdn, n_attn = types.count("linear_attention"), types.count("full_attention")
    gdn_layer, attn_layer = lm.layers[gi], lm.layers[ai]
    L = args.context
    cache = model.create_cache(1, L)
    key = jax.random.key(0)
    H, dk, dv = tc.linear_num_value_heads, tc.linear_key_head_dim, tc.linear_value_head_dim

    rows = []
    for S in (1, args.chunk):
        h = jax.random.normal(key, (1, S, tc.hidden_size), jnp.bfloat16)
        tm = jnp.ones((1, S), bool)
        pos = jnp.array(L - S, jnp.int32)
        pe = lm.rotary_emb(jnp.broadcast_to(jnp.arange(L - S, L)[None, None], (3, 1, S)))
        pe = jax.tree.map(lambda x: x.astype(jnp.bfloat16), pe)
        kvm = jnp.ones((1, L), bool)

        t_gdn = timed(mixer_gdn, gdn_layer.linear_attn, h, cache.layers[gi], tm)
        q = jax.random.normal(key, (1, S, H, dk), jnp.float32) / dk**0.5
        k = q
        v = jax.random.normal(key, (1, S, H, dv), jnp.float32)
        g = -jnp.ones((1, S, H), jnp.float32) * 0.1
        beta = jnp.full((1, S, H), 0.5, jnp.float32)
        state = cache.layers[gi].recurrent_state
        if S == 1:
            t_core = timed(core_step, q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], state)
        else:
            t_core = timed(core_chunk, q, k, v, g, beta, state)
        t_attn = timed(mixer_attn, attn_layer.self_attn, h, pe, cache.layers[ai], pos, kvm)
        t_mlp = timed(mlp, gdn_layer.mlp, h)
        rows.append((S, t_gdn, t_core, t_attn, t_mlp))

    print(f"Saluki layers {gi} (GDN) and {ai} (attention), attention cache {L}; "
          f"{n_gdn} GDN + {n_attn} attention layers, MLP in all {n_gdn + n_attn}")
    print(f"{'tokens':>6s} {'GDN mixer':>10s} {'GDN core':>9s} {'attn mixer':>10s} {'MLP':>7s}   "
          f"per-step totals: {'GDN':>6s} {'(core)':>7s} {'attn':>6s} {'MLP':>6s} {'sum':>7s}  (ms)")
    for S, t_gdn, t_core, t_attn, t_mlp in rows:
        tg, tc_, ta, tmlp = n_gdn * t_gdn, n_gdn * t_core, n_attn * t_attn, (n_gdn + n_attn) * t_mlp
        print(f"{S:6d} {t_gdn:10.3f} {t_core:9.3f} {t_attn:10.3f} {t_mlp:7.3f}   "
              f"                 {tg:6.1f} {tc_:7.1f} {ta:6.1f} {tmlp:6.1f} {tg + ta + tmlp:7.1f}")

    del cache
    for ctx in (64, L):
        c = model.create_cache(1, ctx + 32)
        logits, c = model_step(model, jnp.zeros((1, 1), jnp.int32), c)  # compile
        t0 = time.perf_counter()
        for _ in range(16):
            logits, c = model_step(model, jnp.argmax(logits, -1)[:, None], c)
        jax.block_until_ready(logits)
        print(f"whole-model decode step, cache {ctx + 32}: "
              f"{(time.perf_counter() - t0) / 16 * 1e3:.1f} ms")
        del c, logits


if __name__ == "__main__":
    main()
