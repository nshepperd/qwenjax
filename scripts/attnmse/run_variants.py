"""Overnight anti-collapse experiments on the attnmse cartridge objective.

Variants (all: p=1024, lr 1e-2 WSD, batch 2, same data as the baselines):
  temp    -- prefix attention temperature annealed 4 -> 1 over 1000 steps
  noise   -- instance noise on slot keys, sigma 0.3*RMS -> 0 over 1000 steps
  resets  -- VQ-style dead-slot resets every 250 steps from a corpus-KV
             reservoir, Adam moments zeroed on reset
  rms1    -- unit-RMS reparameterization (per-layer scales fixed at init)
  anchor  -- attnmse + KL to the *init cartridge's* behaviour on
             corpus-irrelevant conversations (abstention anchor)

The unit-RMS parameterization has since become `Cartridge`'s default (it won
the night); the non-rms variants here build their cartridge with
`unit_rms=False` so they still reproduce the raw-parameter runs they were.

Usage: uv run python scripts/attnmse/run_variants.py <variant>  (outputs: runs/attnmse/overnight/)
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
HERE = REPO / "runs/attnmse/overnight"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
import optax

import cartridge as cli
from qwen_jax import chat
from qwen_jax.attnmse import attnmse_loss, attnmse_loss_graduated
from qwen_jax.cache import KVPrefix
from qwen_jax.cartridge import Cartridge
from qwen_jax.distill import Trainer, blockwise_kl, evaluate, make_batch
from qwen_jax.selfstudy import Example, load_examples

P = 1024
SCALE = 128 ** -0.5
STEPS = int(os.environ.get("VARIANT_STEPS", 2000))
LR = 1e-2


def setup(*, unit_rms: bool):
    tokenizer = cli.load_tokenizer()
    corpus = cli.load_corpus(tokenizer, None)
    train = load_examples(str(REPO / "runs/cart/train.jsonl"))
    heldout = load_examples(str(REPO / "runs/cart/heldout.jsonl"))
    model = cli.load_model()
    init = cli.init_cartridge(model, tokenizer, corpus, P, cli.DESCRIPTION, unit_rms=unit_rms)
    shapes = argparse.Namespace(batch=2, context=2176, seq=512, description=cli.DESCRIPTION)
    held = list(cli.batches_from(tokenizer, heldout, shapes, shuffle=False))
    return tokenizer, corpus, train, model, init, shapes, held


def train_loop(model, tokenizer, train, shapes, trainer, cart, held, out,
               steps=STEPS, hook=None, batch_wrap=None):
    state = trainer.init(cart)
    log, recent, step, epoch, t0 = [], [], 0, 0, time.time()
    while step < steps:
        for batch in cli.batches_from(tokenizer, train, shapes, shuffle=True, seed=epoch):
            if batch_wrap is not None:
                batch = batch_wrap(batch, step)
            cart, state, loss = trainer.step(model, cart, state, batch)
            recent.append(float(loss))
            step += 1
            if hook is not None:
                cart, state = hook(cart, state, step)
            if step % 100 == 0 or step == steps:
                row = {"step": step, "loss": float(np.mean(recent)), "time": time.time() - t0}
                if step % 500 == 0 or step == steps:
                    row["heldout_kl"] = evaluate(model, cart, held)
                log.append(row)
                print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in row.items()), flush=True)
                recent = []
            if step >= steps:
                break
        epoch += 1
    return cart, log


def finish(cart, log, name):
    cart.save(HERE / f"{name}.safetensors")
    (HERE / f"{name}.log.json").write_text(json.dumps(log, indent=1))
    print(f"saved {name}", flush=True)


# --------------------------------------------------------------------------
# resets machinery
# --------------------------------------------------------------------------


@jax.jit
def _prefix_mass(K, q, k, mi):
    """(kv_heads, P) attention mass on prefix slots for one layer, one batch."""
    b, s = q.shape[:2]
    causal = jnp.tril(jnp.ones((s, s), bool))
    K_full = jnp.concatenate([jnp.broadcast_to(K[None], (b, *K.shape)), k], axis=1)
    qg = q.reshape(b, s, 8, 4, 128)
    scores = jnp.einsum("bqhgd,bkhd->bhgqk", qg, K_full) * SCALE
    valid = jnp.concatenate([jnp.ones((b, P), bool), mi], axis=1)[:, None, None, None, :]
    cmask = jnp.concatenate([jnp.ones((s, P), bool), causal], axis=1)[None, None, None, :, :]
    scores = jnp.where(valid & cmask, scores, -jnp.inf)
    w = jax.nn.softmax(scores, axis=-1)
    w = w * mi.astype(jnp.float32)[:, None, None, :, None]
    return jnp.sum(w[..., :P], axis=(0, 2, 3))


def make_reset_hook(model, tokenizer, corpus, probe_batches, every=250, thresh=1e-6):
    from study import teacher_layer_inputs

    reservoir = Cartridge.init_from_tokens(model, corpus.head(2048), freeze_first=False,
                                           unit_rms=False)
    resK = np.asarray(reservoir.keys, np.float32)   # (36, 2048, 8, 128), physical
    resV = np.asarray(reservoir.values, np.float32)
    rng = np.random.default_rng(0)
    layer_ids = list(range(36))
    print("capturing probe q/k for reset mass checks...", flush=True)
    sub, lmask = teacher_layer_inputs(model, probe_batches, layer_ids)
    # kept on host; transferred per reset round so they cost the training
    # step no residency
    qs = {li: sub[li]["q"] for li in layer_ids}
    ks = {li: sub[li]["k"] for li in layer_ids}
    mi = jnp.asarray(lmask)

    def hook(cart, state, step):
        if step % every or step >= STEPS:
            return cart, state
        # Physical KV, so the mass check and the reservoir mean the same thing
        # under either parameterization. np.array copies: asarray views are
        # read-only.
        K = np.array(cart.physical_keys, np.float32)
        V = np.array(cart.physical_values, np.float32)
        dead_mask = np.zeros(K.shape[:3], bool)  # (36, P, 8)
        for li in layer_ids:
            mass = np.asarray(_prefix_mass(jnp.asarray(K[li]), jnp.asarray(qs[li]),
                                           jnp.asarray(ks[li]), mi))  # (8, P)
            dead = (mass < thresh * mass.sum(axis=1, keepdims=True)).T  # (P, 8)
            dead[0] = False  # never touch the sink
            dead_mask[li] = dead
        n = int(dead_mask.sum())
        if n:
            for li in layer_ids:
                jj, hh = np.nonzero(dead_mask[li])
                toks = rng.integers(0, resK.shape[1], size=len(jj))
                K[li, jj, hh] = resK[li, toks, hh]
                V[li, jj, hh] = resV[li, toks, hh]
            cart = cart.with_params((jnp.asarray(K) / cart.key_scale,
                                     jnp.asarray(V) / cart.value_scale))
            zmask = jnp.asarray(~dead_mask[:, :, :, None])  # keep live moments

            def zero(x):
                if hasattr(x, "shape") and tuple(x.shape) == K.shape:
                    return x * zmask
                return x

            state = jax.tree.map(zero, state)
        print(f"  [reset @ {step}] revived {n} (slot,head) blocks", flush=True)
        return cart, state

    return hook


# --------------------------------------------------------------------------
# anchor machinery
# --------------------------------------------------------------------------


def anchor_batches(tokenizer, shapes):
    rows = [json.loads(l) for l in (HERE / "anchor.jsonl").read_text().splitlines()]
    examples = [Example(chunk_ids=[], user=r["user"], assistant=r["assistant"]) for r in rows]
    out = []
    for i in range(0, len(examples) - 1, 2):
        b = make_batch(tokenizer, examples[i:i + 2], description="", context=8, seq=256,
                       pad_id=tokenizer.pad_token_id)
        out.append((b.student_ids, b.student_mask))
    return out


def anchor_loss_fn(model, cartridge, packed, *, block=128):
    """KL of the current cartridge against the *init* cartridge on
    corpus-irrelevant conversations. Run on its own steps, interleaved with
    the attnmse objective -- the summed graph does not fit in 16 GB, and the
    interleave ratio plays the role of the mixing weight."""
    a_ids, a_mask, initK, initV = packed
    init_prefix = KVPrefix(keys=initK.astype(model.cache_dtype()),
                           values=initV.astype(model.cache_dtype()))
    t_hidden, _, _ = model.model(input_ids=a_ids, attention_mask=a_mask, prefix=init_prefix)
    t_hidden = jax.lax.stop_gradient(t_hidden)
    s_hidden, _, _ = model.model(input_ids=a_ids, attention_mask=a_mask,
                                 prefix=cartridge.prefix(model.cache_dtype()))
    return blockwise_kl(model.get_lm_head(), s_hidden, t_hidden,
                        a_mask.astype(jnp.bool), block=block)


def run_anchor(model, tokenizer, train, shapes, init, held, name, *, every=4):
    """attnmse steps interleaved with off-corpus anchor steps (1 in `every`).

    Anchor set: 24 conversations, the first 16 trained on, the last 8 held
    out, so the anchor's generalisation is measured rather than assumed. The
    anchor batch index advances on its own counter (an earlier version
    aliased it against the interleave period and only ever saw 3 batches).
    """
    anchors = anchor_batches(tokenizer, shapes)
    a_train, a_held = anchors[:8], anchors[8:]
    initK, initV = init.physical_keys, init.physical_values
    a_steps = min(1500, STEPS)
    opt = optax.adam(cli.wsd_schedule(LR, a_steps, warmup=20, decay_frac=0.2))
    core_tr = Trainer(opt, loss=attnmse_loss)
    anch_tr = Trainer(opt, loss=anchor_loss_fn)
    f_anchor = jax.jit(anchor_loss_fn)
    cart, state = init, core_tr.init(init)
    log, recent, step, epoch, a_i, aloss, t0 = [], [], 0, 0, 0, float("nan"), time.time()

    def held_anchor(c):
        return float(np.mean([float(f_anchor(model, c, (*a, initK, initV))) for a in a_held]))

    while step < a_steps:
        for batch in cli.batches_from(tokenizer, train, shapes, shuffle=True, seed=epoch):
            if step % every == every - 1:
                packed = (*a_train[a_i % len(a_train)], initK, initV)
                a_i += 1
                cart, state, aloss = anch_tr.step(model, cart, state, packed)
            else:
                cart, state, loss = core_tr.step(model, cart, state, batch)
                recent.append(float(loss))
            step += 1
            if step % 100 == 0 or step == a_steps:
                row = {"step": step, "loss": float(np.mean(recent)), "anchor_kl_train": float(aloss),
                       "time": time.time() - t0}
                if step % 500 == 0 or step == a_steps:
                    row["heldout_kl"] = evaluate(model, cart, held)
                    row["anchor_kl_held"] = held_anchor(cart)
                log.append(row)
                print("  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in row.items()), flush=True)
                recent = []
            if step >= a_steps:
                break
        epoch += 1
    finish(cart, log, name)


# --------------------------------------------------------------------------
# variants
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=("temp", "noise", "resets", "rms1", "anchor", "rms1anchor"))
    args = ap.parse_args()
    unit_rms = args.variant in ("rms1", "rms1anchor")
    tokenizer, corpus, train, model, init, shapes, held = setup(unit_rms=unit_rms)
    schedule = cli.wsd_schedule(LR, STEPS, warmup=20, decay_frac=0.2)

    if args.variant == "temp":
        loss = functools.partial(attnmse_loss_graduated, tau0=4.0, noise0=0.0, anneal_steps=1000)
        trainer = Trainer(optax.adam(schedule), loss=loss)
        cart, log = train_loop(model, tokenizer, train, shapes, trainer, init, held, "temp")
        finish(cart, log, "temp")

    elif args.variant == "noise":
        loss = functools.partial(attnmse_loss_graduated, tau0=1.0, noise0=0.3, anneal_steps=1000)
        trainer = Trainer(optax.adam(schedule), loss=loss)
        cart, log = train_loop(model, tokenizer, train, shapes, trainer, init, held, "noise")
        finish(cart, log, "noise")

    elif args.variant == "resets":
        trainer = Trainer(optax.adam(schedule), loss=attnmse_loss)
        gen = cli.batches_from(tokenizer, train, shapes, shuffle=False)
        hook = make_reset_hook(model, tokenizer, corpus, [next(gen), next(gen)])
        cart, log = train_loop(model, tokenizer, train, shapes, trainer, init, held,
                               "resets", hook=hook)
        finish(cart, log, "resets")

    elif args.variant == "rms1":
        # The baseline objective on a unit-RMS cartridge (now the default
        # parameterization; `setup` chose it from the variant name).
        trainer = Trainer(optax.adam(schedule), loss=attnmse_loss)
        cart, log = train_loop(model, tokenizer, train, shapes, trainer, init, held, "rms1")
        finish(cart, log, "rms1")

    elif args.variant in ("anchor", "rms1anchor"):
        run_anchor(model, tokenizer, train, shapes, init, held, args.variant)


if __name__ == "__main__":
    main()
