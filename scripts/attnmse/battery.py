"""Uniform eval battery over every cartridge produced tonight (plus references).

Per cartridge: held-out KL, held-out attnmse, slot utilization at layers
1/15/20/29 (pooled and per-(slot,head)), and greedy samples on the trap
question and one real question. Writes summary.json and samples-overnight.txt.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.68")

import argparse
import json
import sys
from pathlib import Path

REPO = Path("/home/em/Dev/neural/qwenjax")
HERE = REPO / "runs/attnmse/overnight"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np

import cartridge as cli
from qwen_jax import chat
from qwen_jax.attnmse import attnmse_loss
from qwen_jax.cartridge import Cartridge
from qwen_jax.distill import evaluate
from qwen_jax.selfstudy import load_examples
from study import slot_diagnostics, teacher_layer_inputs

LAYERS = [1, 15, 20, 29]
QUESTIONS = [
    "What does KVCache.evict_lru do in cache.py?",
    "In qwen-jax, what does KVCache.write_prefix do, and what does it return?",
]

tokenizer = cli.load_tokenizer()
corpus = cli.load_corpus(tokenizer, None)
train = load_examples(str(REPO / "runs/cart/train.jsonl"))
heldout = load_examples(str(REPO / "runs/cart/heldout.jsonl"))
model = cli.load_model()
shapes = argparse.Namespace(batch=2, context=2176, seq=512, description=cli.DESCRIPTION)
held = list(cli.batches_from(tokenizer, heldout, shapes, shuffle=False))
probe = []
for b in cli.batches_from(tokenizer, train, shapes, shuffle=False):
    probe.append(b)
    if len(probe) >= 8:
        break
sub, lmask = teacher_layer_inputs(model, probe, LAYERS)

CARTS = {
    "init": cli.init_cartridge(model, tokenizer, corpus, 1024, cli.DESCRIPTION),
    "baseline2k": REPO / "runs/attnmse/attnmse-2k.safetensors",
    "temp": HERE / "temp.safetensors",
    "noise": HERE / "noise.safetensors",
    "resets": HERE / "resets.safetensors",
    "rms1": HERE / "rms1.safetensors",
    "anchor": HERE / "anchor.safetensors",
    "rms1anchor": HERE / "rms1anchor.safetensors",
}

f_attn = jax.jit(attnmse_loss)
im_end = tokenizer.convert_tokens_to_ids(chat.IM_END)
summary = {}
samples = []
for name, src in CARTS.items():
    if isinstance(src, Path):
        if not src.exists():
            print(f"{name}: missing, skipped", flush=True)
            continue
        cart = Cartridge.load(src)
    else:
        cart = src
    row = {"heldout_kl": evaluate(model, cart, held),
           "heldout_attnmse": float(np.mean([float(f_attn(model, cart, b)) for b in held]))}
    for li in LAYERS:
        q, k = jnp.asarray(sub[li]["q"]), jnp.asarray(sub[li]["k"])
        o_proj = model.model.language_model.layers[li].self_attn.o_proj
        params = {"K": jnp.asarray(cart.keys[li], jnp.float32),
                  "V": jnp.asarray(cart.values[li], jnp.float32)}
        mass, _ = slot_diagnostics(o_proj, q, k, None, np.asarray(lmask), params)  # (8, P)
        fh = mass / mass.sum(axis=1, keepdims=True)
        m = mass.sum(axis=0)
        f = m / m.sum()
        row[f"layer{li}"] = {
            "eff_slots": float(np.exp(-(f * np.log(f + 1e-12)).sum())),
            "dead_blocks": int((fh < 1e-5).sum()),
            "top_slot": float(np.sort(f)[-1]),
        }
    summary[name] = row
    print(name, json.dumps(row), flush=True)

    for qtext in QUESTIONS:
        suffix = tokenizer.encode(chat.suffix([("user", qtext)], open_assistant=True),
                                  add_special_tokens=False)
        out = model.generate(
            input_ids=jnp.asarray([suffix], dtype=jnp.int32),
            prefix=cart.prefix(model.cache_dtype()),
            max_new_tokens=220, key=jax.random.key(0), temperature=0.0,
            stop_token_id=im_end, pad_token_id=im_end, progress_bar=False,
        )
        gen = out.tokens[0, len(suffix):].tolist()
        if im_end in gen:
            gen = gen[: gen.index(im_end)]
        samples.append(f"\n######## [{name}] {qtext}\n{tokenizer.decode(gen)}")

(HERE / "summary.json").write_text(json.dumps(summary, indent=1))
(HERE / "samples-overnight.txt").write_text("\n".join(samples))
print("battery done", flush=True)
