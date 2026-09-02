"""Frequency-band decomposition of how training moved the cartridge keys.

For each trained cartridge, Delta-K = K_trained - K_init is split by RoPE
frequency pair (pair i = dims (i, i+64), wavelength 2*pi*theta^(i/64)); band
energy is the fraction of ||Delta-K||^2 in each pair, per layer. Values have
no phase, so V is reported as a single number per layer.
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.68")

import json
import sys
from pathlib import Path

REPO = Path("/home/em/Dev/neural/qwenjax")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

import cartridge as cli
from qwen_jax.cartridge import Cartridge

OUT = REPO / "runs/attnmse/landscape"

tokenizer = cli.load_tokenizer()
corpus = cli.load_corpus(tokenizer, None)
model = cli.load_model()
init = cli.init_cartridge(model, tokenizer, corpus, 1024, cli.DESCRIPTION)
K0 = np.asarray(init.physical_keys, np.float32)   # (36, 1024, 8, 128)
V0 = np.asarray(init.physical_values, np.float32)

CARTS = {
    "kl": REPO / "runs/cart/sweep-lr1e-2.safetensors",
    "attnmse": REPO / "runs/attnmse/attnmse-lr1e-2.safetensors",
    "attnmse2k": REPO / "runs/attnmse/attnmse-2k.safetensors",
}

out = {"wavelength": (2 * np.pi * 5e6 ** (np.arange(64) / 64)).tolist()}
for name, path in CARTS.items():
    c = Cartridge.load(path)
    dK = np.asarray(c.physical_keys, np.float32) - K0
    dV = np.asarray(c.physical_values, np.float32) - V0
    # band energy: (36 layers, 64 pairs), summed over slots and kv heads
    e = dK[..., :64] ** 2 + dK[..., 64:] ** 2
    band = e.sum(axis=(1, 2))
    out[name] = {
        "K_band_energy": band.tolist(),
        "K_total": float((dK ** 2).sum(axis=(1, 2, 3)).sum()),
        "K_per_layer": (dK ** 2).sum(axis=(1, 2, 3)).tolist(),
        "V_per_layer": (dV ** 2).sum(axis=(1, 2, 3)).tolist(),
        "K_init_band": (K0[..., :64] ** 2 + K0[..., 64:] ** 2).sum(axis=(1, 2)).tolist(),
    }
    frac = band.sum(axis=0) / band.sum()
    fast = frac[:16].sum()   # wavelength < ~250 tokens
    mid = frac[16:32].sum()  # ~250 -- ~7k tokens
    slow = frac[32:].sum()   # effectively frozen at our scales
    print(f"{name:10s} dK energy: fast(<250tok)={fast:.3f} mid={mid:.3f} frozen={slow:.3f}")

(OUT / "bands.json").write_text(json.dumps(out))
print("wrote", OUT / "bands.json")
