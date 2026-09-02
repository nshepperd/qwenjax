from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

import jax

# Same persistent compilation cache as scripts/cartridge.py (per JAX version,
# since versions sharing a cache upset each other): the model is a jit
# argument, so the test executables are reused across runs.
jax.config.update("jax_compilation_cache_dir", os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.expanduser(f"~/.cache/jax-compilation/{jax.__version__}")))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

import torch

torch.cuda.memory.set_per_process_memory_fraction(0.2)