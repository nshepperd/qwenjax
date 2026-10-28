from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

# XLA's GPU autotuner times candidate GEMM algorithms and keeps the fastest,
# so which one runs -- and its rounding -- changes from process to process.
# The bf16 consistency tests (padded vs unpadded, cached vs full) compare two
# differently-shaped programs at 0.01 JS and flipped between pass and fail
# across identical runs (test_padding_vision: 0.0124 or 0.085 at one position,
# or a pass). With it off the choice is fixed. An explicit
# xla_gpu_autotune_level in XLA_FLAGS still wins.
if "xla_gpu_autotune_level" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"
    ).strip()

import torch

torch.cuda.memory.set_per_process_memory_fraction(0.2)