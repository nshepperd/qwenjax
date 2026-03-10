import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

import torch
torch.cuda.memory.set_per_process_memory_fraction(0.2)