"""Loading Llama-architecture models from HF safetensors checkpoints.

The module tree uses HF's names, so this is the generic walk with no key
translation.
"""
from __future__ import annotations

from pathlib import Path

import safetensors.flax as st
from tqdm import tqdm

from qwen_jax.loading import load_state_dict

from .config import LlamaConfig
from .model import LlamaForCausalLM


def load_llama_jax(model_path: str | Path, error_on_unused: bool = True) -> LlamaForCausalLM:
    """Load a Llama model from an HF model directory of safetensors."""
    model_path = Path(model_path)
    config = LlamaConfig.from_pretrained(model_path)

    state_dict = {}
    for filename in tqdm(sorted(model_path.glob("*.safetensors")), desc="Loading safetensors"):
        state_dict.update(st.load_file(filename))
    # Rotary inv_freq buffers are derived, not loaded; old checkpoints saved them.
    for key in [k for k in state_dict if k.endswith("rotary_emb.inv_freq")]:
        del state_dict[key]

    model = load_state_dict(LlamaForCausalLM(config), state_dict)
    if error_on_unused:
        assert not state_dict, f"Unused keys in state_dict: {sorted(state_dict)}"
    return model


__all__ = ["load_llama_jax"]
