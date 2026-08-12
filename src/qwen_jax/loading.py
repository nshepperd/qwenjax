from __future__ import annotations

import re
from pathlib import Path

import equinox as eqx
import jax
import safetensors.flax as st
from tqdm import tqdm

from qwen_jax.config import Qwen3VLConfig
from qwen_jax.model import Qwen3VLForConditionalGeneration

from . import equinox_utils as eu
from .linear import Linear
from .linear4bit import Linear4bit


def path_to_str(path: jax.tree_util.KeyPath) -> str:
    return jax.tree_util.keystr(path, simple=True, separator=".")


def load_qwen3_jax(model_path: str | Path, error_on_unused: bool = True) -> Qwen3VLForConditionalGeneration:
    model_path = Path(model_path)
    model_conf = Qwen3VLConfig.from_pretrained(pretrained_model_name_or_path=model_path)

    state_dict = {}
    for filename in tqdm(model_path.glob("*.safetensors"), desc="Loading safetensors"):
        state_dict.update(st.load_file(filename))

    quantized_layers = set()
    RE_QUANT = re.compile(r"(.*)\.weight.quant_state.bitsandbytes__.*")
    for key in state_dict:
        m = RE_QUANT.match(key)
        if m:
            quantized_layers.add(m.group(1))

    model = Qwen3VLForConditionalGeneration(model_conf)

    def visit(path: jax.tree_util.KeyPath, module: eqx.Module) -> eqx.Module:
        str_path = path_to_str(path)
        if str_path in quantized_layers:
            if isinstance(module, Linear):
                return Linear4bit(module.in_features, module.out_features, use_bias=module.use_bias)
        return module
    model = eu.mapmod_with_path(visit, model)
    model = model.load_state_dict(state_dict, prefix='')
    if error_on_unused:
        assert len(state_dict) == 0, f"Unused keys in state_dict: {list(state_dict.keys())}"
    return model
