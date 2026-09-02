from __future__ import annotations

import re
from pathlib import Path
from typing import TypeVar

import equinox as eqx
import jax
import safetensors.flax as st
from tqdm import tqdm

from qwen_jax.config import Qwen3VLConfig
from qwen_jax.model import Qwen3VLForConditionalGeneration

from . import equinox_utils as eu
from .linear import Linear
from .linear4bit import Linear4bit
from .param import AbstractParam, path_to_key

T = TypeVar("T", bound=eqx.Module)


def load_state_dict(module: T, state_dict: dict[str, jax.Array]) -> T:
    """Fill every Param in `module` from `state_dict`.

    A Param's path in the tree is its key in the state dict, so no module needs
    a load method of its own -- containers are traversed, Params are loaded.
    Keys are consumed as they are used, so whatever remains in `state_dict`
    afterwards is exactly what went unclaimed.
    """

    def visit(path: jax.tree_util.KeyPath, module: eqx.Module) -> eqx.Module:
        if isinstance(module, AbstractParam):
            return module.load_state_dict(state_dict, path)
        return module

    return eu.mapmod_with_path(visit, module)


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
        str_path = path_to_key(path)
        if str_path in quantized_layers:
            if isinstance(module, Linear):
                return Linear4bit(module.in_features, module.out_features, use_bias=module.use_bias)
        return module
    model = eu.mapmod_with_path(visit, model)
    model = load_state_dict(model, state_dict)
    if error_on_unused:
        assert len(state_dict) == 0, f"Unused keys in state_dict: {list(state_dict.keys())}"
    return model
