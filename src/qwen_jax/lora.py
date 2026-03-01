from .linear4bit import Linear4bit
from pydantic.main import BaseModel
from dataclasses import field
import json
from pathlib import Path
from typing import Sequence, TypeVar
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from jax.tree_util import KeyPath
import safetensors.flax as st
import re

from .linear import Linear

from . import equinox_utils as eu


class LoraConfig(BaseModel):
    base_model_name_or_path: str
    peft_type: str = "LORA"
    r: int
    lora_alpha: float
    lora_dropout: float = 0.0
    target_modules: list[str]

class LoraLinear(eqx.Module):
    """LoRA-augmented Linear layer."""
    base: eqx.Module
    lora_A: Linear
    lora_B: Linear
    alpha: Float[Array, ""] | Float[Array, "r"]
    r: int = field(metadata=dict(static=True))
    dtype: jnp.dtype = field(default=jnp.dtype('float32'), metadata=dict(static=True))

    @jax.remat
    def __call__(self, x: Array) -> Array:
        base_output: Array = self.base(x)  # type: ignore
        lora_output: Array = self.lora_B(
            self.lora_A(x.astype(self.dtype)) * (self.alpha)
        )
        return lora_output.astype(base_output.dtype) + base_output


class LoraAdapterItem(eqx.Module):
    lora_A: Float[Array, "r in_dim"]
    lora_B: Float[Array, "out_dim r"]

class LoraAdapter(eqx.Module):
    weights: dict[str, LoraAdapterItem]
    alpha: Float[Array, ""]
    r: int = field(metadata=dict(static=True))

    def save_pretrained(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {
            "peft_type": "LORA",
            "r": self.r,
            "lora_alpha": float(self.alpha),
            "base_model_name_or_path": "",
        }
        with open(path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
        state_dict = {}
        for name, item in self.weights.items():
            state_dict[f"{name}.lora_A.weight"] = item.lora_A
            state_dict[f"{name}.lora_B.weight"] = item.lora_B
        st.save_file(state_dict, str(path / "adapter_model.safetensors"))

T = TypeVar("T", bound=eqx.Module)
def apply_adapters(module: T, adapters: Sequence[LoraAdapter]) -> T:
    """Apply LoRA adapters to a model."""
    def apply_mod(path: KeyPath, module: eqx.Module):
        path_str = jax.tree_util.keystr(path, simple=True, separator='.')
        matching = [(adapter.r, adapter.alpha, adapter.weights[path_str]) for adapter in adapters if path_str in adapter.weights]
        if not matching:
            return module
        merged_alpha_on_r = jnp.concatenate([jnp.broadcast_to(a/r, (r,)) for r, a, w in matching], axis=0)
        lora_A = jnp.concatenate([w.lora_A for _, _, w in matching], axis=0)
        lora_B = jnp.concatenate([w.lora_B for _, _, w in matching], axis=1)
        r_total = sum(r for r, _, _ in matching)
        return LoraLinear(
            base=module,
            lora_A=eu.new(Linear,
                in_features=lora_A.shape[1],
                out_features=r_total,
                weight=lora_A,
                bias=None,
                use_bias=False,
            ),
            lora_B=eu.new(Linear,
                in_features=r_total,
                out_features=lora_B.shape[0],
                weight=lora_B,
                bias=None,
                use_bias=False,
            ),
            alpha=merged_alpha_on_r,
            r=r_total,
            dtype=lora_A.dtype,
        )
    return eu.mapmod_with_path(apply_mod, module)

def load_lora(path: Path | str, remapping: dict[str, str] | None = None):
    path = Path(path)
    with open(path / "adapter_config.json", "r") as f:
        config = json.load(f)
    assert config['peft_type'] == 'LORA', f"Unsupported peft_type: {config['peft_type']}"
    assert config.get('lora_bias', False) is False, "LoRA bias not supported"
    alpha: Array = jnp.array(config.get("lora_alpha", 1.0), dtype=jnp.float32)
    r: int = config['r']
    if remapping is None:
        remapping = {}
    def unmapname(name: str) -> str:
        for k, v in remapping.items():
            if name.startswith(v):
                return k + name[len(v):]
        return name
    state_dict = st.load_file(str(path / "adapter_model.safetensors"))
    weights = {}
    for name in list(state_dict.keys()):
        m = re.match(r'^(.+)\.lora_A\.weight$', name)
        if m:
            base_name = m.group(1)
            model_name = unmapname(base_name)
            weights[model_name] = LoraAdapterItem(
                lora_A=state_dict.pop(name),
                lora_B=state_dict.pop(f"{base_name}.lora_B.weight"),
            )
    assert not state_dict, f"Unprocessed keys in state_dict: {list(state_dict.keys())}"
    return LoraAdapter(weights=weights, alpha=alpha, r=r)

def rng_iter(key: PRNGKeyArray):
    while True:
        key, subkey = jax.random.split(key)
        yield subkey

def new_lora(model: eqx.Module, config: LoraConfig, key: PRNGKeyArray):
    """Create a new LoRA adapter with random weights."""
    weights = {}
    rng = rng_iter(key)
    def visit(path: KeyPath, module: eqx.Module):
        path_str = jax.tree_util.keystr(path, simple=True, separator='.')
        if any(path_str.endswith(tm) for tm in config.target_modules):
            if isinstance(module, (Linear, Linear4bit)):
                in_features = module.in_features  # type: ignore
                out_features = module.out_features  # type: ignore
                lora_A = jax.random.normal(next(rng), (config.r, in_features), dtype=jnp.float32) * (in_features**-0.5)
                lora_B = jnp.zeros((out_features, config.r), dtype=jnp.float32)
                weights[path_str] = LoraAdapterItem(
                    lora_A=lora_A,
                    lora_B=lora_B,
                )
    eu.mapmod_with_path(visit, model)
    return LoraAdapter(weights=weights, alpha=jnp.array(config.lora_alpha, dtype=jnp.float32), r=config.r)
