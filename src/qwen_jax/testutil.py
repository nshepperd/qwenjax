from typing import TypeVar
from typing import ParamSpec
from typing import Any, Protocol, runtime_checkable
import jax
import torch
import equinox as eqx
from collections import OrderedDict
import sys
import jax.numpy as jnp

import qwen_jax.equinox_utils as eu

def to_jax(x, device=None):
    if isinstance(x, torch.Tensor):
        return jax.device_put(jax.dlpack.from_dlpack(x), device=device if device is not None else jax.devices()[0])
    return x

# ============================================================================
# PyTorch output extraction using hooks
# ============================================================================

def extract_pytorch_outputs(model: torch.nn.Module, *args, **kwargs) -> tuple[Any, OrderedDict[str, jax.Array]]:
    """Extract outputs from every layer using forward hooks."""
    outputs = OrderedDict()
    hooks = []

    def make_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                outputs[name] = to_jax(output)
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    outputs[name] = to_jax(output[0])
        return hook

    for name, module in model.named_modules():
        if name:
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)

    with torch.no_grad():
        out = model(*args, **kwargs)

    for hook in hooks:
        hook.remove()

    return out, outputs

cpu_device = jax.devices('cpu')[0]

def extract_pytorch_outputs_and_inputs(model: torch.nn.Module, *args, **kwargs) -> tuple[Any, OrderedDict[str, jax.Array], OrderedDict[str, jax.Array]]:
    """Extract outputs from every layer using forward hooks."""
    outputs = OrderedDict()
    inputs = OrderedDict()
    hooks = []

    def make_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                outputs[name] = to_jax(output, device=cpu_device)
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    outputs[name] = to_jax(output[0], device=cpu_device)
            if isinstance(input, tuple) and len(input) == 1:
                if isinstance(input[0], torch.Tensor):
                    inputs[name] = to_jax(input[0], device=cpu_device)
        return hook

    for name, module in model.named_modules():
        if name:
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)

    with torch.no_grad():
        out = model(*args, **kwargs)

    for hook in hooks:
        hook.remove()

    return out, outputs, inputs


# ============================================================================
# JAX/Equinox output extraction using module wrapping
# ============================================================================

class WrappedModule(eqx.Module):
    """Wrapper that captures output of the wrapped module's __call__."""
    inner: Any
    path: str = eqx.field(static=True)

    def __getattr__(self, name: str) -> Any:
        """Masquerade as the inner module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        result = self.inner(*args, **kwargs)
        if isinstance(result, jax.Array):
            return eu.label(result, name=self.path)
        return result


def path_to_string(path: jax.tree_util.KeyPath) -> str:
    """Convert JAX path tuple to dot-separated string."""
    parts = []
    for p in path:
        if hasattr(p, 'name'):
            parts.append(str(p.name))
        elif hasattr(p, 'idx'):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return '.'.join(parts)


def wrap_modules_for_capture(model: eqx.Module) -> eqx.Module:
    def wrap(path, mod: eqx.Module):
        path_str = path_to_string(path)
        return WrappedModule(mod, path_str)
    return eu.mapmod_with_path(wrap, model)

P = ParamSpec('P')
R = TypeVar('R')
@runtime_checkable
class CallableModule(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        ...

def extract_jax_outputs_wrapped(model: CallableModule[P, R], *args: P.args, **kwargs: P.kwargs) -> tuple[R, OrderedDict[str, jax.Array]]:
    """Extract outputs by wrapping all modules."""
    # Wrap the model
    wrapped_model = wrap_modules_for_capture(model)

    if not isinstance(wrapped_model, CallableModule):
        raise TypeError("Model is not callable.")

    # Run forward pass
    if sys.gettrace() is not None:
        # If debugging, run it first without jit so we can debug with concrete arrays
        wrapped_model(*args, **kwargs)
    def fwd(wrapped_model, *args, **kwargs):
        return eu.collect(wrapped_model)(*args, **kwargs)
    out, capture = jax.jit(fwd)(wrapped_model, *args, **kwargs)

    return out, capture
