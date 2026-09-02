"""Cartridges: a trained KV prefix standing in for a corpus in context.

After arXiv 2506.06266. A cartridge is a `KVPrefix` whose keys and values are
parameters: `p` slots per layer, initialised from the real KV cache of the
first `p` tokens of the corpus and then optimised so that the model behaves,
downstream of the prefix, as if the whole corpus were in its context window.
The model itself is frozen. See `qwen_jax.selfstudy` for the data and
`qwen_jax.distill` for the objective.

This module is the parameter object only: what a cartridge is made of, how one
is initialised, serialised, and turned into the `KVPrefix` the model attends
to. Keys and values are kept in float32 as the optimiser's master copy and cast
to the cache dtype on the way into attention.

The first slot is frozen. It holds the KV of the first token of the sequence,
which functions as the attention sink; letting it train destabilises the run
(the paper's Appendix A reports accuracy collapsing when it is trainable).

Parameterization. The KV cache's scale is wildly nonuniform across layers
(per-layer key RMS spreads 8x over the stack, value RMS 238x, on Qwen3-VL-8B),
and Adam's per-coordinate step is scale-free only up to the *learning rate*,
which is one number for the whole cartridge. So the parameters are stored at
unit RMS per layer: `keys`/`values` are what the optimiser sees, and the
per-layer `key_scale`/`value_scale`, fixed at initialisation, are multiplied
back in by `prefix()`. At equal budget this trained a markedly better
cartridge than raw KV parameters (held-out KL 0.561 vs 0.624 under the
attention-MSE objective, 2000 steps), so it is the default; `unit_rms=False`
at construction keeps the scales at one, which is the raw parameterization.
`physical_keys`/`physical_values` are the KV the model attends to, for code
that inspects a cartridge rather than trains it.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import safetensors.flax as st
from jaxtyping import Array, Bool, Float, Int

from . import equinox_utils as eu
from .cache import KVPrefix


@dataclasses.dataclass(frozen=True)
class CartridgeMeta:
    """What a cartridge was made from. Static: it rides along under jit unchanged."""

    model: str = ""
    """Name of the base model, for a sanity check at load time."""
    init_tokens: int = 0
    """How many corpus tokens the initialisation ran over (== p)."""
    description: str = ""
    """The text prepended to the corpus in the system prompt, if any."""

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_json(cls, s: str) -> CartridgeMeta:
        data = json.loads(s)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class Cartridge(eqx.Module):
    """Trainable KV prefix. See the module docstring."""

    keys: Float[Array, "layers p kv_heads head_dim"]
    """The optimiser's view of the keys: `physical_keys / key_scale`."""
    values: Float[Array, "layers p kv_heads head_dim"]
    key_scale: Float[Array, "layers 1 1 1"]
    """Per-layer scale fixed at construction (the init KV's RMS, or ones)."""
    value_scale: Float[Array, "layers 1 1 1"]
    trainable: Bool[Array, "p"]
    meta: CartridgeMeta = eqx.field(static=True)
    # An array, not a static int: a static field is part of jit's cache key,
    # and a step counter that changes every step would recompile every step.
    steps: Int[Array, ""] = dataclasses.field(default_factory=lambda: jnp.array(0, jnp.int32))

    @property
    def length(self) -> int:
        return self.keys.shape[1]

    @property
    def num_layers(self) -> int:
        return self.keys.shape[0]

    @property
    def physical_keys(self) -> Float[Array, "layers p kv_heads head_dim"]:
        """The keys the model attends to, in float32."""
        return self.keys * self.key_scale

    @property
    def physical_values(self) -> Float[Array, "layers p kv_heads head_dim"]:
        return self.values * self.value_scale

    def prefix(self, dtype=jnp.bfloat16) -> KVPrefix:
        """The prefix the model attends to, in the cache dtype."""
        return KVPrefix(keys=self.physical_keys.astype(dtype),
                        values=self.physical_values.astype(dtype))

    # --- construction ------------------------------------------------------

    @classmethod
    def from_physical(
        cls,
        keys: Array,
        values: Array,
        *,
        trainable: Array,
        meta: CartridgeMeta,
        steps: Array | int = 0,
        unit_rms: bool = True,
    ) -> Cartridge:
        """Build from the KV the model attends to, choosing the parameterization."""
        keys = jnp.asarray(keys, dtype=jnp.float32)
        values = jnp.asarray(values, dtype=jnp.float32)
        if unit_rms:
            key_scale, value_scale = _layer_rms(keys), _layer_rms(values)
        else:
            key_scale = jnp.ones((keys.shape[0], 1, 1, 1), jnp.float32)
            value_scale = key_scale
        return cls(
            keys=keys / key_scale,
            values=values / value_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            trainable=jnp.asarray(trainable, dtype=jnp.bool),
            meta=meta,
            steps=jnp.asarray(steps, dtype=jnp.int32),
        )

    @classmethod
    def from_prefix(
        cls,
        prefix: KVPrefix,
        *,
        freeze_first: bool = True,
        unit_rms: bool = True,
        meta: CartridgeMeta | None = None,
    ) -> Cartridge:
        trainable = jnp.ones((prefix.length,), dtype=jnp.bool)
        if freeze_first:
            trainable = trainable.at[0].set(False)
        return cls.from_physical(
            prefix.keys, prefix.values, trainable=trainable, unit_rms=unit_rms,
            meta=meta or CartridgeMeta(init_tokens=prefix.length),
        )

    @classmethod
    def init_from_tokens(
        cls,
        model,
        token_ids: Int[Array, "p"] | np.ndarray,
        *,
        freeze_first: bool = True,
        unit_rms: bool = True,
        meta: CartridgeMeta | None = None,
    ) -> Cartridge:
        """Initialise from the model's own KV cache over `token_ids`.

        This is the initialisation the paper found necessary: a randomly
        initialised prefix trains badly, the KV of real tokens trains well.
        Equivalently, before any training the cartridge *is* in-context
        learning over its first `p` tokens, which is a useful baseline.
        """
        ids = jnp.asarray(token_ids, dtype=jnp.int32)[None, :]
        out = model(input_ids=ids, attention_mask=jnp.ones_like(ids), use_cache=True,
                    last_logit_only=True)
        assert out.cache is not None
        prefix = KVPrefix.from_cache(out.cache)
        if meta is None:
            meta = CartridgeMeta(init_tokens=int(ids.shape[1]))
        return cls.from_prefix(prefix, freeze_first=freeze_first, unit_rms=unit_rms, meta=meta)

    # --- training support -------------------------------------------------

    def params(self) -> tuple[Array, Array]:
        """The optimisable leaves, as a tuple the optimiser can hold state for."""
        return (self.keys, self.values)

    def with_params(self, params: tuple[Array, Array]) -> Cartridge:
        keys, values = params
        return eu.replace(self, keys=keys, values=values)

    def mask_grads(self, grads: tuple[Array, Array]) -> tuple[Array, Array]:
        """Zero the gradient on frozen slots.

        A zero gradient is a no-op for Adam-family optimisers (the moments
        stay zero there), so masking here rather than masking the update
        keeps the optimiser generic.
        """
        m = self.trainable[None, :, None, None].astype(grads[0].dtype)
        return (grads[0] * m, grads[1] * m)

    def advance(self, steps: int) -> Cartridge:
        return eu.replace(self, steps=self.steps + steps)

    # --- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """The parameters as the optimiser holds them, plus their scales, so a
        save/load round trip is bit-exact. A file without scale tensors (from
        before the reparameterization) holds the physical KV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        st.save_file(
            {"keys": self.keys, "values": self.values, "key_scale": self.key_scale,
             "value_scale": self.value_scale, "trainable": self.trainable,
             "steps": self.steps},
            str(path),
            metadata={"cartridge": self.meta.to_json()},
        )

    @classmethod
    def load(cls, path: str | Path) -> Cartridge:
        from safetensors import safe_open

        path = Path(path)
        tensors = st.load_file(str(path))
        with safe_open(str(path), framework="flax") as f:
            raw: dict[str, Any] = f.metadata() or {}
        meta = CartridgeMeta.from_json(raw.get("cartridge", "{}"))
        steps = jnp.asarray(tensors.get("steps", 0), dtype=jnp.int32)
        if "key_scale" not in tensors:
            # Legacy file: physical KV. Re-parameterize on the way in.
            return cls.from_physical(
                tensors["keys"], tensors["values"], trainable=tensors["trainable"],
                meta=meta, steps=steps,
            )
        return cls(
            keys=jnp.asarray(tensors["keys"], dtype=jnp.float32),
            values=jnp.asarray(tensors["values"], dtype=jnp.float32),
            key_scale=jnp.asarray(tensors["key_scale"], dtype=jnp.float32),
            value_scale=jnp.asarray(tensors["value_scale"], dtype=jnp.float32),
            trainable=jnp.asarray(tensors["trainable"], dtype=jnp.bool),
            meta=meta,
            steps=steps,
        )


def _layer_rms(x: Float[Array, "layers p kv_heads head_dim"]) -> Float[Array, "layers 1 1 1"]:
    """Per-layer root-mean-square over slots, heads and dims; one for an empty
    or all-zero layer so the division is a no-op there."""
    rms = jnp.sqrt(jnp.mean(x * x, axis=(1, 2, 3), keepdims=True))
    return jnp.where(rms > 0, rms, 1.0)


def compose(model, *cartridges: Cartridge, reposition: bool = True) -> KVPrefix:
    """One prefix from several cartridges, in order.

    With `reposition`, each cartridge's keys are re-rotated to the absolute
    positions they will occupy in the joined prefix, so the result is what the
    model would have cached had it seen the pieces back to back. Without it the
    pieces keep their training-time positions, which is what the paper does.
    Which works better is an empirical question; both are exact operations.
    """
    rotary = model.model.language_model.rotary_emb
    pieces = []
    offset = 0
    for cart in cartridges:
        prefix = cart.prefix(model.cache_dtype())
        if reposition and offset:
            prefix = prefix.shift(offset, rotary)
        pieces.append(prefix)
        offset += cart.length
    return KVPrefix.concat(*pieces)


__all__ = ["Cartridge", "CartridgeMeta", "compose"]
