"""Gated DeltaNet: the linear-attention token mixer of Qwen3.5.

Three of every four Qwen3.5 layers replace softmax attention with a gated
delta rule (Gated DeltaNet, from Qwen3-Next): a fixed-size per-head state
S (head_k_dim x head_v_dim) is decayed by a learned gate and updated with a
delta-rule correction each token, and the output is a read of that state.
Sequence mixing is a short depthwise causal conv plus the recurrence, so
decode carries a small recurrent state instead of a KV cache.

The math follows the reference torch implementations in
`transformers.models.qwen3_next.modeling_qwen3_next` exactly (the FLA-aligned
fallbacks): fp32 recurrence, l2-normalized q/k with eps 1e-6, q scaled by
dk**-0.5. The chunked form replaces the reference's in-chunk sequential
inversion with a unit-triangular solve, which computes the same matrix.

Head-order convention: value heads are in llama.cpp's tiled order -- v head h
pairs with k head (h % num_k_heads) -- so q/k expand with a tile, not a
repeat_interleave. GGUF weights load directly; HF weights are permuted by
`qwen35.loading`.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from einops import rearrange
from jaxtyping import Array, Bool, Float

from qwen_jax.linear import Linear
from qwen_jax.param import Param

from .cache import DeltaNetCacheLayer
from .config import Qwen35TextConfig


def l2norm(x: Array, eps: float = 1e-6) -> Array:
    """FLA-style l2 normalization (note: eps inside the sqrt, unsquared)."""
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def chunk_gated_delta_rule(
    query: Float[Array, "batch seq heads k_dim"],
    key: Float[Array, "batch seq heads k_dim"],
    value: Float[Array, "batch seq heads v_dim"],
    g: Float[Array, "batch seq heads"],
    beta: Float[Array, "batch seq heads"],
    initial_state: Float[Array, "batch heads k_dim v_dim"] | None = None,
    chunk_size: int = 64,
) -> tuple[Float[Array, "batch seq heads v_dim"], Float[Array, "batch heads k_dim v_dim"]]:
    """Chunked gated delta rule over a whole sequence (prefill).

    Returns (output, final_state). Everything runs in float32; the output is
    cast back to value's dtype, the state stays float32.
    """
    with jax.default_matmul_precision("highest"):
        return _chunk_gated_delta_rule(query, key, value, g, beta, initial_state, chunk_size)


def _chunk_gated_delta_rule(query, key, value, g, beta, initial_state, chunk_size):
    # The recurrence lives in float32 on purpose (mamba_ssm_dtype); letting the
    # GPU default these matmuls to tf32 would quietly take back most of that
    # precision, compounding over the sequence. Hence the "highest" scope above.
    initial_dtype = value.dtype
    query = l2norm(query.astype(jnp.float32))
    key = l2norm(key.astype(jnp.float32))
    value = value.astype(jnp.float32)
    beta = beta.astype(jnp.float32)
    g = g.astype(jnp.float32)

    batch, seq_len, num_heads, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size

    # (b, s, h, d) -> (b, h, n_chunks, c, d), zero-padded to whole chunks.
    # Padded positions have k = v = 0 (no state update, garbage output that is
    # sliced off) and g = 0 (no spurious decay past the last real token).
    def chunked(x):
        x = jnp.pad(x, [(0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 2))
        return rearrange(x, "b (n c) h ... -> b h n c ...", c=chunk_size)

    query = chunked(query) * (k_dim ** -0.5)
    key = chunked(key)
    value = chunked(value)
    k_beta = key * chunked(beta)[..., None]
    v_beta = value * chunked(beta)[..., None]
    g = jnp.cumsum(chunked(g), axis=-1)  # (b, h, n, c) log-decay within chunk

    # decay[i, j] = exp(g_i - g_j) for j <= i (0 above the diagonal).
    tril = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool))
    decay_mask = jnp.where(tril, jnp.exp(jnp.tril(g[..., :, None] - g[..., None, :])), 0.0)

    # In-chunk delta-rule propagator: the reference builds T (strictly lower)
    # then runs forward substitution turning it into (I - T)^{-1} - I; adding
    # I gives (I - T)^{-1}, which is what a unit-lower-triangular solve is.
    strict = jnp.tril(jnp.ones((chunk_size, chunk_size), dtype=bool), k=-1)
    T = jnp.where(strict, -(k_beta @ jnp.swapaxes(key, -1, -2)) * decay_mask, 0.0)
    eye = jnp.eye(chunk_size, dtype=T.dtype)
    attn = jsl.solve_triangular(eye - T, jnp.broadcast_to(eye, T.shape), lower=True, unit_diagonal=True)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * jnp.exp(g)[..., None])

    if initial_state is None:
        state = jnp.zeros((batch, num_heads, k_dim, v_dim), dtype=jnp.float32)
    else:
        state = initial_state.astype(jnp.float32)

    # Scan over chunks: (b, h, n, c, d) -> per-step (b, h, c, d).
    def step(state, inputs):
        q_i, k_i, v_i, kcd_i, g_i, decay_i = inputs
        attn_i = (q_i @ jnp.swapaxes(k_i, -1, -2)) * decay_i
        v_new = v_i - kcd_i @ state
        out_i = (q_i * jnp.exp(g_i)[..., None]) @ state + attn_i @ v_new
        g_last = g_i[..., -1]
        state = state * jnp.exp(g_last)[..., None, None] + (
            jnp.swapaxes(k_i * jnp.exp(g_last[..., None] - g_i)[..., None], -1, -2) @ v_new
        )
        return state, out_i

    scan_inputs = jax.tree.map(
        lambda x: jnp.moveaxis(x, 2, 0), (query, key, value, k_cumdecay, g, decay_mask)
    )
    state, out = jax.lax.scan(step, state, scan_inputs)

    out = rearrange(out, "n b h c v -> b (n c) h v")[:, :seq_len]
    return out.astype(initial_dtype), state


def recurrent_gated_delta_rule_step(
    query: Float[Array, "batch heads k_dim"],
    key: Float[Array, "batch heads k_dim"],
    value: Float[Array, "batch heads v_dim"],
    g: Float[Array, "batch heads"],
    beta: Float[Array, "batch heads"],
    state: Float[Array, "batch heads k_dim v_dim"],
) -> tuple[Float[Array, "batch heads v_dim"], Float[Array, "batch heads k_dim v_dim"]]:
    """One decode step of the gated delta rule. State in/out is float32."""
    with jax.default_matmul_precision("highest"):  # see _chunk_gated_delta_rule
        initial_dtype = value.dtype
        query = l2norm(query.astype(jnp.float32)) * (query.shape[-1] ** -0.5)
        key = l2norm(key.astype(jnp.float32))
        value = value.astype(jnp.float32)

        state = state * jnp.exp(g.astype(jnp.float32))[..., None, None]
        kv_mem = jnp.einsum("bhkv,bhk->bhv", state, key)
        delta = (value - kv_mem) * beta.astype(jnp.float32)[..., None]
        state = state + jnp.einsum("bhk,bhv->bhkv", key, delta)
        out = jnp.einsum("bhkv,bhk->bhv", state, query)
        return out.astype(initial_dtype), state


class GatedRMSNorm(eqx.Module):
    """Per-head RMSNorm followed by a SiLU gate (Qwen3NextRMSNormGated).

    Unlike the zero-centered text norms, this weight is stored as-is in both
    HF and GGUF checkpoints.
    """

    weight: Param
    variance_epsilon: float = eqx.field(static=True)

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = Param(hidden_size, array=jnp.ones((hidden_size,)))
        self.variance_epsilon = eps

    def __call__(self, hidden_states: Array, gate: Array) -> Array:
        input_dtype = hidden_states.dtype
        x = hidden_states.astype(jnp.float32)
        variance = jnp.square(x).mean(-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.variance_epsilon)
        x = self.weight() * x.astype(input_dtype)
        x = x * jax.nn.silu(gate.astype(jnp.float32))
        return x.astype(input_dtype)


class CausalConv1d(eqx.Module):
    """Depthwise causal conv over the sequence, matching nn.Conv1d(groups=dim).

    The weight is stored (channels, kernel) -- the GGUF layout, and HF's with
    the singleton middle axis squeezed. No bias (Qwen3.5 convs have none).
    """

    weight: Param
    channels: int = eqx.field(static=True)
    kernel_size: int = eqx.field(static=True)

    def __init__(self, channels: int, kernel_size: int):
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = Param(channels, kernel_size)

    def __call__(
        self,
        x: Float[Array, "batch seq channels"],
        context: Float[Array, "batch kernel_minus_1 channels"],
    ) -> tuple[Float[Array, "batch seq channels"], Float[Array, "batch kernel_minus_1 channels"]]:
        """Convolve x given `context` as left context (zeros at sequence start).

        Returns (silu(conv(x)), new_context) where new_context is the last
        kernel-1 inputs, ready to be cached for the next call.
        """
        seq_len = x.shape[1]
        w = self.weight().astype(x.dtype)  # (channels, kernel)
        x_full = jnp.concatenate([context.astype(x.dtype), x], axis=1)
        y = sum(
            x_full[:, j : j + seq_len] * w[None, None, :, j]
            for j in range(self.kernel_size)
        )
        new_context = x_full[:, -(self.kernel_size - 1):]
        return jax.nn.silu(y), new_context


class Qwen35GatedDeltaNet(eqx.Module):
    """The Gated DeltaNet mixer module (Qwen3.5's split-projection variant).

    `A` holds -exp(A_log) -- the GGUF convention -- so the per-token log-decay
    is g = A * softplus(a + dt_bias), computed in float32.
    """

    in_proj_qkv: Linear
    in_proj_z: Linear
    in_proj_b: Linear
    in_proj_a: Linear
    conv1d: CausalConv1d
    A: Param
    dt_bias: Param
    norm: GatedRMSNorm
    out_proj: Linear

    num_k_heads: int = eqx.field(static=True)
    num_v_heads: int = eqx.field(static=True)
    head_k_dim: int = eqx.field(static=True)
    head_v_dim: int = eqx.field(static=True)

    def __init__(self, config: Qwen35TextConfig):
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        key_dim = self.num_k_heads * self.head_k_dim
        value_dim = self.num_v_heads * self.head_v_dim

        self.in_proj_qkv = Linear(config.hidden_size, 2 * key_dim + value_dim, use_bias=False)
        self.in_proj_z = Linear(config.hidden_size, value_dim, use_bias=False)
        self.in_proj_b = Linear(config.hidden_size, self.num_v_heads, use_bias=False)
        self.in_proj_a = Linear(config.hidden_size, self.num_v_heads, use_bias=False)
        self.conv1d = CausalConv1d(2 * key_dim + value_dim, config.linear_conv_kernel_dim)
        self.A = Param(self.num_v_heads)
        self.dt_bias = Param(self.num_v_heads)
        self.norm = GatedRMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = Linear(value_dim, config.hidden_size, use_bias=False)

    @jax.remat
    def __call__(
        self,
        hidden_states: Float[Array, "batch seq hidden"],
        cache: DeltaNetCacheLayer | None = None,
        token_mask: Bool[Array, "batch seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq hidden"], DeltaNetCacheLayer | None]:
        """Forward pass.

        Args:
            hidden_states: Input (batch, seq, hidden), already layernormed.
            cache: Recurrent state to continue from (and to update).
            token_mask: Validity of *these* positions (batch, seq). Padded
                positions are zeroed before the projections, as in HF; with
                left-padding the zero state decays to zero, so padding is
                exact there.

        Returns:
            (output, new_cache) tuple.
        """
        batch, seq_len, _ = hidden_states.shape
        if token_mask is not None:
            hidden_states = hidden_states * token_mask[..., None].astype(hidden_states.dtype)

        qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        conv_context = (
            cache.conv_state
            if cache is not None
            else jnp.zeros(
                (batch, self.conv1d.kernel_size - 1, self.conv1d.channels), qkv.dtype
            )
        )
        qkv, new_conv_state = self.conv1d(qkv, conv_context)

        key_dim = self.num_k_heads * self.head_k_dim
        query, key, value = jnp.split(qkv, [key_dim, 2 * key_dim], axis=-1)
        query = query.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, seq_len, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)

        beta = jax.nn.sigmoid(b.astype(jnp.float32))
        g = self.A().astype(jnp.float32) * jax.nn.softplus(
            a.astype(jnp.float32) + self.dt_bias().astype(jnp.float32)
        )

        # Tiled head expansion: v head h reads k head (h % num_k_heads).
        ratio = self.num_v_heads // self.num_k_heads
        if ratio > 1:
            query = jnp.tile(query, (1, 1, ratio, 1))
            key = jnp.tile(key, (1, 1, ratio, 1))

        if cache is not None and seq_len == 1:
            core_out, new_state = recurrent_gated_delta_rule_step(
                query[:, 0], key[:, 0], value[:, 0], g[:, 0], beta[:, 0],
                cache.recurrent_state,
            )
            core_out = core_out[:, None]
        else:
            core_out, new_state = chunk_gated_delta_rule(
                query, key, value, g, beta,
                initial_state=cache.recurrent_state if cache is not None else None,
            )

        z = z.reshape(batch, seq_len, self.num_v_heads, self.head_v_dim)
        core_out = self.norm(core_out, z)
        core_out = core_out.reshape(batch, seq_len, -1)

        output = self.out_proj(core_out)
        new_cache = None
        if cache is not None:
            new_cache = DeltaNetCacheLayer(
                conv_state=new_conv_state.astype(cache.conv_state.dtype),
                recurrent_state=new_state,
            )
        return output, new_cache


__all__ = [
    "CausalConv1d",
    "GatedRMSNorm",
    "Qwen35GatedDeltaNet",
    "chunk_gated_delta_rule",
    "l2norm",
    "recurrent_gated_delta_rule_step",
]
