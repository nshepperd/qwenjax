"""Tests for Qwen3.5 (hybrid DeltaNet / gated-attention decoder).

Parity is checked against the transformers reference implementation on tiny
randomly-initialized configs in float32 -- the real checkpoints only exist
quantized here, so the reference modules are the ground truth for the math.
The HF->qwen_jax weight conversion under test is the same code the safetensors
loader uses (head reorder, A_log, zero-centered norms), so these tests cover
the loader's transforms too.
"""
from __future__ import annotations

import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_next.modeling_qwen3_next import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from qwen_jax.equinox_utils import mapmod_with_path
from qwen_jax.loading import load_state_dict
from qwen_jax.param import AbstractParam, path_to_key
from qwen_jax.qwen35 import (
    Qwen35Attention,
    Qwen35Cache,
    Qwen35Config,
    Qwen35ForConditionalGeneration,
    Qwen35GatedDeltaNet,
    Qwen35TextConfig,
    chunk_gated_delta_rule,
    convert_hf_state_dict,
    recurrent_gated_delta_rule_step,
)
from qwen_jax.rope import Qwen3VLTextRotaryEmbedding


@pytest.fixture(autouse=True)
def xla_attention(monkeypatch):
    """Use the XLA attention path: exact float32, no fp16 downcast."""
    monkeypatch.setenv("QWEN_JAX_ATTENTION", "xla")


@pytest.fixture(autouse=True)
def full_f32_matmuls():
    """Match torch: no tf32 in the float32 parity comparisons on GPU."""
    with jax.default_matmul_precision("highest"):
        yield


TINY_TEXT_KWARGS = dict(
    vocab_size=512,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    layer_types=(
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ),
    full_attention_interval=4,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
)
TINY_ROPE = dict(
    rope_type="default",
    rope_theta=10000000.0,
    partial_rotary_factor=0.25,
    mrope_interleaved=True,
    mrope_section=[2, 1, 1],  # sums to rotary_dim / 2 = 4
)


def hf_text_config() -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        **{k: list(v) if isinstance(v, tuple) else v for k, v in TINY_TEXT_KWARGS.items()},
        rope_parameters=dict(TINY_ROPE),
    )
    config._attn_implementation = "eager"
    return config


def jax_text_config() -> Qwen35TextConfig:
    return Qwen35TextConfig(**TINY_TEXT_KWARGS, rope_parameters=dict(TINY_ROPE))


def jax_full_config() -> Qwen35Config:
    return Qwen35Config(
        text_config=jax_text_config(),
        vision_config=dict(
            depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
            out_hidden_size=64, deepstack_visual_indexes=(),
        ),
    )


def to_jax(x: torch.Tensor) -> jax.Array:
    return jnp.asarray(x.detach().cpu().numpy())


def load_converted(module, hf_module: torch.nn.Module, prefix: str):
    """Load an HF submodule's weights into a qwen_jax module.

    The HF state dict is keyed as if the submodule sat at `prefix` inside the
    full model, run through the real conversion, then re-relativized.
    """
    hf_sd = {
        f"{prefix}.{k}": to_jax(v) for k, v in hf_module.state_dict().items()
    }
    converted = convert_hf_state_dict(hf_sd, jax_text_config())
    out_prefix = prefix.replace("model.", "model.language_model.", 1) + "."
    relative = {}
    for key, value in converted.items():
        assert key.startswith(out_prefix), key
        relative[key.removeprefix(out_prefix)] = value
    module = load_state_dict(module, relative)
    assert not relative, f"Unused keys: {sorted(relative)}"
    return module


# =============================================================================
# Delta rule kernels
# =============================================================================


def random_delta_inputs(key, batch=2, seq=37, heads=4, dk=16, dv=16):
    ks = jax.random.split(key, 5)
    q = jax.random.normal(ks[0], (batch, seq, heads, dk), dtype=jnp.float32)
    k = jax.random.normal(ks[1], (batch, seq, heads, dk), dtype=jnp.float32)
    v = jax.random.normal(ks[2], (batch, seq, heads, dv), dtype=jnp.float32)
    # Realistic ranges: beta in (0, 1), g moderately negative log-decay.
    beta = jax.nn.sigmoid(jax.random.normal(ks[3], (batch, seq, heads)))
    g = -jax.nn.softplus(jax.random.normal(ks[4], (batch, seq, heads)))
    return q, k, v, g, beta


@pytest.mark.parametrize("initial", [False, True])
def test_chunk_delta_rule_vs_torch(initial):
    q, k, v, g, beta = random_delta_inputs(jax.random.key(0))
    state0 = None
    state0_pt = None
    if initial:
        state0 = jax.random.normal(jax.random.key(9), (2, 4, 16, 16)) * 0.1
        state0_pt = torch.from_numpy(np.array(state0))

    out_pt, state_pt = torch_chunk_gated_delta_rule(
        *(torch.from_numpy(np.array(x)) for x in (q, k, v, g, beta)),
        chunk_size=16,
        initial_state=state0_pt,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out, state = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=state0, chunk_size=16)
    np.testing.assert_allclose(out, out_pt.numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(state, state_pt.numpy(), atol=1e-4, rtol=1e-4)


def test_recurrent_step_vs_torch():
    q, k, v, g, beta = random_delta_inputs(jax.random.key(1), seq=1)
    state0 = jax.random.normal(jax.random.key(2), (2, 4, 16, 16)) * 0.1

    out_pt, state_pt = torch_recurrent_gated_delta_rule(
        *(torch.from_numpy(np.array(x)) for x in (q, k, v, g, beta)),
        initial_state=torch.from_numpy(np.array(state0)),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    out, state = recurrent_gated_delta_rule_step(
        q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], state0
    )
    np.testing.assert_allclose(out[:, None], out_pt.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(state, state_pt.numpy(), atol=1e-5, rtol=1e-5)


def test_chunk_matches_recurrent():
    """The chunked prefill and the token-by-token recurrence are the same map."""
    q, k, v, g, beta = random_delta_inputs(jax.random.key(3), seq=21)
    out_chunk, state_chunk = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=8)

    state = jnp.zeros((2, 4, 16, 16), dtype=jnp.float32)
    outs = []
    for t in range(q.shape[1]):
        out_t, state = recurrent_gated_delta_rule_step(
            q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state
        )
        outs.append(out_t)
    out_rec = jnp.stack(outs, axis=1)
    np.testing.assert_allclose(out_chunk, out_rec, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(state_chunk, state, atol=1e-4, rtol=1e-4)


# =============================================================================
# Modules vs HF reference
# =============================================================================


@torch.no_grad()
def test_deltanet_module_vs_hf():
    torch.manual_seed(0)
    hf = Qwen3_5GatedDeltaNet(hf_text_config(), layer_idx=0).float().eval()
    ours = load_converted(
        Qwen35GatedDeltaNet(jax_text_config()), hf, "model.layers.0.linear_attn"
    )

    x = torch.randn(2, 33, 64)
    out_pt = hf(x)
    out, _ = ours(to_jax(x))
    np.testing.assert_allclose(np.asarray(out), out_pt.numpy(), atol=2e-4, rtol=1e-3)


@torch.no_grad()
def test_attention_module_vs_hf():
    torch.manual_seed(1)
    config = hf_text_config()
    hf = Qwen3_5Attention(config, layer_idx=3).float().eval()
    ours = load_converted(
        Qwen35Attention(jax_text_config()), hf, "model.layers.3.self_attn"
    )

    batch, seq = 2, 19
    x = torch.randn(batch, seq, 64)

    rotary = Qwen3VLTextRotaryEmbedding(jax_text_config())
    position_ids = jnp.broadcast_to(jnp.arange(seq)[None, None, :], (3, batch, seq))
    cos, sin = rotary(position_ids)

    causal = torch.full((seq, seq), float("-inf")).triu(1)[None, None]
    out_pt, _ = hf(
        x,
        position_embeddings=(
            torch.from_numpy(np.array(cos)), torch.from_numpy(np.array(sin))
        ),
        attention_mask=causal,
    )

    out, _ = ours(
        to_jax(x),
        position_embeddings=(cos, sin),
        kv_mask=jnp.ones((batch, seq), dtype=bool),
    )
    np.testing.assert_allclose(np.asarray(out), out_pt.numpy(), atol=2e-4, rtol=1e-3)


# =============================================================================
# Full model vs HF, and cache consistency
# =============================================================================


@pytest.fixture(scope="module")
def tiny_models():
    torch.manual_seed(2)
    hf = Qwen3_5ForCausalLM(hf_text_config()).float().eval()

    config = jax_full_config()
    model = Qwen35ForConditionalGeneration(config)
    hf_sd = {k: to_jax(v) for k, v in hf.state_dict().items()}
    state_dict = convert_hf_state_dict(hf_sd, config.text_config)

    # The text-only HF model has nothing for the vision tower; leave those
    # params unset rather than fail the walk over them.
    def fill(path, module):
        if not isinstance(module, AbstractParam):
            return module
        if path_to_key(path) not in state_dict:
            return module
        return module.load_state_dict(state_dict, path)

    model = mapmod_with_path(fill, model)
    assert not state_dict, f"Unused keys: {sorted(state_dict)}"
    return hf, model


@torch.no_grad()
def test_model_vs_hf(tiny_models):
    hf, model = tiny_models
    torch.manual_seed(3)
    input_ids = torch.randint(0, 512, (2, 41))

    out_pt = hf(input_ids=input_ids).logits
    out = model(input_ids=to_jax(input_ids)).logits
    np.testing.assert_allclose(np.asarray(out), out_pt.numpy(), atol=2e-3, rtol=1e-2)


@torch.no_grad()
def test_model_vs_hf_padded(tiny_models):
    """Left padding: parity with HF under an explicit shared position basis.

    Batch must be > 1: HF's apply_mask_to_padding_states skips masking the
    DeltaNet inputs for single-item batches, feeding pad embeddings through
    the recurrence, and we deliberately don't reproduce that quirk.
    """
    hf, model = tiny_models
    torch.manual_seed(4)
    pad, seq = 7, 29
    ids = torch.randint(0, 512, (2, pad + seq))
    mask = torch.ones(2, pad + seq, dtype=torch.long)
    mask[0, :pad] = 0  # row 0 left-padded, row 1 full

    # HF's text-only path numbers positions by index regardless of padding;
    # pass the same explicit positions to both.
    positions = torch.arange(pad + seq)[None].expand(2, -1)
    out_pt = hf(input_ids=ids, attention_mask=mask, position_ids=positions).logits

    position_ids = jnp.broadcast_to(jnp.arange(pad + seq)[None, None, :], (3, 2, pad + seq))
    out = model(
        input_ids=to_jax(ids),
        attention_mask=to_jax(mask),
        position_ids=position_ids,
    ).logits
    np.testing.assert_allclose(
        np.asarray(out)[0, pad:], out_pt.numpy()[0, pad:], atol=2e-3, rtol=1e-2
    )
    np.testing.assert_allclose(
        np.asarray(out)[1], out_pt.numpy()[1], atol=2e-3, rtol=1e-2
    )


def test_cache_consistency(tiny_models):
    """Prefill+chunk+decode through the hybrid cache matches one full pass."""
    _, model = tiny_models
    input_ids = jax.random.randint(jax.random.key(5), (2, 40), 0, 512)

    full = model(input_ids=input_ids).logits

    cache = Qwen35Cache.create(
        model.config.text_config, batch_size=2, max_seq_len=40, dtype=jnp.float32
    )
    out1 = model(input_ids=input_ids[:, :25], cache=cache)
    out2 = model(input_ids=input_ids[:, 25:33], cache=out1.cache)  # chunked continuation
    logits = [out1.logits, out2.logits]
    cache = out2.cache
    for t in range(33, 40):  # single-token decode
        out = model(input_ids=input_ids[:, t : t + 1], cache=cache)
        cache = out.cache
        logits.append(out.logits)
    stitched = jnp.concatenate(logits, axis=1)
    np.testing.assert_allclose(np.asarray(stitched), np.asarray(full), atol=1e-3, rtol=1e-2)


def test_generate(tiny_models):
    """generate() runs end to end and its tokens re-score consistently."""
    _, model = tiny_models
    input_ids = jax.random.randint(jax.random.key(6), (1, 9), 0, 512)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=12,
        temperature=0.0,
        key=jax.random.key(0),
        progress_bar=False,
        return_logits=True,
    )
    assert output.tokens.shape == (1, 21)

    full = model(input_ids=output.tokens[:, :-1]).logits
    np.testing.assert_allclose(
        np.asarray(output.logits),
        np.asarray(full[:, 8:]),
        atol=1e-3,
        rtol=1e-2,
    )


# =============================================================================
# GGUF end-to-end (needs the real files; skipped when absent)
# =============================================================================

QWEN35_GGUF = Path(os.environ.get("QWEN35_GGUF", "/data/models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf"))
QWEN35_CONFIG = Path(os.environ.get("QWEN35_CONFIG", "/data/models/Qwen3.5-9B"))


@pytest.mark.skipif(not QWEN35_GGUF.exists(), reason="Qwen3.5 GGUF not available")
def test_gguf_generate():
    from transformers import AutoTokenizer

    from qwen_jax.qwen35 import load_qwen35_gguf

    model = load_qwen35_gguf(QWEN35_CONFIG, QWEN35_GGUF)
    tokenizer = AutoTokenizer.from_pretrained(QWEN35_CONFIG)
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="np", return_dict=True
    )["input_ids"]
    # A thinking model spends tokens reasoning before it answers; give it room.
    output = model.generate(
        input_ids=jnp.asarray(input_ids),
        max_new_tokens=96,
        temperature=0.0,
        key=jax.random.key(0),
        progress_bar=False,
    )
    text = tokenizer.decode(output.tokens[0, input_ids.shape[1]:])
    print(text)
    assert "Paris" in text
