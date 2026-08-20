"""Tests for loading Qwen3-VL from GGUF checkpoints.

The name translation is the part that can silently do the wrong thing: a
transposed or misfiled weight still loads, still has the right shape, and still
generates plausible-looking text. So the integration tests check the loaded
values against the safetensors checkpoint the GGUF was quantized from, tensor
by tensor, rather than checking only that the load completes.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import safetensors.flax as st
from gguf_jax import QuantizedArray

from qwen_jax.gguf import (
    UnmappedTensor,
    gguf_state_dict,
    load_qwen3_gguf,
    text_key,
    vision_key,
)
from qwen_jax.param import params_with_paths, path_to_key

HF_PATH = Path("/data/models/Qwen3-VL-8B-Instruct")
GGUF_DIR = Path("/data/models/Qwen3-VL-8B-Instruct-GGUF")
GGUF_PATH = GGUF_DIR / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH = GGUF_DIR / "mmproj-BF16.gguf"

DEEPSTACK = (8, 16, 24)

needs_weights = pytest.mark.skipif(
    not (GGUF_PATH.exists() and MMPROJ_PATH.exists() and HF_PATH.exists()),
    reason="local Qwen3-VL-8B-Instruct GGUF + safetensors checkpoints not present",
)


# =============================================================================
# Name translation
# =============================================================================


@pytest.mark.parametrize(
    ("gguf_name", "expected"),
    [
        ("token_embd.weight", "model.language_model.embed_tokens.weight"),
        ("output.weight", "lm_head.weight"),
        ("output_norm.weight", "model.language_model.norm.weight"),
        ("blk.7.attn_q.weight", "model.language_model.layers.7.self_attn.q_proj.weight"),
        ("blk.7.attn_output.weight", "model.language_model.layers.7.self_attn.o_proj.weight"),
        ("blk.12.ffn_gate.weight", "model.language_model.layers.12.mlp.gate_proj.weight"),
        ("blk.12.ffn_norm.weight", "model.language_model.layers.12.post_attention_layernorm.weight"),
        ("blk.0.attn_k_norm.weight", "model.language_model.layers.0.self_attn.k_norm.weight"),
    ],
)
def test_text_key(gguf_name, expected):
    assert text_key(gguf_name) == expected


@pytest.mark.parametrize(
    ("gguf_name", "expected"),
    [
        ("v.blk.3.attn_qkv.weight", "model.visual.blocks.3.attn.qkv.weight"),
        ("v.blk.3.attn_out.bias", "model.visual.blocks.3.attn.proj.bias"),
        ("v.blk.3.ln2.weight", "model.visual.blocks.3.norm2.weight"),
        ("v.blk.3.ffn_down.weight", "model.visual.blocks.3.mlp.linear_fc2.weight"),
        ("mm.0.weight", "model.visual.merger.linear_fc1.weight"),
        ("mm.2.bias", "model.visual.merger.linear_fc2.bias"),
        ("v.post_ln.weight", "model.visual.merger.norm.weight"),
        # The mmproj names deepstack mergers by vision block; the model holds
        # them in a list, so 8/16/24 must land on 0/1/2 in that order.
        ("v.deepstack.8.fc1.weight", "model.visual.deepstack_merger_list.0.linear_fc1.weight"),
        ("v.deepstack.16.norm.bias", "model.visual.deepstack_merger_list.1.norm.bias"),
        ("v.deepstack.24.fc2.weight", "model.visual.deepstack_merger_list.2.linear_fc2.weight"),
    ],
)
def test_vision_key(gguf_name, expected):
    assert vision_key(gguf_name, DEEPSTACK) == expected


@pytest.mark.parametrize(
    "gguf_name", ["blk.0.nonsense.weight", "token_embed.weight", "blk.x.attn_q.weight"]
)
def test_unmapped_text_tensor_raises(gguf_name):
    with pytest.raises(UnmappedTensor):
        text_key(gguf_name)


def test_unmapped_deepstack_block_raises():
    """A merger on a block the config does not list is a mismatch, not a skip."""
    with pytest.raises(UnmappedTensor):
        vision_key("v.deepstack.12.fc1.weight", DEEPSTACK)


# =============================================================================
# Loaded values vs the checkpoint the GGUF came from
# =============================================================================


@pytest.fixture(scope="module", autouse=True)
def on_cpu():
    """Run these on the host.

    They are about names and values, not kernels, and they need the whole model
    dequantized to float32 -- which is 32 GB, well past the slice of the GPU the
    conftest hands out.
    """
    import jax

    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.fixture(scope="module")
def reference_state_dict() -> dict[str, np.ndarray]:
    sd = {}
    for f in sorted(HF_PATH.glob("*.safetensors")):
        sd.update(st.load_file(f))
    return sd


@pytest.fixture(scope="module")
def loaded_state_dict() -> dict:
    return gguf_state_dict(GGUF_PATH, MMPROJ_PATH, DEEPSTACK, dtype=jnp.float32)


def _dense(v) -> np.ndarray:
    if isinstance(v, QuantizedArray):
        v = v.dequantize(jnp.float32)
    return np.asarray(v, dtype=np.float32)


@needs_weights
def test_every_tensor_is_accounted_for(reference_state_dict, loaded_state_dict):
    """No tensor goes unmapped in either direction."""
    assert set(loaded_state_dict) == set(reference_state_dict)


@needs_weights
def test_vision_tower_is_exact(reference_state_dict, loaded_state_dict):
    """The mmproj is BF16, so the vision weights should survive the round trip
    bit for bit -- including the patch embedding, whose conv3d kernel the mmproj
    splits across two tensors and which we stack back together."""
    for key, value in loaded_state_dict.items():
        if not key.startswith("model.visual"):
            continue
        ref = np.asarray(reference_state_dict[key], dtype=np.float32)
        got = _dense(value)
        assert got.shape == ref.shape, key
        np.testing.assert_array_equal(got, ref, err_msg=key)


@needs_weights
def test_quantized_weights_are_close(reference_state_dict, loaded_state_dict):
    """Every quantized text weight should be a recognisable version of the
    original. Quantization costs a few percent of relative error; a transposed
    or misfiled tensor costs essentially all of the cosine similarity, so the
    threshold only has to separate those two cases."""
    worst = []
    for key, value in loaded_state_dict.items():
        if not isinstance(value, QuantizedArray):
            continue
        ref = np.asarray(reference_state_dict[key], dtype=np.float32).ravel()
        got = _dense(value).ravel()
        assert got.shape == ref.shape, key
        cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
        worst.append((cos, key))
    assert worst, "no quantized tensors found"
    worst.sort()
    assert worst[0][0] > 0.99, f"worst cosine similarity: {worst[:3]}"


@needs_weights
def test_load_fills_every_param():
    model = load_qwen3_gguf(HF_PATH, GGUF_PATH, MMPROJ_PATH)
    unset = [path_to_key(p) for p, v in params_with_paths(model) if not v.is_set()]
    assert unset == []


@needs_weights
def test_load_without_mmproj_leaves_vision_unset():
    """Text-only work should not require the vision half to be present."""
    model = load_qwen3_gguf(HF_PATH, GGUF_PATH, None)
    unset = {path_to_key(p) for p, v in params_with_paths(model) if not v.is_set()}
    assert unset
    assert all(k.startswith("model.visual") for k in unset)
