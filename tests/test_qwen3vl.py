"""Tests for Qwen3-VL model (vision + text)."""

from typing import Any
from qwen_jax.utils.pjit import pjit
import pytest
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jaxtyping import Array
from PIL import Image
from pydantic import BaseModel
from pydantic.config import ConfigDict
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as HFModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

import qwen_jax.equinox_utils as eu
import qwen_jax.testutil as tu
from qwen_jax.cache import KVCache
from qwen_jax.equinox_utils import mapmod_with_path
from qwen_jax.testutil import to_jax
from qwen_jax.model import Qwen3VLForConditionalGeneration

# =============================================================================
# Functions and fixtures
# =============================================================================

# Path to local Qwen3-VL model
QWEN3VL_MODEL_PATH = "/data/models/Qwen3-VL-2B-Instruct"


@pytest.fixture
def prng_key():
    """Get a PRNG key for JAX initialization."""
    return jax.random.PRNGKey(42)

@pytest.fixture(scope='module')
def hf_model() -> HFModel:
    hf_model = HFModel.from_pretrained(QWEN3VL_MODEL_PATH, torch_dtype=torch.float32)
    return hf_model

@pytest.fixture(scope='module')
def jax_model():
    from qwen_jax.loading import load_qwen3_jax
    return load_qwen3_jax(QWEN3VL_MODEL_PATH)

@pytest.fixture(scope='module')
def processor() -> Qwen3VLProcessor:
    return Qwen3VLProcessor.from_pretrained(QWEN3VL_MODEL_PATH)

def convert_inputs(inputs) -> dict:
    return {k: to_jax(v) for k, v in dict(inputs).items()}

def to_torch(x: jax.Array) -> torch.Tensor:
    return torch.from_numpy(np.array(x))

def jitbound(f: eqx._module._prebuilt.BoundMethod):
    return eqx._module._prebuilt.BoundMethod(jax.jit(f.__func__), f.__self__)

def _H(logits):
    return -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1)
@pjit(device=jax.devices('cpu')[0])
def jsdiv(logits1, logits2):
    logits1 = jnp.float32(logits1)
    logits2 = jnp.float32(logits2)
    M = jnp.log(0.5 * (jax.nn.softmax(logits1) + jax.nn.softmax(logits2)))
    return _H(M) - 0.5 * (_H(logits1) + _H(logits2))

@pytest.fixture
def large_image_inputs():
    processor = Qwen3VLProcessor.from_pretrained(QWEN3VL_MODEL_PATH)

    image1 = Image.new("RGB", (3832, 2084), color="white")
    image2 = Image.new("RGB", (512, 512), color="black")
    txt = processor.apply_chat_template(
        [{
            "role": "user",
            "content": [
                {"type": "image", "url": None},
                {"type": "image", "url": None},
                {"type": "text", "text": "What animal is on the candy?"}
            ]
        }]
    )
    inputs = processor(text=txt, images=[[image1, image2]], return_tensors="pt")
    return inputs

# =============================================================================
# Weight Loading Tests (require HuggingFace model)
# =============================================================================

def test_loaded_models(hf_model: HFModel, jax_model: Qwen3VLForConditionalGeneration):
    hf_params = hf_model.state_dict()
    def visit(path: jax.tree_util.KeyPath, leaf: Any):
        str_path = jax.tree_util.keystr(path, simple=True, separator=".")
        if isinstance(leaf, Array) and str_path in hf_params:
            np.testing.assert_allclose(leaf, hf_params[str_path].detach().cpu().numpy())
        return leaf
    jax.tree_util.tree_map_with_path(visit, jax_model)

def test_vision(hf_model: HFModel, jax_model: Qwen3VLForConditionalGeneration):
    processor = Qwen3VLProcessor.from_pretrained(QWEN3VL_MODEL_PATH)
    image1 = Image.new("RGB", (256, 256), color="white")
    image2 = Image.new("RGB", (256, 256), color="black")
    txt = processor.apply_chat_template(
        [{
            "role": "user",
            "content": [
                {"type": "image", "url": None},
                {"type": "image", "url": None},
                {"type": "text", "text": "What animal is on the candy?"}
            ]
        }]
    )
    inputs = processor(text=txt, images=[[image1, image2]], return_tensors="pt")
    with torch.no_grad():
        output_pt, layers_pt = tu.extract_pytorch_outputs(hf_model, **inputs)
    inputs = {k: to_jax(v) for k, v in dict(inputs).items()}
    output_jax, layers_jax = tu.extract_jax_outputs_wrapped(jax_model, **inputs)

    for name, jax_val in layers_jax.items():
        if name in layers_pt:
            pt_val = layers_pt[name].squeeze()
            jax_val = jax_val.squeeze()

            if jax_val.shape != pt_val.shape:
                print(f'<{name}> Shape mismatch: {jax_val.shape} vs {pt_val.shape}')
                continue
            abs_diff = np.abs(jax_val - pt_val)
            max_abs_diff = float(np.max(abs_diff))
            mean_abs_diff = float(np.mean(abs_diff))

            close = np.isclose(jax_val, pt_val, rtol=1e-2, atol=1e-2)
            mismatch_pct = 100 * (1 - close.mean())
            print(f'<{name}> mean_diff={mean_abs_diff:.6f} max_diff={max_abs_diff:.6f} mismatch={mismatch_pct:.1f}%')

    np.testing.assert_allclose(
        jsdiv(output_jax.logits, output_pt.logits.detach().cpu().numpy()),
        0.0,
        rtol=1e-2, atol=1e-2
    )

def test_vision_2(hf_model: HFModel, jax_model: Qwen3VLForConditionalGeneration):
    processor = Qwen3VLProcessor.from_pretrained(QWEN3VL_MODEL_PATH)
    image1 = Image.new("RGB", (512, 1024), color="white")
    txt = processor.apply_chat_template(
        [{
            "role": "user",
            "content": [
                {"type": "image", "url": None},
                {"type": "text", "text": "What animal is on the candy?"}
            ]
        }]
    )
    inputs = processor(text=txt, images=[[image1]], return_tensors="pt")
    with torch.no_grad():
        output_pt, layers_out_pt, layers_in_pt = tu.extract_pytorch_outputs_and_inputs(hf_model, **inputs)

    jax_layers = {}
    def visit(path, module):
        str_path = jax.tree_util.keystr(path, simple=True, separator=".")
        jax_layers[str_path] = module
        return module
    mapmod_with_path(visit, jax_model)

    assert layers_in_pt
    assert layers_out_pt

    mismatch = {}
    mismatched_layers = []
    for name in layers_in_pt:
        if name not in layers_out_pt:
            print(f'Input layer {name} not in output layers')
            continue
        if name not in jax_layers:
            print(f'Layer {name} not found in JAX model')
            continue
        try:
            pt_val = layers_out_pt[name]
            jax_val = jax_layers[name](to_jax(layers_in_pt[name]))
        except Exception as e:
            print(f'Error processing layer {name}: {e}')
            continue

        if jax_val.shape != pt_val.shape:
            print(f'<{name}> Shape mismatch: {jax_val.shape} vs {pt_val.shape}')
            continue
        abs_diff = np.abs(jax_val - pt_val)
        max_abs_diff = float(np.max(abs_diff))
        mean_abs_diff = float(np.mean(abs_diff))

        close = np.isclose(jax_val, pt_val, rtol=1e-3, atol=1e-3)
        mismatch_pct = 100 * (1 - close.mean())
        mismatch[name] = mismatch_pct
        print(f'<{name}> mean_diff={mean_abs_diff:.6f} max_diff={max_abs_diff:.6f} mismatch={mismatch_pct:.1f}%')
        if mismatch_pct > 1.0:
            mismatched_layers.append(name)

    assert not mismatched_layers # all(m == 0.0 for m in mismatch.values()), f'Mismatched layers: {mismatch}'


def test_get_rope_index(hf_model: HFModel, jax_model: Qwen3VLForConditionalGeneration, large_image_inputs):
    inputs = large_image_inputs
    hf_rope_index = hf_model.model.get_rope_index(
        input_ids=inputs['input_ids'],
        image_grid_thw=inputs['image_grid_thw'],
        attention_mask=inputs['attention_mask'],
    )
    inputs = {k: to_jax(v) for k, v in dict(inputs).items()}
    jax_rope_index = jitbound(jax_model.model.get_rope_index)(
        input_ids=inputs['input_ids'],
        image_grid_thw=inputs['image_grid_thw'],
        attention_mask=inputs['attention_mask'],
    )
    np.testing.assert_array_equal(
        jax_rope_index[0],
        hf_rope_index[0].numpy(),
    )
    np.testing.assert_array_equal(
        jax_rope_index[1],
        hf_rope_index[1].numpy(),
    )

@torch.no_grad()
def test_fast_pos_embed_interpolate(hf_model: HFModel, jax_model: Qwen3VLForConditionalGeneration, large_image_inputs):
    grid_thw = to_jax(large_image_inputs['image_grid_thw'])
    total_patches = int(np.sum(np.prod(grid_thw, axis=-1)))
    hf_embed = hf_model.model.visual.fast_pos_embed_interpolate(
        to_torch(grid_thw),
    )
    jax_embed = jax_model.model.visual.fast_pos_embed_interpolate(
        grid_thw,
        total_patches,
    )
    np.testing.assert_allclose(
        np.array(jax_embed),
        hf_embed.detach().cpu().numpy(),
        rtol=1e-3, atol=1e-3
    )

@torch.no_grad()
def test_rot_pos_emb(hf_model: HFModel, jax_model: Qwen3VLForConditionalGeneration, large_image_inputs):
    grid_thw = to_jax(large_image_inputs['image_grid_thw'])
    total_patches = int(np.sum(np.prod(grid_thw, axis=-1)))
    hf_embed = hf_model.model.visual.rot_pos_emb(
        to_torch(grid_thw),
    )
    jax_embed = jax_model.model.visual.rot_pos_emb(
        grid_thw,
        total_patches,
    )
    np.testing.assert_allclose(
        np.array(jax_embed),
        hf_embed.detach().cpu().numpy(),
        rtol=1e-3, atol=1e-3
    )


def test_vision_gen(jax_model: Qwen3VLForConditionalGeneration):
    processor = Qwen3VLProcessor.from_pretrained(QWEN3VL_MODEL_PATH)
    image1 = Image.new("RGB", (256, 256), color="white")
    image2 = Image.new("RGB", (256, 256), color="black")
    txt = processor.apply_chat_template(
        [{
            "role": "user",
            "content": [
                {"type": "image", "url": None},
                {"type": "image", "url": None},
                {"type": "text", "text": "What animal is on the candy?"}
            ]
        }]
    )
    inputs = processor(text=txt, images=[[image1, image2]], return_tensors="pt")
    inputs = {k: to_jax(v) for k, v in dict(inputs).items()}

    output = jax_model.generate(
        **inputs,
        max_new_tokens=96,
        temperature=0.7,
        key=jax.random.key(0),
        return_logits=True,
    )
    print("Generated IDs:", output.tokens)
    out_str = processor.decode(output.tokens[0, inputs['input_ids'].shape[1]:])
    print(out_str)

    full_inputs = inputs | {'input_ids': output.tokens[:, :-1], 'attention_mask': jnp.ones_like(output.tokens[:, :-1])}
    output_full = jax_model(**full_inputs)
    np.testing.assert_allclose(
        jsdiv(output.logits, output_full.logits[:, inputs['input_ids'].shape[1]-1:]),
        0.0,
        rtol=1e-2, atol=1e-2
    )

class TextImageInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed = True)
    input_ids: Array # bsz seq
    attention_mask: Array # bsz seq
    image_grid_thw: Array # num_images 3
    pixel_values: Array # total_patches 3*t*h*w

def test_vision_with_cache(jax_model: Qwen3VLForConditionalGeneration):
    processor = Qwen3VLProcessor.from_pretrained(QWEN3VL_MODEL_PATH)
    image1 = Image.new("RGB", (256, 256), color="white")
    image2 = Image.new("RGB", (256, 256), color="black")
    txt = processor.apply_chat_template(
        [{
            "role": "user",
            "content": [
                {"type": "image", "url": None},
                {"type": "image", "url": None},
                {"type": "text", "text": "What animal is on the candy?"}
            ]
        }]
    )
    inputs = processor(text=txt, images=[[image1, image2]], return_tensors="pt")
    inputs = {k: to_jax(v) for k, v in dict(inputs).items()}
    inputs = TextImageInputs.model_validate(inputs)

    output_full = jax_model(**inputs.dict())

    cache = KVCache.create(
        num_layers=jax_model.model.config.text_config.num_hidden_layers,
        batch_size=inputs.input_ids.shape[0],
        max_seq_len=inputs.input_ids.shape[1],
        num_kv_heads=jax_model.model.config.text_config.num_key_value_heads,
        head_dim=jax_model.model.config.text_config.head_dim,
        dtype=jnp.bfloat16,
    )
    cache_position = jnp.array(0, dtype=jnp.int32)
    output1 = jax_model(
        input_ids=inputs.input_ids[:, :-4],
        attention_mask=inputs.attention_mask[:, :-4],
        image_grid_thw=inputs.image_grid_thw,
        pixel_values=inputs.pixel_values,
        cache=cache,
    )
    cache = output1.cache
    cache_position += inputs.input_ids[:, :-4].shape[1]
    output2 = jax_model(
        input_ids=inputs.input_ids[:, -4:],
        attention_mask=inputs.attention_mask[:, -4:],
        cache=cache,
        rope_deltas=output1.rope_deltas,
    )

    div = jsdiv(
        output_full.logits,
        jnp.concatenate([output1.logits, output2.logits], axis=1),
    )

    np.testing.assert_allclose(
        div,
        0.0,
        rtol=1e-2, atol=1e-2
    )

def test_padding(jax_model: Qwen3VLForConditionalGeneration, processor: Qwen3VLProcessor):
    txt = processor.apply_chat_template(
        [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Example text without image."}
            ]
        }]
    )
    inputs = convert_inputs(processor(text=txt, return_tensors="pt"))
    output = jax_model(**inputs)
    inputs_padded = convert_inputs(processor(text=txt, return_tensors="pt", max_length=100, padding='max_length', truncation=True, padding_side='left'))
    output_padded = jax_model(**inputs_padded)
    assert inputs_padded['input_ids'].shape[1] > inputs['input_ids'].shape[1]
    np.testing.assert_allclose(
        jsdiv(output.logits, output_padded.logits[:, -output.logits.shape[1]:]),
        0.0,
        rtol=1e-2, atol=1e-2
    )

def test_padding_vision(jax_model: Qwen3VLForConditionalGeneration, processor: Qwen3VLProcessor):
    image1 = Image.new("RGB", (256, 256), color="white")
    image2 = Image.new("RGB", (256, 256), color="black")
    txt = processor.apply_chat_template(
        [{
            "role": "user",
            "content": [
                {"type": "image", "url": None},
                {"type": "image", "url": None},
                {"type": "text", "text": "Example text without image."}
            ]
        }]
    )
    inputs = convert_inputs(processor(text=txt, images=[[image1, image2]], return_tensors="pt"))
    output = jax_model(**inputs)
    inputs_padded = convert_inputs(processor(text=txt, images=[[image1, image2]], return_tensors="pt", max_length=1000, padding='max_length', truncation=True, padding_side='left'))
    output_padded = jax_model(**inputs_padded)
    np.testing.assert_allclose(
        jsdiv(output.logits, output_padded.logits[:, -output.logits.shape[1]:]),
        0.0,
        rtol=1e-2, atol=1e-2
    )

def test_padding_vision_with_cache(jax_model: Qwen3VLForConditionalGeneration, processor: Qwen3VLProcessor):
    image = Image.new("RGB", (256, 256), color="white")
    txt = processor.apply_chat_template(
        [{
            "role": "user",
            "content": [
                {"type": "image", "url": None},
                {"type": "text", "text": "Example text with image."},
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This is a response from the assistant."}
            ]
        }
        ]
    )
    inputs = convert_inputs(processor(text=txt, return_tensors="pt"))
    txt1 = processor.apply_chat_template(
        [{
            "role": "user",
            "content": [
                {"type": "image", "url": None},
                {"type": "text", "text": "Example text with image."},
            ]
        }]
    )
    txt2 = processor.apply_chat_template(
        [{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This is a response from the assistant."}
            ]
        }
        ]
    )
    inputs1_padded = convert_inputs(processor(text=txt1, return_tensors="pt", padding=True, pad_to_multiple_of=128, padding_side='left'))
    inputs2_padded = convert_inputs(processor(text=txt2, return_tensors="pt", padding=True, pad_to_multiple_of=128, padding_side='left'))
    mask = jnp.concatenate([inputs1_padded['attention_mask'], inputs2_padded['attention_mask']], axis=1)
    inputs2_padded['attention_mask'] = mask
    output = jax_model(**inputs)
    output1_padded = jax_model(**inputs1_padded, use_cache=True)
    output2_padded = jax_model(**inputs2_padded, cache=output1_padded.cache.resize(mask.shape[1]), rope_deltas=output1_padded.rope_deltas)
    np.testing.assert_allclose(
        jsdiv(output.logits, jnp.concatenate([output1_padded.logits, output2_padded.logits], axis=1)[jnp.bool(mask)]),
        0.0,
        rtol=1e-2, atol=1e-2
    )
