"""Tests for the Llama-architecture decoder.

Parity against the transformers reference on tiny randomly-initialized
configs in float32. The HF state dict loads with no conversion, so these cover
the safetensors loader's key layout too.
"""
from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

from qwen_jax.cache import KVCache
from qwen_jax.llama import LlamaConfig, LlamaForCausalLM
from qwen_jax.loading import load_state_dict


@pytest.fixture(autouse=True)
def xla_attention(monkeypatch):
    """Use the XLA attention path: exact float32, no fp16 downcast."""
    monkeypatch.setenv("QWEN_JAX_ATTENTION", "xla")


@pytest.fixture(autouse=True)
def full_f32_matmuls():
    """Match torch: no tf32 in the float32 parity comparisons on GPU."""
    with jax.default_matmul_precision("highest"):
        yield


TINY_KWARGS = dict(
    vocab_size=512,
    hidden_size=64,
    intermediate_size=160,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    rms_norm_eps=1e-5,
    rope_theta=10000000.0,
)


def to_jax(x: torch.Tensor) -> jax.Array:
    return jnp.asarray(x.detach().cpu().numpy())


def make_models(tie: bool, seed: int = 2):
    torch.manual_seed(seed)
    hf_config = HFLlamaConfig(**TINY_KWARGS, tie_word_embeddings=tie)
    hf_config._attn_implementation = "eager"
    hf = HFLlamaForCausalLM(hf_config).float().eval()

    # Through the same to_dict -> validate path as LlamaConfig.from_pretrained.
    config = LlamaConfig.model_validate(hf_config.to_dict())
    state_dict = {k: to_jax(v) for k, v in hf.state_dict().items()}
    if tie:
        state_dict.pop("lm_head.weight", None)
    model = load_state_dict(LlamaForCausalLM(config), state_dict)
    assert not state_dict, f"Unused keys: {sorted(state_dict)}"
    return hf, model


@pytest.fixture(scope="module")
def tiny_models():
    return make_models(tie=False)


def test_config_from_hf_dict():
    config = LlamaConfig.model_validate(HFLlamaConfig(**TINY_KWARGS).to_dict())
    assert config.head_dim == 16
    assert config.rope_theta == 10000000.0
    assert config.num_key_value_heads == 2


def test_config_refuses_rope_scaling():
    hf_config = HFLlamaConfig(
        **TINY_KWARGS,
        rope_scaling=dict(
            rope_type="llama3", factor=8.0, low_freq_factor=1.0,
            high_freq_factor=4.0, original_max_position_embeddings=8192,
        ),
    )
    with pytest.raises(NotImplementedError, match="llama3"):
        LlamaConfig.model_validate(hf_config.to_dict())


@pytest.mark.parametrize("tie", [False, True])
@torch.no_grad()
def test_model_vs_hf(tie):
    hf, model = make_models(tie=tie)
    torch.manual_seed(3)
    input_ids = torch.randint(0, 512, (2, 37))

    out_pt = hf(input_ids=input_ids).logits
    out = model(input_ids=to_jax(input_ids)).logits
    np.testing.assert_allclose(np.asarray(out), out_pt.numpy(), atol=2e-4, rtol=1e-3)


@torch.no_grad()
def test_model_vs_hf_left_padded(tiny_models):
    """Left padding with default positions matches HF on the real tokens.

    HF shifts positions by the padding; we leave them at the index. RoPE only
    sees position differences, so the two agree wherever the mask is 1.
    """
    hf, model = tiny_models
    torch.manual_seed(4)
    pad, seq = 6, 23
    ids = torch.randint(0, 512, (2, pad + seq))
    mask = torch.ones(2, pad + seq, dtype=torch.long)
    mask[0, :pad] = 0

    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    out_pt = hf(input_ids=ids, attention_mask=mask, position_ids=positions).logits
    out = model(input_ids=to_jax(ids), attention_mask=to_jax(mask)).logits
    np.testing.assert_allclose(
        np.asarray(out)[0, pad:], out_pt.numpy()[0, pad:], atol=2e-4, rtol=1e-3
    )
    np.testing.assert_allclose(np.asarray(out)[1], out_pt.numpy()[1], atol=2e-4, rtol=1e-3)


def test_last_logit_only(tiny_models):
    _, model = tiny_models
    input_ids = jax.random.randint(jax.random.key(7), (2, 15), 0, 512)
    mask = jnp.ones((2, 15), jnp.int32).at[1, 11:].set(0)  # row 1 right-padded

    full = model(input_ids=input_ids, attention_mask=mask).logits
    last = model(input_ids=input_ids, attention_mask=mask, last_logit_only=True).last_logits
    np.testing.assert_allclose(np.asarray(last[0]), np.asarray(full[0, 14]), atol=1e-5)
    np.testing.assert_allclose(np.asarray(last[1]), np.asarray(full[1, 10]), atol=1e-5)


def test_cache_consistency(tiny_models):
    """Prefill + chunked continuation + single-token decode matches one pass."""
    _, model = tiny_models
    input_ids = jax.random.randint(jax.random.key(5), (2, 40), 0, 512)
    full = model(input_ids=input_ids).logits

    cache = model.create_cache(2, 40, dtype=jnp.float32)
    out1 = model(input_ids=input_ids[:, :25], cache=cache)
    out2 = model(input_ids=input_ids[:, 25:33], cache=out1.cache)
    logits = [out1.logits, out2.logits]
    cache = out2.cache
    for t in range(33, 40):
        out = model(input_ids=input_ids[:, t : t + 1], cache=cache)
        cache = out.cache
        logits.append(out.logits)
    stitched = jnp.concatenate(logits, axis=1)
    np.testing.assert_allclose(np.asarray(stitched), np.asarray(full), atol=1e-4, rtol=1e-3)
    assert isinstance(cache, KVCache) and int(cache.position) == 40


def test_generate(tiny_models):
    """generate() runs end to end and its logits re-score consistently."""
    _, model = tiny_models
    input_ids = jax.random.randint(jax.random.key(6), (1, 9), 0, 512)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=12,
        temperature=0.0,
        key=jax.random.key(0),
        progress_bar=False,
        return_logits=True,
        cache=model.create_cache(1, 21, dtype=jnp.float32),
    )
    assert output.tokens.shape == (1, 21)
    full = model(input_ids=output.tokens[:, :-1]).logits
    np.testing.assert_allclose(
        np.asarray(output.logits), np.asarray(full[:, 8:]), atol=1e-4, rtol=1e-3
    )


def llama_cpp_permute(weights: np.ndarray, n_head: int, n_head_kv: int | None) -> np.ndarray:
    """LlamaModel.permute from llama.cpp's convert_hf_to_gguf.py, verbatim."""
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))


GGUF_NAMES = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


@torch.no_grad()
def test_gguf_vs_hf(tmp_path):
    """A llama-arch GGUF (q/k permuted, Q8_0 linears) matches HF on the same
    quantized weights: the name table and the q/k unpermute are both right.

    HF runs the Q8_0 round-tripped weights, so the comparison isolates layout
    from quantization error.
    """
    import gguf
    from gguf.constants import GGMLQuantizationType as QT

    from qwen_jax.llama import load_llama_gguf

    torch.manual_seed(9)
    hf_config = HFLlamaConfig(**TINY_KWARGS)
    hf_config._attn_implementation = "eager"
    hf = HFLlamaForCausalLM(hf_config).float().eval()
    hf_config.save_pretrained(tmp_path)

    writer = gguf.GGUFWriter(tmp_path / "tiny.gguf", "llama")
    n_head, n_kv = hf_config.num_attention_heads, hf_config.num_key_value_heads
    for key, param in hf.state_dict().items():
        w = param.numpy()
        if w.ndim == 2:
            # Round-trip HF's copy through Q8_0 so both models see the same values.
            q8 = gguf.quants.quantize(w, QT.Q8_0)
            w = gguf.quants.dequantize(q8, QT.Q8_0).reshape(w.shape)
            param.copy_(torch.from_numpy(w))
        m = re.match(r"model\.layers\.(\d+)\.(.*)$", key)
        name = f"blk.{m.group(1)}.{GGUF_NAMES[m.group(2)]}" if m else GGUF_NAMES[key]
        if key.endswith("q_proj.weight"):
            w = llama_cpp_permute(w, n_head, n_head)
        elif key.endswith("k_proj.weight"):
            w = llama_cpp_permute(w, n_head, n_kv)
        if w.ndim == 2:
            writer.add_tensor(name, gguf.quants.quantize(w, QT.Q8_0), raw_dtype=QT.Q8_0)
        else:
            writer.add_tensor(name, w.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    model = load_llama_gguf(tmp_path, tmp_path / "tiny.gguf", dtype=jnp.float32, fused=False)
    torch.manual_seed(10)
    input_ids = torch.randint(0, 512, (2, 19))
    out_pt = hf(input_ids=input_ids).logits
    out = model(input_ids=to_jax(input_ids)).logits
    np.testing.assert_allclose(np.asarray(out), out_pt.numpy(), atol=2e-4, rtol=1e-3)


def test_unpermute_inverts_llama_cpp():
    from qwen_jax.llama.gguf import unpermute_rows

    for n_head, n_kv in [(4, 4), (4, 2), (56, 8)]:
        rows = n_kv * 128
        w = np.arange(rows * 3).reshape(rows, 3)
        permuted = llama_cpp_permute(w, n_head, n_kv)
        np.testing.assert_array_equal(permuted[unpermute_rows(n_kv, rows)], w)


@torch.no_grad()
def test_greedy_matches_hf_generate(tiny_models):
    hf, model = tiny_models
    torch.manual_seed(8)
    input_ids = torch.randint(0, 512, (1, 11))
    ref = hf.generate(input_ids, max_new_tokens=10, do_sample=False, pad_token_id=0)
    out = model.generate(
        input_ids=to_jax(input_ids),
        max_new_tokens=10,
        temperature=0.0,
        key=jax.random.key(0),
        progress_bar=False,
        cache=model.create_cache(1, 21, dtype=jnp.float32),
    )
    np.testing.assert_array_equal(np.asarray(out.tokens), ref.numpy())
