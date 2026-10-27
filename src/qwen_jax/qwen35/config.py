"""Pydantic configuration models for Qwen3.5 (dense).

Qwen3.5 is the Qwen3-Next lineage made natively multimodal: a hybrid stack of
Gated DeltaNet (linear attention) layers with a gated full-attention layer
every `full_attention_interval`-th position, partial RoPE on the attention
layers, and the Qwen3-VL vision encoder without DeepStack.

Head-order convention: everywhere in this package, DeltaNet value heads are in
llama.cpp's *tiled* order (v head h pairs with k head h % num_k_heads), not
HF's grouped order (h // (num_v/num_k)). GGUF files store weights this way;
the HF-side conversion in `qwen35.loading` reorders. This choice is what lets
quantized GGUF tensors load without any row/column surgery on packed data.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from qwen_jax.config import Qwen3VLVisionConfig, RopeScalingConfig, lift_rope_theta


class Qwen35TextConfig(BaseModel):
    """Configuration for the Qwen3.5 text decoder."""

    model_config = ConfigDict(frozen=True)

    vocab_size: int = 248320
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_act: str = "silu"
    max_position_embeddings: int = 262144
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_output_gate: bool = True

    # Hybrid layer layout. layer_types wins when present; otherwise derived
    # from full_attention_interval (every interval-th layer is full attention).
    layer_types: tuple[str, ...] | None = None
    full_attention_interval: int = 4

    # Gated DeltaNet dimensions.
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    # The MTP head is one extra decoder layer for self-speculative decoding;
    # it is not implemented here (llama.cpp drops it from GGUFs too).
    mtp_num_hidden_layers: int = 1

    rope_theta: float = 10000000.0
    rope_scaling: RopeScalingConfig | None = Field(
        default=None,
        validation_alias=AliasChoices("rope_parameters", "rope_scaling"),
    )

    _lift_rope_theta = model_validator(mode="before")(lift_rope_theta)

    def resolved_layer_types(self) -> tuple[str, ...]:
        """One of "linear_attention"/"full_attention" per layer."""
        if self.layer_types is not None:
            assert len(self.layer_types) == self.num_hidden_layers
            return self.layer_types
        return tuple(
            "full_attention" if (i + 1) % self.full_attention_interval == 0
            else "linear_attention"
            for i in range(self.num_hidden_layers)
        )


class Qwen35Config(BaseModel):
    """Configuration for the Qwen3.5 multimodal model."""

    model_config = ConfigDict(frozen=True)

    text_config: Qwen35TextConfig
    vision_config: Qwen3VLVisionConfig
    image_token_id: int = 248056
    video_token_id: int = 248057
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    tie_word_embeddings: bool = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs) -> Qwen35Config:
        """Load the configuration from a pretrained model directory."""
        from transformers import Qwen3_5Config as HFQwen35Config

        hf_config = HFQwen35Config.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        data = hf_config.to_dict()
        # Qwen3.5 has no DeepStack; HF's vision config deletes the attribute
        # outright, so make its absence mean "none" rather than letting the
        # shared Qwen3VLVisionConfig default (8, 16, 24) leak in.
        vision = dict(data.get("vision_config") or {})
        vision.setdefault("deepstack_visual_indexes", ())
        data["vision_config"] = vision
        return cls.model_validate(data)


__all__ = ["Qwen35Config", "Qwen35TextConfig"]
