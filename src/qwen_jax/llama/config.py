"""Pydantic configuration for Llama-architecture decoders.

Covers HF `LlamaForCausalLM` checkpoints (Llama 2/3 and the many models that
reuse the architecture, e.g. Nanbeige4.3). Only unscaled RoPE is implemented;
a checkpoint asking for llama3/yarn/etc. scaling is refused at load rather
than run with the wrong frequencies.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from qwen_jax.config import RopeScalingConfig, lift_rope_theta


class LlamaConfig(BaseModel):
    """Configuration for a Llama decoder."""

    model_config = ConfigDict(frozen=True)

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    # HF defaults this to hidden_size // num_attention_heads; see _default_head_dim.
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = None

    rope_theta: float = 10000.0
    rope_scaling: RopeScalingConfig | None = Field(
        default=None,
        validation_alias=AliasChoices("rope_parameters", "rope_scaling"),
    )

    _lift_rope_theta = model_validator(mode="before")(lift_rope_theta)

    @model_validator(mode="before")
    @classmethod
    def _default_head_dim(cls, data):
        if isinstance(data, dict) and data.get("head_dim") is None and "hidden_size" in data:
            heads = data.get("num_attention_heads", 32)
            data = {**data, "head_dim": data["hidden_size"] // heads}
        return data

    @model_validator(mode="after")
    def _check_supported(self):
        if self.rope_scaling is not None and self.rope_scaling.rope_type != "default":
            raise NotImplementedError(
                f"RoPE scaling {self.rope_scaling.rope_type!r} is not implemented"
            )
        if self.hidden_act != "silu":
            raise NotImplementedError(f"hidden_act {self.hidden_act!r} is not implemented")
        if self.mlp_bias:
            raise NotImplementedError("mlp_bias is not implemented")
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs) -> LlamaConfig:
        """Load the configuration from an HF model directory or hub id."""
        from transformers import LlamaConfig as HFLlamaConfig

        hf_config = HFLlamaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.model_validate(hf_config.to_dict())


__all__ = ["LlamaConfig"]
