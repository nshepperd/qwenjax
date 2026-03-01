"""Pydantic configuration models for Qwen3-VL."""

from pathlib import Path
from pydantic import BaseModel, ConfigDict
from transformers import Qwen3VLConfig as HFQwen3VLConfig

class Qwen3VLVisionConfig(BaseModel):
    """Configuration for the Qwen3-VL vision encoder."""

    model_config = ConfigDict(frozen=True)

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)
    initializer_range: float = 0.02


class RopeScalingConfig(BaseModel):
    """Configuration for RoPE scaling."""

    model_config = ConfigDict(frozen=True)

    rope_type: str = "default"
    factor: float | None = None
    original_max_position_embeddings: int | None = None
    attention_factor: float | None = None
    beta_fast: float | None = None
    beta_slow: float | None = None
    short_factor: tuple[float, ...] | None = None
    long_factor: tuple[float, ...] | None = None
    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    mrope_section: tuple[int, int, int] | None = None
    mrope_interleaved: bool = True

class Qwen3VLTextConfig(BaseModel):
    """Configuration for the Qwen3-VL text decoder."""

    model_config = ConfigDict(frozen=True)

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 5000000.0
    rope_scaling: RopeScalingConfig | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0

class QuantizationConfig(BaseModel):
    _load_in_4bit: bool
    _load_in_8bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_storage: str
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    llm_int8_enable_fp32_cpu_offload: bool
    llm_int8_has_fp16_weight: bool
    llm_int8_skip_modules: list[str]
    llm_int8_threshold: float
    load_in_4bit: bool
    load_in_8bit: bool
    quant_method: str

class Qwen3VLConfig(BaseModel):
    """Configuration for the Qwen3-VL multimodal model."""

    model_config = ConfigDict(frozen=True)

    text_config: Qwen3VLTextConfig
    vision_config: Qwen3VLVisionConfig
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False
    quantization_config: QuantizationConfig | None = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs) -> "Qwen3VLConfig":
        """Load the configuration from a pretrained model."""
        hf_config = HFQwen3VLConfig.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        return cls.model_validate(hf_config.to_dict())
