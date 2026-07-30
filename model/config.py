from dataclasses import dataclass, fields
from typing import List, Optional, Tuple


def _from_dict(cls, d):
    """Build a dataclass from a dict, ignoring unknown keys (so missing keys
    fall back to defaults — e.g. norm_topk_prob, which the checkpoint omits)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class VisionConfig:
    image_size: int = 1024
    model_name: str = "deeplip_b_l"
    model_type: str = "vision"

    clip_layers: int = 24
    clip_hidden_size: int = 1024
    clip_num_heads: int = 16
    clip_patch_size: int = 14
    clip_image_size: int = 224
    clip_ffn_hidden_size: int = 4096
    clip_seq_length: int = 256
    clip_use_flash_attn: bool = False
    clip_hidden_dropout: float = 0.0
    clip_attention_dropout: float = 0.0
    clip_layernorm_epsilon: float = 1e-5
    clip_pre_layernorm_epsilon: float = 1e-5

    sam_embed_dim: int = 768
    sam_depth: int = 12
    sam_num_heads: int = 12
    sam_global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11)
    sam_window_size: int = 14
    sam_patch_size: int = 16
    sam_out_chans: int = 256
    sam_mlp_ratio: float = 4.0
    sam_downsample_channels: Tuple[int, ...] = (512, 1024)


@dataclass
class ProjectorConfig:
    input_dim: int = 2048
    model_type: str = "mlp_projector"
    n_embed: int = 1280
    projector_type: str = "linear"


@dataclass
class LanguageConfig:
    architectures: List[str] = None
    bos_token_id: int = 0
    eos_token_id: int = 1
    first_k_dense_replace: int = 1
    hidden_size: int = 1280
    intermediate_size: int = 6848
    kv_lora_rank: Optional[int] = None
    lm_head: bool = True
    max_position_embeddings: int = 32768
    moe_intermediate_size: int = 896
    n_group: int = 1
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_attention_heads: int = 10
    num_experts_per_tok: int = 6
    num_hidden_layers: int = 12
    num_key_value_heads: int = 10
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    rm_head: bool = False
    topk_group: int = 1
    topk_method: str = "greedy"
    use_mla: bool = False
    v_head_dim: int = 128
    vocab_size: int = 129280
    sliding_window_size: int = 128
    sliding_window: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    routed_scaling_factor: float = 1.0
    scoring_func: str = "softmax"
    aux_loss_alpha: float = 0.001
    seq_aux: bool = True
    # DeepseekV2Config default is False, and the checkpoint config.json does not
    # override it. Normalizing the top-k weights (True) inflates every MoE
    # layer's output ~2x and corrupts fine-grained predictions.
    norm_topk_prob: bool = False
    moe_layer_freq: int = 1
    hidden_act: str = "silu"

    def __post_init__(self):
        if self.architectures is None:
            self.architectures = ["DeepseekOCRForCausalLM"]

    @classmethod
    def from_dict(cls, d):
        return _from_dict(cls, d)


@dataclass
class UnlimitedOCRConfig:
    model_type: str = "unlimited-ocr"
    architectures: List[str] = None
    bos_token_id: int = 0
    eos_token_id: int = 1
    hidden_size: int = 1280
    vocab_size: int = 129280
    candidate_resolutions: List[List[int]] = None
    global_view_pos: str = "head"
    tile_tag: str = "2D"
    torch_dtype: str = "bfloat16"

    vision_config: VisionConfig = None
    projector_config: ProjectorConfig = None
    language_config: LanguageConfig = None

    def __post_init__(self):
        if self.architectures is None:
            self.architectures = ["UnlimitedOCRForCausalLM"]
        if self.candidate_resolutions is None:
            self.candidate_resolutions = [[1024, 1024]]
        if self.vision_config is None:
            self.vision_config = VisionConfig()
        if self.projector_config is None:
            self.projector_config = ProjectorConfig()
        if self.language_config is None:
            self.language_config = LanguageConfig()

    @classmethod
    def from_dict(cls, d):
        cfg = _from_dict(cls, d)
        cfg.language_config = LanguageConfig.from_dict(d.get("language_config"))
        cfg.projector_config = _from_dict(ProjectorConfig, d.get("projector_config"))
        cfg.vision_config = VisionConfig()  # architecture constants
        return cfg

    @classmethod
    def from_json(cls, path):
        import json

        with open(path) as f:
            return cls.from_dict(json.load(f))
