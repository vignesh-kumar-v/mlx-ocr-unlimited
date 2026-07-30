"""DeepSeek-V2 MoE language model (the LLM backbone of DeepSeek-OCR).

This checkpoint runs with ``use_mla=False``, so attention is standard
grouped/multi-head attention with plain Llama-style RoPE (not Multi-head Latent
Attention). Decode uses a sliding window that always retains the prompt/image
tokens; see ``model/cache.py``.
"""

import mlx.core as mx
import mlx.nn as nn

from .config import LanguageConfig
from .moe import SwitchGLU


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class DeepseekV2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))

    def __call__(self, x):
        # Gating is computed in float32, matching the reference.
        logits = mx.matmul(x.astype(mx.float32), self.weight.astype(mx.float32).T)
        scores = mx.softmax(logits, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        return inds, weights.astype(x.dtype)


class DeepseekV2MoE(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.gate = MoEGate(config)
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts
        )
        self.shared_experts = DeepseekV2MLP(
            config.hidden_size, config.moe_intermediate_size * config.n_shared_experts
        )

    def __call__(self, x):
        inds, weights = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * weights[..., None]).sum(axis=-2)
        return y + self.shared_experts(x)


class DeepseekV2Attention(nn.Module):
    def __init__(self, config: LanguageConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.v_head_dim = config.v_head_dim
        self.scale = self.head_dim**-0.5
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.v_head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, config.hidden_size, bias=False
        )

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        q = (
            self.q_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, L, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, L, self.num_kv_heads, self.v_head_dim)
            .transpose(0, 2, 1, 3)
        )

        offset = cache.offset if cache is not None else 0
        q = mx.fast.rope(
            q,
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offset,
        )
        k = mx.fast.rope(
            k,
            self.head_dim,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offset,
        )

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.v_head_dim)
        return self.o_proj(out)


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: LanguageConfig, layer_idx: int):
        super().__init__()
        self.self_attn = DeepseekV2Attention(config)

        is_moe = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        )
        self.mlp = (
            DeepseekV2MoE(config)
            if is_moe
            else DeepseekV2MLP(config.hidden_size, config.intermediate_size)
        )
        self.input_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, mask=None, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))
