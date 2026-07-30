"""Tests for the DeepSeek-OCR MLX port.

Fast unit tests always run. The end-to-end test is skipped unless converted
weights (``mlx-weights/``) and the sample PDF are present.
"""

import os
import sys

import mlx.core as mx
import mlx.nn as nn
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "mlx-weights", "weights.safetensors")
PDF = os.path.join(ROOT, "papers", "1706.03762v7.pdf")


def test_switchglu_matches_per_expert_loop():
    """The batched gather_mm MoE must equal an explicit per-expert computation."""
    from model.moe import SwitchGLU

    mx.random.seed(0)
    n_experts, hidden, inter, top_k, tokens = 8, 32, 48, 3, 10
    glu = SwitchGLU(hidden, inter, n_experts)
    mx.eval(glu.parameters())

    x = mx.random.normal((tokens, hidden))
    inds = mx.stack(
        [mx.random.permutation(n_experts)[:top_k] for _ in range(tokens)], axis=0
    )
    y = glu(x, inds)  # [tokens, top_k, hidden]

    gate, up, down = glu.gate_proj.weight, glu.up_proj.weight, glu.down_proj.weight
    ref = mx.zeros((tokens, top_k, hidden))
    for t in range(tokens):
        for j in range(top_k):
            e = int(inds[t, j])
            h = nn.silu(x[t] @ gate[e].T) * (x[t] @ up[e].T)
            ref[t, j] = h @ down[e].T

    assert float(mx.abs(y - ref).max()) < 1e-4


def test_moe_gate_topk_unnormalized():
    """norm_topk_prob=False: gate weights are raw softmax probs of the top-k experts."""
    from model.config import LanguageConfig
    from model.deepseek_v2 import MoEGate

    cfg = LanguageConfig()
    assert cfg.norm_topk_prob is False
    gate = MoEGate(cfg)
    gate.weight = mx.random.normal((cfg.n_routed_experts, cfg.hidden_size))

    x = mx.random.normal((1, 4, cfg.hidden_size))
    inds, weights = gate(x)
    assert inds.shape == (1, 4, cfg.num_experts_per_tok)
    scores = mx.softmax(x[0, 0] @ gate.weight.T, axis=-1, precise=True)
    assert abs(float(weights[0, 0].sum()) - float(scores[inds[0, 0]].sum())) < 1e-4


def test_config_from_json_defaults_norm_false():
    from model.config import LanguageConfig

    # checkpoint config.json omits norm_topk_prob -> must default to False
    cfg = LanguageConfig.from_dict({"hidden_size": 1280, "n_routed_experts": 64})
    assert cfg.norm_topk_prob is False


@pytest.mark.skipif(
    not (os.path.exists(WEIGHTS) and os.path.exists(PDF)),
    reason="weights/PDF not present",
)
def test_single_page_ocr():
    import ocr
    from model.loader import load
    from utils.pdf import pdf_to_images

    model, tokenizer = load(os.path.join(ROOT, "mlx-weights"))
    page = pdf_to_images(PDF, dpi=200)[0]
    text = ocr.infer(
        model,
        tokenizer,
        page,
        prompt="<image>Free OCR.",
        image_size=1024,
        max_tokens=256,
    )
    text = ocr.to_markdown(text)
    assert "Attention Is All You Need" in text
    assert "avaswani@google.com" in text
