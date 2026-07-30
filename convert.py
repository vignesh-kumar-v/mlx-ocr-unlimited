"""Convert HuggingFace DeepSeek-OCR (baidu/Unlimited-OCR) weights to MLX.

Two layout changes are applied for efficient MLX inference:
  * Per-expert MoE weights (``...mlp.experts.{e}.{gate,up,down}_proj.weight``)
    are stacked into ``...mlp.switch_mlp.{gate,up,down}_proj.weight`` of shape
    ``[n_experts, out, in]`` for the batched ``gather_mm`` MoE.
  * Conv2d weights are transposed OIHW -> OHWI for ``mlx.nn.Conv2d``.
"""

import argparse
import os
import re
import shutil
from collections import defaultdict

import mlx.core as mx
from safetensors import safe_open

_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
_DTYPES = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}


def convert(hf_path, mlx_path, dtype="fp16"):
    os.makedirs(mlx_path, exist_ok=True)
    target = _DTYPES[dtype]

    st = os.path.join(hf_path, "model-00001-of-000001.safetensors")
    if not os.path.exists(st):
        st = os.path.join(hf_path, "model.safetensors")

    weights = {}
    experts = defaultdict(dict)  # (mlp_prefix, proj) -> {expert_idx: array}

    # The checkpoint is bfloat16, which numpy can't represent, so load via torch.
    def load(f, key):
        return mx.array(f.get_tensor(key).float().numpy())

    with safe_open(st, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "rotary_emb.inv_freq" in key:
                continue

            m = _EXPERT_RE.match(key)
            if m:
                mlp_prefix, e_idx, proj = m.group(1), int(m.group(2)), m.group(3)
                experts[(mlp_prefix, proj)][e_idx] = load(f, key)
                continue

            arr = load(f, key)
            if key.endswith(".weight") and arr.ndim == 4:  # Conv2d OIHW -> OHWI
                arr = arr.transpose(0, 2, 3, 1)
            weights[key] = arr.astype(target)

    # Stack experts into SwitchLinear layout [n_experts, out, in].
    for (mlp_prefix, proj), by_idx in experts.items():
        stacked = mx.stack([by_idx[i] for i in range(len(by_idx))], axis=0)
        weights[f"{mlp_prefix}.switch_mlp.{proj}.weight"] = stacked.astype(target)

    out_file = os.path.join(mlx_path, "weights.safetensors")
    mx.save_safetensors(out_file, weights)

    # Copy config + tokenizer so the MLX directory is self-contained.
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        src = os.path.join(hf_path, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(mlx_path, name))

    print(f"Converted {len(weights)} tensors ({dtype}) -> {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert DeepSeek-OCR weights to MLX")
    p.add_argument("--hf-path", required=True, help="HuggingFace model directory")
    p.add_argument("--mlx-path", default="mlx-weights", help="Output directory")
    p.add_argument(
        "--dtype", default="fp16", choices=list(_DTYPES), help="Storage dtype"
    )
    args = p.parse_args()
    convert(args.hf_path, args.mlx_path, args.dtype)
