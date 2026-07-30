"""Robust model + tokenizer loading for DeepSeek-OCR.

Accepts a converted MLX directory (from ``convert.py``), a local HuggingFace
checkpoint, or a HuggingFace repo id (downloaded + converted on demand).
"""

import os

import mlx.core as mx
from transformers import PreTrainedTokenizerFast

from .config import UnlimitedOCRConfig
from .unlimited_ocr import UnlimitedOCRForCausalLM

HF_REPO = "baidu/Unlimited-OCR"
_DTYPES = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}


def _resolve_mlx_dir(model_path):
    """Return a directory containing weights.safetensors + config.json + tokenizer.json."""
    if model_path and os.path.isdir(model_path):
        hf_dir = model_path
    else:
        from huggingface_hub import snapshot_download

        hf_dir = snapshot_download(model_path or HF_REPO)

    # Already an MLX conversion (local dir or a hosted repo of converted weights).
    if os.path.exists(os.path.join(hf_dir, "weights.safetensors")):
        return hf_dir

    mlx_dir = os.path.join(hf_dir, "mlx")
    if not os.path.exists(os.path.join(mlx_dir, "weights.safetensors")):
        from convert import convert

        convert(hf_dir, mlx_dir)
    return mlx_dir


def load(model_path="mlx-weights", dtype=None):
    """Load the OCR model and tokenizer. ``dtype`` optionally recasts weights."""
    mlx_dir = _resolve_mlx_dir(model_path)

    config = UnlimitedOCRConfig.from_json(os.path.join(mlx_dir, "config.json"))
    model = UnlimitedOCRForCausalLM(config)
    model.load_weights(os.path.join(mlx_dir, "weights.safetensors"))  # strict
    if dtype is not None:
        model.set_dtype(_DTYPES[dtype])
    mx.eval(model.parameters())
    model.eval()

    tokenizer = load_tokenizer(mlx_dir)
    return model, tokenizer


def load_tokenizer(path):
    # AutoTokenizer misroutes this checkpoint to LlamaTokenizer and corrupts the
    # byte-level BPE; load the fast tokenizer from tokenizer.json directly.
    return PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)
