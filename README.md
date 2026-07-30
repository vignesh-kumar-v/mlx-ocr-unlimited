# DeepSeek-OCR on Apple MLX

A native [Apple MLX](https://github.com/ml-explore/mlx) port of **DeepSeek-OCR**
(published as [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR)),
a vision-language model for document OCR, layout parsing, and grounding. Runs
on Apple Silicon with no PyTorch/CUDA at inference time.

## Architecture

```
             SAM ViT-B  ─┐
image ──►                ├─► concat(2048) ─► Linear ─► DeepSeek-V2 MoE ─► text
             CLIP ViT-L ─┘                    (1280)      (12 layers)
```

| Component | Details |
|-----------|---------|
| **SAM ViT-B** | 12 layers, 768-dim, windowed + global attention, decomposed relative position bias, conv neck → 1024-dim |
| **CLIP ViT-L** | 24 layers, 1024-dim, quick-GELU; consumes SAM features as patch embeddings |
| **Projector** | Linear 2048 → 1280 over concatenated CLIP + SAM features |
| **DeepSeek-V2 LLM** | 12 layers, MoE with 64 routed experts (top-6) + 2 shared, standard MHA (`use_mla=False`), RoPE, 128-token decode sliding window |

## Install

```bash
pip install -r requirements.txt        # runtime + torch (for conversion)
```

## Weights

Pre-converted fp16 weights are hosted at
[`Vignesh-5756/Unlimited-OCR-mlx-fp16`](https://huggingface.co/Vignesh-5756/Unlimited-OCR-mlx-fp16)
— pass the repo id to `--model` and they are downloaded on first use (no
PyTorch needed):

```bash
python ocr.py --model Vignesh-5756/Unlimited-OCR-mlx-fp16 --image doc.png
```

To convert the original checkpoint yourself (stacks MoE experts, transposes
convolutions; requires torch):

```bash
hf download baidu/Unlimited-OCR --local-dir hf-weights
python convert.py --hf-path hf-weights --mlx-path mlx-weights   # --dtype {fp16,bf16,fp32}
```

The `mlx-weights/` directory is self-contained (weights + config + tokenizer).

## Usage

### CLI

```bash
# Single image
python ocr.py --image doc.png --prompt "<image>Free OCR."

# High-resolution single image (crop / "Gundam" tiling)
python ocr.py --image dense.png --prompt "<image>Free OCR." --crop

# Multi-page PDF → markdown
python ocr.py --pdf paper.pdf --prompt "<image> Multi page parsing." --output out.md

# Layout grounding with rendered bounding boxes
python ocr.py --image doc.png --prompt "<image><|grounding|>OCR this image." --save-results out/
```

Useful flags: `--model` (MLX dir or HF id), `--dtype {bf16,fp16,fp32}`,
`--image-size`, `--max-tokens`, `--temperature`, `--no-repeat-ngram-size`,
`--ngram-window`.

### Python

```python
from model.loader import load
from processing.image import load_image
import ocr

model, tokenizer = load("mlx-weights")
image = load_image("doc.png").convert("RGB")
text = ocr.infer(model, tokenizer, image, prompt="<image>Free OCR.")
print(ocr.to_markdown(text))
```

## Performance

The batched-`gather_mm` MoE, bounded sliding-window KV cache, and fused
`mx.fast.rope` bring decode to **~175 tok/s** on a single image and **~120
tok/s** for multi-page parsing (larger always-attended image context), roughly
**6–8× the initial naive port**.

Fidelity to the reference PyTorch/CUDA output on the 15-page *Attention Is All
You Need* PDF, by compute precision:

| `--dtype` | Memory | Similarity |
|-----------|--------|------------|
| `fp16` (default) | ~6 GB | **0.99** |
| `fp32` | ~12 GB | **0.9998** (exact reproduction) |
| `bf16` | ~6 GB | 0.95 |

fp16 is the default: the extra mantissa (vs bf16) matters for long greedy
generation. Use `--dtype fp32` to reproduce the reference near-exactly; the
remaining fp16 gap is concentrated on one dense numeric table.

## Implementation notes

- **MoE** (`model/moe.py`): experts are stored stacked `[n_experts, out, in]`
  and evaluated with a single `mx.gather_mm` (sorted by expert), not a Python
  loop.
- **Sliding-window cache** (`model/cache.py`): keeps every prompt/image token
  and rotates only a 128-token decode window, matching the reference ring
  buffer while bounding memory.
- **Vision interpolation** (`model/vision_encoder.py`): positional / relative
  position embeddings are resized with cached separable resize matrices
  (analytic linear + PIL-probed bicubic, bit-identical to `torch.F.interpolate`)
  applied to all channels in one matmul.
- **Tokenizer**: load with `PreTrainedTokenizerFast` (via `model.loader.load`).
  `AutoTokenizer` misroutes this checkpoint to `LlamaTokenizer`, which corrupts
  the byte-level BPE (wrong ids on encode, stripped spaces on decode).

## License

MIT
