"""DeepSeek-OCR on Apple MLX — command-line interface and Python API.

Examples
--------
Single image (Free OCR):
    python ocr.py --image doc.png --prompt "<image>Free OCR."

High-resolution single image (crop / "Gundam" tiling):
    python ocr.py --image dense.png --prompt "<image>Free OCR." --crop

Multi-page PDF -> markdown:
    python ocr.py --pdf paper.pdf --prompt "<image> Multi page parsing." --output out.md

Grounding / layout with rendered boxes:
    python ocr.py --image doc.png --prompt "<image><|grounding|>OCR this image." --save-results out/
"""

import argparse
import math
import os
import time

import mlx.core as mx
from PIL import ImageOps

from inference.generate import generate
from model.loader import load
from processing.image import BasicImageTransform, dynamic_preprocess, load_image
from utils.pdf import pdf_to_images
from utils.visualization import process_image_with_refs, re_match

IMAGE_TOKEN_ID = 128815
_MEAN = (0.5, 0.5, 0.5)
_PAD = tuple(int(x * 255) for x in _MEAN)


def _transform():
    return BasicImageTransform(mean=_MEAN, std=_MEAN, normalize=True)


def _image_tokens(n):
    """One image's placeholder token block: n rows of (n patches + newline) + view sep."""
    return ([IMAGE_TOKEN_ID] * n + [IMAGE_TOKEN_ID]) * n + [IMAGE_TOKEN_ID]


def infer(
    model,
    tokenizer,
    image,
    prompt="<image>Free OCR.",
    crop=False,
    base_size=1024,
    image_size=640,
    dtype=None,
    **gen_kwargs,
):
    """OCR a single PIL image. ``crop`` enables high-res tiling."""
    if dtype is None:
        dtype = model.lm_head.weight.dtype
    tfm = _transform()
    patch_size, downsample = 16, 4
    before, after = (prompt.split("<image>") + [""])[:2]

    tokens = tokenizer.encode(before, add_special_tokens=False)
    seq_mask = [False] * len(tokens)

    w_crop, h_crop = 1, 1
    crop_tiles = []
    if crop and (image.size[0] > 640 or image.size[1] > 640):
        crop_tiles, (w_crop, h_crop) = dynamic_preprocess(image, image_size=image_size)

    global_view = ImageOps.pad(image, (base_size, base_size), color=_PAD)
    images_ori = mx.stack([tfm(global_view).astype(dtype)], axis=0)
    images_crop = (
        mx.stack([tfm(t).astype(dtype) for t in crop_tiles], axis=0)
        if crop_tiles
        else mx.zeros((1, 3, base_size, base_size), dtype=dtype)
    )

    nq = math.ceil((image_size // patch_size) / downsample)
    nqb = math.ceil((base_size // patch_size) / downsample)
    img_tokens = _image_tokens(nqb)
    if w_crop > 1 or h_crop > 1:
        img_tokens += ([IMAGE_TOKEN_ID] * (nq * w_crop) + [IMAGE_TOKEN_ID]) * (
            nq * h_crop
        )
    tokens += img_tokens
    seq_mask += [True] * len(img_tokens)

    tail = tokenizer.encode(after, add_special_tokens=False)
    tokens += tail
    seq_mask += [False] * len(tail)

    input_ids = mx.array([[0] + tokens], dtype=mx.int32)
    seq_mask = mx.array([[False] + seq_mask], dtype=mx.bool_)
    spatial = mx.array([[w_crop, h_crop]], dtype=mx.int32)

    gen, _ = generate(
        model,
        input_ids,
        images=[(images_crop, images_ori)],
        images_seq_mask=seq_mask,
        images_spatial_crop=spatial,
        eos_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )
    return _decode(tokenizer, gen)


def infer_multi(
    model,
    tokenizer,
    images,
    prompt="<image> Multi page parsing.",
    image_size=1024,
    dtype=None,
    **gen_kwargs,
):
    """OCR multiple pages in a single sequence (produces <PAGE> separators)."""
    if dtype is None:
        dtype = model.lm_head.weight.dtype
    tfm = _transform()
    patch_size, downsample = 16, 4
    nq = math.ceil((image_size // patch_size) / downsample)
    before, after = (prompt.split("<image>") + [""])[:2]

    tokens = tokenizer.encode(before, add_special_tokens=False)
    seq_mask = [False] * len(tokens)
    images_list, spatial = [], []
    for image in images:
        if image_size <= 640:
            image = image.resize((image_size, image_size))
        global_view = ImageOps.pad(image, (image_size, image_size), color=_PAD)
        images_list.append(tfm(global_view).astype(dtype))
        spatial.append([1, 1])
        block = _image_tokens(nq)
        tokens += block
        seq_mask += [True] * len(block)

    tail = tokenizer.encode(after, add_special_tokens=False)
    tokens += tail
    seq_mask += [False] * len(tail)

    input_ids = mx.array([[0] + tokens], dtype=mx.int32)
    seq_mask = mx.array([[False] + seq_mask], dtype=mx.bool_)
    images_ori = mx.stack(images_list, axis=0)
    dummy_crop = mx.zeros((1, 3, image_size, image_size), dtype=dtype)

    gen, _ = generate(
        model,
        input_ids,
        images=[(dummy_crop, images_ori)],
        images_seq_mask=seq_mask,
        images_spatial_crop=mx.array(spatial, dtype=mx.int32),
        eos_token_id=tokenizer.eos_token_id,
        **gen_kwargs,
    )
    return _decode(tokenizer, gen)


def _decode(tokenizer, gen):
    text = tokenizer.decode(gen, skip_special_tokens=False)
    for stop in ("<｜end▁of▁sentence｜>", "\n\n<|end_of_text|>"):
        if text.endswith(stop):
            text = text[: -len(stop)]
    return text.strip()


def to_markdown(text, save_dir=None, images=None):
    """Strip grounding tags to clean markdown; optionally render boxed crops."""
    pages = text.split("<PAGE>")
    multi = len(pages) > 1
    chunks = pages[1:] if multi else [text]
    out = []
    for idx, page in enumerate(chunks):
        page = page.strip()
        matches_ref, matches_images, matches_other = re_match(page)
        if save_dir is not None and images is not None and idx < len(images):
            prefix = f"page_{idx}_" if multi else ""
            result = process_image_with_refs(
                images[idx].copy(), matches_ref, save_dir, image_prefix=prefix
            )
            result.save(os.path.join(save_dir, f"result_with_boxes_{idx}.jpg"))
        for i, m in enumerate(matches_images):
            page = page.replace(
                m, f"![](images/{'page_%d_' % idx if multi else ''}{i}.jpg)\n"
            )
        for m in matches_other:
            page = (
                page.replace(m, "")
                .replace("\\coloneqq", ":=")
                .replace("\\eqqcolon", "=:")
            )
        out.append(page)
    return ("<PAGE>\n" + "\n<PAGE>\n".join(out)) if multi else out[0]


def main():
    p = argparse.ArgumentParser(description="DeepSeek-OCR (MLX)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a single image")
    src.add_argument("--pdf", help="Path to a PDF (multi-page)")
    p.add_argument("--prompt", default=None, help="Prompt with an <image> placeholder")
    p.add_argument("--model", default="mlx-weights", help="MLX dir or HF repo id")
    p.add_argument("--output", help="Write result markdown to this file")
    p.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    p.add_argument("--crop", action="store_true", help="High-res tiling (single image)")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-repeat-ngram-size", type=int, default=0)
    p.add_argument("--ngram-window", type=int, default=0)
    p.add_argument(
        "--save-results", metavar="DIR", help="Render bounding boxes into DIR"
    )
    p.add_argument("--dpi", type=int, default=300, help="PDF rasterization DPI")
    args = p.parse_args()

    t0 = time.time()
    model, tokenizer = load(args.model, dtype=args.dtype)
    dtype = model.lm_head.weight.dtype
    print(f"Loaded model in {time.time() - t0:.1f}s ({dtype})")

    gen_kwargs = dict(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        ngram_window=args.ngram_window,
        verbose=True,
    )

    if args.pdf:
        prompt = args.prompt or "<image> Multi page parsing."
        image_size = args.image_size or 1024
        images = [
            load_image(pth).convert("RGB")
            for pth in pdf_to_images(args.pdf, dpi=args.dpi)
        ]
        # sensible default for long multi-page docs
        if args.no_repeat_ngram_size == 0:
            gen_kwargs.update(no_repeat_ngram_size=35, ngram_window=1024)
        raw = infer_multi(
            model,
            tokenizer,
            images,
            prompt=prompt,
            image_size=image_size,
            dtype=dtype,
            **gen_kwargs,
        )
    else:
        prompt = args.prompt or "<image>Free OCR."
        image_size = args.image_size or 640
        images = [load_image(args.image).convert("RGB")]
        raw = infer(
            model,
            tokenizer,
            images[0],
            prompt=prompt,
            crop=args.crop,
            image_size=image_size,
            dtype=dtype,
            **gen_kwargs,
        )

    if args.save_results:
        os.makedirs(os.path.join(args.save_results, "images"), exist_ok=True)
    result = to_markdown(raw, save_dir=args.save_results, images=images)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\nWrote {args.output}")
    else:
        print("\n" + result)


if __name__ == "__main__":
    main()
