import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def re_match(text):
    ref_pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(ref_pattern, text, re.DOTALL)

    det_pattern = r"(<\|det\|>\s*([A-Za-z_][\w-]*)\s*(\[[^\]]+\])\s*<\|/det\|>)"
    for full_match, label, box in re.findall(det_pattern, text, re.DOTALL):
        matches.append((full_match, label, box))

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if a_match[1].strip() == "image" or "<|ref|>image<|/ref|>" in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_str = ref_text[2]
        cor_list = json.loads(cor_str)
        if cor_list and isinstance(cor_list[0], (int, float)):
            cor_list = [cor_list]
    except Exception:
        return None
    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, output_path, image_prefix=""):
    image_width, image_height = image.size

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (
                    np.random.randint(0, 200),
                    np.random.randint(0, 200),
                    np.random.randint(0, 255),
                )
                color_a = color + (20,)

                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == "image":
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            os.makedirs(f"{output_path}/images", exist_ok=True)
                            cropped.save(
                                f"{output_path}/images/{image_prefix}{img_idx}.jpg"
                            )
                        except Exception:
                            pass
                        img_idx += 1

                    try:
                        if label_type == "title":
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )

                        text_x = x1
                        text_y = max(0, y1 - 15)

                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle(
                            [text_x, text_y, text_x + text_width, text_y + text_height],
                            fill=(255, 255, 255, 30),
                        )
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except Exception:
                        pass
        except Exception:
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_path, image_prefix=""):
    result_image = draw_bounding_boxes(
        image, ref_texts, output_path, image_prefix=image_prefix
    )
    return result_image
