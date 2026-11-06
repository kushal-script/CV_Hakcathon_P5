from __future__ import annotations

import numpy as np
from PIL import Image

Box = tuple[float, float, float, float]


def pad_box(box: Box, pad: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - pad * bw, x2 + pad * bw
    y1, y2 = y1 - pad * bh, y2 + pad * bh
    x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
    x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0, 0, width, height
    return x1, y1, x2, y2


def jitter_box(box: Box, scale: float, shift: float, rng: np.random.Generator) -> Box:
    x1, y1, x2, y2 = box
    cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    cx += rng.uniform(-shift, shift) * bw
    cy += rng.uniform(-shift, shift) * bh
    bw *= 1 + rng.uniform(-scale, scale)
    bh *= 1 + rng.uniform(-scale, scale)
    return cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2


def make_crop(image: Image.Image, box: Box, size: int, pad: float = 0.10) -> Image.Image:
    w, h = image.size
    region = pad_box(box, pad, w, h)
    return image.crop(region).resize((size, size), Image.BILINEAR)


def cache_crop(image: Image.Image, box: Box, pad: float, max_side: int) -> tuple[Image.Image, Box]:
    w, h = image.size
    x1, y1, x2, y2 = pad_box(box, pad, w, h)
    crop = image.crop((x1, y1, x2, y2))
    s = min(1.0, max_side / max(crop.size))
    if s < 1.0:
        crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.BILINEAR)
    inner = ((box[0] - x1) * s, (box[1] - y1) * s, (box[2] - x1) * s, (box[3] - y1) * s)
    return crop, inner
