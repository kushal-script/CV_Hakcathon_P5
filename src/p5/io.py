from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

MAX_PIXELS = 50_000_000
MIN_SIDE = 32
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class InputError(ValueError):
    pass


@dataclass
class ImageIn:
    image: Image.Image
    rgb: np.ndarray
    original_size: tuple[int, int]
    scale: float


def load_image(data: bytes, max_side: int = 1600) -> ImageIn:
    try:
        im = Image.open(io.BytesIO(data))
        im.seek(0)
        im = ImageOps.exif_transpose(im)
    except (UnidentifiedImageError, OSError) as e:
        raise InputError("This file is not an image we can read. Please upload a JPG, PNG, WebP or HEIC photo.") from e
    except Image.DecompressionBombError as e:
        raise InputError("This image is too large. Please upload a photo under 50 megapixels.") from e
    if min(im.size) < MIN_SIDE:
        raise InputError(f"This image is too small ({im.width}×{im.height}). Please upload at least {MIN_SIDE} px per side.")
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    original = im.size
    s = min(1.0, max_side / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return ImageIn(im, np.asarray(im), original, s)
