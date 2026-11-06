from __future__ import annotations

import cv2
import numpy as np

FAMILIES = ("gaussian_noise", "motion_blur", "defocus_blur", "lowlight", "contrast", "jpeg_compression", "pixelate", "fog")
HELD_OUT = ("speckle_noise", "gaussian_blur")
SEVERITIES = (1, 3, 5)


def _f(rgb):
    return rgb.astype(np.float32) / 255.0


def _u8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def gaussian_noise(rgb, s, rng):
    c = [0.08, 0.12, 0.18, 0.26, 0.38][s - 1]
    return _u8(_f(rgb) + rng.normal(0, c, rgb.shape))


def speckle_noise(rgb, s, rng):
    c = [0.15, 0.2, 0.35, 0.45, 0.6][s - 1]
    x = _f(rgb)
    return _u8(x + x * rng.normal(0, c, rgb.shape))


def motion_blur(rgb, s, rng):
    k = [5, 9, 13, 17, 23][s - 1] * max(1, max(rgb.shape[:2]) // 256)
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(-45, 45)), 1.0)
    kernel = cv2.warpAffine(kernel, m, (k, k))
    return cv2.filter2D(rgb, -1, kernel / max(kernel.sum(), 1e-6))


def defocus_blur(rgb, s, rng):
    r = [3, 4, 6, 8, 10][s - 1] * max(1, max(rgb.shape[:2]) // 256)
    k = 2 * r + 1
    kernel = np.zeros((k, k), np.float32)
    cv2.circle(kernel, (r, r), r, 1.0, -1)
    return cv2.filter2D(rgb, -1, kernel / kernel.sum())


def gaussian_blur(rgb, s, rng):
    sigma = [1, 2, 3, 4, 6][s - 1] * max(1, max(rgb.shape[:2]) // 256)
    return cv2.GaussianBlur(rgb, (0, 0), sigma)


def lowlight(rgb, s, rng):
    g = [1.5, 2.0, 3.0, 3.5, 4.0][s - 1]
    exposure = [0.6, 0.45, 0.3, 0.22, 0.15][s - 1]
    x = (_f(rgb) ** g) * exposure
    peak = [120, 80, 50, 35, 25][s - 1]
    x = rng.poisson(x * peak) / peak
    return _u8(x + rng.normal(0, 0.01 * s, rgb.shape))


def contrast(rgb, s, rng):
    c = [0.4, 0.3, 0.2, 0.1, 0.05][s - 1]
    x = _f(rgb)
    m = x.mean(axis=(0, 1), keepdims=True)
    return _u8((x - m) * c + m)


def jpeg_compression(rgb, s, rng):
    q = [25, 18, 15, 10, 7][s - 1]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def pixelate(rgb, s, rng):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][s - 1]
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (max(1, int(w * c)), max(1, int(h * c))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def fog(rgb, s, rng):
    a, smooth = [(0.25, 3), (0.35, 3), (0.5, 2.5), (0.6, 2), (0.7, 1.7)][s - 1]
    h, w = rgb.shape[:2]
    noise = rng.random((max(4, h // 16), max(4, w // 16))).astype(np.float32)
    noise = cv2.GaussianBlur(cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), smooth * 8)
    noise = (noise - noise.min()) / max(noise.max() - noise.min(), 1e-6)
    x = _f(rgb)
    return _u8(x * (1 - a) + a * (0.6 + 0.4 * noise[..., None]))


FUNCS = {f.__name__: f for f in (gaussian_noise, speckle_noise, motion_blur, defocus_blur, gaussian_blur, lowlight,
                                 contrast, jpeg_compression, pixelate, fog)}


def corrupt(rgb: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    return FUNCS[name](rgb, severity, np.random.default_rng(seed))


def occlude(rgb: np.ndarray, box, fraction: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x1, y1, x2, y2 = (int(v) for v in box)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    side_w, side_h = int(bw * np.sqrt(fraction)), int(bh * np.sqrt(fraction))
    ox = x1 + int(rng.integers(0, max(1, bw - side_w)))
    oy = y1 + int(rng.integers(0, max(1, bh - side_h)))
    out = rgb.copy()
    out[oy:oy + side_h, ox:ox + side_w] = 128
    return out


def downsample(rgb: np.ndarray, box, side: int) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in box)
    region = rgb[y1:y2, x1:x2]
    if region.size == 0:
        return rgb
    s = side / max(region.shape[:2])
    small = cv2.resize(region, (max(1, int(region.shape[1] * s)), max(1, int(region.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = rgb.copy()
    out[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    return out
