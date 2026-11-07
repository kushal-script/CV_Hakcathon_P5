from __future__ import annotations

import cv2
import numpy as np

from p5.types import Enhanced, Quality


def gray_world(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    gain = means.mean() / np.maximum(means, 1e-3)
    return np.clip(f * gain, 0, 255).astype(np.uint8)


def clahe_l(rgb: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def auto_gamma(rgb: np.ndarray, target: float = 0.45) -> np.ndarray:
    mean = max(rgb.mean() / 255.0, 1e-3)
    g = float(np.clip(np.log(target) / np.log(mean), 0.3, 1.0))
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return lut[rgb]


def denoise(rgb: np.ndarray, sigma: float, scale: float = 0.8) -> np.ndarray:
    h = float(np.clip(sigma * 255.0 * scale, 3, 30))
    return cv2.fastNlMeansDenoisingColored(rgb, None, h, h, 7, 21)


def enhance(rgb: np.ndarray, q: Quality, clip: float = 2.0, tile: int = 8, nlm_scale: float = 0.8,
            noisy: float = 0.035, gamma_target: float = 0.45) -> Enhanced:
    from p5.classical.quality import noise_sigma
    out, applied = rgb, []
    if "cast" in q.flags:
        out, applied = gray_world(out), applied + ["white_balance"]
    if "dark" in q.flags:
        out, applied = auto_gamma(out, gamma_target), applied + ["gamma"]
    sigma = q.noise_sigma
    if applied and "dark" in q.flags:
        sigma = noise_sigma(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)) / 255.0
    if "noisy" in q.flags or sigma > noisy:
        out, applied = denoise(out, sigma, nlm_scale), applied + ["denoise"]
    if "dark" in q.flags or "low_contrast" in q.flags:
        out, applied = clahe_l(out, clip if "dark" not in q.flags else clip * 0.75, tile), applied + ["clahe"]
    return Enhanced(out, applied)
