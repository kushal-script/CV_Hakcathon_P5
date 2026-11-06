from __future__ import annotations

import cv2
import numpy as np

from p5.types import Quality

DEFAULTS = {"dark": 0.30, "low_contrast": 0.12, "cast": 0.12, "noisy": 0.035, "blurry": 60.0}


def noise_sigma(gray: np.ndarray) -> float:
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    resp = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    return float(np.sqrt(np.pi / 2) * np.mean(np.abs(resp)) / 6.0)


def analyse(rgb: np.ndarray, th: dict | None = None, max_side: int = 512) -> Quality:
    th = {**DEFAULTS, **(th or {})}
    s = min(1.0, max_side / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    f = small.astype(np.float32) / 255.0
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    means = f.reshape(-1, 3).mean(0)
    cast = float(np.abs(means - means.mean()).max() / max(means.mean(), 1e-3))
    gray = (luma * 255).astype(np.uint8)
    q = Quality(
        luma_mean=float(luma.mean()), luma_p5=float(np.percentile(luma, 5)), luma_p95=float(np.percentile(luma, 95)),
        rms_contrast=float(luma.std()), cast=cast, noise_sigma=noise_sigma(gray) / 255.0,
        sharpness=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
    )
    if q.luma_mean < th["dark"]:
        q.flags.append("dark")
    if q.rms_contrast < th["low_contrast"]:
        q.flags.append("low_contrast")
    if q.cast > th["cast"]:
        q.flags.append("cast")
    if q.noise_sigma > th["noisy"]:
        q.flags.append("noisy")
    if q.sharpness < th["blurry"]:
        q.flags.append("blurry")
    return q
