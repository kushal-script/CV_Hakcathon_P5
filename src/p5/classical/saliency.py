from __future__ import annotations

import cv2
import numpy as np

from p5.classical.refine import largest_contour, refine
from p5.types import Roi


def saliency_map(rgb: np.ndarray, side: int = 128) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    f = np.fft.fft2(small)
    log_amp, phase = np.log(np.abs(f) + 1e-8), np.angle(f)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal, (9, 9), 2.5)
    sal = (sal - sal.min()) / max(sal.max() - sal.min(), 1e-8)
    return cv2.resize(sal, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def classical_roi(rgb: np.ndarray) -> Roi:
    h, w = rgb.shape[:2]
    sal = (saliency_map(rgb) * 255).astype(np.uint8)
    _, binary = cv2.threshold(sal, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    c = largest_contour(binary)
    if c is None or cv2.contourArea(c) < 0.02 * w * h:
        box = (0.1 * w, 0.1 * h, 0.9 * w, 0.9 * h)
    else:
        x, y, bw, bh = cv2.boundingRect(c)
        box = (float(x), float(y), float(x + bw), float(y + bh))
    return refine(rgb, box, source="saliency")
