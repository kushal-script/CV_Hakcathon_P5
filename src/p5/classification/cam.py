from __future__ import annotations

import cv2
import numpy as np


def class_activation_map(features: np.ndarray, fc_weight: np.ndarray, class_id: int, size: tuple[int, int]) -> np.ndarray:
    cam = np.tensordot(fc_weight[class_id], features, axes=(0, 0))
    cam = np.maximum(cam, 0)
    cam = cam / max(float(cam.max()), 1e-8)
    return cv2.resize(cam.astype(np.float32), size, interpolation=cv2.INTER_CUBIC)


def overlay(rgb: np.ndarray, cam: np.ndarray, strength: float = 0.75) -> np.ndarray:
    c = np.clip(cam, 0, 1)
    heat = cv2.cvtColor(cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB).astype(np.float32)
    base = rgb.astype(np.float32) * (0.55 + 0.45 * (1 - c[..., None]))
    a = (strength * c)[..., None]
    return np.clip(base * (1 - a) + heat * a, 0, 255).astype(np.uint8)
