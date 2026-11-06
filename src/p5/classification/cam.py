from __future__ import annotations

import cv2
import numpy as np


def class_activation_map(features: np.ndarray, fc_weight: np.ndarray, class_id: int, size: tuple[int, int]) -> np.ndarray:
    cam = np.tensordot(fc_weight[class_id], features, axes=(0, 0))
    cam = np.maximum(cam, 0)
    cam = cam / max(float(cam.max()), 1e-8)
    return cv2.resize(cam.astype(np.float32), size, interpolation=cv2.INTER_CUBIC)


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)
