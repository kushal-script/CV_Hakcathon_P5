from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

AUTO_NOTE = "Three-wheelers are not in the car catalogues, so no make or model is given."


class VehicleType:
    def __init__(self, path: Path):
        d = json.loads(Path(path).read_text())
        self.w = np.asarray(d["coef"], dtype=np.float32)
        self.b = float(d["intercept"])
        self.threshold = float(d["threshold"])

    def autorickshaw(self, emb: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-(float(np.dot(emb, self.w)) + self.b))))


def load(models: Path) -> VehicleType | None:
    f = Path(models) / "vehicle_type.json"
    return VehicleType(f) if f.exists() else None


def crop_quality(crop: np.ndarray, pad: float) -> dict:
    side = max(crop.shape[:2]) / (1 + 2 * pad)
    f = crop.astype(np.float32) / 255.0
    luma = float(np.median(0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]))
    s = 224 / max(crop.shape[:2])
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    sharp = float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())
    return {"side_px": round(side, 1), "luma": round(luma, 3), "sharpness": round(sharp, 1)}


def unnamed_reason(q: dict, cfg: dict) -> str | None:
    if q["side_px"] < cfg["name_min_side"]:
        return f"too small in the frame to name the make or model ({q['side_px']:.0f} px across, naming needs {cfg['name_min_side']})"
    if q["luma"] < cfg["name_min_luma"]:
        return "too dark to name the make or model"
    if q["sharpness"] < cfg["name_min_sharpness"]:
        return "too blurred to name the make or model"
    return None
