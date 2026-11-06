from __future__ import annotations

import cv2
import numpy as np

HUES = ((10, "red"), (22, "orange"), (34, "yellow"), (85, "green"), (128, "blue"), (160, "purple"), (180, "red"))


def vehicle_colour(crop: np.ndarray) -> dict:
    h, w = crop.shape[:2]
    body = crop[int(0.35 * h):int(0.8 * h), int(0.15 * w):int(0.85 * w)]
    if body.size == 0:
        body = crop
    hsv = cv2.cvtColor(body, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
    bright = hsv[hsv[:, 2] > 45]
    px = bright if len(bright) > 0.25 * len(hsv) else hsv
    s, v = float(np.median(px[:, 1])), float(np.median(px[:, 2]))
    if s < 50 or v < 50:
        name = "white" if v > 190 else "silver" if v > 140 else "grey" if v > 75 else "black"
        return {"name": name, "saturation": s, "value": v}
    chroma = px[px[:, 1] >= 50]
    hue = float(np.median(chroma[:, 0])) if len(chroma) else 0.0
    name = next(n for limit, n in HUES if hue < limit)
    if name in ("red", "orange") and v < 110:
        name = "maroon" if name == "red" else "brown"
    return {"name": name, "saturation": s, "value": v, "hue": hue}
