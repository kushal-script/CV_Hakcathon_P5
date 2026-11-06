from __future__ import annotations

import cv2
import numpy as np

COLOURS = {"class": (46, 160, 67), "family": (46, 160, 67), "make": (219, 154, 4), "abstain": (219, 154, 4),
           "ood_make": (207, 34, 46), "ood": (207, 34, 46), "secondary": (110, 118, 129)}


def draw_box(rgb: np.ndarray, box, text: str, colour, below: bool = False) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    t = max(2, int(round(max(rgb.shape[:2]) / 400)))
    cv2.rectangle(rgb, (x1, y1), (x2, y2), colour, t)
    scale = max(0.5, max(rgb.shape[:2]) / 1400)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, t - 1))
    ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 10
    if below:
        ty = y2 + th + 8 if y2 + th + 12 < rgb.shape[0] else y2 - 6
    cv2.rectangle(rgb, (x1, ty - th - 6), (x1 + tw + 8, ty + base - 2), colour, -1)
    cv2.putText(rgb, text, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, t - 1), cv2.LINE_AA)


def fit(rgb: np.ndarray, max_side: int) -> np.ndarray:
    s = min(1.0, max_side / max(rgb.shape[:2]))
    return rgb if s == 1 else cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def jpeg(rgb: np.ndarray, max_side: int = 512, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fit(rgb, max_side), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, contour, colour=(46, 160, 67)) -> np.ndarray:
    out = rgb.copy()
    tint = np.zeros_like(rgb)
    tint[mask > 0] = colour
    out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)
    if contour is not None:
        cv2.drawContours(out, [contour.astype(np.int32)], -1, colour, max(2, max(rgb.shape[:2]) // 300))
    return out


def edges_panel(edges: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)


def hsv_hist_panel(hist: np.ndarray, width: int = 512, height: int = 160) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, np.uint8)
    h = hist.reshape(8, 4, 4)
    n = h.size
    bar = width / n
    top = max(float(h.max()), 1e-6)
    k = 0
    for hi in range(8):
        for si in range(4):
            for vi in range(4):
                hsv = np.uint8([[[int((hi + 0.5) * 180 / 8), int((si + 0.5) * 256 / 4), int((vi + 0.5) * 256 / 4)]]])
                colour = tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])
                bh = int((h[hi, si, vi] / top) * (height - 10))
                cv2.rectangle(canvas, (int(k * bar), height - bh), (int((k + 1) * bar) - 1, height), colour, -1)
                k += 1
    return canvas
