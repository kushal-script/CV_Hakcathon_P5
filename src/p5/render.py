from __future__ import annotations

import cv2
import numpy as np

COLOURS = {"class": (47, 90, 67), "family": (47, 90, 67), "make": (154, 106, 20), "abstain": (154, 106, 20),
           "ood_make": (155, 58, 38), "ood": (155, 58, 38), "secondary": (87, 83, 78)}


def draw_box(rgb: np.ndarray, box, text: str, colour, below: bool = False) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    t = max(2, int(round(max(rgb.shape[:2]) / 400)))
    cv2.rectangle(rgb, (x1, y1), (x2, y2), colour, t)
    scale = max(0.5, max(rgb.shape[:2]) / 1400)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, t - 1))
    ty = y1 - 6 if y1 - th - 10 > 0 else y1 + th + 10
    if below:
        ty = y2 + th + 8 if y2 + th + 12 < rgb.shape[0] else y2 - 6
    lx = max(0, min(x1, rgb.shape[1] - tw - 8))
    cv2.rectangle(rgb, (lx, ty - th - 6), (lx + tw + 8, ty + base - 2), colour, -1)
    cv2.putText(rgb, text, (lx + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, t - 1), cv2.LINE_AA)


def fit(rgb: np.ndarray, max_side: int) -> np.ndarray:
    s = min(1.0, max_side / max(rgb.shape[:2]))
    return rgb if s == 1 else cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def jpeg(rgb: np.ndarray, max_side: int = 512, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(fit(rgb, max_side), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def mask_overlay(rgb: np.ndarray, mask: np.ndarray, contour, colour=(47, 90, 67)) -> np.ndarray:
    out = rgb.copy()
    tint = np.zeros_like(rgb)
    tint[mask > 0] = colour
    out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)
    if contour is not None:
        cv2.drawContours(out, [contour.astype(np.int32)], -1, colour, max(2, max(rgb.shape[:2]) // 300))
    return out


def edges_panel(edges: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2RGB)


HUE_LABELS = ("red", "yellow", "green", "teal", "cyan", "blue", "violet", "magenta", "neutral")


def hsv_hist_panel(hist: np.ndarray, width: int = 512, height: int = 170) -> np.ndarray:
    paper, ink, faint = (251, 250, 247), (29, 28, 26), (201, 196, 184)
    canvas = np.full((height, width, 3), paper, np.uint8)
    h = hist.reshape(8, 4, 4)
    chroma = h[:, 1:, 1:].sum(axis=(1, 2))
    grey = h[:, 0, :].sum() + h[:, :, 0].sum() - h[:, 0, 0].sum()
    bars = list(chroma) + [grey]
    top = max(max(bars), 1e-6)
    plot_h, base = height - 34, height - 26
    n = len(bars)
    bw = width / n
    for i, v in enumerate(bars):
        bh = int(v / top * plot_h)
        x0, x1 = int(i * bw + 6), int((i + 1) * bw - 6)
        cv2.rectangle(canvas, (x0, base - bh), (x1, base), ink if i < 8 else faint, -1)
    cv2.line(canvas, (0, base), (width, base), faint, 1)
    for i, name in enumerate(HUE_LABELS[:n]):
        size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        cv2.putText(canvas, name, (int(i * bw + (bw - size[0]) / 2), height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (123, 119, 110), 1, cv2.LINE_AA)
    return canvas
