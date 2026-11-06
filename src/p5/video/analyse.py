from __future__ import annotations

import base64
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from p5 import render
from p5.classical.colour import vehicle_colour
from p5.classification.classifier import Classifier
from p5.confidence.policy import Thresholds, decide
from p5.confidence.temperature import softmax_t
from p5.detection.detector import Detector
from p5.taxonomy import Taxonomy
from p5.video.tracker import Track, Tracker, suppress_nested

COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
FINE_GRAINED = {2, 7}


@dataclass
class VideoOptions:
    target_fps: float = 8.0
    max_side: int = 1280
    max_seconds: float = 180.0
    allowed_direction: tuple[float, float] | None = None
    wrong_way_cos: float = -0.3
    min_travel_frac: float = 0.08
    best_crops: int = 3


@dataclass
class _Crops:
    items: list[tuple[float, np.ndarray]] = field(default_factory=list)

    def offer(self, score: float, crop: np.ndarray, k: int) -> None:
        self.items.append((score, crop))
        self.items.sort(key=lambda t: -t[0])
        del self.items[k:]


def embedding(feature_map: np.ndarray) -> np.ndarray:
    v = feature_map.reshape(feature_map.shape[0], -1).mean(1) if feature_map.ndim == 3 else feature_map.mean(0)
    return v / (np.linalg.norm(v) + 1e-9)


def compass(dx: float, dy: float) -> str:
    angle = (math.degrees(math.atan2(dx, -dy)) + 360) % 360
    names = ["up", "up-right", "right", "down-right", "down", "down-left", "left", "up-left"]
    return names[int((angle + 22.5) // 45) % 8]


def jpeg_b64(rgb: np.ndarray, side: int = 160) -> str:
    return base64.b64encode(render.jpeg(rgb, side, 80)).decode()


class VideoAnalyser:
    def __init__(self, coco_detector: Detector, classifier: Classifier, tax: Taxonomy, th: Thresholds):
        self.det, self.clf, self.tax, self.th = coco_detector, classifier, tax, th

    def run(self, path: str, opts: VideoOptions | None = None, progress: Callable[[float], None] | None = None) -> dict:
        opts = opts or VideoOptions()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("This file is not a video we can read. Please upload an MP4, MOV or AVI file.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stride = max(1, round(fps / opts.target_fps))
        limit = int(min(total or 10**9, opts.max_seconds * fps))
        scale = min(1.0, opts.max_side / max(width, height, 1))
        tracker = Tracker(max_age=max(5, int(2 * opts.target_fps)))
        crops: dict[int, _Crops] = {}
        keyframe = None
        keyframes: list[np.ndarray] = []
        marks = {int(limit * f) // stride * stride for f in (0.1, 0.5, 0.9)} if limit < 10**9 else set()
        idx = processed = 0
        while idx < limit:
            ok = cap.grab()
            if not ok:
                break
            if idx % stride == 0:
                ok, bgr = cap.retrieve()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if scale < 1.0:
                    rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                dets = suppress_nested(self.det(rgb, conf=0.15, classes=set(COCO_VEHICLES)))
                active = tracker.update(dets, idx)
                for t in active:
                    if t.cls not in FINE_GRAINED:
                        continue
                    x1, y1, x2, y2 = t.history[-1][1]
                    area = (x2 - x1) * (y2 - y1) / (rgb.shape[0] * rgb.shape[1])
                    if area < 0.004:
                        continue
                    pad = 0.10
                    bw, bh = x2 - x1, y2 - y1
                    c = rgb[int(max(0, y1 - pad * bh)):int(min(rgb.shape[0], y2 + pad * bh)),
                            int(max(0, x1 - pad * bw)):int(min(rgb.shape[1], x2 + pad * bw))]
                    if c.size:
                        crops.setdefault(t.tid, _Crops()).offer(t.history[-1][2] * area, c.copy(), opts.best_crops)
                if keyframe is None or (processed == 5):
                    keyframe = rgb.copy()
                if idx in marks and len(keyframes) < 3:
                    keyframes.append(rgb.copy())
                processed += 1
                if progress and limit:
                    progress(min(0.95, idx / limit))
            idx += 1
        cap.release()
        frame_hw = keyframe.shape[:2] if keyframe is not None else (int(height * scale), int(width * scale))
        tracks = [self._summarise(t, crops.get(t.tid), fps, frame_hw, opts) for t in tracker.all_tracks()]
        tracks = [t for t in tracks if t is not None]
        tracks.sort(key=lambda t: t["first_t"])
        counts = Counter(t["class"] for t in tracks)
        per_minute = Counter(int(t["first_t"] // 60) for t in tracks)
        alerts = [{"type": "wrong_way", "track_id": t["id"], "t": t["first_t"], "label": t["label"]}
                  for t in tracks if t.get("wrong_way")]
        if progress:
            progress(1.0)
        return {
            "meta": {"fps": fps, "frames": total, "width": width, "height": height, "duration_s": total / fps if fps else None,
                     "analysed_s": min(total, limit) / fps if fps else None, "processed_frames": processed, "stride": stride,
                     "frame_size": [frame_hw[1], frame_hw[0]], "allowed_direction": opts.allowed_direction},
            "tracks": tracks,
            "counts": {"by_class": dict(counts), "per_minute": {str(k): v for k, v in sorted(per_minute.items())}, "total": len(tracks)},
            "alerts": alerts,
            "keyframe": jpeg_b64(keyframe, 960) if keyframe is not None else None,
            "_keyframes": keyframes or ([keyframe] if keyframe is not None else []),
        }

    def _summarise(self, t: Track, crops: _Crops | None, fps: float, frame_hw, opts: VideoOptions) -> dict | None:
        hist = t.history
        if len(hist) < 4:
            return None
        cx = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        (x0, y0), (x1, y1) = cx(hist[0][1]), cx(hist[-1][1])
        dx, dy = x1 - x0, y1 - y0
        travel = math.hypot(dx, dy)
        diag = math.hypot(*frame_hw)
        dur = max((hist[-1][0] - hist[0][0]) / fps, 1e-6)
        out = {
            "id": t.tid, "class": COCO_VEHICLES.get(t.cls, str(t.cls)), "first_t": hist[0][0] / fps, "last_t": hist[-1][0] / fps,
            "n_detections": len(hist), "mean_score": float(np.mean([h[2] for h in hist])),
            "boxes": [[round(f / fps, 3), *[round(v, 1) for v in b]] for f, b, _ in hist],
            "direction": {"dx": dx, "dy": dy, "compass": compass(dx, dy) if travel > 0 else "static",
                          "travel_frac": travel / diag, "speed_px_s": travel / dur},
            "label": COCO_VEHICLES.get(t.cls, "vehicle"), "decision": None, "colour": None, "thumbnail": None, "wrong_way": False,
        }
        if opts.allowed_direction and travel / diag >= opts.min_travel_frac:
            ax, ay = opts.allowed_direction
            norm = math.hypot(ax, ay) or 1.0
            cos = (dx * ax + dy * ay) / (travel * norm)
            out["direction"]["cos_to_allowed"] = cos
            out["wrong_way"] = bool(cos <= opts.wrong_way_cos)
        if crops and crops.items:
            best = crops.items[0][1]
            out["thumbnail"] = jpeg_b64(best)
            out["colour"] = vehicle_colour(best)
            logits, embs = [], []
            for _, c in crops.items:
                crop = np.asarray(Image.fromarray(c).resize((self.clf.size, self.clf.size), Image.BILINEAR))
                z, f = self.clf.run([crop])
                logits.append(z[0])
                embs.append(embedding(f[0]))
            mean_logits = np.mean(logits, axis=0)
            e = np.mean(embs, axis=0)
            out["embedding"] = (e / (np.linalg.norm(e) + 1e-9)).astype(np.float32).tolist()
            out["probs"] = softmax_t(mean_logits, self.th.t_d).astype(np.float32).tolist()
            d = decide(mean_logits, self.tax, self.th)
            out["decision"] = {"level": d.level, "label": d.label, "confidence": d.confidence, "make": d.make,
                               "top5": d.top5[:3], "note": d.note}
            colour = out["colour"]["name"]
            out["label"] = f"{colour} {d.label}" if d.level in ("class", "family", "make") else f"{colour} {out['class']}"
        return out
