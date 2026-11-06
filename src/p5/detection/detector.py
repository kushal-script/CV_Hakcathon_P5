from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    score: float
    cls: int = 0


def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) / 2, (size - nw) / 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[int(round(top - 0.1)):int(round(top - 0.1)) + nh, int(round(left - 0.1)):int(round(left - 0.1)) + nw] = resized
    return canvas, r, (int(round(left - 0.1)), int(round(top - 0.1)))


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    idx = cv2.dnn.NMSBoxes([[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes],
                           scores.astype(float).tolist(), 0.0, iou_thr)
    return [int(i) for i in np.array(idx).reshape(-1)]


class Detector:
    def __init__(self, path: Path, size: int = 640, threads: int = 2, iou: float = 0.7):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.size, self.iou = size, iou

    def __call__(self, rgb: np.ndarray, conf: float = 0.05, classes: set[int] | None = None) -> list[Detection]:
        canvas, r, (dx, dy) = letterbox(rgb, self.size)
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.input: x})[0][0]
        if out.ndim == 2 and out.shape[-1] == 6:
            boxes, scores, cls = out[:, :4], out[:, 4], out[:, 5].astype(int)
            sel = scores >= conf
            if classes is not None:
                sel &= np.isin(cls, list(classes))
            keep = np.nonzero(sel)[0].tolist()
        else:
            pred = out.T if out.shape[0] < out.shape[1] else out
            boxes_c, cls_scores = pred[:, :4], pred[:, 4:]
            if classes is not None:
                mask = np.full(cls_scores.shape[1], -1.0, np.float32)
                mask[[c for c in classes if c < cls_scores.shape[1]]] = 0.0
                cls_scores = np.where(mask >= 0, cls_scores, 0.0)
            cls_all = cls_scores.argmax(1)
            scores = cls_scores.max(1)
            sel = scores >= conf
            cx, cy, bw, bh = boxes_c[sel].T
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
            scores, cls = scores[sel], cls_all[sel]
            keep = nms(boxes, scores, self.iou) if len(scores) else []
        h, w = rgb.shape[:2]
        dets = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            x1, x2 = np.clip([(x1 - dx) / r, (x2 - dx) / r], 0, w)
            y1, y2 = np.clip([(y1 - dy) / r, (y2 - dy) / r], 0, h)
            if x2 - x1 > 2 and y2 - y1 > 2:
                dets.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(scores[i]), int(cls[i])))
        return sorted(dets, key=lambda d: -d.score)
