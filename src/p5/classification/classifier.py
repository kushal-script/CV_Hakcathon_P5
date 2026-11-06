from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from p5.classification.crop import make_crop


@dataclass
class DeepOut:
    logits: np.ndarray
    features: np.ndarray
    crop: np.ndarray


class Classifier:
    def __init__(self, onnx_path: Path, meta_path: Path, threads: int = 2, size: int | None = None):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.meta = json.loads(meta_path.read_text())
        self.size = size or self.meta["size"]
        self.pad = self.meta["pad"]
        self.mean = np.array(self.meta["mean"], np.float32)[:, None, None]
        self.std = np.array(self.meta["std"], np.float32)[:, None, None]

    def prepare(self, image: Image.Image, box) -> np.ndarray:
        crop = make_crop(image, box, self.size, self.pad)
        return np.asarray(crop)

    def run(self, crops: list[np.ndarray], flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack([c.transpose(2, 0, 1) for c in crops]).astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        logits, feats = self.sess.run(None, {"input": x})
        if flip:
            lf, _ = self.sess.run(None, {"input": x[..., ::-1].copy()})
            logits = (logits + lf) / 2
        return logits, feats

    def __call__(self, image: Image.Image, box, flip: bool = False) -> DeepOut:
        crop = self.prepare(image, box)
        logits, feats = self.run([crop], flip)
        return DeepOut(logits[0], feats[0], crop)
