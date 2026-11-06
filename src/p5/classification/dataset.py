from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from p5.classification.crop import jitter_box, make_crop


def clahe_or_gamma(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=float(rng.uniform(1.5, 3.0)), tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    g = float(np.exp(rng.uniform(np.log(0.5), np.log(2.5))))
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return lut[rgb]


def down_up(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    f = rng.uniform(0.25, 0.75)
    small = img.resize((max(8, int(img.width * f)), max(8, int(img.height * f))), Image.BILINEAR)
    return small.resize(img.size, Image.BILINEAR)


class CropDataset(Dataset):
    def __init__(self, index: pd.DataFrame, crop_dir: Path, size: int, mean, std, train: bool,
                 pad: float = 0.10, jitter: tuple[float, float] = (0.10, 0.05), aug: bool = True,
                 use_full: bool = False, full_dir: Path | None = None, seed: int = 42):
        self.df = index.reset_index(drop=True)
        self.crop_dir, self.full_dir = crop_dir, full_dir
        self.size, self.pad, self.jitter = size, pad, jitter
        self.train, self.aug, self.use_full = train, aug and train, use_full
        self.seed = seed
        self.labels = self.df["class_id"].to_numpy() if "class_id" in self.df else None
        self.randaug = v2.RandAugment(num_ops=2, magnitude=7)
        self.normalize = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])
        self.erase = v2.RandomErasing(p=0.25, value=0)

    def __len__(self) -> int:
        return len(self.df)

    def _rng(self, i: int) -> np.random.Generator:
        info = torch.utils.data.get_worker_info()
        base = torch.initial_seed() if info is None else info.seed
        return np.random.default_rng((base + i) % (2**32))

    def load(self, i: int, rng: np.random.Generator | None = None) -> Image.Image:
        r = self.df.iloc[i]
        iid = int(r["id"])
        if self.use_full:
            img = Image.open(self.full_dir / f"{iid:05d}.jpg").convert("RGB")
            return img.resize((self.size, self.size), Image.BILINEAR)
        img = Image.open(self.crop_dir / f"{iid:05d}.jpg").convert("RGB")
        box = (r["bx1"], r["by1"], r["bx2"], r["by2"])
        if self.train and rng is not None:
            box = jitter_box(box, self.jitter[0], self.jitter[1], rng)
        return make_crop(img, box, self.size, self.pad)

    def __getitem__(self, i: int):
        rng = self._rng(i) if self.train else None
        img = self.load(i, rng)
        if self.aug:
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if rng.random() < 0.2:
                img = down_up(img, rng)
            if rng.random() < 0.2:
                img = Image.fromarray(clahe_or_gamma(np.asarray(img), rng))
        t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
        if self.aug:
            t = self.randaug(t)
        t = self.normalize(t)
        if self.aug:
            t = self.erase(t)
        y = int(self.labels[i]) if self.labels is not None else -1
        return t, y, int(self.df.iloc[i]["id"])
