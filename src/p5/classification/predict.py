from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from p5.classification.crop import make_crop
from p5.classification.model import build
from p5.data.split import read_ids


class InputDataset(Dataset):
    def __init__(self, ids: list[int], source, size: int, pad: float, mean, std):
        self.ids, self.source, self.size, self.pad = ids, source, size, pad
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, k: int):
        img = self.source(self.ids[k], self.size, self.pad)
        t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        return (t - self.mean) / self.std, self.ids[k]


def top_boxes(path: Path) -> dict[int, tuple]:
    df = pd.read_csv(path)
    df = df.sort_values("score", ascending=False).drop_duplicates("id")
    return {int(r.id): (r.x1, r.y1, r.x2, r.y2) for r in df.itertuples()}


def make_sources(cache: Path, split: str, det_csv: Path | None):
    index = pd.read_csv(cache / f"index_{'train' if split == 'val' else split}.csv").set_index("id")
    folder = "train" if split == "val" else split
    crop_dir, full_dir = cache / "crops" / folder, cache / "full" / folder
    pred = top_boxes(det_csv) if det_csv and det_csv.exists() else {}

    def gt(i, size, pad):
        r = index.loc[i]
        return make_crop(Image.open(crop_dir / f"{i:05d}.jpg").convert("RGB"), (r.bx1, r.by1, r.bx2, r.by2), size, pad)

    def full(i, size, pad):
        return Image.open(full_dir / f"{i:05d}.jpg").convert("RGB").resize((size, size), Image.BILINEAR)

    def predicted(i, size, pad):
        img = Image.open(full_dir / f"{i:05d}.jpg").convert("RGB")
        if i not in pred:
            return img.resize((size, size), Image.BILINEAR)
        return make_crop(img, pred[i], size, pad)

    return {"gt": gt, "full": full, "pred": predicted}, index


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cache classifier logits for a split and input type")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--inputs", nargs="+", default=["gt", "pred", "full"])
    p.add_argument("--det-dir", type=Path, default=None)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build(state["backbone"], state["num_classes"], pretrained=False)
    model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    amp = torch.bfloat16 if device.type == "cuda" else None
    val_ids = read_ids(args.assets / "splits" / "val_ids.txt")
    args.out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        det_csv = None
        if args.det_dir:
            det_csv = args.det_dir / ("det_ft_val.csv" if split == "val" else "det_ft_test.csv")
        sources, index = make_sources(args.cache, split, det_csv)
        ids = sorted(val_ids) if split == "val" else sorted(index.index.tolist())
        for name in args.inputs:
            ds = InputDataset(ids, sources[name], state["size"], 0.10, state["mean"], state["std"])
            dl = DataLoader(ds, args.batch, num_workers=args.workers, pin_memory=device.type == "cuda")
            logits, flipped, order = [], [], []
            with torch.no_grad():
                for x, i in dl:
                    x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                    with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                        logits.append(model(x).float().cpu())
                        flipped.append(model(torch.flip(x, dims=[3])).float().cpu())
                    order.append(i)
            np.savez_compressed(args.out / f"logits_{split}_{name}.npz", logits=torch.cat(logits).numpy(),
                                logits_flip=torch.cat(flipped).numpy(), ids=torch.cat(order).numpy())
            print(json.dumps({"split": split, "input": name, "n": len(ids)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
