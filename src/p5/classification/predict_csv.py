from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from p5.classification.dataset import CropDataset
from p5.classification.export import load_model
from p5.classification.train import pick_device, predict


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cache classifier logits for the rows of a split CSV")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--crops", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    device = pick_device("auto")
    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    model, state = load_model(args.ckpt)
    model = model.to(device).to(memory_format=torch.channels_last)
    df = pd.read_csv(args.split_csv)
    df["id"] = np.arange(len(df))
    args.out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        part = df[df.split == split].reset_index(drop=True)
        ds = CropDataset(part, args.crops, state["size"], state["mean"], state["std"], train=False)
        dl = DataLoader(ds, args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
        out = predict(model, dl, device, amp, flip=True)
        np.savez_compressed(args.out / f"{split}_gt.npz", **out, files=part.file.to_numpy())
        print(json.dumps({"split": split, "n": len(part)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
