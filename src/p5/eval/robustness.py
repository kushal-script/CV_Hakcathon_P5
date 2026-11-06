from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from p5.classical.preprocess import enhance
from p5.classical.quality import analyse
from p5.classification.crop import make_crop
from p5.classification.model import build
from p5.data.corrupt import FAMILIES, HELD_OUT, SEVERITIES, corrupt, downsample, occlude


def conditions() -> list[tuple[str, str, int]]:
    out = [("clean", "clean", 0)]
    out += [(f"{f}_s{s}", f, s) for f in FAMILIES + HELD_OUT for s in SEVERITIES]
    out += [(f"resolution_{r}", "resolution", r) for r in (112, 56, 28)]
    out += [(f"occlusion_{int(p * 100)}", "occlusion", int(p * 100)) for p in (0.10, 0.25, 0.50)]
    return out


class CondDataset(Dataset):
    def __init__(self, ids, full_dir: Path, gt: dict, kind: str, value: int, m2: bool):
        self.ids, self.full_dir, self.gt, self.kind, self.value, self.m2 = ids, full_dir, gt, kind, value, m2

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        i = self.ids[k]
        rgb = np.asarray(Image.open(self.full_dir / f"{i:05d}.jpg").convert("RGB"))
        if self.kind == "resolution":
            rgb = downsample(rgb, self.gt[i], self.value)
        elif self.kind == "occlusion":
            rgb = occlude(rgb, self.gt[i], self.value / 100, seed=i)
        elif self.kind != "clean":
            rgb = corrupt(rgb, self.kind, self.value, seed=i)
        applied = []
        if self.m2:
            e = enhance(rgb, analyse(rgb))
            rgb, applied = e.rgb, e.applied
        return rgb, i, ",".join(applied)


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="E14: end-to-end robustness logits under corruptions (no labels read)")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--detector", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=1960)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--m2", action="store_true", help="apply the classical quality-gated enhancement before detection")
    args = p.parse_args(argv)

    from ultralytics import YOLO

    index = pd.read_csv(args.cache / "index_test.csv")
    ids = sorted(index.id.sample(args.n, random_state=42).tolist())
    gt = {}
    for r in index.itertuples():
        s = min(1.0, 512 / max(r.orig_w, r.orig_h))
        gt[int(r.id)] = (r.x1 * s, r.y1 * s, r.x2 * s, r.y2 * s)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "subset_ids.txt").write_text("\n".join(map(str, ids)) + "\n")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build(state["backbone"], state["num_classes"], pretrained=False)
    model.load_state_dict(state["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    mean = torch.tensor(state["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(state["std"], device=device).view(1, 3, 1, 1)
    det = YOLO(str(args.detector))
    size = state["size"]

    def classify(crops):
        x = torch.from_numpy(np.stack([np.asarray(c) for c in crops])).to(device).permute(0, 3, 1, 2).float() / 255.0
        x = ((x - mean) / std).contiguous(memory_format=torch.channels_last)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return model(x).float().cpu().numpy()

    for name, kind, value in conditions():
        if args.only and name not in args.only:
            continue
        tag = f"{name}{'_m2' if args.m2 else ''}"
        out = args.out / f"{tag}.npz"
        if out.exists():
            continue
        dl = DataLoader(CondDataset(ids, args.cache / "full" / "test", gt, kind, value, args.m2), batch_size=64,
                        num_workers=args.workers, collate_fn=collate)
        e2e, gtl, boxes, scores, order, applied = [], [], [], [], [], []
        for imgs, bids, app in dl:
            res = det.predict([im[..., ::-1] for im in imgs], conf=0.05, verbose=False, device=0 if device.type == "cuda" else None)
            crops_p, crops_g = [], []
            for im, i, r in zip(imgs, bids, res):
                pil = Image.fromarray(im)
                if len(r.boxes):
                    k = int(r.boxes.conf.argmax())
                    b, sc = r.boxes.xyxy[k].tolist(), float(r.boxes.conf[k])
                else:
                    b, sc = [0, 0, im.shape[1], im.shape[0]], 0.0
                boxes.append(b)
                scores.append(sc)
                crops_p.append(make_crop(pil, b, size, 0.10))
                crops_g.append(make_crop(pil, gt[i], size, 0.10))
            e2e.append(classify(crops_p))
            gtl.append(classify(crops_g))
            order += bids
            applied += app
        np.savez_compressed(out, ids=np.array(order), logits_e2e=np.concatenate(e2e), logits_gt=np.concatenate(gtl),
                            boxes=np.array(boxes), det_score=np.array(scores), applied=np.array(applied))
        print(json.dumps({"condition": tag, "n": len(order)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
