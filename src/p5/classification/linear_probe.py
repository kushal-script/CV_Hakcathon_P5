from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from p5.classification.model import build, data_config
from p5.classification.predict import InputDataset, make_sources
from p5.data.split import read_ids
from p5.eval.classification import summary
from p5.taxonomy import Taxonomy


@torch.no_grad()
def features(model, ids, source, size, mean, std, device, flip=False, workers=8, batch=64) -> np.ndarray:
    dl = DataLoader(InputDataset(ids, source, size, 0.10, mean, std), batch, num_workers=workers)
    out = []
    for x, _ in dl:
        x = x.to(device)
        if flip:
            x = torch.flip(x, dims=[3])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out.append(model.pool(model.forward_features(x)).float().cpu())
    return torch.cat(out).numpy()


def train_head(xtr, ytr, xva, yva, wd, device, epochs=300, lr=1e-3):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    t = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32, device=device)
    xt, xv = t(xtr), t(xva)
    yt = torch.tensor(ytr, device=device)
    head = nn.Linear(xt.shape[1], 196).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        perm = torch.randperm(len(yt), device=device)
        for k in range(0, len(yt), 1024):
            i = perm[k:k + 1024]
            loss = nn.functional.cross_entropy(head(xt[i]), yt[i], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        return head, (mu, sd), head(xv).cpu().numpy()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Frozen-feature linear probe for a backbone")
    p.add_argument("--backbone", required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--det-dir", type=Path, default=None)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tax = Taxonomy.load(args.assets / "taxonomy")
    model = build(args.backbone, len(tax.names), pretrained=True).to(device).eval()
    cfg = data_config(model)
    index = pd.read_csv(args.cache / "index_train.csv").set_index("id")
    val_ids = sorted(read_ids(args.assets / "splits" / "val_ids.txt"))
    tr_ids = sorted(set(index.index) - set(val_ids))
    src_train, _ = make_sources(args.cache, "val", args.det_dir / "det_ft_val.csv" if args.det_dir else None)
    src_test, test_index = make_sources(args.cache, "test", args.det_dir / "det_ft_test.csv" if args.det_dir else None)
    f = lambda ids, src, flip=False: features(model, ids, src, args.size, cfg["mean"], cfg["std"], device, flip, args.workers)

    xtr = np.concatenate([f(tr_ids, src_train["gt"]), f(tr_ids, src_train["gt"], True)])
    ytr = np.concatenate([index.loc[tr_ids, "class_id"].to_numpy()] * 2)
    xva, yva = f(val_ids, src_train["gt"]), index.loc[val_ids, "class_id"].to_numpy()
    xva_pred = f(val_ids, src_train["pred"]) if args.det_dir else None
    test_ids = sorted(test_index.index.tolist())
    xte = {name: f(test_ids, src_test[name]) for name in (("gt", "pred") if args.det_dir else ("gt",))}

    grid = {}
    for wd in (1e-4, 1e-2, 0.1, 0.5):
        _, _, zv = train_head(xtr, ytr, xva, yva, wd, device)
        grid[wd] = summary(zv, yva, tax)["top1"]
        print(json.dumps({"wd": wd, "val_top1": grid[wd]}), flush=True)
    wd = max(grid, key=grid.get)
    head, (mu, sd), zv = train_head(xtr, ytr, xva, yva, wd, device)
    apply = lambda x: head(torch.tensor((x - mu) / sd, dtype=torch.float32, device=device)).detach().cpu().numpy()
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "logits_val_gt.npz", logits=zv, labels=yva, ids=np.array(val_ids))
    if xva_pred is not None:
        np.savez_compressed(args.out / "logits_val_pred.npz", logits=apply(xva_pred), labels=yva, ids=np.array(val_ids))
    for name, x in xte.items():
        np.savez_compressed(args.out / f"logits_test_{name}.npz", logits=apply(x), ids=np.array(test_ids))
    metrics = {"backbone": args.backbone, "size": args.size, "feature_dim": int(xtr.shape[1]), "wd_grid_val_top1": grid,
               "best_wd": wd, "val_gt": summary(zv, yva, tax)}
    if xva_pred is not None:
        metrics["val_pred"] = summary(apply(xva_pred), yva, tax)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({"final_probe": metrics}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
