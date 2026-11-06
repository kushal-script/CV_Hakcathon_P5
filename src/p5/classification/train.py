from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from p5.classification.dataset import CropDataset
from p5.classification.model import build, data_config
from p5.data.split import assert_disjoint, read_ids
from p5.eval.classification import summary
from p5.taxonomy import Taxonomy


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a backbone on Stanford Cars crops")
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--backbone", default="tresnet_l.miil_in1k_448")
    p.add_argument("--size", type=int, default=368)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=float, default=2.0)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-aug", action="store_true")
    p.add_argument("--full-image", action="store_true")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--llrd", type=float, default=1.0)
    p.add_argument("--split-csv", type=Path, default=None)
    p.add_argument("--crops", type=Path, default=None)
    p.add_argument("--init", type=Path, default=None)
    return p.parse_args(argv)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def param_groups(model: nn.Module, lr: float, head_lr: float, wd: float, llrd: float = 1.0) -> list[dict]:
    head = {id(p) for p in model.get_classifier().parameters()}
    if llrd < 1.0 and hasattr(model, "layer_id"):
        top = model.num_layers + 1
        groups: dict[tuple, list] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            lid = model.layer_id(name)
            rate = head_lr if id(p) in head else lr * llrd ** (top - lid)
            decay = 0.0 if p.ndim <= 1 else wd
            groups.setdefault((rate, decay), []).append(p)
        return [{"params": ps, "lr": r, "weight_decay": d} for (r, d), ps in groups.items()]
    groups = {k: [] for k in ("body_decay", "body_no", "head_decay", "head_no")}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head else "body"
        kind = "no" if p.ndim <= 1 else "decay"
        groups[f"{part}_{kind}"].append(p)
    return [
        {"params": groups["body_decay"], "lr": lr, "weight_decay": wd},
        {"params": groups["body_no"], "lr": lr, "weight_decay": 0.0},
        {"params": groups["head_decay"], "lr": head_lr, "weight_decay": wd},
        {"params": groups["head_no"], "lr": head_lr, "weight_decay": 0.0},
    ]


def lr_lambda(total: int, warmup: int, floor: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))
    return f


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype, flip: bool = True):
    model.eval()
    logits, flipped, labels, ids = [], [], [], []
    for x, y, i in loader:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits.append(model(x).float().cpu())
            if flip:
                flipped.append(model(torch.flip(x, dims=[3])).float().cpu())
        labels.append(y)
        ids.append(i)
    out = {"logits": torch.cat(logits).numpy(), "labels": torch.cat(labels).numpy(), "ids": torch.cat(ids).numpy()}
    if flip:
        out["logits_flip"] = torch.cat(flipped).numpy()
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    amp_dtype = None
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)

    tax = Taxonomy.load(args.assets / "taxonomy")
    if args.split_csv:
        index = pd.read_csv(args.split_csv)
        index = index[index.split != "test"].reset_index(drop=True)
        index["id"] = np.arange(len(index))
        tr_df, va_df = index[index.split == "train"], index[index.split == "val"]
        assert_disjoint(tr_df.file, va_df.file)
    else:
        index = pd.read_csv(args.cache / "index_train.csv")
        val_ids = set(read_ids(args.assets / "splits" / "val_ids.txt"))
        tr_df, va_df = index[~index.id.isin(val_ids)], index[index.id.isin(val_ids)]
        assert_disjoint(tr_df.id, va_df.id)
    if args.limit:
        tr_df, va_df = tr_df.sample(min(args.limit, len(tr_df)), random_state=0), va_df.head(max(32, args.limit // 8))

    model = build(args.backbone, len(tax.names), pretrained=True)
    if args.init:
        src = torch.load(args.init, map_location="cpu", weights_only=False)["model"]
        head = {k for k in model.state_dict() if k.startswith("head.fc.") or k.startswith("fc.") or k.startswith("head.")}
        body = {k: v for k, v in src.items() if k not in head and k in model.state_dict() and v.shape == model.state_dict()[k].shape}
        missing = model.load_state_dict(body, strict=False).missing_keys
        print(json.dumps({"init": str(args.init), "loaded": len(body), "fresh": missing}), flush=True)
    cfg = data_config(model)
    mean, std = list(cfg["mean"]), list(cfg["std"])
    model = model.to(device).to(memory_format=torch.channels_last)

    crop_dir = args.crops if args.crops else args.cache / "crops" / "train"
    common = dict(crop_dir=crop_dir, full_dir=args.cache / "full" / "train" if args.cache else None, size=args.size,
                  mean=mean, std=std, use_full=args.full_image)
    train_ds = CropDataset(tr_df, train=True, aug=not args.no_aug, seed=args.seed, **common)
    val_ds = CropDataset(va_df, train=False, **common)
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, pin_memory=pin,
                          drop_last=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, args.batch * 2, shuffle=False, num_workers=args.workers, pin_memory=pin)

    opt = torch.optim.AdamW(param_groups(model, args.lr, args.head_lr, args.wd, args.llrd))
    steps = args.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(steps, int(args.warmup * len(train_dl)), 1e-6 / args.lr))
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)
    crit = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    step_fn = torch.compile(model) if args.compile else model

    meta = {**vars(args), "mean": mean, "std": std, "interpolation": cfg.get("interpolation", "bilinear"),
            "num_classes": len(tax.names), "train_n": len(tr_df), "val_n": len(va_df), "device": str(device),
            "amp": str(amp_dtype), "torch": torch.__version__}
    (args.out / "config.json").write_text(json.dumps(meta, indent=1, default=str))

    start_epoch, history = 0, []
    ckpt = args.out / "last.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, history = state["epoch"] + 1, state["history"]
        print(json.dumps({"resumed_from_epoch": start_epoch}), flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, seen, loss_sum, correct = time.time(), 0, 0.0, 0
        for x, y, _ in train_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = step_fn(x)
                loss = crit(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            seen += len(y)
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
        rec = {"epoch": epoch + 1, "train_loss": loss_sum / seen, "train_acc": correct / seen,
               "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1), "img_s": round(seen / (time.time() - t0), 1)}
        last = epoch + 1 == args.epochs
        if last or (args.eval_every and (epoch + 1) % args.eval_every == 0):
            pv = predict(model, val_dl, device, amp_dtype, flip=False)
            rec.update({f"val_{k}": v for k, v in summary(pv["logits"], pv["labels"], tax).items() if k in ("top1", "top5")})
        history.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "history": history}, ckpt)

    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)
    pv = predict(model, val_dl, device, amp_dtype, flip=True)
    np.savez_compressed(args.out / "val_logits.npz", **pv)
    metrics = {"split": "val", "n": int(len(pv["labels"])),
               "single": summary(pv["logits"], pv["labels"], tax),
               "flip_tta": summary(pv["logits"] + pv["logits_flip"], pv["labels"], tax)}
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    torch.save({"model": model.state_dict(), "backbone": args.backbone, "size": args.size, "mean": mean, "std": std,
                "num_classes": len(tax.names)}, args.out / "model.pt")
    print(json.dumps({"final": metrics}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
