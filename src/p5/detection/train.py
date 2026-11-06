from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from p5.data.split import read_ids
from p5.eval.detection import detection_report, iou

COCO_VEHICLES = {2: "car", 5: "bus", 7: "truck"}


def scaled_gt(index: pd.DataFrame, max_side: int = 512) -> dict[int, list[float]]:
    out = {}
    for r in index.itertuples():
        s = min(1.0, max_side / max(r.orig_w, r.orig_h))
        out[int(r.id)] = [r.x1 * s, r.y1 * s, r.x2 * s, r.y2 * s]
    return out


def predict(model, paths: list[Path], ids: list[int], classes: set[int] | None, conf: float, batch: int = 64,
            device=None) -> dict[int, list[tuple[list[float], float]]]:
    preds: dict[int, list] = {}
    for k in range(0, len(paths), batch):
        chunk = [str(p) for p in paths[k:k + batch]]
        results = model.predict(chunk, conf=conf, max_det=100, verbose=False, device=device)
        for iid, r in zip(ids[k:k + batch], results):
            keep = []
            for b, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                if classes is None or int(c) in classes:
                    keep.append((b, float(s)))
            preds[iid] = keep
    return preds


def to_frame(preds: dict[int, list]) -> pd.DataFrame:
    rows = [(i, *b, s) for i, ps in preds.items() for b, s in ps]
    return pd.DataFrame(rows, columns=["id", "x1", "y1", "x2", "y2", "score"])


def yolo_line(box, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    return f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def build_dataset(root: Path, index: pd.DataFrame, full_dir: Path, gt: dict, pseudo: dict, train_ids: set[int],
                  val_ids: set[int], negatives: list[Path]) -> Path:
    from PIL import Image
    if root.exists():
        shutil.rmtree(root)
    for part in ("train", "val"):
        (root / "images" / part).mkdir(parents=True)
        (root / "labels" / part).mkdir(parents=True)
    n_pseudo = 0
    for r in index.itertuples():
        iid = int(r.id)
        part = "val" if iid in val_ids else "train" if iid in train_ids else None
        if part is None:
            continue
        src = full_dir / f"{iid:05d}.jpg"
        dst = root / "images" / part / src.name
        dst.symlink_to(src.resolve())
        with Image.open(src) as im:
            w, h = im.size
        lines = [yolo_line(gt[iid], w, h)]
        if part == "train":
            extra = pseudo.get(iid, [])
            n_pseudo += len(extra)
            lines += [yolo_line(b, w, h) for b in extra]
        (root / "labels" / part / f"{iid:05d}.txt").write_text("\n".join(lines) + "\n")
    for p in negatives:
        (root / "images" / "train" / p.name).symlink_to(p.resolve())
        (root / "labels" / "train" / (p.stem + ".txt")).write_text("")
    cfg = {"path": str(root), "train": "images/train", "val": "images/val", "names": {0: "vehicle"}}
    (root / "data.yaml").write_text(yaml.safe_dump(cfg))
    print(json.dumps({"dataset": str(root), "pseudo_boxes": n_pseudo, "negatives": len(negatives)}), flush=True)
    return root / "data.yaml"


def pseudo_labels(zs: dict, gt: dict, min_conf: float = 0.5, max_iou: float = 0.3, min_area: float = 0.01,
                  sizes: dict | None = None) -> dict[int, list]:
    out = {}
    for iid, ps in zs.items():
        w, h = sizes[iid]
        keep = [b for b, s in ps if s >= min_conf and iou(b, gt[iid]) < max_iou
                and (b[2] - b[0]) * (b[3] - b[1]) >= min_area * w * h]
        if keep:
            out[iid] = keep
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="E03 zero-shot and E06 fine-tuned single-class vehicle detector")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", default="yolo26s.pt")
    p.add_argument("--negatives", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    from PIL import Image
    from ultralytics import YOLO

    args.out, args.cache, args.assets = args.out.resolve(), args.cache.resolve(), args.assets.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.cache / "index_train.csv")
    test_index = pd.read_csv(args.cache / "index_test.csv")
    if args.limit:
        index, test_index = index.head(args.limit), test_index.head(args.limit)
    val_ids = set(read_ids(args.assets / "splits" / "val_ids.txt")) & set(index.id)
    train_ids = set(index.id) - val_ids
    full_dir, test_dir = args.cache / "full" / "train", args.cache / "full" / "test"
    gt = scaled_gt(index)
    ids = index.id.tolist()
    paths = [full_dir / f"{i:05d}.jpg" for i in ids]
    sizes = {}
    for i, path in zip(ids, paths):
        with Image.open(path) as im:
            sizes[i] = im.size

    zs_model = YOLO(args.weights)
    zs = predict(zs_model, paths, ids, set(COCO_VEHICLES), conf=0.001, device=args.device)
    to_frame(zs).to_csv(args.out / "det_zs_train.csv", index=False)
    e03 = detection_report({i: zs[i] for i in val_ids}, {i: gt[i] for i in val_ids})
    print(json.dumps({"E03_zero_shot_val": e03}), flush=True)

    pseudo = pseudo_labels({i: zs[i] for i in train_ids}, gt, sizes=sizes)
    negatives = sorted((args.negatives / "train").glob("*.jpg")) if args.negatives else []
    data_yaml = build_dataset(args.out / "yolo_data", index, full_dir, gt, pseudo, train_ids, val_ids, negatives)

    model = YOLO(args.weights)
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                project=str(args.out), name="e06", exist_ok=True, seed=42, deterministic=False, plots=True,
                device=args.device, verbose=False)
    best = Path(model.trainer.best)
    shutil.copy(best, args.out / "detector.pt")
    ft = YOLO(str(best))
    ft_val = predict(ft, [full_dir / f"{i:05d}.jpg" for i in sorted(val_ids)], sorted(val_ids), None, 0.001, device=args.device)
    e06 = detection_report(ft_val, {i: gt[i] for i in val_ids})
    print(json.dumps({"E06_finetuned_val": e06}), flush=True)
    to_frame(ft_val).to_csv(args.out / "det_ft_val.csv", index=False)

    test_ids = test_index.id.tolist()
    test_paths = [test_dir / f"{i:05d}.jpg" for i in test_ids]
    to_frame(predict(ft, test_paths, test_ids, None, 0.001, device=args.device)).to_csv(args.out / "det_ft_test.csv", index=False)
    to_frame(predict(zs_model, test_paths, test_ids, set(COCO_VEHICLES), 0.001, device=args.device)).to_csv(
        args.out / "det_zs_test.csv", index=False)

    if args.negatives:
        for name in ("dev", "test"):
            neg = sorted((args.negatives / name).glob("*.jpg"))
            for tag, m, cls in (("ft", ft, None), ("zs", zs_model, set(COCO_VEHICLES))):
                pr = predict(m, neg, list(range(len(neg))), cls, 0.001, device=args.device)
                to_frame(pr).assign(file=lambda d: d.id.map(lambda k: neg[k].name)).to_csv(
                    args.out / f"det_{tag}_coco_neg_{name}.csv", index=False)

    onnx_path = ft.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=17)
    shutil.copy(onnx_path, args.out / "detector.onnx")
    (args.out / "metrics.json").write_text(json.dumps({"E03_zero_shot_val": e03, "E06_finetuned_val": e06,
                                                       "pseudo_images": len(pseudo), "negatives": len(negatives)}, indent=1))
    print("detector done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
