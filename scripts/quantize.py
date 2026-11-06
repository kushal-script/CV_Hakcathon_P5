#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import onnxruntime as ort
import pandas as pd
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from p5.classification.crop import make_crop
from p5.data.split import read_ids
from p5.eval.classification import summary
from p5.taxonomy import Taxonomy


def batches(index: pd.DataFrame, crop_dir: Path, meta: dict, bs: int):
    mean = np.array(meta["mean"], np.float32)[:, None, None]
    std = np.array(meta["std"], np.float32)[:, None, None]
    rows = list(index.itertuples())
    for k in range(0, len(rows), bs):
        xs = []
        for r in rows[k:k + bs]:
            img = Image.open(crop_dir / f"{int(r.id):05d}.jpg").convert("RGB")
            c = np.asarray(make_crop(img, (r.bx1, r.by1, r.bx2, r.by2), meta["size"], meta["pad"]))
            xs.append(((c.transpose(2, 0, 1).astype(np.float32) / 255.0) - mean) / std)
        yield np.stack(xs)


class Reader(CalibrationDataReader):
    def __init__(self, gen):
        self.it = iter(gen)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {"input": x}


def evaluate(path: Path, index: pd.DataFrame, crop_dir: Path, meta: dict, threads: int) -> tuple[np.ndarray, float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    out, times = [], []
    for x in batches(index, crop_dir, meta, 1):
        t = time.perf_counter()
        out.append(sess.run(["logits"], {"input": x})[0])
        times.append(time.perf_counter() - t)
    return np.concatenate(out), float(np.median(times) * 1000)


def main() -> int:
    p = argparse.ArgumentParser(description="E11: static INT8 quantization of the classifier with accuracy and latency check")
    p.add_argument("--export", type=Path, required=True, help="folder with classifier.onnx and classifier_meta.json")
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--calib", type=int, default=320)
    p.add_argument("--eval", type=int, default=0, help="0 = full val split")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--method", default="percentile", choices=["minmax", "entropy", "percentile"])
    args = p.parse_args()

    meta = json.loads((args.export / "classifier_meta.json").read_text())
    tax = Taxonomy.load(args.assets / "taxonomy")
    index = pd.read_csv(args.cache / "index_train.csv")
    val_ids = set(read_ids(args.assets / "splits" / "val_ids.txt"))
    train_df, val_df = index[~index.id.isin(val_ids)], index[index.id.isin(val_ids)]
    if args.eval:
        val_df = val_df.sample(args.eval, random_state=0)
    crop_dir = args.cache / "crops" / "train"

    fp32, pre, int8 = args.export / "classifier.onnx", args.export / "classifier_pre.onnx", args.export / "classifier_int8.onnx"
    quant_pre_process(str(fp32), str(pre), skip_symbolic_shape=True)
    calib = train_df.sample(args.calib, random_state=42)
    method = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
              "percentile": CalibrationMethod.Percentile}[args.method]
    quantize_static(str(pre), str(int8), Reader(batches(calib, crop_dir, meta, 16)), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=method, extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    pre.unlink(missing_ok=True)

    y = val_df.class_id.to_numpy()
    res = {"n": int(len(y)), "threads": args.threads, "calib": args.calib, "method": args.method}
    for name, path in (("fp32", fp32), ("int8", int8)):
        z, ms = evaluate(path, val_df, crop_dir, meta, args.threads)
        res[name] = {**summary(z, y, tax), "latency_p50_ms": ms, "size_mb": round(path.stat().st_size / 2**20, 1)}
        np.savez_compressed(args.export / f"val_logits_{name}.npz", logits=z, labels=y, ids=val_df.id.to_numpy())
        print(json.dumps({name: res[name]}), flush=True)
    res["delta_top1"] = res["int8"]["top1"] - res["fp32"]["top1"]
    (args.export / "quantization.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
