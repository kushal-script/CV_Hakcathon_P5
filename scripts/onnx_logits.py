#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import onnxruntime as ort
import pandas as pd

from p5.classification.predict import make_sources
from p5.data.split import read_ids


def main() -> int:
    p = argparse.ArgumentParser(description="Logits from an exported ONNX classifier for a split and input type")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--input", choices=["gt", "pred", "full"], default="pred")
    p.add_argument("--det-dir", type=Path, default=None)
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--assets", type=Path, default=ROOT / "assets")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    meta = json.loads(args.meta.read_text())
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(str(args.model), opts, providers=["CPUExecutionProvider"])
    det_csv = None
    if args.det_dir:
        det_csv = args.det_dir / ("det_ft_val.csv" if args.split == "val" else "det_ft_test.csv")
    sources, index = make_sources(args.cache, args.split, det_csv)
    if args.split == "val":
        ids = sorted(read_ids(args.assets / "splits" / "val_ids.txt"))
    else:
        ids = sorted(pd.read_csv(args.cache / "index_test.csv").id.tolist())
    mean = np.array(meta["mean"], np.float32)[:, None, None]
    std = np.array(meta["std"], np.float32)[:, None, None]
    src = sources[args.input]
    out, t0 = [], time.time()
    for k in range(0, len(ids), args.batch):
        xs = [((np.asarray(src(i, meta["size"], meta["pad"])).transpose(2, 0, 1).astype(np.float32) / 255.0) - mean) / std
              for i in ids[k:k + args.batch]]
        out.append(sess.run(["logits"], {"input": np.stack(xs)})[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    extra = {"labels": index.loc[ids, "class_id"].to_numpy()} if args.split == "val" else {}
    np.savez_compressed(args.out, logits=np.concatenate(out), ids=np.array(ids), **extra)
    print(json.dumps({"split": args.split, "input": args.input, "n": len(ids), "sec": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
