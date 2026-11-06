#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="Assemble an additional catalogue (classifier, calibration, taxonomy) under models/")
    p.add_argument("--export", type=Path, required=True, help="folder with classifier(_int8).onnx, classifier_meta.json and head.npz")
    p.add_argument("--precision", choices=["fp32", "int8"], default="int8")
    p.add_argument("--thresholds", type=Path, required=True)
    p.add_argument("--taxonomy", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(args.export / ("classifier_int8.onnx" if args.precision == "int8" else "classifier.onnx"), out / "classifier.onnx")
    meta = json.loads((args.export / "classifier_meta.json").read_text())
    meta.update(precision=args.precision, sha256=sha(out / "classifier.onnx"))
    (out / "classifier_meta.json").write_text(json.dumps(meta, indent=1))
    shutil.copy(args.export / "head.npz", out / "head.npz")
    shutil.copy(args.thresholds, out / "thresholds.yaml")
    shutil.copytree(args.taxonomy, out / "taxonomy")
    if args.report:
        shutil.copy(args.report, out / "report.json")
    manifest = {f.name: {"bytes": f.stat().st_size, "sha256": sha(f)} for f in sorted(out.glob("*")) if f.is_file()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
