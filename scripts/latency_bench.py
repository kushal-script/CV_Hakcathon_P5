#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from p5.pipeline import AppConfig, Pipeline


def cpu_name() -> str:
    import subprocess
    for cmd in (["sysctl", "-n", "machdep.cpu.brand_string"], ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
            if out:
                return out
        except OSError:
            pass
    return platform.machine()


def bench(models: Path, files: list[Path], threads: int) -> dict:
    pipe = Pipeline(AppConfig(models=models, threads=threads))
    pipe.run(files[0].read_bytes())
    stages: dict[str, list[float]] = {}
    for f in files:
        r = pipe.run(f.read_bytes())
        for k, v in r.timings_ms.items():
            stages.setdefault(k, []).append(v)
    return {k: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))} for k, v in stages.items()}


def main() -> int:
    p = argparse.ArgumentParser(description="Per-stage latency of the full product pipeline")
    p.add_argument("--images", type=Path, default=ROOT / "data" / "stanford_cars" / "cars_test")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--fp32", type=Path, default=None, help="FP32 classifier.onnx to compare against the packaged model")
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "latency.json")
    args = p.parse_args()

    files = sorted(args.images.glob("*.jpg"))[:: max(1, len(list(args.images.glob('*.jpg'))) // args.n)][: args.n]
    models = ROOT / "models"
    meta = json.loads((models / "classifier_meta.json").read_text())
    report = {"cpu": cpu_name(), "threads": args.threads, "n": len(files),
              meta["precision"]: bench(models, files, args.threads)}
    if args.fp32:
        with tempfile.TemporaryDirectory() as tmp:
            alt = Path(tmp) / "models"
            shutil.copytree(models, alt)
            shutil.copy(args.fp32, alt / "classifier.onnx")
            report["fp32"] = bench(alt, files, args.threads)
    args.out.write_text(json.dumps(report, indent=1))
    for prec in ("int8", "fp32"):
        if prec in report:
            print(prec, {k: round(v["p50"]) for k, v in report[prec].items()}, "| total p95", round(report[prec]["total"]["p95"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
