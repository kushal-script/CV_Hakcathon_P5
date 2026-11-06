#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_pipe = None


def init(models: str, enhance: bool) -> None:
    global _pipe
    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2
    cv2.setNumThreads(1)
    from p5.pipeline import AppConfig, Pipeline
    _pipe = Pipeline(AppConfig(models=Path(models), threads=1, panels=False, comparison=False, keep_logits=True,
                               enhance=enhance))


def run(job: tuple[int, str]) -> dict:
    iid, path = job
    r = _pipe.run(Path(path).read_bytes())
    v = r.vehicles[0] if r.vehicles else None
    return {"id": iid, "status": r.status, "enhancement": r.enhancement, "timings_ms": r.timings_ms,
            "box": v.box if v else None, "detection_score": v.detection_score if v else None,
            "level": v.decision["level"] if v else "no_vehicle", "label": v.decision["label"] if v else None,
            "class_id": v.decision["class_id"] if v else None, "confidence": v.decision["confidence"] if v else None,
            "n_vehicles": len(r.vehicles), "logits": r.primary_logits, "error": r.error}


def main() -> int:
    p = argparse.ArgumentParser(description="Run the deployed pipeline over a folder of images (no labels read)")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--models", type=Path, default=ROOT / "models")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-enhance", action="store_true")
    args = p.parse_args()

    files = sorted(args.images.glob("*.jpg"))
    if args.limit:
        files = files[:args.limit]
    jobs = [(int(f.stem) if f.stem.isdigit() else k, str(f)) for k, f in enumerate(files)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers, initializer=init, initargs=(str(args.models), not args.no_enhance)) as pool, \
            args.out.open("w") as fh:
        for k, rec in enumerate(pool.imap(run, jobs, chunksize=8)):
            fh.write(json.dumps(rec) + "\n")
            if (k + 1) % 1000 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    print(f"wrote {args.out} ({len(jobs)} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
