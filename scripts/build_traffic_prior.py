#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "traffic_raw"
UA = "VahanaDrishti/0.3 (https://github.com/kushal-script/CV_Hakcathon_P5; traffic calibration)"
SOURCES = {
    "travel_times": "https://www.qgistutorials.com/downloads/bangalore-wards-2019-3-OnlyWeekdays-HourlyAggregate.csv",
    "wards": "https://www.qgistutorials.com/downloads/bangalore_wards.json",
}
BANDS = (3_000, 6_000, 12_000)


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=300) as r:
            dest.write_bytes(r.read())
    return dest


def centroid(geom: dict) -> tuple[float, float]:
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    area = cx = cy = 0.0
    for poly in polys:
        ring = poly[0]
        for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
            c = x0 * y1 - x1 * y0
            area += c
            cx += (x0 + x1) * c
            cy += (y0 + y1) * c
    return cx / (3 * area), cy / (3 * area)


def osrm_table(base: str, pts: list[tuple[float, float]], block: int, pause: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(pts)
    dur, dist = np.full((n, n), np.nan), np.full((n, n), np.nan)
    blocks = [list(range(i, min(n, i + block))) for i in range(0, n, block)]
    for bi in blocks:
        for bj in blocks:
            idx = bi if bi == bj else bi + bj
            coords = ";".join(f"{pts[k][0]:.5f},{pts[k][1]:.5f}" for k in idx)
            src = ";".join(str(i) for i in range(len(bi)))
            dst = ";".join(str(i) for i in (range(len(bi)) if bi == bj else range(len(bi), len(bi) + len(bj))))
            url = f"{base}/table/v1/driving/{coords}?sources={src}&destinations={dst}&annotations=duration,distance"
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=120) as r:
                        res = json.loads(r.read())
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(5 * (attempt + 1))
            d = np.array(res["durations"], dtype=float)
            m = np.array(res["distances"], dtype=float)
            dur[np.ix_(bi, bj)] = d
            dist[np.ix_(bi, bj)] = m
            time.sleep(pause)
    return dur, dist


def band_of(meters: float) -> int:
    return int(sum(meters >= b for b in BANDS[1:]))


def holdout(src: int, dst: int, share: float) -> bool:
    h = int(hashlib.sha1(f"{min(src, dst)}-{max(src, dst)}".encode()).hexdigest()[:8], 16)
    return (h % 1000) < share * 1000


def main() -> int:
    p = argparse.ArgumentParser(description="Hour-of-day travel-time slowdown for Bengaluru from Uber Movement, measured against OSRM free-flow times")
    p.add_argument("--osrm", default="https://router.project-osrm.org")
    p.add_argument("--block", type=int, default=50)
    p.add_argument("--pause", type=float, default=1.5)
    p.add_argument("--min-distance-m", type=float, default=BANDS[0])
    p.add_argument("--ratio-range", type=float, nargs=2, default=[0.5, 8.0])
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--out", type=Path, default=ROOT / "assets" / "bengaluru" / "traffic_prior.json")
    args = p.parse_args()

    tt = fetch(SOURCES["travel_times"], RAW / "bangalore-wards-2019-3-OnlyWeekdays-HourlyAggregate.csv")
    wards = json.loads(fetch(SOURCES["wards"], RAW / "bangalore_wards.json").read_text())
    ids = [int(f["properties"]["MOVEMENT_ID"]) for f in wards["features"]]
    pts = [centroid(f["geometry"]) for f in wards["features"]]
    index = {wid: k for k, wid in enumerate(ids)}
    cache = RAW / "osrm_freeflow.npz"
    if cache.exists():
        z = np.load(cache)
        dur, dist = z["dur"], z["dist"]
    else:
        dur, dist = osrm_table(args.osrm, pts, args.block, args.pause)
        np.savez_compressed(cache, dur=dur, dist=dist, ids=np.array(ids))

    fit = defaultdict(list)
    test = []
    kept = skipped = 0
    with tt.open() as fh:
        for r in csv.DictReader(fh):
            s, d = index.get(int(r["sourceid"])), index.get(int(r["dstid"]))
            if s is None or d is None or s == d:
                skipped += 1
                continue
            ff, meters = dur[s, d], dist[s, d]
            obs = float(r["geometric_mean_travel_time"])
            if not (np.isfinite(ff) and ff > 0 and meters >= args.min_distance_m and obs > 0):
                skipped += 1
                continue
            ratio = obs / ff
            if not (args.ratio_range[0] <= ratio <= args.ratio_range[1]):
                skipped += 1
                continue
            kept += 1
            h, b = int(r["hod"]), band_of(meters)
            if holdout(s, d, args.holdout):
                test.append((h, b, ff, obs))
            else:
                fit[(h, b)].append(math.log(ratio))

    cells = {}
    for (h, b), vals in sorted(fit.items()):
        v = np.array(vals)
        cells[f"{h}:{b}"] = {"hour": h, "band": b, "n": int(len(v)), "median_log_ratio": float(np.median(v)),
                             "p10_log_ratio": float(np.percentile(v, 10)), "p90_log_ratio": float(np.percentile(v, 90))}
    obs = np.array([o for *_, o in test])
    ff = np.array([f for _, _, f, _ in test])
    pred = np.array([f * math.exp(cells[f"{h}:{b}"]["median_log_ratio"]) for h, b, f, _ in test])
    lo = np.array([f * math.exp(cells[f"{h}:{b}"]["p10_log_ratio"]) for h, b, f, _ in test])
    hi = np.array([f * math.exp(cells[f"{h}:{b}"]["p90_log_ratio"]) for h, b, f, _ in test])
    report = {
        "test_rows": int(len(test)),
        "free_flow": {"mae_s": float(np.mean(np.abs(ff - obs))), "median_abs_pct_error": float(np.median(np.abs(ff - obs) / obs))},
        "calibrated": {"mae_s": float(np.mean(np.abs(pred - obs))), "median_abs_pct_error": float(np.median(np.abs(pred - obs) / obs))},
        "interval_80_coverage": float(np.mean((obs >= lo) & (obs <= hi))),
    }
    hourly = {}
    for h in range(24):
        vals = [x for (hh, _), xs in fit.items() if hh == h for x in xs]
        if vals:
            hourly[str(h)] = {"median_ratio": float(math.exp(np.median(vals))), "p10_ratio": float(math.exp(np.percentile(vals, 10))),
                              "p90_ratio": float(math.exp(np.percentile(vals, 90))), "n": len(vals)}
    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"name": "Uber Movement, Bangalore wards, 2019 Q3, weekdays, travel times by hour of day",
                   "attribution": "Data retrieved from Uber Movement, (c) 2019 Uber Technologies, Inc., https://movement.uber.com",
                   "licence": "Creative Commons Attribution Non-Commercial (Uber Movement data terms); Uber Movement has since been discontinued",
                   "urls": SOURCES, "travel_time_field": "geometric_mean_travel_time"},
        "free_flow": {"engine": "OSRM", "url": args.osrm, "profile": "driving", "between": "area-weighted ward centroids"},
        "filters": {"min_distance_m": args.min_distance_m, "ratio_range": args.ratio_range, "rows_kept": kept, "rows_skipped": skipped},
        "bands_m": list(BANDS),
        "holdout": {"share_of_ward_pairs": args.holdout, **report},
        "hourly": hourly,
        "cells": cells,
        "wards": len(ids),
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"rows_kept": kept, "holdout": report, "peak": max(hourly.items(), key=lambda kv: kv[1]["median_ratio"])}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
