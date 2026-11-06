from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import asdict, replace

import numpy as np

from .filter import Localizer, Params
from .graph import RoadGraph, wrap

_G: RoadGraph | None = None
_OBS: list | None = None
_CACHE: dict = {}


def _init_worker(path: str, obs: list) -> None:
    global _G, _OBS
    if path not in _CACHE:
        _CACHE[path] = RoadGraph(path)
    _G, _OBS = _CACHE[path], obs


def _obs(records: list[dict]) -> list[tuple]:
    out, tp = [], records[0]["t"]
    for r in records[1:]:
        dt = r["t"] - tp
        tp = r["t"]
        if dt > 0:
            out.append((dt, r["psi"] if r.get("ok") else None, bool(r.get("stationary", False)), r.get("ground_rate")))
    return out


def _run(args):
    dirs, n, params, seed, want_path = args
    loc = Localizer(_G, n, params, seed=seed)
    loc.init(dirs, _OBS[0][4] if len(_OBS[0]) > 4 else 0.0)
    for dt, psi, still, rate, *_ in _OBS[1:]:
        loc.step(dt, psi, still, rate)
    hs = loc.hypotheses(radius=50.0, top=3)
    res = {"ll": float(loc.loglik_total), "hyps": [[h.x, h.y, h.mass] for h in hs], "seed": seed}
    if want_path and hs:
        x, y, _ = loc.positions()
        near = np.nonzero(np.hypot(x - hs[0].x, y - hs[0].y) <= 50.0)[0]
        res["path"] = loc.backtrack(near) if len(near) else []
        res["k"] = float(np.median(loc.k[near])) if len(near) else 1.0
        res["c"] = float(np.median(loc.c[near])) if len(near) else None
    return res


def _pool_map(fn, jobs, workers):
    if workers <= 1 or len(jobs) <= 1:
        return [fn(j) for j in jobs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, len(jobs)), initializer=_init_worker, initargs=(_G.path, _OBS)) as pool:
        return pool.map(fn, jobs, chunksize=1)


def decoy_centres(graph: RoadGraph, centre_xy, radius_m: float, k: int, seed: int, min_sep_m: float = 5000.0) -> list:
    rng = np.random.default_rng(seed)
    w = graph.d_len / graph.d_len.sum()
    out = []
    for _ in range(200 * k):
        if len(out) >= k:
            break
        d = int(rng.choice(graph.D, p=w))
        x, y = graph.d_mid_xy[d]
        if math.hypot(x - centre_xy[0], y - centre_xy[1]) < min_sep_m or any(math.hypot(x - a, y - b) < 2 * radius_m for a, b in out):
            continue
        out.append((float(x), float(y)))
    return out


def decoys(graph: RoadGraph, records: list[dict], centre_xy, radius_m: float, params: Params, k: int = 4, runs: int = 3,
           per_m: float = 1.0, n_lo: int = 150_000, n_hi: int = 400_000, workers: int = 4, seed: int = 7) -> list[dict]:
    global _G, _OBS
    _G = graph
    _OBS = [(0.0, None, False, None, records[0]["psi"])] + _obs(records)
    jobs, owners, centres = [], [], decoy_centres(graph, centre_xy, radius_m, k, seed)
    for i, (x, y) in enumerate(centres):
        area = graph.within(x, y, radius_m)
        n = particles_for(graph, area, per_m, n_lo, n_hi)
        for r in range(runs):
            jobs.append((area, n, params, 1000 + 10 * i + r, False))
            owners.append(i)
    res = _pool_map(_run, jobs, workers)
    out = []
    for i, (x, y) in enumerate(centres):
        lls = [r["ll"] for r, o in zip(res, owners) if o == i and math.isfinite(r["ll"])]
        lon, lat = graph.proj.to_lonlat(x, y)
        out.append({"lat": round(float(lat), 5), "lon": round(float(lon), 5), "best_ll": max(lls) if lls else -math.inf})
    return out


def particles_for(graph: RoadGraph, dirs: np.ndarray, per_m: float, lo: int, hi: int) -> int:
    return int(np.clip(per_m * graph.d_len[dirs].sum(), lo, hi))


def search(graph: RoadGraph, records: list[dict], center_xy: tuple[float, float], radius_m: float, params: Params,
           per_m: float = 1.0, n_lo: int = 150_000, n_hi: int = 400_000, std_runs: int = 4, alpha_runs: int = 2,
           alt_runs: int = 2, exclude_m: float = 150.0, workers: int = 4, progress=None) -> dict:
    global _G, _OBS
    _G = graph
    obs = _obs(records)
    _OBS = [(0.0, None, False, None, records[0]["psi"])] + obs
    area = graph.within(center_xy[0], center_xy[1], radius_m)
    if len(area) == 0:
        return {"status": "not_applicable", "reason": "There are no mapped roads inside the search area."}
    n = particles_for(graph, area, per_m, n_lo, n_hi)
    alpha = replace(params, mode_cell_m=250.0)
    mains = _pool_map(_run, [(area, n, params if s < std_runs else alpha, 11 + s, True) for s in range(std_runs + alpha_runs)], workers)
    if progress:
        progress(0.7)
    A = max(mains, key=lambda m: m["ll"])
    if not A["hyps"] or not math.isfinite(A["ll"]):
        return {"status": "not_located", "reason": "No route in the search area matches the camera movement.", "particles": n}
    start = A["path"][0] if A.get("path") else (0.0, A["hyps"][0][0], A["hyps"][0][1], -1)
    mid = graph.d_mid_xy[area]
    rest = area[np.hypot(mid[:, 0] - start[1], mid[:, 1] - start[2]) > exclude_m]
    p_rest = float(graph.d_len[rest].sum() / graph.d_len[area].sum()) if len(rest) else 0.0
    alts = _pool_map(_run, [(rest, n, params if s % 2 == 0 else alpha, 101 + s, True) for s in range(alt_runs)], workers) if len(rest) else []
    if progress:
        progress(0.95)
    B = max(alts, key=lambda m: m["ll"]) if alts else None
    llA = A["ll"]
    llB = B["ll"] + math.log(p_rest) if B and p_rest > 0 and math.isfinite(B["ll"]) else -math.inf
    best, other = (A, B) if llA >= llB else (B, A)
    gap = abs(llA - llB) if math.isfinite(llB) else math.inf
    p_best = 1.0 / (1.0 + math.exp(-gap)) if math.isfinite(gap) else 1.0
    mass = best["hyps"][0][2]
    bx, by = best["hyps"][0][0], best["hyps"][0][1]
    agree = sum(1 for m in mains if m["hyps"] and math.hypot(m["hyps"][0][0] - bx, m["hyps"][0][1] - by) <= 50.0)
    moving = [o for o in obs if not o[2]]
    evidence = {"agreeing_runs": agree, "main_runs": len(mains), "moving_s": float(sum(o[0] for o in moving)),
                "turn_deg": float(math.degrees(sum(abs(b - a) for a, b in zip([None] + [o[1] for o in obs], [o[1] for o in obs]) if a is not None and b is not None)))}
    return {"best": best, "other": other, "p_best": p_best, "mass": mass, "confidence": p_best * mass, "evidence": evidence,
            "main_tops": [m["hyps"][0][:2] + [m["ll"]] if m["hyps"] else None for m in mains],
            "log10_lr": gap / math.log(10) if math.isfinite(gap) else None, "particles": n, "runs": len(mains) + len(alts),
            "main_lls": [m["ll"] for m in mains], "alt_lls": [m["ll"] for m in alts], "p_rest": p_rest,
            "area_dirs": int(len(area)), "area_km": round(float(graph.d_len[area].sum()) / 1000, 1)}
