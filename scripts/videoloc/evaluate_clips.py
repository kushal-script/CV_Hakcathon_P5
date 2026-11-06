import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from p5.geo.videoloc import UNKNOWN_CAMERA
from p5.geo.videoloc.filter import Params
from p5.geo.videoloc.graph import RoadGraph
from p5.geo.videoloc.search import search

G = None
CFG = None


def init_worker(graph, cfg):
    global G, CFG
    G = RoadGraph(graph)
    CFG = cfg


def job(a):
    f, t0, L, cid = a
    rows = [json.loads(l) for l in open(f)]
    rows = [r for r in rows if t0 <= r["t"] <= t0 + L]
    if CFG.get("psi_scale"):
        rows = [{**r, "psi": r["psi"] * CFG["psi_scale"], **({"ground_rate": r["ground_rate"] * CFG["rate_scale"]} if r.get("ground_rate") is not None else {})} for r in rows]
    rng = np.random.default_rng(cid)
    tx, ty = G.proj.to_xy(np.array([r["lon"] for r in rows]), np.array([r["lat"] for r in rows]))
    ang, rad = rng.uniform(0, 2 * np.pi), CFG["offset_m"] * np.sqrt(rng.uniform())
    if CFG.get("k_band"):
        p = Params(outlier=CFG["outlier"], k_values=tuple(CFG["k_band"]), scale_prior=tuple(CFG["c_prior"]))
    elif CFG.get("unknown_camera"):
        p = Params(outlier=CFG["outlier"], k_values=UNKNOWN_CAMERA["k_values"], scale_prior=UNKNOWN_CAMERA["scale_prior"])
    else:
        p = Params(outlier=CFG["outlier"], scale_prior=(1.65, 1.65), scale_prior_log_sd=0.15)
    t_start = time.time()
    res = search(G, rows, (float(tx[0] + rad * np.cos(ang)), float(ty[0] + rad * np.sin(ang))), CFG["radius"], p,
                 per_m=CFG["per_m"], n_lo=CFG["n_lo"], n_hi=CFG["n_hi"], workers=1)
    if "best" not in res:
        return {"file": f.split("/")[-1][:21], "t0": t0, "L": L, "conf": 0.0, "err": 1e9, "status": res.get("status")}
    hx, hy, m = res["best"]["hyps"][0]
    ev = res["evidence"]
    errs_main = [round(float(np.hypot(t[0] - tx[-1], t[1] - ty[-1])), 1) if t else None for t in res["main_tops"]]
    return {"agree": ev["agreeing_runs"], "runs": ev["main_runs"], "vo_turn": round(ev["turn_deg"], 1), "moving_s": round(ev["moving_s"], 1),
            "llA": round(res["main_lls"] and max(res["main_lls"]), 2), "gap": None if res["log10_lr"] is None else round(res["log10_lr"], 3), "main_errs": errs_main,"file": f.split("/")[-1][:21], "t0": t0, "L": L, "cid": cid, "conf": round(res["confidence"], 4), "p_best": round(res["p_best"], 4),
            "mass": round(m, 4), "err": round(float(np.hypot(hx - tx[-1], hy - ty[-1])), 1), "particles": res["particles"],
            "k": res["best"].get("k"), "c": res["best"].get("c"), "moved_m": round(float(np.sum(np.hypot(np.diff(tx), np.diff(ty)))), 1),
            "abs_turn_deg": round(float(np.degrees(np.sum(np.abs(np.diff(np.unwrap(np.array([r["yaw"] for r in rows]))))))), 1),
            "secs": round(time.time() - t_start, 1)}


if __name__ == "__main__":
    graph, cfg = sys.argv[1], json.load(open(sys.argv[2]))
    jobs, cid = [], cfg.get("cid0", 9000)
    if cfg.get("clips"):
        for f, s, L in cfg["clips"]:
            jobs.append((f, float(s), float(L), cid))
            cid += 1
    for f in cfg.get("files", []):
        rows = [json.loads(l) for l in open(f)]
        T = rows[-1]["t"]
        starts = [0.0] if T < 80 else [0.0, max(0.0, T - 60.0)]
        for s in starts:
            if T - s >= 55:
                jobs.append((f, s, 60.0, cid))
                cid += 1
    print("clips", len(jobs), flush=True)
    with Pool(cfg["procs"], initializer=init_worker, initargs=(graph, cfg)) as pool:
        res = []
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            print(json.dumps(r), flush=True)
    json.dump(res, open(cfg["out"], "w"))
