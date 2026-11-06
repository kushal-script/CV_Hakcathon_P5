import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from p5.geo.videoloc.filter import Params
from p5.geo.videoloc.graph import RoadGraph
from p5.geo.videoloc.search import search

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class Router:
    def __init__(self, g: RoadGraph):
        self.g = g
        n = int(max(g.d_start.max(), g.d_end.max())) + 1
        order = np.lexsort((g.d_len, g.d_end, g.d_start))
        best = {}
        for d in order:
            key = (int(g.d_start[d]), int(g.d_end[d]))
            if key not in best:
                best[key] = int(d)
        self.edge_of = best
        rows = np.array([k[0] for k in best]); cols = np.array([k[1] for k in best])
        self.A = csr_matrix((g.d_len[list(best.values())] + 1e-3, (rows, cols)), shape=(n, n))
        self.n = n

    def route(self, src: int, dst: int):
        dist, pred = dijkstra(self.A, directed=True, indices=src, return_predecessors=True)
        if not np.isfinite(dist[dst]):
            return None
        nodes = [dst]
        while nodes[-1] != src:
            nodes.append(int(pred[nodes[-1]]))
        nodes = nodes[::-1]
        return [self.edge_of[(a, b)] for a, b in zip(nodes, nodes[1:])]


def path_polyline(g: RoadGraph, dirs: list[int], step=1.0):
    xs, ys = [], []
    for d in dirs:
        s = np.arange(0.0, g.d_len[d], step)
        x, y, _ = g.point(np.full(len(s), d), s)
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    seg = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    br = np.arctan2(np.diff(x), np.diff(y))
    br = np.concatenate([br, br[-1:]])
    return x, y, cum, br


def smooth_heading(br, cum, arc_m):
    un = np.unwrap(br)
    out = np.empty_like(un)
    half = arc_m / 2
    lo = np.searchsorted(cum, cum - half)
    hi = np.searchsorted(cum, cum + half, side="right")
    cs = np.concatenate([[0.0], np.cumsum(un)])
    out = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return out



G = R = None
NOISE = []
CFG = None
H_CAM = 1.65


def load_noise(files):
    out = []
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        yaw = np.unwrap(np.array([r["yaw"] for r in rows]))
        gt = -np.diff(yaw, prepend=yaw[0])
        dy = np.array([r["dyaw"] for r in rows])
        v = np.array([r["vf"] for r in rows])
        gr = np.array([r.get("ground_rate", np.nan) for r in rows])
        lr = np.where((v > 1) & np.isfinite(gr) & (gr > 0), np.log(np.where(gr > 0, gr, 1) * H_CAM / np.maximum(v, 1e-3)), np.nan)
        out.append({"head_err": dy - gt, "speed_lr": lr, "speed": v, "t": np.array([r["t"] for r in rows])})
    return out


def init_worker(graph, noise_files, cfg):
    global G, R, NOISE, CFG
    G = RoadGraph(graph)
    R = Router(G)
    NOISE = load_noise(noise_files)
    CFG = cfg


def make_clip(rng, L, area_center, radius):
    nz = NOISE[rng.integers(len(NOISE))]
    n_steps = int(L * 5)
    for _ in range(80):
        if len(nz["t"]) <= n_steps + 1:
            nz = NOISE[rng.integers(len(NOISE))]
            continue
        i0 = int(rng.integers(0, len(nz["t"]) - n_steps - 1))
        v = np.clip(nz["speed"][i0:i0 + n_steps], 0, None)
        dt = np.diff(nz["t"][i0:i0 + n_steps + 1])
        need = float(np.sum(v * dt)) + 50
        area = G.within(area_center[0], area_center[1], radius)
        src = int(rng.choice(area, p=G.d_len[area] / G.d_len[area].sum()))
        mid = G.d_mid_xy[area]
        far = area[np.hypot(mid[:, 0] - G.d_mid_xy[src, 0], mid[:, 1] - G.d_mid_xy[src, 1]) > max(need * 0.7, 300)]
        if len(far) == 0:
            continue
        dst = int(rng.choice(far))
        route = R.route(int(G.d_end[src]), int(G.d_start[dst]))
        if not route:
            continue
        dirs = [src] + route
        x, y, cum, br = path_polyline(G, dirs)
        start = float(rng.uniform(0, G.d_len[src]))
        dist = start + np.concatenate([[0.0], np.cumsum(v[:-1] * dt[:-1])])
        if dist[-1] >= cum[-1] - 5:
            continue
        head = smooth_heading(br, cum, rng.uniform(8.0, 20.0))
        idx = np.searchsorted(cum, dist)
        rel = head[idx] - head[idx[0]]
        err = np.concatenate([[0.0], np.cumsum(nz["head_err"][i0 + 1:i0 + n_steps])])
        psi = rel + err
        lr = nz["speed_lr"][i0:i0 + n_steps]
        t = np.concatenate([[0.0], np.cumsum(dt[:-1])])
        recs = []
        for k in range(n_steps):
            r = {"t": float(t[k]), "psi": float(psi[k]), "ok": True}
            if v[k] < 0.3:
                r["stationary"] = True
            elif np.isfinite(lr[k]):
                r["ground_rate"] = float(v[k] / H_CAM * np.exp(lr[k]))
            recs.append(r)
        return recs, x[idx], y[idx]
    return None


def job(a):
    cid, L, cx, cy = a
    rng = np.random.default_rng(cid)
    made = make_clip(rng, L, (cx, cy), CFG["radius"] * 0.6)
    if made is None:
        return {"cid": cid, "skipped": True}
    recs, tx, ty = made
    ang, rad = rng.uniform(0, 2 * np.pi), CFG["offset_m"] * np.sqrt(rng.uniform())
    t0 = time.time()
    p = Params(outlier=CFG["outlier"], scale_prior=(H_CAM, H_CAM), scale_prior_log_sd=0.15)
    res = search(G, recs, (float(tx[0] + rad * np.cos(ang)), float(ty[0] + rad * np.sin(ang))), CFG["radius"], p, per_m=CFG["per_m"],
                 n_lo=CFG["n_lo"], n_hi=CFG["n_hi"], workers=1)
    if "best" not in res:
        return {"cid": cid, "L": L, "conf": 0.0, "err": None, "status": res.get("status")}
    hx, hy, m = res["best"]["hyps"][0]
    lon, lat = G.proj.to_lonlat(cx, cy)
    ev = res["evidence"]
    return {"agree": ev["agreeing_runs"], "runs": ev["main_runs"], "vo_turn": round(ev["turn_deg"], 1), "moving_s": round(ev["moving_s"], 1),
            "gap": None if res["log10_lr"] is None else round(res["log10_lr"], 3),
            "main_errs": [round(float(np.hypot(t[0] - tx[-1], t[1] - ty[-1])), 1) if t else None for t in res["main_tops"]], "cid": cid, "L": L, "area": [round(float(lat), 4), round(float(lon), 4)], "conf": round(res["confidence"], 4), "p_best": round(res["p_best"], 4),
            "mass": round(m, 4), "err": round(float(np.hypot(hx - tx[-1], hy - ty[-1])), 1), "particles": res["particles"], "roads_km": res["area_km"],
            "secs": round(time.time() - t0, 1)}


if __name__ == "__main__":
    graph, cfg = sys.argv[1], json.load(open(sys.argv[2]))
    noise_files = cfg["noise"]
    g0 = RoadGraph(graph)
    rs = np.random.default_rng(cfg["seed"])
    picks = rs.choice(g0.D, size=cfg["per_length"] * len(cfg["lengths"]), p=g0.d_len / g0.d_len.sum())
    centers = [tuple(map(float, g0.proj.to_lonlat(*g0.d_mid_xy[d]))) for d in picks]
    jobs = []
    cid = 0
    for L in cfg["lengths"]:
        for k in range(cfg["per_length"]):
            lon, lat = centers[cid]
            jobs.append((5000 + cid, L, lon, lat))
            cid += 1
    jobs = [(c, L, *map(float, g0.proj.to_xy(lon, lat))) for c, L, lon, lat in jobs]
    with Pool(cfg["procs"], initializer=init_worker, initargs=(graph, noise_files, cfg)) as pool:
        res = []
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            print(json.dumps(r), flush=True)
    json.dump(res, open(cfg["out"], "w"))
