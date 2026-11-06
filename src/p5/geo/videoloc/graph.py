from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

R_EARTH = 6371008.8


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Projection:
    lat0: float
    lon0: float

    def to_xy(self, lon, lat):
        k = R_EARTH * math.cos(math.radians(self.lat0))
        return np.radians(np.asarray(lon, dtype=float) - self.lon0) * k, np.radians(np.asarray(lat, dtype=float) - self.lat0) * R_EARTH

    def to_lonlat(self, x, y):
        k = R_EARTH * math.cos(math.radians(self.lat0))
        return self.lon0 + np.degrees(np.asarray(x, dtype=float) / k), self.lat0 + np.degrees(np.asarray(y, dtype=float) / R_EARTH)


class RoadGraph:
    def __init__(self, path: str | Path):
        self.path = str(path)
        z = np.load(path, allow_pickle=True)
        coords, offsets = z["coords"], z["offsets"].astype(np.int64)
        u, v, oneway = z["u"].astype(np.int64), z["v"].astype(np.int64), z["oneway"]
        self.street_names = z["names"]
        self.classes = list(z["classes"])
        self.source = {k: str(z[k]) for k in ("osm_source", "osm_date") if k in z.files}
        self.proj = Projection(float(coords[:, 1].mean()), float(coords[:, 0].mean()))
        px, py = self.proj.to_xy(coords[:, 0], coords[:, 1])
        fwd, bwd = np.nonzero(oneway != -1)[0], np.nonzero(oneway != 1)[0]
        self.d_edge = np.concatenate([fwd, bwd]).astype(np.int32)
        self.d_sign = np.concatenate([np.ones(len(fwd)), -np.ones(len(bwd))]).astype(np.int8)
        D = len(self.d_edge)
        self.D = D
        e = self.d_edge
        self.d_name = z["name"][e]
        self.d_hw = z["hw"][e]
        self.d_start = np.where(self.d_sign > 0, u[e], v[e])
        self.d_end = np.where(self.d_sign > 0, v[e], u[e])
        ns = (offsets[e + 1] - offsets[e] - 1).astype(np.int64)
        first = np.cumsum(ns) - ns
        owner = np.repeat(np.arange(D), ns)
        k = np.arange(ns.sum()) - first[owner]
        base = offsets[e][owner]
        forward = self.d_sign[owner] > 0
        a = np.where(forward, base + k, base + ns[owner] - k)
        b = np.where(forward, a + 1, a - 1)
        dx, dy = px[b] - px[a], py[b] - py[a]
        ln = np.hypot(dx, dy)
        tiny = ln < 1e-3
        ux = np.where(tiny, 0.0, dx / np.where(tiny, 1.0, ln))
        uy = np.where(tiny, 1.0, dy / np.where(tiny, 1.0, ln))
        ln = np.where(tiny, 1e-3, ln)
        csum = np.cumsum(ln)
        self.seg_x0, self.seg_y0, self.seg_ux, self.seg_uy, self.seg_len = px[a], py[a], ux, uy, ln
        self.seg_cum = csum - ln - (csum[first] - ln[first])[owner]
        self.seg_owner = owner.astype(np.int32)
        self.seg_bearing = np.arctan2(ux, uy)
        self.d_len = np.add.reduceat(ln, first)
        self.d_first_seg = np.concatenate([first, [len(ln)]])
        self.d_base = np.concatenate([[0.0], np.cumsum(self.d_len + 1.0)[:-1]])
        self.seg_key = self.d_base[owner] + self.seg_cum
        self.start_bear = self.seg_bearing[first]
        self.end_bear = self.seg_bearing[self.d_first_seg[1:] - 1]
        n_nodes = int(max(self.d_start.max(), self.d_end.max())) + 1
        order = np.argsort(self.d_start, kind="stable")
        out_ptr = np.searchsorted(self.d_start[order], np.arange(n_nodes + 1))
        out_idx = order.astype(np.int32)
        cnt = out_ptr[self.d_end + 1] - out_ptr[self.d_end]
        self.succ_ptr = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        j = np.arange(self.succ_ptr[-1]) - np.repeat(self.succ_ptr[:-1], cnt)
        self.succ_idx = out_idx[np.repeat(out_ptr[self.d_end], cnt) + j]
        mx, my, _ = self.point(np.arange(D), self.d_len / 2)
        self.d_mid_xy = np.stack([mx, my], 1)

    def locate(self, d, s):
        s = np.clip(s, 0.0, np.maximum(self.d_len[d] - 1e-6, 0.0))
        seg = np.searchsorted(self.seg_key, self.d_base[d] + s, side="right") - 1
        seg = np.clip(seg, self.d_first_seg[d], self.d_first_seg[d + 1] - 1)
        return seg, s - self.seg_cum[seg]

    def point(self, d, s):
        seg, t = self.locate(d, s)
        return self.seg_x0[seg] + t * self.seg_ux[seg], self.seg_y0[seg] + t * self.seg_uy[seg], self.seg_bearing[seg]

    def within(self, x, y, radius):
        return np.nonzero(np.hypot(self.d_mid_xy[:, 0] - x, self.d_mid_xy[:, 1] - y) <= radius + self.d_len / 2)[0]

    def name_at(self, d) -> str:
        n = self.street_names[int(self.d_name[int(d)])]
        return str(n) if n else ""
