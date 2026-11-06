from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import RoadGraph, wrap


def turn(a, side):
    w = wrap(a)
    return np.where(np.abs(np.abs(w) - np.pi) < 1e-6, side * np.pi, w)


@dataclass
class Params:
    sigma_obs: float = np.radians(4.0)
    outlier: float = 0.05
    bias_q: float = np.radians(1.0) ** 2
    turn_half_m: float = 7.0
    corr_s: float = 1.34
    init_heading_sd: float = np.radians(45.0)
    speed_sd_rate: float = 0.224
    scale_q: float = 0.0004
    scale_prior: tuple = (0.5, 3.0)
    scale_prior_log_sd: float | None = None
    k_values: tuple = (1.0,)
    accel_sd: float = 1.5
    v_max: float = 33.0
    uturn_side: float = 1.0
    resample_frac: float = 0.5
    mode_cell_m: float = 0.0
    mode_alpha: float = 0.5
    ckpt_every_s: float = 5.0
    anneal_s: float = 30.0


@dataclass
class Hypothesis:
    x: float
    y: float
    mass: float
    radius_m: float


class Localizer:
    def __init__(self, graph: RoadGraph, n: int, params: Params | None = None, seed: int = 0):
        self.g, self.n, self.p = graph, n, params or Params()
        self.rng = np.random.default_rng(seed)
        self.events: list = []
        self.t = 0.0
        self.next_ckpt = 0.0
        self.anc = np.arange(n)
        self.loglik_total = 0.0

    def _successor(self, d):
        g = self.g
        a, b = g.succ_ptr[d], g.succ_ptr[d + 1]
        m = b - a
        pick = a + np.floor(self.rng.random(len(d)) * np.maximum(m, 1)).astype(np.int64)
        return np.where(m > 0, g.succ_idx[np.minimum(pick, len(g.succ_idx) - 1)], -1).astype(np.int32)

    def init(self, dirs: np.ndarray, psi0: float):
        g, n, rng, p = self.g, self.n, self.rng, self.p
        w = g.d_len[dirs]
        self.d = dirs[rng.choice(len(dirs), size=n, p=w / w.sum())].astype(np.int32)
        self.s = rng.random(n) * g.d_len[self.d]
        self.v = rng.uniform(0, 15.0, n)
        self.k = rng.choice(np.asarray(p.k_values, dtype=float), size=n)
        lo, hi = p.scale_prior
        self.c = np.exp(rng.normal(np.log(lo), p.scale_prior_log_sd, n)) if p.scale_prior_log_sd is not None \
            else np.exp(rng.uniform(np.log(lo), np.log(hi), n))
        self.nxt = self._successor(self.d)
        self.pb = g.start_bear[self.d].copy()
        self.mu_b = wrap(self.k * psi0 - self.heading())
        self.P_b = np.full(n, p.init_heading_sd ** 2)
        self.logw = np.full(n, -np.log(n))
        self._checkpoint(force=True)

    def heading(self):
        g, a, side = self.g, self.p.turn_half_m, self.p.uturn_side
        _, _, here = g.point(self.d, self.s)
        L = g.d_len[self.d]
        th = here.copy()
        near_end = (self.s > L - a) & (self.nxt >= 0)
        near_start = (self.s < a) & ~near_end
        if near_end.any():
            i = np.nonzero(near_end)[0]
            w = np.clip((self.s[i] - (L[i] - a)) / (2 * a), 0, 0.5)
            th[i] = wrap(here[i] + w * turn(g.start_bear[self.nxt[i]] - here[i], side))
        if near_start.any():
            i = np.nonzero(near_start)[0]
            w = np.clip(0.5 + self.s[i] / (2 * a), 0.5, 1.0)
            th[i] = wrap(self.pb[i] + w * turn(here[i] - self.pb[i], side))
        return th

    def _move(self, dt, rate, stationary):
        g, rng, p = self.g, self.rng, self.p
        if stationary:
            self.v = np.zeros(self.n)
        elif rate is not None:
            sd = p.speed_sd_rate / np.sqrt(dt)
            self.v = np.minimum(self.c * rate * np.exp(rng.normal(-0.5 * sd * sd, sd, self.n)), p.v_max)
        else:
            self.v = np.minimum(np.abs(self.v + rng.normal(0, p.accel_sd * dt, self.n)), p.v_max)
        self.c = self.c * np.exp(rng.normal(0, np.sqrt(p.scale_q * dt), self.n))
        self.s = self.s + self.v * dt
        over = np.nonzero(self.s >= g.d_len[self.d])[0]
        while len(over):
            dead = self.nxt[over] < 0
            self.logw[over[dead]] = -np.inf
            self.s[over[dead]] = g.d_len[self.d[over[dead]]] - 1e-3
            live = over[~dead]
            self.s[live] -= g.d_len[self.d[live]]
            self.pb[live] = g.end_bear[self.d[live]]
            self.d[live] = self.nxt[live]
            self.nxt[live] = self._successor(self.d[live])
            over = live[self.s[live] >= g.d_len[self.d[live]]]

    def step(self, dt: float, psi: float | None, stationary: bool = False, rate: float | None = None):
        p = self.p
        self.t += dt
        self._move(dt, rate, stationary)
        self.P_b = self.P_b + p.bias_q * dt
        ll = np.zeros(self.n)
        if psi is not None:
            r = wrap(self.k * psi - self.heading() - self.mu_b)
            lam = min(1.0, dt / p.corr_s) if p.corr_s > 0 else 1.0
            if p.anneal_s > 0:
                lam *= min(1.0, self.t / p.anneal_s)
            S = self.P_b + p.sigma_obs ** 2
            good = (1 - p.outlier) * np.exp(-0.5 * r * r / S) / np.sqrt(2 * np.pi * S)
            bad = p.outlier / (2 * np.pi)
            ll += lam * np.log(good + bad)
            K = good / (good + bad) * self.P_b / (self.P_b + p.sigma_obs ** 2 / lam)
            self.mu_b = wrap(self.mu_b + K * r)
            self.P_b = (1 - K) * self.P_b
        lw = self.logw + ll
        m = np.max(lw)
        if not np.isfinite(m):
            self.logw = np.full(self.n, -np.inf)
            self.loglik_total = -np.inf
            return
        tot = m + np.log(np.exp(lw - m).sum())
        self.loglik_total += tot
        self.logw = lw - tot
        w = np.exp(self.logw)
        if 1.0 / np.sum(w * w) < p.resample_frac * self.n:
            self._resample(w)
        self._checkpoint()

    def _resample(self, w):
        n, rng, p = self.n, self.rng, self.p
        if p.mode_cell_m <= 0:
            c = np.cumsum(w)
            c[-1] = 1.0
            idx = np.searchsorted(c, (rng.random() + np.arange(n)) / n)
            new_logw = np.full(n, -np.log(n))
        else:
            x, y, _ = self.g.point(self.d, self.s)
            key = np.floor(x / p.mode_cell_m).astype(np.int64) * 1_000_003 + np.floor(y / p.mode_cell_m).astype(np.int64)
            _, lab = np.unique(key, return_inverse=True)
            lab = lab.ravel()
            W = np.bincount(lab, weights=w)
            alloc = W ** p.mode_alpha
            alloc *= n / alloc.sum()
            alloc[alloc < 1] = 0.0
            alloc *= n / alloc.sum()
            counts = np.floor(alloc).astype(np.int64)
            short = n - counts.sum()
            if short > 0:
                counts[np.argsort(-(alloc - counts))[:short]] += 1
            order = np.argsort(lab, kind="stable")
            ls, ws = lab[order], w[order]
            cum = np.cumsum(ws)
            start = np.concatenate([[0.0], np.cumsum(W)[:-1]])
            gkey = ls + np.minimum((cum - start[ls]) / np.maximum(W[ls], 1e-300), 1.0)
            cl = np.repeat(np.arange(len(W)), counts)
            j = np.arange(n) - np.repeat(np.cumsum(counts) - counts, counts)
            u = cl + (rng.random(len(W))[cl] + j) / np.maximum(counts[cl], 1)
            idx = order[np.minimum(np.searchsorted(gkey, u), n - 1)]
            new_logw = np.log(np.maximum(W[cl], 1e-300) / counts[cl])
            new_logw -= np.log(np.exp(new_logw - new_logw.max()).sum()) + new_logw.max()
        for name in ("d", "s", "v", "k", "c", "nxt", "pb", "mu_b", "P_b"):
            setattr(self, name, getattr(self, name)[idx])
        self.anc = self.anc[idx]
        self.logw = new_logw

    def _checkpoint(self, force=False):
        if force or self.t >= self.next_ckpt - 1e-9:
            self.events.append((self.t, self.d.copy(), self.s.astype(np.float32), self.anc.astype(np.int32)))
            self.anc = np.arange(self.n)
            self.next_ckpt = self.t + self.p.ckpt_every_s

    def positions(self):
        x, y, _ = self.g.point(self.d, self.s)
        return x, y, np.exp(self.logw)

    def hypotheses(self, radius=50.0, top=3, cell=25.0):
        x, y, w = self.positions()
        return cluster_hypotheses(x, y, w, radius, top, cell)

    def backtrack(self, final_idx: np.ndarray):
        idx = self.anc[final_idx]
        out = []
        for t, d, s, anc in reversed(self.events):
            dd, ss = d[idx], s[idx].astype(float)
            x, y, _ = self.g.point(dd, ss)
            vals, cnt = np.unique(dd, return_counts=True)
            out.append((t, float(np.median(x)), float(np.median(y)), int(vals[np.argmax(cnt)])))
            idx = anc[idx]
        return out[::-1]


def cluster_hypotheses(x, y, w, radius=50.0, top=3, cell=25.0):
    keys = np.stack([np.floor(x / cell), np.floor(y / cell)], 1).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    remaining = np.ones(len(x), dtype=bool)
    hyps = []
    for _ in range(top):
        cw = np.bincount(inv[remaining], weights=w[remaining], minlength=inv.max() + 1)
        c = int(np.argmax(cw))
        if cw[c] <= 0:
            break
        sel = remaining & (inv == c)
        cx, cy = np.average(x[sel], weights=w[sel] + 1e-300), np.average(y[sel], weights=w[sel] + 1e-300)
        near = remaining & (np.hypot(x - cx, y - cy) <= radius)
        mass = float(w[near].sum())
        if mass <= 0:
            break
        cx, cy = np.average(x[near], weights=w[near] + 1e-300), np.average(y[near], weights=w[near] + 1e-300)
        hyps.append(Hypothesis(float(cx), float(cy), mass, radius))
        remaining &= np.hypot(x - cx, y - cy) > 2 * radius
    return hyps
