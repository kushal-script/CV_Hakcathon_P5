from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))
Z90 = 1.2815515655446004
RATIO_RANGE = (0.5, 8.0)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class TrafficModel:
    def __init__(self, prior: dict | None, journeys: list[dict] | None = None, shrink: float = 10.0, min_holdout: int = 20):
        self.prior = prior
        self.cells = {(c["hour"], c["band"]): c for c in (prior or {}).get("cells", {}).values()}
        self.bands = (prior or {}).get("bands_m", [3000, 6000, 12000])
        self.shrink, self.min_holdout = shrink, min_holdout
        self.local = {"n": 0, "offset_log": 0.0}
        self.local = self.fit_local(journeys or [])

    @classmethod
    def load(cls, path: Path, journeys: list[dict] | None = None, **kw) -> "TrafficModel":
        prior = json.loads(path.read_text()) if path.exists() else None
        return cls(prior, journeys, **kw)

    @property
    def available(self) -> bool:
        return bool(self.cells)

    def band(self, meters: float) -> int:
        return int(sum(meters >= b for b in self.bands[1:]))

    def cell(self, hour: int, meters: float) -> dict | None:
        b = self.band(meters)
        for cand in (b, b - 1, b + 1, b - 2, b + 2):
            c = self.cells.get((hour, cand))
            if c:
                return c
        return None

    def residuals(self, journeys: list[dict]) -> np.ndarray:
        out = []
        for j in journeys:
            hour = datetime.fromtimestamp(j["depart_ts"], IST).hour
            c = self.cell(hour, j["distance_m"])
            if c:
                out.append(math.log(j["observed_s"] / j["freeflow_s"]) - c["median_log_ratio"])
        return np.array(out)

    @staticmethod
    def usable(journeys: list[dict]) -> list[dict]:
        return [j for j in journeys if j.get("freeflow_s", 0) > 0 and j.get("observed_s", 0) > 0
                and RATIO_RANGE[0] <= j["observed_s"] / j["freeflow_s"] <= RATIO_RANGE[1]]

    def offset(self, journeys: list[dict]) -> float:
        r = self.residuals(journeys)
        return float(r.sum() / (len(r) + self.shrink)) if len(r) else 0.0

    def fit_local(self, journeys: list[dict]) -> dict:
        used = sorted(self.usable(journeys), key=lambda j: j["depart_ts"])
        by_source: dict[str, int] = {}
        for j in used:
            by_source[j["source"]] = by_source.get(j["source"], 0) + 1
        out = {"n": len(used), "ignored": len(journeys) - len(used), "by_source": by_source, "offset_log": 0.0, "factor": 1.0, "holdout": None}
        if not self.available or not used:
            return out
        out["offset_log"] = self.offset(used)
        out["factor"] = math.exp(out["offset_log"])
        if len(used) >= self.min_holdout:
            k = int(round(0.8 * len(used)))
            fit, test = used[:k], used[k:]
            off = self.offset(fit)
            obs = np.array([j["observed_s"] for j in test])
            ff = np.array([j["freeflow_s"] for j in test])
            prior_only = np.array([self.predict(j["freeflow_s"], j["distance_m"], j["depart_ts"], offset=0.0)["p50"] for j in test])
            corrected = [self.predict(j["freeflow_s"], j["distance_m"], j["depart_ts"], offset=off) for j in test]
            p50 = np.array([c["p50"] for c in corrected])
            lo, hi = np.array([c["p10"] for c in corrected]), np.array([c["p90"] for c in corrected])
            mape = lambda pred: float(np.median(np.abs(pred - obs) / obs))
            out["holdout"] = {"fit": len(fit), "test": len(test), "split": "time ordered, newest 20% held out",
                              "free_flow_mdape": mape(ff), "prior_mdape": mape(prior_only), "corrected_mdape": mape(p50),
                              "free_flow_mae_s": float(np.mean(np.abs(ff - obs))), "prior_mae_s": float(np.mean(np.abs(prior_only - obs))),
                              "corrected_mae_s": float(np.mean(np.abs(p50 - obs))), "interval_80_coverage": float(np.mean((obs >= lo) & (obs <= hi)))}
        return out

    def predict(self, freeflow_s: float, distance_m: float, depart_ts: float | None = None, offset: float | None = None) -> dict:
        ts = depart_ts if depart_ts is not None else time.time()
        when = datetime.fromtimestamp(ts, IST)
        c = self.cell(when.hour, distance_m) if self.available else None
        if c is None or freeflow_s <= 0:
            return {"calibrated": False, "p10": freeflow_s, "p50": freeflow_s, "p90": freeflow_s, "factor": 1.0, "hour": when.hour,
                    "basis": "free-flow estimate (no travel-time data for this case)"}
        off = self.local["offset_log"] if offset is None else offset
        p10, p50, p90 = (freeflow_s * math.exp(c[k] + off) for k in ("p10_log_ratio", "median_log_ratio", "p90_log_ratio"))
        n = self.local["n"]
        basis = f"{when:%H}:00 Bengaluru weekday profile (Uber Movement 2019)"
        if n:
            basis += f", corrected by {n} journey{'s' if n != 1 else ''} observed here"
        if when.weekday() >= 5:
            basis += "; weekend trips use the weekday profile"
        return {"calibrated": True, "p10": p10, "p50": p50, "p90": p90, "factor": p50 / freeflow_s, "hour": when.hour, "basis": basis}

    def prob_within(self, freeflow_s: float, distance_m: float, depart_ts: float | None, seconds: float) -> float:
        p = self.predict(freeflow_s, distance_m, depart_ts)
        if not p["calibrated"] or seconds <= 0:
            return 1.0 if seconds >= freeflow_s else 0.0
        sigma = max(0.05, (math.log(p["p90"]) - math.log(p["p10"])) / (2 * Z90))
        return normal_cdf((math.log(seconds) - math.log(p["p50"])) / sigma)

    def summary(self) -> dict:
        prior = self.prior or {}
        return {"prior": {k: prior.get(k) for k in ("source", "free_flow", "filters", "holdout", "hourly", "generated", "wards")} if prior else None,
                "local": self.local, "shrink_pseudo_journeys": self.shrink, "min_holdout": self.min_holdout}
