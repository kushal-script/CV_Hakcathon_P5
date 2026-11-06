import math
from datetime import datetime

import pytest

from p5.geo.traffic import IST, TrafficModel
from p5.identity import Identity

PRIOR = {"bands_m": [3000, 6000, 12000],
         "cells": {f"{h}:{b}": {"hour": h, "band": b, "n": 100, "median_log_ratio": math.log(3.0 if h == 18 else 1.5),
                                "p10_log_ratio": math.log(2.0 if h == 18 else 1.2), "p90_log_ratio": math.log(4.0 if h == 18 else 2.0)}
                   for h in range(24) for b in range(3)}}


def at(hour: int, day: int = 4) -> float:
    return datetime(2026, 11, day, hour, 10, tzinfo=IST).timestamp()


def journeys(n: int, factor: float, hour: int = 18) -> list[dict]:
    return [{"source": "sighting", "depart_ts": at(hour) + k * 86400 * 7, "observed_s": 600 * factor, "freeflow_s": 600.0, "distance_m": 5000.0}
            for k in range(n)]


def test_prior_scales_free_flow_by_hour():
    m = TrafficModel(PRIOR)
    p = m.predict(600, 5000, at(18))
    assert p["calibrated"] and abs(p["p50"] - 1800) < 1e-6 and abs(p["p10"] - 1200) < 1e-6 and abs(p["p90"] - 2400) < 1e-6
    assert abs(m.predict(600, 5000, at(3))["p50"] - 900) < 1e-6
    assert "weekday profile" in m.predict(600, 5000, at(18, day=8))["basis"]


def test_without_prior_it_stays_free_flow():
    m = TrafficModel(None, journeys(30, 3.5))
    p = m.predict(600, 5000, at(18))
    assert not p["calibrated"] and p["p50"] == 600


def test_local_journeys_shift_the_prior_with_shrinkage():
    m = TrafficModel(PRIOR, journeys(40, 3.0 * math.exp(0.2)))
    assert m.local["n"] == 40
    assert abs(m.local["offset_log"] - 0.2 * 40 / 50) < 1e-9
    h = m.local["holdout"]
    assert h["test"] == 8 and h["corrected_mdape"] < h["prior_mdape"] < h["free_flow_mdape"]


def test_implausible_journeys_are_ignored():
    m = TrafficModel(PRIOR, journeys(5, 20.0) + journeys(5, 3.0))
    assert m.local["n"] == 5 and m.local["ignored"] == 5


def test_probability_of_fitting_in_a_gap():
    m = TrafficModel(PRIOR)
    assert abs(m.prob_within(600, 5000, at(18), 1800) - 0.5) < 1e-6
    assert m.prob_within(600, 5000, at(18), 900) < 0.05
    assert m.prob_within(600, 5000, at(18), 3600) > 0.95


def test_route_scoring_uses_calibrated_times(monkeypatch):
    from p5.investigate import routes
    route = {"geometry": [[77.59, 12.97], [77.62, 12.98]], "duration_s": 600.0, "distance_m": 5000.0}
    m = TrafficModel(PRIOR)
    slow, why = routes._score(dict(route), {"places": []}, [], set(), at(18), 700, m)
    fine, _ = routes._score(dict(route), {"places": []}, [], set(), at(18), 1800, m)
    assert fine > slow and any("usually takes" in r for r in why)
    r = dict(route)
    routes._score(r, {"places": []}, [], set(), at(18), None, m)
    assert r["typical_s"] == [1200, 1800, 2400] and r["freeflow_s"] == 600


def test_track_windows_become_journeys(monkeypatch):
    from p5.investigate import journeys as J
    monkeypatch.setattr(J, "fastest", lambda a, b: {"duration_s": 240.0, "distance_m": 3100.0})
    pts = [{"lat": 12.97 + 0.003 * k, "lon": 77.59, "epoch": 1_800_000_000 + 60 * k} for k in range(30)]
    rows = J.from_track("c:cam", pts)
    assert rows and all(r["observed_s"] > 0 and r["freeflow_s"] == 240.0 for r in rows)
    assert len({r["ref"] for r in rows}) == len(rows)


def test_import_validation():
    from p5.investigate.journeys import parse_import
    text = ("from_lat,from_lon,to_lat,to_lon,depart,seconds\n"
            "12.9763,77.5983,12.9352,77.6245,2026-11-04T18:05:00,1500\n"
            "12.2958,76.6394,12.9352,77.6245,2026-11-04T18:05:00,1500\n"
            "12.9763,77.5983,12.9352,77.6245,not-a-date,1500\n")
    rows, errors = parse_import(text)
    assert len(rows) == 1 and len(errors) == 2
    assert parse_import("a,b\n1,2\n")[0] == []


def test_journeys_are_scoped_and_deletable(tmp_path):
    from p5.store import Store
    s = Store(tmp_path / "v.db")
    central = Identity(1, "ps-a", "station", "ps-a", "central", "A")
    row = {"ref": "x", "from_lat": 12.97, "from_lon": 77.59, "to_lat": 12.93, "to_lon": 77.62, "depart_ts": at(18), "observed_s": 1500,
           "freeflow_s": 600, "distance_m": 5000}
    assert s.add_journeys(central, "sighting", [row]) == 1
    assert s.add_journeys(central, "sighting", [{**row, "observed_s": 1600}]) == 1
    admin_c = Identity(9, "admin-central", "division_admin", None, "central", "Admin")
    admin_n = Identity(10, "admin-north", "division_admin", None, "north", "Admin")
    page = s.journeys(admin_c)
    assert page["total"] == 1 and page["journeys"][0]["observed_s"] == 1600
    assert s.journeys(admin_n)["total"] == 0
    jid = page["journeys"][0]["id"]
    assert not s.delete_journey(admin_n, jid) and s.delete_journey(admin_c, jid)
    assert s.all_journeys() == []
