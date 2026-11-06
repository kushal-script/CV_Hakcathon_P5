import time

import pytest

from p5.identity import Identity, sign, verify
from p5.investigate.service import CaseStore
from p5.region import Boundary
from p5.store import Store

KEY = b"k" * 40


def ident(role="station", station="ps-a", division="central", uid=1):
    return Identity(uid, f"user{uid}", role, station if role == "station" else None, division, "Test")


def test_identity_signature_and_freshness():
    enc, sig = sign({"uid": 3, "u": "ps-a", "r": "station", "s": "ps-a", "d": "central", "n": "A", "ts": time.time()}, KEY)
    assert verify(enc, sig, KEY).station_id == "ps-a"
    assert verify(enc, sig[:-1] + ("0" if sig[-1] != "0" else "1"), KEY) is None
    assert verify(enc, sig, b"x" * 40) is None
    old, osig = sign({"uid": 3, "u": "ps-a", "r": "station", "s": "ps-a", "d": "central", "n": "A", "ts": time.time() - 3600}, KEY)
    assert verify(old, osig, KEY) is None
    bad, bsig = sign({"uid": 3, "u": "x", "r": "station", "s": None, "d": "central", "n": "A", "ts": time.time()}, KEY)
    assert verify(bad, bsig, KEY) is None
    root, rsig = sign({"uid": 3, "u": "x", "r": "superuser", "s": None, "d": "central", "n": "A", "ts": time.time()}, KEY)
    assert verify(root, rsig, KEY) is None


def test_events_are_scoped_to_station_and_division(tmp_path):
    s = Store(tmp_path / "v.db")
    a, b = ident(station="ps-a", uid=1), ident(station="ps-b", uid=2)
    other = ident(station="ps-z", division="north", uid=3)
    s.record(a, "identify", "car a", detections=[{"label": "X", "level": "class", "confidence": 0.9}])
    s.record(b, "identify", "car b")
    s.record(other, "identify", "car z")
    assert [e["summary"] for e in s.events(a)["events"]] == ["car a"]
    admin_c = ident(role="division_admin", division="central", uid=9)
    assert sorted(e["summary"] for e in s.events(admin_c)["events"]) == ["car a", "car b"]
    admin_n = ident(role="division_admin", division="north", uid=10)
    assert [e["summary"] for e in s.events(admin_n)["events"]] == ["car z"]
    eid = s.events(admin_c)["events"][0]["id"]
    assert s.event(admin_n, eid) is None
    assert s.delete_event(admin_n, eid) is False
    assert s.delete_event(admin_c, eid) is True
    assert len(s.events(admin_c)["events"]) == 1


def test_overview_counts_only_own_division(tmp_path):
    s = Store(tmp_path / "v.db")
    s.record(ident(station="ps-a"), "identify", "x", detections=[{"label": "M", "make": "Tata", "level": "class", "confidence": 0.95, "catalogue": "india"}])
    s.record(ident(station="ps-z", division="north"), "identify", "y", detections=[{"label": "N", "level": "class", "confidence": 0.5}])
    ov = s.overview(ident(role="division_admin"), time.time() - 86400, time.time() + 60)
    assert ov["kpis"]["identifications"] == 1 and ov["kpis"]["vehicles"] == 1
    assert ov["top_makes"] == [{"make": "Tata", "n": 1}]
    assert sum(b["n"] for b in ov["confidence_hist"]) == 1


def test_case_access_rules():
    case = {"id": "c1", "station_id": "ps-a", "division_id": "central"}
    assert CaseStore.allowed(case, ident(station="ps-a"))
    assert not CaseStore.allowed(case, ident(station="ps-b"))
    assert CaseStore.allowed(case, ident(role="division_admin", division="central"))
    assert not CaseStore.allowed(case, ident(role="division_admin", division="north"))
    assert not CaseStore.allowed({**case, "deleted_at": 1.0}, ident(station="ps-a"))
    assert not CaseStore.allowed({"id": "legacy"}, ident(role="division_admin"))


def test_boundary_membership():
    sq = [[(77.0, 12.0), (78.0, 12.0), (78.0, 13.0), (77.0, 13.0), (77.0, 12.0)]]
    b = Boundary([sq], "test")
    assert b.contains(12.5, 77.5)
    assert not b.contains(13.5, 77.5)
    assert not b.contains(float("nan"), 77.5)
