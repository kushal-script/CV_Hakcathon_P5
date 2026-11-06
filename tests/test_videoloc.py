import json
import math
import subprocess

import numpy as np
import pytest

from p5.geo.videoloc import metadata
from p5.geo.videoloc.filter import Params
from p5.geo.videoloc.graph import RoadGraph, wrap
from p5.geo.videoloc.search import search

LAT0, LON0 = 12.97, 77.59


def lonlat(x, y):
    return LON0 + x / (111320 * math.cos(math.radians(LAT0))), LAT0 + y / 110540


def network(tmp_path, copies=1, seed=3, gap=3000.0, name="g.npz"):
    rng = np.random.default_rng(seed)
    n, step = 6, 130.0
    jitter = rng.uniform(-40, 40, (n, n, 2))
    keep_h = rng.random((n, n)) > 0.15
    keep_v = rng.random((n, n)) > 0.15
    nodes, u, v, coords, offsets = [], [], [], [], [0]
    for c in range(copies):
        base = len(nodes)
        for i in range(n):
            for j in range(n):
                nodes.append(lonlat(c * gap + j * step + jitter[i, j, 0], i * step + jitter[i, j, 1]))
        for i in range(n):
            for j in range(n):
                for di, dj, keep in ((0, 1, keep_h), (1, 0, keep_v)):
                    if i + di < n and j + dj < n and keep[i, j]:
                        a, b = base + i * n + j, base + (i + di) * n + (j + dj)
                        u.append(a)
                        v.append(b)
                        coords.extend([nodes[a], nodes[b]])
                        offsets.append(len(coords))
    E = len(u)
    path = tmp_path / name
    np.savez_compressed(path, node_xy=np.array(nodes), u=np.array(u, dtype=np.int32), v=np.array(v, dtype=np.int32), oneway=np.zeros(E, dtype=np.int8),
                        coords=np.array(coords), offsets=np.array(offsets, dtype=np.int64), hw=np.zeros(E, dtype=np.int8), name=np.zeros(E, dtype=np.int32),
                        names=np.array([""], dtype=object), classes=np.array(["residential"], dtype=object), osm_source=np.array("test"), osm_date=np.array("test"))
    return RoadGraph(path)


def drive(g, rng, seconds=70.0, speed=8.0, dt=0.2):
    d = int(rng.integers(g.D))
    dirs, total = [d], g.d_len[d]
    while total < speed * seconds + 60:
        a, b = g.succ_ptr[d], g.succ_ptr[d + 1]
        options = [x for x in g.succ_idx[a:b] if g.d_end[x] != g.d_start[d]] or list(g.succ_idx[a:b])
        d = int(options[rng.integers(len(options))])
        dirs.append(d)
        total += g.d_len[d]
    xs, ys = [], []
    for d in dirs:
        s = np.arange(0.0, g.d_len[d], 1.0)
        x, y, _ = g.point(np.full(len(s), d), s)
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    br = np.unwrap(np.arctan2(np.diff(x), np.diff(y)))
    k = np.ones(15) / 15
    br = np.convolve(np.pad(br, 7, mode="edge"), k, mode="valid")
    n = int(seconds / dt)
    dist = 5 + speed * dt * np.arange(n)
    i = np.minimum(dist.astype(int), len(br) - 1)
    psi = br[i] - br[i[0]]
    recs = [{"t": k * dt, "psi": float(psi[k]), "ok": True, "ground_rate": speed / 1.65} for k in range(n)]
    return recs, x[i[-1]], y[i[-1]]


def params():
    return Params(scale_prior=(1.65, 1.65), scale_prior_log_sd=0.05, anneal_s=10.0)


def test_graph_respects_direction_and_geometry(tmp_path):
    g = network(tmp_path)
    assert g.D == 2 * len(np.load(tmp_path / "g.npz")["u"])
    d = 0
    x, y, b = g.point(np.array([d]), np.array([0.0]))
    sx, sy = g.proj.to_xy(*np.load(tmp_path / "g.npz")["coords"][0])
    assert abs(x[0] - sx) < 1e-6 and abs(y[0] - sy) < 1e-6
    succ = g.succ_idx[g.succ_ptr[d]:g.succ_ptr[d + 1]]
    assert all(g.d_start[s] == g.d_end[d] for s in succ)


def test_finds_a_drive_on_an_irregular_network(tmp_path):
    g = network(tmp_path)
    recs, tx, ty = drive(g, np.random.default_rng(5))
    cx, cy = g.d_mid_xy.mean(0)
    res = search(g, recs, (float(cx), float(cy)), 2000.0, params(), n_lo=20000, n_hi=20000, std_runs=2, alpha_runs=0, alt_runs=1, workers=1)
    hx, hy, _ = res["best"]["hyps"][0]
    assert math.hypot(hx - tx, hy - ty) < 40
    assert res["confidence"] > 0.9


def test_reports_ambiguity_when_two_places_look_the_same(tmp_path):
    g = network(tmp_path, copies=2)
    recs, tx, ty = drive(g, np.random.default_rng(5))
    cx, cy = g.d_mid_xy.mean(0)
    res = search(g, recs, (float(cx), float(cy)), 4000.0, params(), n_lo=40000, n_hi=40000, std_runs=2, alpha_runs=0, alt_runs=2, workers=1)
    assert res["confidence"] < 0.9


def test_embedded_gps_track_is_read(monkeypatch):
    out = [{"SourceFile": "v.mp4", "Doc1:GPSLatitude": 12.9716, "Doc1:GPSLongitude": 77.5946, "Doc1:SampleTime": 0.0, "Doc1:GPSDateTime": "2026:11:06 12:00:00Z",
            "Doc2:GPSLatitude": 12.9720, "Doc2:GPSLongitude": 77.5950, "Doc2:SampleTime": 1.0, "Doc2:GPSDateTime": "2026:11:06 12:00:01Z", "Main:Duration": 2.0}]
    monkeypatch.setattr(metadata.shutil, "which", lambda name: "/usr/bin/exiftool")
    monkeypatch.setattr(metadata.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(out).encode(), b""))
    r = metadata.embedded_gps("v.mp4")
    assert r["source"].startswith("track") and len(r["points"]) == 2
    assert r["points"][1]["t"] == 1.0 and r["points"][1]["epoch"] - r["points"][0]["epoch"] == pytest.approx(1.0)


def test_single_location_and_missing_tool(monkeypatch):
    monkeypatch.setattr(metadata.shutil, "which", lambda name: None)
    assert metadata.embedded_gps("v.mp4") is None
    out = [{"Main:GPSCoordinates": "12.9716 77.5946 920"}]
    monkeypatch.setattr(metadata.shutil, "which", lambda name: "/usr/bin/exiftool")
    monkeypatch.setattr(metadata.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(out).encode(), b""))
    r = metadata.embedded_gps("v.mp4")
    assert r["points"][0]["lat"] == pytest.approx(12.9716) and len(r["points"]) == 1


ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
KEY = "s" * 48


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import importlib
    import os
    if not (ROOT / "models" / "classifier.onnx").exists():
        pytest.skip("needs the packaged models")
    os.environ["P5_INTERNAL_SECRET"] = KEY
    os.environ["P5_DATA_DIR"] = str(tmp_path_factory.mktemp("geo"))
    from fastapi.testclient import TestClient
    server = importlib.import_module("app.server")
    with TestClient(server.app) as c:
        yield c


def who(station="ps-a"):
    import time
    from p5.identity import sign
    enc, sig = sign({"uid": 1, "u": "user", "r": "station", "s": station, "d": "central", "n": "Test", "ts": time.time()}, KEY.encode())
    return {"x-p5-identity": enc, "x-p5-identity-sig": sig}


def test_geolocate_endpoint_checks_identity_and_inputs(client):
    cid = client.post("/api/cases", data={"name": "Footage"}, headers=who()).json()["id"]
    url = f"/api/cases/{cid}/cameras/nocam/geolocate"
    good = {"lat": 12.9716, "lon": 77.5946, "radius_m": 1500, "hfov_deg": 70}
    assert client.post(url, data=good).status_code == 401
    assert client.post(url, data={**good, "lat": 19.07, "lon": 72.87}, headers=who()).status_code == 422
    assert client.post(url, data={**good, "hfov_deg": 5}, headers=who()).status_code == 422
    assert client.post(url, data={**good, "radius_m": 9000}, headers=who()).status_code == 422
    assert client.post(url, data=good, headers=who()).status_code == 404
    assert client.post(url, data=good, headers=who("ps-b")).status_code == 404
    assert client.post(f"/api/cases/{cid}/cameras/nocam/geolocation/accept", headers=who()).status_code == 404


def test_signboards_support_or_contradict_a_route(tmp_path):
    import gzip
    from p5.geo.videoloc import signs as S
    rows = [{"n": "empire restaurant koramangala", "label": "Empire Restaurant", "x": 77.6241, "y": 12.9352, "c": "amenity=restaurant", "o": "node/1"},
            {"n": "forum south bengaluru", "label": "Forum South Bengaluru", "x": 77.5700, "y": 12.8900, "c": "shop=mall", "o": "way/2"}]
    f = tmp_path / "p.json.gz"
    f.write_bytes(gzip.compress(json.dumps(rows).encode()))
    places = S.load_places(f)
    route = [{"t": 0.0, "lat": 12.9350, "lon": 77.6238}, {"t": 30.0, "lat": 12.9360, "lon": 77.6250}]
    read = [{"t": 10.0, "text": "EMPIRE RESTAURANT KORAMANGALA", "norm": S.norm("EMPIRE RESTAURANT KORAMANGALA"), "ocr_conf": 0.9},
            {"t": 20.0, "text": "Forum South Bengaluru", "norm": S.norm("Forum South Bengaluru"), "ocr_conf": 0.9}]
    sup = S.match_signs(read, route, places)
    con = S.conflicting_signs(read, route, places)
    assert [s["matched"] for s in sup] == ["Empire Restaurant"] and sup[0]["distance_m"] < 100
    assert [s["matched"] for s in con] == ["Forum South Bengaluru"] and con[0]["distance_from_route_m"] > 5000
    assert not S.distinctive(S.norm("Main Road")) and S.distinctive(S.norm("Lavelle Road"))
