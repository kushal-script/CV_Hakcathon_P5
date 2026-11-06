import importlib
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "classifier.onnx").exists(), reason="needs the packaged models")
KEY = "s" * 48


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["P5_INTERNAL_SECRET"] = KEY
    os.environ["P5_DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    from fastapi.testclient import TestClient
    server = importlib.import_module("app.server")
    with TestClient(server.app) as c:
        yield c


def headers(role="station", station="ps-a", division="central"):
    from p5.identity import sign
    enc, sig = sign({"uid": 1, "u": "user", "r": role, "s": station if role == "station" else None, "d": division, "n": "Test", "ts": time.time()}, KEY.encode())
    return {"x-p5-identity": enc, "x-p5-identity-sig": sig}


def test_health_is_public(client):
    assert client.get("/api/health").status_code == 200


def test_identity_is_required(client):
    assert client.get("/api/activity").status_code == 401
    bad = headers()
    bad["x-p5-identity-sig"] = "0" * 64
    assert client.get("/api/activity", headers=bad).status_code == 401


def test_admin_endpoints_need_admin_role(client):
    assert client.get("/api/admin/overview", headers=headers()).status_code == 403
    r = client.get("/api/admin/overview", headers=headers(role="division_admin"))
    assert r.status_code == 200 and "kpis" in r.json()


def test_cases_are_private_to_their_station(client):
    r = client.post("/api/cases", data={"name": "Theft"}, headers=headers(station="ps-a"))
    assert r.status_code == 200
    cid = r.json()["id"]
    assert client.get(f"/api/cases/{cid}", headers=headers(station="ps-a")).status_code == 200
    assert client.get(f"/api/cases/{cid}", headers=headers(station="ps-b")).status_code == 404
    assert client.get(f"/api/cases/{cid}", headers=headers(role="division_admin", division="central")).status_code == 200
    assert client.get(f"/api/cases/{cid}", headers=headers(role="division_admin", division="north")).status_code == 404
    assert client.patch(f"/api/cases/{cid}", json={"station_id": "ps-b"}, headers=headers(station="ps-a")).status_code == 403
