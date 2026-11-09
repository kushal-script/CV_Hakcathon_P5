import json
import struct

import cv2
import numpy as np

from p5.classification.vehicle_type import VehicleType, crop_quality, unnamed_reason
from p5.geo.route import mp4_location
from p5.geo.videoloc.trace import sign_anchor, trace_video
from p5.identity import Identity
from p5.investigate.service import CaseStore

CFG = {"name_min_side": 112, "name_min_luma": 0.08, "name_min_sharpness": 20.0}


def textured(h, w, level=0.5, seed=0):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(level, 0.2, (h, w, 3)), 0, 1)
    return (img * 255).astype(np.uint8)


def test_naming_needs_a_large_bright_sharp_crop():
    assert unnamed_reason(crop_quality(textured(200, 260), 0.10), CFG) is None
    assert "too small" in unnamed_reason(crop_quality(textured(60, 80), 0.10), CFG)
    assert unnamed_reason(crop_quality(textured(200, 260, level=0.03), 0.10), CFG) == "too dark to name the make or model"
    blurred = cv2.GaussianBlur(textured(200, 260), (0, 0), 12)
    assert unnamed_reason(crop_quality(blurred, 0.10), CFG) == "too blurred to name the make or model"


def test_vehicle_type_probability(tmp_path):
    f = tmp_path / "vt.json"
    f.write_text(json.dumps({"coef": [2.0, 0.0], "intercept": -1.0, "threshold": 0.7}))
    vt = VehicleType(f)
    assert vt.autorickshaw(np.array([1.0, 0.0])) > 0.7
    assert vt.autorickshaw(np.array([0.0, 1.0])) < 0.5


def test_location_formats(tmp_path):
    a = tmp_path / "a.mp4"
    a.write_bytes(b"\x00" * 40 + b"\xa9xyz\x00\x12\x15\xc7+12.9753+077.6050/" + b"\x00" * 40)
    assert mp4_location(str(a)) == {"lat": 12.9753, "lon": 77.605, "source": "video metadata"}
    b = tmp_path / "b.webm"
    b.write_bytes(b"\x00" * 40 + b"LOCATION\x44\x87\x92+13.0110+077.5836/" + b"\x00" * 40)
    assert mp4_location(str(b))["lat"] == 13.011
    c = tmp_path / "c.mp4"
    body = b"\x00\x00\x00\x00" + b"\x15\xc7" + b"\x00" + b"\x00" + struct.pack(">iii", int(77.5836 * 65536), int(13.011 * 65536), 0) + b"earth\x00\x00"
    c.write_bytes(b"\x00" * 40 + struct.pack(">I", len(body) + 8) + b"loci" + body + b"\x00" * 40)
    loc = mp4_location(str(c))
    assert abs(loc["lat"] - 13.011) < 1e-4 and abs(loc["lon"] - 77.5836) < 1e-4


def test_sign_anchors_need_a_distinctive_place():
    rows = [{"n": "cpri", "label": "CPRI", "x": 77.57, "y": 13.03, "c": "office=research"},
            {"n": "indianoil", "label": "IndianOil", "x": 77.6, "y": 13.0, "c": "amenity=fuel"}] + \
           [{"n": "ganesha temple", "label": "Ganesha Temple", "x": 77.5 + k / 100, "y": 12.9, "c": "amenity=place_of_worship"} for k in range(5)]
    where = {}
    for i, r in enumerate(rows):
        where.setdefault(r["n"], []).append(i)
    places = {"rows": rows, "where": where}
    assert sign_anchor({"t": 4.0, "text": "CPRI", "norm": "cpri"}, places, None)["lat"] == 13.03
    assert sign_anchor({"t": 4.0, "text": "IndianOil", "norm": "indianoil"}, places, None) is None
    assert sign_anchor({"t": 4.0, "text": "Ganesha Temple", "norm": "ganesha temple"}, places, None) is None
    far = {"t": 0.0, "lat": 12.80, "lon": 77.40}
    assert sign_anchor({"t": 5.0, "text": "CPRI", "norm": "cpri"}, places, far) is None


def test_video_without_background_evidence_is_not_traced(tmp_path):
    path = tmp_path / "plain.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for k in range(10):
        w.write(np.full((48, 64, 3), 40 + k, np.uint8))
    w.release()
    t = trace_video(str(path))
    assert t["status"] == "not_traced" and t["route"] is None and t["points"] == []


def test_video_jobs_are_private_to_their_owner(tmp_path):
    store = CaseStore(tmp_path, None, None, None, None)
    store.jobs["v1"] = {"id": "v1", "kind": "video", "status": "done", "owner": 7, "station_id": "ps-a", "division_id": "central"}
    me = Identity(7, "ps-a", "station", "ps-a", "central", "A")
    other = Identity(8, "ps-a-2", "station", "ps-a", "central", "A2")
    assert store.job("v1", me)["status"] == "done"
    try:
        store.job("v1", other)
        raise AssertionError("another account read the job")
    except KeyError:
        pass
