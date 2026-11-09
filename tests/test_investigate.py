import numpy as np

from p5.config import hunt_model

from p5.classical.colour import vehicle_colour
from p5.detection.detector import Detection
from p5.geo.route import parse_track, position_at
from p5.investigate.hints import parse_places, parse_times
from p5.investigate.hunt import colour_match, score
from p5.investigate.routes import dist_to_line_m
from p5.video.tracker import Tracker, suppress_nested


def test_tracker_keeps_identity_across_frames():
    tr = Tracker(min_hits=2)
    for f in range(10):
        tr.update([Detection((10 + 5 * f, 10, 60 + 5 * f, 50), 0.9, 2), Detection((300 - 5 * f, 100, 350 - 5 * f, 140), 0.9, 2)], f)
    tracks = tr.all_tracks()
    assert len(tracks) == 2
    assert all(t.hits == 10 for t in tracks)


def test_nested_boxes_are_suppressed():
    big = Detection((0, 0, 100, 100), 0.9, 2)
    inner = Detection((20, 20, 40, 40), 0.6, 2)
    other = Detection((200, 0, 260, 60), 0.8, 2)
    assert len(suppress_nested([big, inner, other])) == 2


def test_colour_names():
    white = np.full((60, 120, 3), 235, np.uint8)
    red = np.zeros((60, 120, 3), np.uint8)
    red[..., 0] = 200
    assert vehicle_colour(white)["name"] == "white"
    assert vehicle_colour(red)["name"] == "red"
    assert colour_match("white", "silver") == 0.5 and colour_match("red", "blue") == 0.0


def test_hint_parsing():
    places = parse_places("Seen near Brigade Road, heading towards Indiranagar. Avoiding Silk Board.")
    roles = {p["text"]: p["role"] for p in places}
    assert roles == {"Brigade Road": "via", "Indiranagar": "destination", "Silk Board": "avoid"}
    times = parse_times("after 9:30 pm and before 23:15")
    assert [t["minutes"] for t in times] == [21 * 60 + 30, 23 * 60 + 15]
    assert times[0]["qualifier"] == "after"


def test_score_prefers_identical_track():
    e = np.zeros(8)
    e[0] = 1
    target = {"embedding": e.tolist(), "probs": [1.0, 0.0], "colour": {"name": "blue"}}
    same = {"embedding": e.tolist(), "probs": [1.0, 0.0], "colour": {"name": "blue"}}
    other = {"embedding": np.roll(e, 1).tolist(), "probs": [0.0, 1.0], "colour": {"name": "red"}}
    assert score(target, same)["score"] >= hunt_model()["threshold"] > score(target, other)["score"]


def test_track_parsing_and_interpolation():
    pts = parse_track(b"lat,lon,t\n12.9752,77.6072,0\n12.9730,77.6069,20\n", "x.csv")
    lat, lon = position_at(pts, 10)
    assert abs(lat - 12.9741) < 1e-4
    assert dist_to_line_m((12.9741, 77.60705), [[77.6072, 12.9752], [77.6069, 12.9730]]) < 5


def test_video_summary_plurals():
    from p5.investigate.service import video_summary
    res = {"counts": {"total": 13, "by_class": {"bus": 11, "car": 1, "motorcycle": 1}}, "alerts": []}
    assert video_summary(res) == "13 vehicles tracked (11 buses, 1 car, 1 motorcycle)"
    one = {"counts": {"total": 1, "by_class": {"truck": 1}}, "alerts": []}
    assert video_summary(one) == "1 vehicle tracked (1 truck)"
