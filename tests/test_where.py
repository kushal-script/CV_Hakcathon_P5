import io

import pytest
from PIL import Image

from p5.geo.where import names_in, where_taken


class Read:
    def __init__(self, pairs):
        self.txts = [t for t, _ in pairs]
        self.scores = [s for _, s in pairs]


def ocr_of(pairs):
    return lambda rgb: Read(pairs)


def places_with(names, spacing_deg=0.05):
    rows, where = [], {}
    for i, (name, n) in enumerate(names):
        for k in range(n):
            where.setdefault(name.lower(), []).append(len(rows))
            rows.append({"label": name, "c": "highway=primary", "x": 77.6 + 0.2 * i + spacing_deg * k, "y": 12.97})
    return {"rows": rows, "where": where}


def plain_jpeg(exif=None):
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 120, 120)).save(buf, "JPEG", **({"exif": exif} if exif is not None else {}))
    return buf.getvalue()


def never():
    raise AssertionError("the image should not be read when the photo has GPS")


def inside_city(lat, lon):
    return 12.8 < lat < 13.2 and 77.4 < lon < 77.8


def test_gps_in_photo_is_reported_without_reading_text():
    exif = Image.Exif()
    exif[0x8825] = {1: "N", 2: (12.0, 58.0, 32.52), 3: "E", 4: (77.0, 36.0, 22.32)}
    exif[0x8769] = {36867: "2026:11:07 09:30:00", 36881: "+05:30"}
    exif[271], exif[272] = "TestMake", "TestModel"
    out = where_taken(plain_jpeg(exif), never, ocr_of([]), places_with([]), inside_city)
    assert out["status"] == "gps"
    assert out["gps"]["lat"] == pytest.approx(12.9757, abs=1e-4)
    assert out["gps"]["lon"] == pytest.approx(77.6062, abs=1e-4)
    assert out["gps"]["inside_bengaluru"] is True
    assert out["taken_at"] == "2026-11-07T09:30:00+05:30" and out["time_zone_recorded"] is True
    assert out["camera"] == "TestMake TestModel"


def test_exact_place_name_on_a_sign_is_a_clue():
    out = where_taken(plain_jpeg(), lambda: None, ocr_of([("BRIGADE ROAD BENGALURU 560025", 0.95)]), places_with([("Brigade Road", 1)]), inside_city)
    assert out["status"] == "clues"
    assert [c["name"] for c in out["clues"]] == ["Brigade Road"]
    assert out["clues"][0]["places_in_city"] == 1 and len(out["clues"][0]["places"]) == 1


def test_road_stored_in_many_pieces_is_one_place():
    out = where_taken(plain_jpeg(), lambda: None, ocr_of([("Brigade Road", 0.95)]), places_with([("Brigade Road", 10)], spacing_deg=0.0005), inside_city)
    assert out["status"] == "clues" and out["clues"][0]["places_in_city"] == 1
    place = out["clues"][0]["places"][0]
    assert place["kinds"] == ["road"] and 200 < place["extent_m"] < 300


def test_name_shared_by_many_places_is_not_a_location():
    out = where_taken(plain_jpeg(), lambda: None, ocr_of([("Cubbon Park", 0.9)]), places_with([("Cubbon Park", 5)]), inside_city)
    assert out["status"] == "unknown"
    assert out["clues"][0]["places_in_city"] == 5 and "places" not in out["clues"][0]
    assert "shared by more than" in out["reason"]


def test_text_that_names_no_place_gives_unknown():
    out = where_taken(plain_jpeg(), lambda: None, ocr_of([("Fresh Juice Centre", 0.9)]), places_with([("Brigade Road", 1)]), inside_city)
    assert out["status"] == "unknown" and out["texts"] == ["Fresh Juice Centre"]
    assert "none of it is the name of a place" in out["reason"]


def test_no_text_and_no_gps_gives_unknown():
    out = where_taken(plain_jpeg(), lambda: None, ocr_of([]), places_with([("Brigade Road", 1)]), inside_city)
    assert out["status"] == "unknown" and "No readable" in out["reason"]


def test_low_confidence_text_is_ignored():
    out = where_taken(plain_jpeg(), lambda: None, ocr_of([("Brigade Road", 0.3)]), places_with([("Brigade Road", 1)]), inside_city)
    assert out["status"] == "unknown" and out["texts"] == []


class Boxed(Read):
    def __init__(self, pairs, boxes):
        super().__init__(pairs)
        self.boxes = boxes


def box(x0, x1, y0=100, y1=160):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_words_split_by_ocr_are_joined_into_their_line():
    ocr = lambda rgb: Boxed([("BRIGADE", 0.95), ("ROAD", 0.95)], [box(40, 400), box(450, 640)])
    places = places_with([("Brigade Road", 1), ("Brigade", 1)])
    out = where_taken(plain_jpeg(), lambda: None, ocr, places, inside_city)
    assert out["texts"] == ["BRIGADE ROAD"]
    assert [c["name"] for c in out["clues"]] == ["Brigade Road"]


def test_signs_far_apart_on_one_row_stay_separate():
    ocr = lambda rgb: Boxed([("CHURCH", 0.95), ("STREET", 0.95)], [box(40, 300), box(1200, 1450)])
    out = where_taken(plain_jpeg(), lambda: None, ocr, places_with([("Church Street", 1)]), inside_city)
    assert out["texts"] == ["CHURCH", "STREET"] and out["status"] == "unknown"


def test_generic_words_alone_never_match():
    where = places_with([("Brigade Road", 1), ("Road", 1), ("Main Road", 1)])["where"]
    assert names_in("BRIGADE ROAD", where) == ["brigade road"]
    assert names_in("MAIN ROAD", where) == []


def test_heat_spreads_one_name_over_its_place():
    out = where_taken(plain_jpeg(), lambda: None, ocr_of([("Brigade Road", 0.95)]), places_with([("Brigade Road", 10)], spacing_deg=0.0005), inside_city)
    assert len(out["heat"]) == 10
    assert sum(w for _, _, w in out["heat"]) == pytest.approx(1.0)


def test_heat_splits_a_shared_name_between_its_places():
    out = where_taken(plain_jpeg(), lambda: None, ocr_of([("Church Street", 0.95)]), places_with([("Church Street", 3)]), inside_city)
    assert out["clues"][0]["places_in_city"] == 3
    assert [round(w, 4) for _, _, w in out["heat"]] == [0.3333] * 3


def test_two_clues_add_their_heat():
    ocr = lambda rgb: Boxed([("BRIGADE ROAD", 0.95), ("CUBBON PARK", 0.95)], [box(40, 600), box(40, 600, 400, 460)])
    out = where_taken(plain_jpeg(), lambda: None, ocr, places_with([("Brigade Road", 1), ("Cubbon Park", 1)]), inside_city)
    assert sum(w for _, _, w in out["heat"]) == pytest.approx(2.0)


def test_no_clue_means_no_heat():
    out = where_taken(plain_jpeg(), lambda: None, ocr_of([("Fresh Juice Centre", 0.9)]), places_with([("Brigade Road", 1)]), inside_city)
    assert out["status"] == "unknown" and "heat" not in out
