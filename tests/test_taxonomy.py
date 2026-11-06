from pathlib import Path

from p5.taxonomy import Taxonomy, parse_name

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "taxonomy"


def test_counts():
    t = Taxonomy.load(ASSETS)
    assert len(t.names) == 196
    assert len(t.makes) == 49
    assert len(t.display_makes) == 47
    assert len(t.family_names) == 189


def test_multi_word_makes():
    makes = ["AM General", "Aston Martin", "Land Rover", "Audi"]
    assert parse_name("AM General Hummer SUV 2000", makes) == ("AM General", "Hummer SUV", 2000)
    assert parse_name("Land Rover Range Rover SUV 2012", makes) == ("Land Rover", "Range Rover SUV", 2012)


def test_year_twins_share_family():
    t = Taxonomy.load(ASSETS)
    for a, b in t.year_twins:
        assert t.family_of[a] == t.family_of[b]
        assert t.year_of[a] != t.year_of[b]


def test_rebadge_names():
    t = Taxonomy.load(ASSETS)
    names = [t.names[i] for i in t.rebadge_twins["express_savana"]]
    assert any("Express" in n for n in names) and any("Savana" in n for n in names)
