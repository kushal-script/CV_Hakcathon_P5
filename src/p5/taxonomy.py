from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MULTI_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")
DISPLAY_MERGE = {"AM General": "HUMMER", "Ram": "Dodge"}
YEAR_TWINS_1B = [(23, 24), (42, 43), (83, 84), (94, 95), (113, 114), (126, 127), (190, 191)]
REBADGE_TWINS_1B = {
    "express_savana": [64, 71, 119],
    "sprinter": [88, 166],
    "tahoe_yukon_hybrid": [62, 120],
    "aspen_durango": [76, 95],
    "gmt360": [48, 68, 143],
    "town_country_ram_cv": [78, 174],
}


def parse_name(name: str, makes: list[str]) -> tuple[str, str, int]:
    tokens = name.split()
    year = int(tokens[-1])
    make = max((m for m in makes if name.startswith(m + " ")), key=len)
    model = name[len(make):].rsplit(" ", 1)[0].strip()
    return make, model, year


def candidate_makes(names: list[str]) -> list[str]:
    makes = set()
    for n in names:
        multi = next((m for m in MULTI_WORD_MAKES if n.startswith(m + " ")), None)
        makes.add(multi or n.split()[0])
    return sorted(makes)


@dataclass
class Taxonomy:
    names: list[str]
    make_of: list[str]
    model_of: list[str]
    year_of: list[int]
    display_make_of: list[str]
    family_of: list[int]
    family_names: list[str]
    year_twins: list[tuple[int, int]] = field(default_factory=list)
    rebadge_twins: dict[str, list[int]] = field(default_factory=dict)

    @property
    def makes(self) -> list[str]:
        return sorted(set(self.make_of))

    @property
    def display_makes(self) -> list[str]:
        return sorted(set(self.display_make_of))

    @classmethod
    def from_names(cls, names: list[str]) -> "Taxonomy":
        makes = candidate_makes(names)
        parsed = [parse_name(n, makes) for n in names]
        make_of = [p[0] for p in parsed]
        model_of = [p[1] for p in parsed]
        year_of = [p[2] for p in parsed]
        display = [DISPLAY_MERGE.get(m, m) for m in make_of]
        stems = [n.rsplit(" ", 1)[0] for n in names]
        family_names: list[str] = []
        family_of = []
        for s in stems:
            if s not in family_names:
                family_names.append(s)
            family_of.append(family_names.index(s))
        twins = [(a - 1, b - 1) for a, b in YEAR_TWINS_1B]
        rebadge = {k: [i - 1 for i in v] for k, v in REBADGE_TWINS_1B.items()}
        return cls(names, make_of, model_of, year_of, display, family_of, family_names, twins, rebadge)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        classes = [
            {"id": i, "name": n, "make": self.make_of[i], "model": self.model_of[i], "year": self.year_of[i],
             "display_make": self.display_make_of[i], "family": self.family_of[i]}
            for i, n in enumerate(self.names)
        ]
        (directory / "classes.json").write_text(json.dumps(classes, indent=1))
        (directory / "families.json").write_text(json.dumps(self.family_names, indent=1))
        groups = {"year_twins": self.year_twins, "rebadge_twins": self.rebadge_twins}
        (directory / "groups.json").write_text(json.dumps(groups, indent=1))

    @classmethod
    def load(cls, directory: Path) -> "Taxonomy":
        classes = json.loads((directory / "classes.json").read_text())
        families = json.loads((directory / "families.json").read_text())
        groups = json.loads((directory / "groups.json").read_text())
        return cls(
            names=[c["name"] for c in classes],
            make_of=[c["make"] for c in classes],
            model_of=[c["model"] for c in classes],
            year_of=[c["year"] for c in classes],
            display_make_of=[c["display_make"] for c in classes],
            family_of=[c["family"] for c in classes],
            family_names=families,
            year_twins=[tuple(p) for p in groups["year_twins"]],
            rebadge_twins=groups["rebadge_twins"],
        )
