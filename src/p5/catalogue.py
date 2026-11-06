from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from p5.classification.classifier import Classifier
from p5.confidence.policy import Decision, Thresholds
from p5.taxonomy import Taxonomy

LEVEL_RANK = {"class": 0, "family": 1, "make": 2, "abstain": 3, "ood_make": 4, "ood": 5, "no_vehicle": 6}


@dataclass
class Catalogue:
    key: str
    title: str
    tax: Taxonomy
    th: Thresholds
    classifier: Classifier
    fc_weight: np.ndarray

    @classmethod
    def load(cls, key: str, title: str, folder: Path, threads: int, serve_size: int | None = None) -> "Catalogue":
        tax = Taxonomy.load(folder / "taxonomy")
        return cls(key, title.format(n=len(tax.names)), tax, Thresholds.load(folder / "thresholds.yaml"),
                   Classifier(folder / "classifier.onnx", folder / "classifier_meta.json", threads, serve_size),
                   np.load(folder / "head.npz")["fc_weight"])

    def same_input(self, other: "Catalogue") -> bool:
        a, b = self.classifier, other.classifier
        return a.size == b.size and a.pad == b.pad


def primary(decisions: list[Decision], prefer: int = 0) -> int:
    def key(k: int):
        d = decisions[k]
        return LEVEL_RANK.get(d.level, 9), -d.confidence, k != prefer
    return min(range(len(decisions)), key=key)


def tagged(d: Decision, cat: Catalogue) -> dict:
    return {**asdict(d), "catalogue": cat.key, "catalogue_title": cat.title}
