from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ClassicalModel:
    mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray
    w: np.ndarray
    b: np.ndarray
    temperature: float = 1.0

    def scores(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        z = (z - self.pca_mean) @ self.components.T
        return z @ self.w.T + self.b

    def proba(self, x: np.ndarray) -> np.ndarray:
        s = self.scores(x) / self.temperature
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, scale=self.scale, pca_mean=self.pca_mean, components=self.components,
                 w=self.w, b=self.b, temperature=np.array(self.temperature))

    @classmethod
    def load(cls, path: Path) -> "ClassicalModel":
        d = np.load(path)
        return cls(d["mean"], d["scale"], d["pca_mean"], d["components"], d["w"], d["b"], float(d["temperature"]))


def fit(x: np.ndarray, y: np.ndarray, n_components: int = 256, c: float = 0.05, seed: int = 42) -> ClassicalModel:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    scaler = StandardScaler().fit(x)
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    z = (x - scaler.mean_) / scale
    pca = PCA(n_components=min(n_components, z.shape[1], z.shape[0]), random_state=seed).fit(z)
    keep = pca.explained_variance_ > 1e-6 * pca.explained_variance_[0]
    comps = pca.components_[keep] / np.sqrt(pca.explained_variance_[keep])[:, None]
    zp = (z - pca.mean_) @ comps.T
    svm = LinearSVC(C=c, max_iter=5000, random_state=seed).fit(zp, y)
    return ClassicalModel(scaler.mean_, scale, pca.mean_, comps, svm.coef_, svm.intercept_)
