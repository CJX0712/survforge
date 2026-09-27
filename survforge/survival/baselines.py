"""Naive baselines: seeded random risk and single-feature clinical guess."""

from __future__ import annotations

import numpy as np

from ..core.types import SurvDataset


class RandomRiskBackend:
    """Baseline: seeded random risk scores (must be far below every model)."""

    name = "naive_random"
    tier = "baseline"

    def fit(self, train: SurvDataset, seed: int = 0) -> RandomRiskBackend:
        self._seed = int(seed)
        return self

    def predict_risk(self, X) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        return rng.random(np.asarray(X).shape[0])


class FirstFeatureBackend:
    """Baseline: risk = x0 (uses only the strongest single covariate)."""

    name = "naive_x0"
    tier = "baseline"

    def fit(self, train: SurvDataset, seed: int = 0) -> FirstFeatureBackend:
        return self

    def predict_risk(self, X) -> np.ndarray:
        return np.asarray(X, dtype=np.float64)[:, 0].copy()
