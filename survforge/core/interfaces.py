"""Backend protocol — the single contract every backend must honour.

Contract semantics: ``fit(train)`` then ``predict_risk(X)`` returns scores
where HIGHER = HIGHER risk (shorter survival).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import SurvDataset


@runtime_checkable
class SurvivalBackend(Protocol):
    name: str
    tier: str

    def fit(self, train: SurvDataset, seed: int) -> SurvivalBackend: ...

    def predict_risk(self, X) -> np.ndarray: ...
