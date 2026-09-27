"""Spline-Cox backend: Cox PH on restricted cubic spline basis.

Classical flexible-Cox method (Durrleman & Simon 1989; Harrell's rms
methodology) implemented on the Tier-1 numpy engine — captures smooth
nonlinear effects the linear Cox misses, complementing tree members.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E300BackendError
from ..core.types import SurvDataset
from .numpy_cox import cox_fit
from .splines import spline_expand


class SplineCoxBackend:
    name = "spline_cox"
    tier = "tier1"

    def __init__(self, n_knots: int = 5, ridge: float = 1e-3) -> None:
        self.n_knots = n_knots
        self.ridge = ridge
        self.beta_: np.ndarray | None = None

    def fit(self, train: SurvDataset, seed: int = 0) -> SplineCoxBackend:
        Xe = spline_expand(train.X, self.n_knots)
        self.beta_, _ = cox_fit(Xe, train.time, train.event, ridge=self.ridge)
        return self

    def predict_risk(self, X) -> np.ndarray:
        if self.beta_ is None:
            raise E300BackendError("SplineCoxBackend not fitted")
        Xe = spline_expand(np.asarray(X, dtype=np.float64), self.n_knots)
        return Xe @ self.beta_
