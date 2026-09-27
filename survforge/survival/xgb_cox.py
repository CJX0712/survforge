"""Tier-0 backend: XGBoost survival:cox objective (boosted partial likelihood).

Same SOTA package as the AFT member but a different loss (Cox partial
likelihood vs accelerated failure time) -> comparable strength with
decorrelated errors, which is what the fusion flagship needs.
Labels: time if event, -time if censored (xgboost convention).
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E300BackendError
from ..core.types import SurvDataset

try:  # optional Tier-0 dependency
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    XGB_AVAILABLE = False


class XgbCoxBackend:
    name = "xgb_cox"
    tier = "tier0"

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 3,
        learning_rate: float = 0.08,
        min_child_weight: float = 10.0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self._booster = None

    def fit(self, train: SurvDataset, seed: int = 0) -> XgbCoxBackend:
        if not XGB_AVAILABLE:
            raise E300BackendError("xgboost is not installed")
        y = np.where(train.event, train.time, -train.time).astype(np.float32)
        dtrain = xgb.DMatrix(train.X, label=y)
        params = {
            "objective": "survival:cox",
            "tree_method": "hist",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "seed": int(seed),
            "nthread": 1,
        }
        self._booster = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        return self

    def predict_risk(self, X) -> np.ndarray:
        if self._booster is None:
            raise E300BackendError("XgbCoxBackend not fitted")
        # hazard ratio; higher = higher risk
        return self._booster.predict(xgb.DMatrix(np.asarray(X, dtype=np.float64)))
