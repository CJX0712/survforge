"""Tier-0 backend: XGBoost Accelerated Failure Time (gradient-boosted trees).

Inductive bias is complementary to the linear Cox family — this is what the
fusion flagship exploits. Wheel prior: xgboost ships cp313 win_amd64 wheels
(verified). Censored samples use interval labels [t, +inf).
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


class XgbAftBackend:
    name = "xgb_aft"
    tier = "tier0"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.06,
        min_child_weight: float = 10.0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self._booster = None

    def fit(self, train: SurvDataset, seed: int = 0) -> XgbAftBackend:
        if not XGB_AVAILABLE:
            raise E300BackendError("xgboost is not installed")
        dtrain = xgb.DMatrix(train.X)
        lower = train.time.astype(np.float32)
        upper = np.where(train.event, train.time, np.inf).astype(np.float32)
        dtrain.set_float_info("label_lower_bound", lower)
        dtrain.set_float_info("label_upper_bound", upper)
        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": 1.20,
            "tree_method": "hist",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": 0.90,
            "seed": int(seed),
            "nthread": 1,  # determinism on constrained Windows
        }
        self._booster = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        return self

    def predict_risk(self, X) -> np.ndarray:
        if self._booster is None:
            raise E300BackendError("XgbAftBackend not fitted")
        dtest = xgb.DMatrix(np.asarray(X, dtype=np.float64))
        pred_time = self._booster.predict(dtest)
        # AFT predicts survival time in original scale; shorter time = higher risk
        return -np.log(np.clip(pred_time, 1e-6, None))
