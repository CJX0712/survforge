"""Tier-0 backend: lifelines Cox proportional hazards (Breslow/Newton)."""

from __future__ import annotations

import numpy as np

from ..core.errors import E300BackendError
from ..core.types import SurvDataset

try:  # optional Tier-0 dependency
    import pandas as pd
    from lifelines import CoxPHFitter

    LIFELINES_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by fallback test
    LIFELINES_AVAILABLE = False


class LifelinesCoxBackend:
    name = "lifelines_cox"
    tier = "tier0"

    def __init__(self, penalizer: float = 0.01) -> None:
        self.penalizer = penalizer
        self._model = None

    def fit(self, train: SurvDataset, seed: int = 0) -> LifelinesCoxBackend:
        if not LIFELINES_AVAILABLE:
            raise E300BackendError("lifelines is not installed")
        df = pd.DataFrame(np.asarray(train.X, dtype=np.float64))
        df.columns = [f"f{i}" for i in range(train.n_features)]
        df["_time"] = train.time
        df["_event"] = train.event.astype(int)
        cph = CoxPHFitter(penalizer=self.penalizer)
        cph.fit(df, duration_col="_time", event_col="_event")
        self._model = cph
        self._columns = list(df.columns[:-2])
        return self

    def predict_risk(self, X) -> np.ndarray:
        if self._model is None:
            raise E300BackendError("LifelinesCoxBackend not fitted")
        df = pd.DataFrame(np.asarray(X, dtype=np.float64))
        df.columns = self._columns
        # log partial hazard = linear predictor; higher = higher risk
        return self._model.predict_log_partial_hazard(df).to_numpy(dtype=np.float64)
