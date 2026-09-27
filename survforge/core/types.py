"""Shared data types (dataclasses only, stdlib + numpy)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SurvDataset:
    """Censored survival dataset.

    time : observed time (event time if event else censoring time)
    event : boolean event indicator (True = event observed)
    true_risk : optional ground-truth DGP risk (failure analysis only,
                never visible to models)
    """

    X: np.ndarray
    time: np.ndarray
    event: np.ndarray
    feature_names: Sequence[str] = field(default_factory=tuple)
    true_risk: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.time = np.asarray(self.time, dtype=np.float64)
        self.event = np.asarray(self.event, dtype=bool)
        if self.X.ndim != 2:
            raise ValueError("X must be 2-D (n_samples, n_features)")
        n = self.X.shape[0]
        if self.time.shape != (n,) or self.event.shape != (n,):
            raise ValueError("time/event must be 1-D of length n_samples")
        if not np.all(np.isfinite(self.X)):
            raise ValueError("X contains non-finite values")
        if np.any(self.time <= 0):
            raise ValueError("survival times must be positive")
        if self.feature_names is None or len(self.feature_names) == 0:
            self.feature_names = tuple(f"x{i}" for i in range(self.X.shape[1]))

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def split(self, idx: np.ndarray) -> SurvDataset:
        return SurvDataset(
            X=self.X[idx],
            time=self.time[idx],
            event=self.event[idx],
            feature_names=self.feature_names,
            true_risk=(None if self.true_risk is None else self.true_risk[idx]),
        )


@dataclass
class BackendResult:
    """One backend evaluated on one holdout split."""

    backend: str
    c_index: float
    ipcw_c_index: float
    fit_sec: float
    pred_sec: float
    tier: str  # "tier0" | "tier1" | "baseline"


@dataclass
class BenchmarkRow:
    backend: str
    tier: str
    c_mean: float
    c_std: float
    ipcw_mean: float
    ipcw_std: float
    significant: bool | None = None
