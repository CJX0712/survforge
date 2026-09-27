"""Flagship: C-index-aware rank-fusion ensemble.

Fits heterogeneous members (linear Cox family + boosted-tree AFT), converts
each member's risk scores to average ranks, and fuses with weights selected
on validation concordance. Weight modes: uniform (ablation) / cweighted
(ablation) / hpo (flagship, Optuna TPE).
"""

from __future__ import annotations

import time as _time

import numpy as np

from ..core.errors import E300BackendError
from ..core.types import SurvDataset
from ..eval.metrics import rank_average
from ..hpo.tune import tune_weights


class FusionEnsemble:
    name = "fusion_hpo"
    tier = "tier0"

    def __init__(self, member_factories, weight_mode: str = "hpo") -> None:
        if weight_mode not in ("uniform", "cweighted", "hpo"):
            raise E300BackendError(f"unknown weight_mode {weight_mode!r}")
        self.member_factories = list(member_factories)
        self.weight_mode = weight_mode
        self.members_ = []
        self.weights_: dict | None = None
        self.member_names_: list[str] = []
        self.fit_sec_ = 0.0
        self._hpo_trials = 0
        self._hpo_timeout = 0.0

    def fit(
        self, train: SurvDataset, seed: int = 0, val: SurvDataset | None = None
    ) -> FusionEnsemble:
        if len(self.member_factories) < 2:
            raise E300BackendError("fusion needs >= 2 member backends")
        t0 = _time.perf_counter()
        self.members_ = []
        self.member_names_ = []
        for i, factory in enumerate(self.member_factories):
            member = factory().fit(train, seed + 17 * i)
            self.members_.append(member)
            self.member_names_.append(member.name)

        val_scores = {m.name: m.predict_risk(val.X) for m in self.members_}
        self.val_scores_ = val_scores

        if self.weight_mode == "uniform":
            n = len(self.members_)
            self.weights_ = {m: 1.0 / n for m in self.member_names_}
        elif self.weight_mode == "cweighted":
            c_vals = {}
            for m, name in zip(self.members_, self.member_names_, strict=True):
                from ..eval.metrics import harrell_c_index

                c_vals[name] = harrell_c_index(val.time, val.event, val_scores[name])
            floor = min(c_vals.values()) - 1e-6
            raw = {n: max(v - floor, 1e-3) for n, v in c_vals.items()}
            tot = sum(raw.values())
            self.weights_ = {n: raw[n] / tot for n in raw}
        else:  # hpo
            self.weights_ = tune_weights(
                val_scores,
                val,
                trials=self._hpo_trials,
                timeout=self._hpo_timeout,
                seed=seed,
            )
        self.fit_sec_ = _time.perf_counter() - t0
        return self

    def with_hpo_budget(self, trials: int, timeout: float) -> FusionEnsemble:
        self._hpo_trials = trials
        self._hpo_timeout = timeout
        return self

    def predict_risk(self, X) -> np.ndarray:
        if not self.members_ or self.weights_ is None:
            raise E300BackendError("FusionEnsemble not fitted")
        X = np.asarray(X, dtype=np.float64)
        fused = np.zeros(X.shape[0])
        for m in self.members_:
            w = self.weights_.get(m.name, 0.0)
            fused += w * rank_average(m.predict_risk(X))
        return fused
