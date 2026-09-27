"""Optuna weight tuning for the rank-fusion ensemble."""

from __future__ import annotations

import numpy as np

from ..core.types import SurvDataset
from ..eval.metrics import harrell_c_index, rank_average


def _fuse(val_scores: dict, weights: dict, val: SurvDataset) -> float:
    fused = np.zeros(val.n_samples)
    for name, w in weights.items():
        fused += w * rank_average(val_scores[name])
    return harrell_c_index(val.time, val.event, fused)


def tune_weights(
    val_scores: dict,
    val: SurvDataset,
    trials: int,
    timeout: float,
    seed: int,
) -> dict:
    """Tune per-member weights on validation C-index (Optuna TPE).

    Fallbacks (honest degradation): trials<=0 or timeout<=0 or Optuna error
    -> C-index-proportional weights.
    """
    names = sorted(val_scores.keys())
    if len(names) < 2:
        return {n: 1.0 for n in names}

    # fallback: c-weighted
    c_vals = {n: harrell_c_index(val.time, val.event, val_scores[n]) for n in names}
    floor = min(c_vals.values()) - 1e-6
    raw = {n: max(c_vals[n] - floor, 1e-3) for n in names}
    tot = sum(raw.values())
    cweights = {n: raw[n] / tot for n in names}

    if trials <= 0 or timeout <= 0:
        return cweights

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=int(seed)),
        )

        def objective(trial):
            ws = {n: trial.suggest_float(n, 0.0, 1.0) for n in names}
            tot_w = sum(ws.values())
            if tot_w <= 0:
                return 0.5
            ws = {n: w / tot_w for n, w in ws.items()}
            return _fuse(val_scores, ws, val)

        study.optimize(
            objective,
            n_trials=int(trials),
            timeout=float(timeout),
            show_progress_bar=False,
        )
        best = dict(study.best_params)
        tot_best = sum(best.values())
        if tot_best <= 0:
            return cweights
        return {n: w / tot_best for n, w in best.items()}
    except Exception:
        return cweights
