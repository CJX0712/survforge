"""Synthetic survival DGP (fixed seed, reproducible).

Design intent (documented in docs/architecture.md):
- linear part   -> Cox-linear signal
- nonlinear part (sin / hinge-interaction / quadratic) -> tree-based signal
Fusion of both signal families is what the flagship ensemble exploits.
Difficulty knobs (beta_scale, censor_strength) are tuned to a "sweet spot"
where backends differ enough for fusion to add value (not saturated, not
impossible) — same discipline as previous Forge deliveries.
"""

from __future__ import annotations

import numpy as np

from ..core.types import SurvDataset


def make_survival(
    n: int,
    seed: int = 0,
    d: int = 8,
    beta_scale: float = 1.0,
    censor_strength: float = 0.4,
    t_cap: float = 40.0,
) -> SurvDataset:
    """Weibull hazard with nonlinear PH risk + informative censoring.

    h(t|x) = k * lam * t^(k-1) * exp(risk(x))
    risk(x) = beta_scale * [0.9*x0 + 0.7*x1 + sin(1.7*x2)
                            + 0.8*max(x3,0)*x4 - 0.9*x5^2]
    Censoring: exponential with rate depending on risk (informative),
    plus administrative censoring cap ``t_cap``.
    """
    rng = np.random.default_rng(seed)
    if d < 6:
        raise ValueError("DGP needs d >= 6 informative-structure features")
    X = rng.normal(size=(n, d))
    linear = 0.9 * X[:, 0] + 0.7 * X[:, 1]
    nonlinear = (
        np.sin(1.7 * X[:, 2])
        + 0.8 * np.maximum(X[:, 3], 0.0) * X[:, 4]
        - 0.9 * X[:, 5] ** 2
    )
    true_risk = beta_scale * (linear + nonlinear)

    k = 1.2  # Weibull shape
    lam = 0.05  # Weibull scale baseline
    u = rng.uniform(size=n)
    # T = (-ln U / (lam * exp(risk)))^(1/k)
    T = (-np.log(u) / (lam * np.exp(true_risk))) ** (1.0 / k)

    # informative censoring: heavier censoring pressure for low-risk subjects
    rate_c = 0.045 * np.exp(-censor_strength * (true_risk - true_risk.mean()))
    C_exp = rng.exponential(scale=1.0 / rate_c)
    C_admin = rng.uniform(t_cap * 0.5, t_cap * 1.5, size=n)
    C = np.minimum(C_exp, C_admin)

    time = np.minimum(T, C)
    event = T <= C
    return SurvDataset(
        X=X,
        time=time,
        event=event,
        feature_names=tuple(f"x{i}" for i in range(d)),
        true_risk=true_risk,
    )


def train_val_holdout(ds: SurvDataset, n_train: int, n_val: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(ds.n_samples)
    tr = ds.split(idx[:n_train])
    va = ds.split(idx[n_train : n_train + n_val])
    ho = ds.split(idx[n_train + n_val :])
    return tr, va, ho
