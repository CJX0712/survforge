"""Tier-1 offline fallback: pure-numpy Cox PH (Efron ties, Newton-Raphson).

Zero non-numpy dependencies. This is the offline guarantee of SurvForge.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E300BackendError
from ..core.types import SurvDataset

_RIDGE = 1e-6
_MAX_ITER = 100
_TOL = 1e-8


def _tie_blocks(ts: np.ndarray):
    """Yield (start, end) half-open index blocks of equal times."""
    n = ts.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ts[j + 1] == ts[i]:
            j += 1
        yield i, j + 1
        i = j + 1


def cox_fit(
    X: np.ndarray, time: np.ndarray, event: np.ndarray, ridge: float = _RIDGE
) -> tuple[np.ndarray, float]:
    """Fit Cox PH by Newton-Raphson with Efron handling of tied times.

    Returns (beta, loglik). A tiny ridge guards singularity.
    """
    n, d = X.shape
    if n < 3 or int(event.sum()) < 2:
        raise E300BackendError("not enough events to fit Cox model")

    order = np.argsort(time, kind="stable")
    Xs, ts, es = X[order], time[order], event[order]
    eye = np.eye(d)
    beta = np.zeros(d)
    loglik = -np.inf

    for _ in range(_MAX_ITER):
        eta = Xs @ beta
        w = np.exp(np.clip(eta, -30.0, 30.0))
        # suffix risk sets (ascending time order): risk set of subject i
        # = all subjects with time >= ts[i]
        S0 = np.cumsum(w[::-1])[::-1]
        S1 = np.cumsum((w[:, None] * Xs)[::-1], axis=0)[::-1]
        S2 = np.cumsum((w[:, None, None] * Xs[:, :, None] * Xs[:, None, :])[::-1], axis=0)[::-1]

        loglik_new = 0.0
        grad = np.zeros(d)
        hess = np.zeros((d, d))
        for i, j in _tie_blocks(ts):
            ev_idx = [t for t in range(i, j) if es[t]]
            m = len(ev_idx)
            if m == 0:
                continue
            d0 = S0[i]
            d1 = S1[i]
            d2 = S2[i]
            x_ev = Xs[ev_idx]
            w_ev = w[ev_idx]
            s2_ev = np.einsum("ni,nj->nij", x_ev, x_ev)
            loglik_new += float(np.sum(eta[ev_idx]))
            for l in range(m):
                frac = l / m
                r0 = d0 - frac * w_ev[l]
                r1 = d1 - frac * x_ev[l]
                r2 = d2 - frac * s2_ev[l]
                r0 = max(r0, 1e-12)
                loglik_new -= float(np.log(r0))
                grad += x_ev[l] - r1 / r0
                hess += r2 / r0 - np.outer(r1, r1) / (r0 * r0)

        try:
            # hess is the (positive definite) information matrix I;
            # Newton ascent on loglik: beta <- beta + I^-1 * grad
            step = np.linalg.solve(hess + ridge * eye, grad)
        except np.linalg.LinAlgError:
            step = grad / (np.abs(hess).max() + 1e-12)
        beta_new = beta + step
        converged = np.max(np.abs(beta_new - beta)) < _TOL
        beta = beta_new
        loglik = loglik_new
        if converged:
            break

    return beta, float(loglik)


class NumpyCoxBackend:
    """Tier-1 fallback backend. Risk = X @ beta (higher = worse)."""

    name = "numpy_cox"
    tier = "tier1"

    def __init__(self) -> None:
        self.beta_: np.ndarray | None = None
        self.loglik_: float | None = None

    def fit(self, train: SurvDataset, seed: int = 0) -> NumpyCoxBackend:
        self.beta_, self.loglik_ = cox_fit(train.X, train.time, train.event)
        return self

    def predict_risk(self, X) -> np.ndarray:
        if self.beta_ is None:
            raise E300BackendError("NumpyCoxBackend not fitted")
        return np.asarray(X, dtype=np.float64) @ self.beta_
