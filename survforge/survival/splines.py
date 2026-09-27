"""Restricted cubic spline basis (Harrell rcSpline style, classical method:
Durrleman & Simon 1989). Pure numpy, k knots -> k-1 columns per feature.
"""

from __future__ import annotations

import numpy as np


def nspline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    k = len(knots)
    out = [x]
    for j in range(k - 2):
        t = np.clip(x, knots[j], knots[-2])
        term = np.maximum(t - knots[j], 0) ** 3
        term -= (
            np.maximum(x - knots[-2], 0) ** 3 * (knots[-2] - knots[j]) / (knots[-1] - knots[-2])
        )
        out.append(term)
    return np.column_stack(out)


def spline_expand(X: np.ndarray, n_knots: int = 5) -> np.ndarray:
    """Per-feature natural cubic spline basis on quantile knots."""
    cols = []
    for j in range(X.shape[1]):
        xj = X[:, j]
        knots = np.unique(np.quantile(xj, np.linspace(0.0, 1.0, n_knots)))
        if len(knots) < 4:  # degenerate column: fall back to linear
            cols.append(xj[:, None])
        else:
            cols.append(nspline_basis(xj, knots))
    return np.hstack(cols)
