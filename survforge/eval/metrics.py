"""Model-agnostic metrics: Harrell's C-index + simplified Uno IPCW C-index.

Own implementations (no sklearn dependency in the eval path; cross-validated
against lifelines in the test suite).
"""

from __future__ import annotations

import numpy as np

from ..core.errors import E400EvalError


def _km_censoring_survival(
    time: np.ndarray, event: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate of the censoring survival function G(t).

    Treats censoring as the event. Returns (unique_times, G(t)).
    """
    order = np.argsort(time, kind="stable")
    t_sorted = time[order]
    c_sorted = (~event[order]).astype(np.float64)  # censoring = "event" here
    uniq = np.unique(t_sorted)
    n = len(t_sorted)
    surv = 1.0
    out_t, out_s = [], []
    k = 0
    for ut in uniq:
        at_risk = n - k
        # subjects censored (c-event) exactly at ut
        d = c_sorted[t_sorted == ut].sum()
        if at_risk > 0:
            surv *= 1.0 - d / at_risk
        out_t.append(ut)
        out_s.append(surv)
        k += int((t_sorted == ut).sum())
    return np.asarray(out_t), np.asarray(out_s)


def _G_at(uniq_t: np.ndarray, surv: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Right-continuous step evaluation of G at query times."""
    idx = np.searchsorted(uniq_t, query, side="right") - 1
    out = np.ones_like(query, dtype=np.float64)
    valid = idx >= 0
    out[valid] = surv[idx[valid]]
    return out


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's concordance index. Higher risk = shorter survival.

    Comparable pair (i, j): time_i < time_j and event_i is True.
    Tied risk counts 0.5.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=np.float64)
    if time.shape != risk.shape or time.shape != event.shape:
        raise E400EvalError("time/event/risk shape mismatch")

    concordant = 0.0
    comparable = 0
    n = len(time)
    for i in range(n):
        if not event[i]:
            continue
        later = time > time[i]
        if not np.any(later):
            continue
        r_later = risk[later]
        comparable += int(later.sum())
        concordant += float(np.sum(risk[i] > r_later))
        concordant += 0.5 * float(np.sum(risk[i] == r_later))
    if comparable == 0:
        raise E400EvalError("no comparable pairs — cannot estimate C-index")
    return concordant / comparable


def ipcw_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    """Simplified Uno's IPCW C-index.

    Same comparable pairs as Harrell; each pair involving subject i as the
    earlier/event case is weighted by 1/G(time_i) where G is the KM estimate
    of the censoring survival. With no censoring this reduces to Harrell.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    risk = np.asarray(risk, dtype=np.float64)

    uniq_t, surv = _km_censoring_survival(time, event)
    g_at = _G_at(uniq_t, surv, time)

    weighted_conc = 0.0
    weight_sum = 0.0
    n = len(time)
    for i in range(n):
        if not event[i] or g_at[i] <= 0:
            continue
        later = time > time[i]
        if not np.any(later):
            continue
        w = 1.0 / g_at[i]
        r_later = risk[later]
        weight_sum += w * float(later.sum())
        weighted_conc += w * (
            float(np.sum(risk[i] > r_later)) + 0.5 * float(np.sum(risk[i] == r_later))
        )
    if weight_sum <= 0:
        raise E400EvalError("no IPCW-comparable pairs")
    return weighted_conc / weight_sum


def rank_average(scores: np.ndarray) -> np.ndarray:
    """Average ranks in [0, 1] (ties get the mean rank). Higher = higher score."""
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # average ties
    sorted_scores = scores[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    if n > 1:
        ranks = ranks / (n - 1)
    return ranks
