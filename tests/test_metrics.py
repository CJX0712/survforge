"""Metrics tests: Harrell C (cross-checked vs lifelines), IPCW, ranks."""

import numpy as np
import pytest

from survforge.eval.metrics import harrell_c_index, ipcw_c_index, rank_average


def _toy_data():
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([True, True, False, True, False, True])
    risk = np.array([0.9, 0.8, 0.3, 0.7, 0.1, 0.2])
    return time, event, risk


def test_perfect_and_inverse():
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([True, True, True, True])
    risk = np.array([4.0, 3.0, 2.0, 1.0])
    assert harrell_c_index(time, event, risk) == 1.0
    assert harrell_c_index(time, event, -risk) == 0.0


def test_random_half():
    rng = np.random.default_rng(0)
    time = np.arange(1, 501, dtype=float)
    event = np.ones(500, dtype=bool)
    risk = rng.random(500)
    c = harrell_c_index(time, event, risk)
    assert 0.42 <= c <= 0.58


def test_vs_lifelines():
    pytest.importorskip("lifelines")
    from lifelines.utils import concordance_index

    rng = np.random.default_rng(42)
    n = 300
    time = rng.exponential(5, n)
    event = rng.random(n) < 0.7
    risk = rng.random(n)
    mine = harrell_c_index(time, event, risk)
    theirs = concordance_index(time, -risk, event)
    assert abs(mine - theirs) < 1e-9


def test_ipcw_equals_harrell_no_censoring():
    rng = np.random.default_rng(1)
    time = rng.exponential(5, 400)
    event = np.ones(400, dtype=bool)
    risk = rng.random(400)
    assert abs(ipcw_c_index(time, event, risk) - harrell_c_index(time, event, risk)) < 1e-12


def test_ipcw_with_censoring_runs():
    rng = np.random.default_rng(2)
    time = rng.exponential(5, 400)
    event = rng.random(400) < 0.7
    c = ipcw_c_index(time, event, rng.random(400))
    assert 0.0 <= c <= 1.0


def test_rank_average_ties():
    r = rank_average(np.array([1.0, 2.0, 2.0, 3.0]))
    assert r[0] == 0.0 and r[3] == 1.0
    assert r[1] == r[2] == pytest.approx((1 / 3 + 2 / 3) / 2)


def test_shape_mismatch_raises():
    with pytest.raises(Exception):
        harrell_c_index(np.array([1.0, 2.0]), np.array([True]), np.array([1.0]))
