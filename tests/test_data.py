"""Synthetic DGP tests: shapes, reproducibility, censoring behaviour."""

import numpy as np

from survforge.data.synthetic import make_survival, train_val_holdout


def test_shapes_and_reproducibility():
    a = make_survival(200, seed=3, d=8)
    b = make_survival(200, seed=3, d=8)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.time, b.time)
    assert a.n_samples == 200 and a.n_features == 8


def test_event_fraction_sane():
    ds = make_survival(2000, seed=5, d=8)
    frac = ds.event.mean()
    assert 0.35 <= frac <= 0.95


def test_true_risk_correlates_with_time():
    ds = make_survival(3000, seed=9, d=8)
    ev = ds.event
    # among events: higher true risk -> shorter time (negative rank corr)
    t = ds.time[ev]
    r = ds.true_risk[ev]
    corr = np.corrcoef(t, -r)[0, 1]
    assert corr > 0.3


def test_holdout_partition():
    ds = make_survival(600, seed=2, d=8)
    tr, va, ho = train_val_holdout(ds, 300, 150, seed=2)
    assert tr.n_samples == 300 and va.n_samples == 150 and ho.n_samples == 150
    all_idx = np.concatenate([tr.X[:, 0], va.X[:, 0], ho.X[:, 0]])
    assert len(all_idx) == 600


def test_dgp_needs_features():
    import pytest

    with pytest.raises(ValueError):
        make_survival(100, seed=1, d=3)
