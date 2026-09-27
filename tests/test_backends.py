"""Backend tests: contract, cross-validation numpy-Cox vs lifelines, tiers."""

import numpy as np
import pytest

from survforge.survival.numpy_cox import NumpyCoxBackend
from survforge.survival.registry import (
    available_backends,
    build_single_backends,
    fusion_member_factories,
    lifelines_available,
    xgb_available,
)


def test_numpy_cox_recovers_risk_direction(split_data):
    tr, va, ho = split_data
    m = NumpyCoxBackend().fit(tr, seed=0)
    risk = m.predict_risk(ho.X)
    corr = np.corrcoef(risk, ho.true_risk)[0, 1]
    assert corr > 0.5


def test_numpy_cox_matches_lifelines(split_data):
    if not lifelines_available():
        pytest.skip("lifelines not installed")
    tr, va, ho = split_data
    from survforge.survival.lifelines_backend import LifelinesCoxBackend

    r1 = NumpyCoxBackend().fit(tr, seed=0).predict_risk(ho.X)
    r2 = LifelinesCoxBackend().fit(tr, seed=0).predict_risk(ho.X)
    corr = np.corrcoef(r1, r2)[0, 1]
    assert corr > 0.99


def test_numpy_cox_ties_handled(split_data):
    tr, _, _ = split_data
    tr2 = tr.split(np.arange(tr.n_samples))
    tr2.time = np.round(tr.time)  # force ties
    beta, ll = __import__("survforge.survival.numpy_cox", fromlist=["cox_fit"]).cox_fit(
        tr2.X, tr2.time, tr2.event
    )
    assert np.all(np.isfinite(beta))
    assert np.isfinite(ll)


def test_not_fitted_raises():
    with pytest.raises(Exception):
        NumpyCoxBackend().predict_risk(np.zeros((3, 8)))


def test_spline_cox_beats_linear_on_pure_nonlinear():
    """Deterministic property: on a pure-nonlinearity DGP (risk = sin(2x0) +
    0.5*x1^2) the linear Cox cannot represent the signal at all, while the
    spline basis can — spline-Cox must clearly win."""
    import numpy as np

    from survforge.core.types import SurvDataset
    from survforge.data.synthetic import train_val_holdout
    from survforge.eval.metrics import harrell_c_index
    from survforge.survival.spline_cox import SplineCoxBackend

    rng = np.random.default_rng(13)
    n = 2000
    X = rng.normal(size=(n, 8))
    risk = np.sin(2.0 * X[:, 0]) + 0.5 * X[:, 1] ** 2
    lam = 0.05
    T = (-np.log(rng.uniform(size=n)) / (lam * np.exp(risk))) ** (1 / 1.2)
    C = rng.uniform(10, 50, size=n)
    ds = SurvDataset(X=X, time=np.minimum(T, C), event=T <= C, true_risk=risk)
    tr, va, ho = train_val_holdout(ds, 1200, 300, seed=13)

    c_lin = harrell_c_index(
        ho.time, ho.event, NumpyCoxBackend().fit(tr, 0).predict_risk(ho.X)
    )
    c_spl = harrell_c_index(
        ho.time, ho.event, SplineCoxBackend().fit(tr, 0).predict_risk(ho.X)
    )
    assert c_spl > c_lin + 0.05


def test_all_single_backends_contract(split_data):
    tr, va, ho = split_data
    from survforge.core.config import SurvConfig
    from survforge.eval.metrics import harrell_c_index

    for factory in build_single_backends(SurvConfig()):
        m = factory().fit(tr, seed=1)
        risk = m.predict_risk(ho.X)
        assert risk.shape == (ho.n_samples,)
        assert np.all(np.isfinite(risk))
        c = harrell_c_index(ho.time, ho.event, risk)
        if m.name == "naive_random":
            assert 0.3 < c < 0.7
        else:
            assert c > 0.5, f"{m.name} below random: {c}"


def test_fusion_member_factories_diverse():
    factories = fusion_member_factories()
    names = {f().name for f in factories}
    assert len(names) >= 2


def test_xgb_backends_if_available(split_data):
    if not xgb_available():
        pytest.skip("xgboost not installed")
    tr, va, ho = split_data
    from survforge.survival.xgb_aft import XgbAftBackend
    from survforge.survival.xgb_cox import XgbCoxBackend

    for cls in (XgbAftBackend, XgbCoxBackend):
        m = cls().fit(tr, seed=3)
        risk = m.predict_risk(ho.X)
        assert risk.shape == (ho.n_samples,)
        assert np.corrcoef(risk, ho.true_risk)[0, 1] > 0.3


def test_available_backends_probe():
    avail = available_backends()
    assert avail["numpy_cox"] is True and avail["spline_cox"] is True
