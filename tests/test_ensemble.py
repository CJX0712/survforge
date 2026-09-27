"""Ensemble + HPO tests."""

import numpy as np
import pytest

from survforge.hpo.tune import tune_weights
from survforge.survival.ensemble import FusionEnsemble
from survforge.survival.registry import fusion_member_factories


def _fit_ensemble(split_data, mode):
    tr, va, ho = split_data
    ens = FusionEnsemble(fusion_member_factories(), weight_mode=mode)
    ens.with_hpo_budget(5, 5.0)
    ens.fit(tr, seed=0, val=va)
    return ens, tr, va, ho


def test_fusion_fit_predict(split_data):
    ens, tr, va, ho = _fit_ensemble(split_data, "cweighted")
    risk = ens.predict_risk(ho.X)
    assert risk.shape == (ho.n_samples,)
    assert np.all(np.isfinite(risk))


def test_weights_sum_to_one(split_data):
    for mode in ("uniform", "cweighted", "hpo"):
        ens, tr, va, ho = _fit_ensemble(split_data, mode)
        assert abs(sum(ens.weights_.values()) - 1.0) < 1e-9
        assert all(w >= 0 for w in ens.weights_.values())


def test_uniform_weights(split_data):
    ens, *_ = _fit_ensemble(split_data, "uniform")
    assert set(ens.weights_.values()) == {1.0 / len(ens.member_names_)}


def test_not_fitted_raises(split_data):
    ens = FusionEnsemble(fusion_member_factories())
    with pytest.raises(Exception):
        ens.predict_risk(np.zeros((3, 8)))


def test_bad_weight_mode():
    with pytest.raises(Exception):
        FusionEnsemble([lambda: None, lambda: None], weight_mode="bogus")


def test_tune_weights_fallback(split_data):
    tr, va, ho = split_data
    from survforge.survival.registry import fusion_member_factories

    members = [f().fit(tr, 0) for f in fusion_member_factories()]
    vs = {m.name: m.predict_risk(va.X) for m in members}
    w = tune_weights(vs, va, trials=0, timeout=0.0, seed=0)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert set(w.keys()) == set(vs.keys())


def test_tune_weights_optuna_path(split_data):
    tr, va, ho = split_data
    from survforge.survival.registry import fusion_member_factories

    members = [f().fit(tr, 0) for f in fusion_member_factories()]
    vs = {m.name: m.predict_risk(va.X) for m in members}
    w = tune_weights(vs, va, trials=4, timeout=10.0, seed=0)
    assert abs(sum(w.values()) - 1.0) < 1e-9
