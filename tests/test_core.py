"""Core layer tests: seed, config, errors, types."""

import numpy as np
import pytest

from survforge.core.config import SurvConfig
from survforge.core.errors import E100ConfigError, E200DataError, SurvForgeError
from survforge.core.seed import set_all
from survforge.core.types import SurvDataset


def test_seed_deterministic():
    set_all(123)
    a = np.random.rand(5)
    set_all(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_config_defaults():
    cfg = SurvConfig()
    assert cfg.seeds == (101, 202, 303)
    assert cfg.gate_delta_c == 0.05


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("ENV_SURVFORGE_HPO_TRIALS", "7")
    monkeypatch.setenv("ENV_SURVFORGE_SEEDS", "5,6")
    cfg = SurvConfig()
    assert cfg.hpo_trials == 7
    assert cfg.seeds == (5, 6)


def test_config_env_bad_value(monkeypatch):
    monkeypatch.setenv("ENV_SURVFORGE_HPO_TRIALS", "abc")
    with pytest.raises(E100ConfigError):
        SurvConfig()


def test_error_codes():
    assert E100ConfigError("x").code == "E100"
    assert E200DataError("x").code == "E200"
    with pytest.raises(SurvForgeError):
        raise E100ConfigError("boom")


def test_survdataset_validation():
    with pytest.raises(ValueError):
        SurvDataset(
            X=np.zeros((3, 2)),
            time=np.array([1.0, 2.0]),
            event=np.array([True, False, True]),
        )


def test_survdataset_split():
    ds = SurvDataset(
        X=np.arange(12, dtype=float).reshape(6, 2),
        time=np.array([1.0, 2, 3, 4, 5, 6]),
        event=np.array([True] * 6),
    )
    sub = ds.split(np.array([0, 2, 4]))
    assert sub.n_samples == 3
    assert sub.n_features == 2
    assert len(sub.feature_names) == 2
