"""Backend registry with availability probing (optional deps degrade gracefully).

Fusion members are chosen for error decorrelation at comparable strength:
spline-Cox (smooth, numpy) + xgb AFT (interval-loss trees) + xgb Cox
(partial-likelihood trees). Classic strong baseline = lifelines Cox PH
(fallback numpy Cox if lifelines missing).
"""

from __future__ import annotations

from functools import lru_cache

from .baselines import FirstFeatureBackend, RandomRiskBackend
from .ensemble import FusionEnsemble
from .numpy_cox import NumpyCoxBackend
from .spline_cox import SplineCoxBackend


@lru_cache(maxsize=1)
def lifelines_available() -> bool:
    try:
        from .lifelines_backend import LIFELINES_AVAILABLE

        return bool(LIFELINES_AVAILABLE)
    except Exception:
        return False


@lru_cache(maxsize=1)
def xgb_available() -> bool:
    try:
        from .xgb_aft import XGB_AVAILABLE

        return bool(XGB_AVAILABLE)
    except Exception:
        return False


def available_backends() -> dict:
    return {
        "numpy_cox": True,
        "spline_cox": True,
        "lifelines_cox": lifelines_available(),
        "xgb_aft": xgb_available(),
        "xgb_cox": xgb_available(),
    }


def _mk_lifelines():
    from .lifelines_backend import LifelinesCoxBackend

    return LifelinesCoxBackend()


def _mk_xgb_aft():
    from .xgb_aft import XgbAftBackend

    return XgbAftBackend(
        n_estimators=150, max_depth=3, learning_rate=0.08, min_child_weight=10.0
    )


def _mk_xgb_cox():
    from .xgb_cox import XgbCoxBackend

    return XgbCoxBackend()


def build_single_backends(cfg) -> list:
    """Registry of baselines + all available single models (default configs)."""
    backends = [
        RandomRiskBackend,
        FirstFeatureBackend,
        NumpyCoxBackend,
        SplineCoxBackend,
    ]
    if lifelines_available():
        backends.append(_mk_lifelines)
    if xgb_available():
        backends.extend([_mk_xgb_aft, _mk_xgb_cox])
    return backends


def fusion_member_factories() -> list:
    """Diverse-family members: spline-Cox + tree backends (available only)."""
    factories = [SplineCoxBackend]
    if xgb_available():
        factories.extend([_mk_xgb_aft, _mk_xgb_cox])
    elif lifelines_available():
        factories.append(_mk_lifelines)
    if len(factories) < 2:  # extreme fallback: never happens (numpy always on)
        factories.append(NumpyCoxBackend)
    return factories


def build_fusion(cfg, weight_mode: str = "hpo") -> FusionEnsemble:
    ens = FusionEnsemble(fusion_member_factories(), weight_mode=weight_mode)
    ens.with_hpo_budget(cfg.hpo_trials, cfg.hpo_timeout)
    return ens


def classic_baseline_name() -> str:
    return "lifelines_cox" if lifelines_available() else "numpy_cox"
