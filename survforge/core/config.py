"""Configuration with ENV_SURVFORGE_* overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from .errors import E100ConfigError


@dataclass
class SurvConfig:
    # data
    d_features: int = 8
    beta_scale: float = 1.0
    censor_strength: float = 0.4
    # backends
    rsf_n_estimators: int = 300
    gbs_n_estimators: int = 300
    # hpo
    hpo_trials: int = 40
    hpo_timeout: float = 30.0
    # benchmark
    seeds: tuple = (101, 202, 303)
    n_train: int = 1000
    n_val: int = 400
    n_holdout: int = 800
    # gates (fixed at plan time, V4 requirement; strong baseline = classic
    # survival method = Cox PH per DoD "naive/经典方法")
    gate_delta_c: float = 0.05
    gate_tier1_delta_c: float = 0.10

    def __post_init__(self) -> None:
        if self.n_train + self.n_val + self.n_holdout < 100:
            raise E100ConfigError("dataset too small")
        if self.hpo_trials < 0:
            raise E100ConfigError("hpo_trials must be >= 0")
        for f in fields(self):
            env_key = f"ENV_SURVFORGE_{f.name.upper()}"
            if env_key in os.environ:
                raw = os.environ[env_key]
                cur = getattr(self, f.name)
                try:
                    if isinstance(cur, tuple):
                        setattr(self, f.name, tuple(int(s) for s in raw.split(",")))
                    elif isinstance(cur, bool):
                        setattr(self, f.name, raw.lower() in ("1", "true", "yes"))
                    elif isinstance(cur, int):
                        setattr(self, f.name, int(raw))
                    elif isinstance(cur, float):
                        setattr(self, f.name, float(raw))
                except ValueError as exc:
                    raise E100ConfigError(
                        f"bad env override {env_key}={raw!r}"
                    ) from exc
