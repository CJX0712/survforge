"""Global determinism: single seed entry point."""

from __future__ import annotations

import random

import numpy as np

_CURRENT_SEED: int = 42


def set_all(seed: int) -> int:
    """Seed python.random and numpy. Library-level seeds are set by each
    backend constructor (they receive the seed explicitly)."""
    global _CURRENT_SEED
    _CURRENT_SEED = int(seed)
    random.seed(_CURRENT_SEED)
    np.random.seed(_CURRENT_SEED % (2**32))
    return _CURRENT_SEED


def current_seed() -> int:
    return _CURRENT_SEED
