"""Shared fixtures."""

import pytest

from survforge.data.synthetic import make_survival


@pytest.fixture()
def small_data():
    return make_survival(300, seed=7, d=8)


@pytest.fixture()
def split_data():
    ds = make_survival(500, seed=11, d=8)
    from survforge.data.synthetic import train_val_holdout

    return train_val_holdout(ds, 250, 100, seed=11)
