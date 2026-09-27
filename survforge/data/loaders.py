"""CSV/file loader. Columns: time, event(0/1), features."""

from __future__ import annotations

import csv

import numpy as np

from ..core.errors import E200DataError
from ..core.types import SurvDataset


def load_csv(path: str) -> SurvDataset:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    if not rows:
        raise E200DataError(f"empty csv: {path}")
    try:
        t_idx = header.index("time")
        e_idx = header.index("event")
    except ValueError as exc:
        raise E200DataError("csv must contain 'time' and 'event' columns") from exc
    feat_idx = [i for i in range(len(header)) if i not in (t_idx, e_idx)]
    arr = np.array([[float(v) for v in r] for r in rows], dtype=np.float64)
    return SurvDataset(
        X=arr[:, feat_idx],
        time=arr[:, t_idx],
        event=arr[:, e_idx] > 0.5,
        feature_names=tuple(header[i] for i in feat_idx),
    )
