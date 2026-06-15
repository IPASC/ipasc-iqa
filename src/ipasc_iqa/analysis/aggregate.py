"""Per-dataset aggregate statistics over stored per-image results.

The result store keeps one value per (dataset, image, measure). For reporting we
also want the dataset-level mean and standard deviation of each measure, which
this module computes from those per-image records. Non-finite values (e.g. an
infinite PSNR for an identical pair) are excluded from the statistics.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from ipasc_iqa.core.result import ResultRecord


def measure_stats(records: Iterable[ResultRecord]) -> dict[str, dict[str, float]]:
    """Return ``{measure: {"n", "mean", "std"}}`` over finite per-image values."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        if rec.value is None or not np.isfinite(rec.value):
            continue
        buckets[rec.measure].append(rec.value)

    out: dict[str, dict[str, float]] = {}
    for measure, values in buckets.items():
        arr = np.asarray(values, dtype=float)
        out[measure] = {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
        }
    return out
