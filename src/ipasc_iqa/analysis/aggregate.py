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

def measure_directions(measures: Iterable[object]) -> dict[str, bool]:
    """Return {measure_name: higher_is_better} from result metadata."""
    directions: dict[str, bool] = {}

    for measure in measures:
        name = getattr(measure, "name", None)
        metadata = getattr(measure, "metadata", {}) or {}

        if name is None:
            continue

        directions[name] = bool(metadata.get("higher_is_better", True))

    return directions


def rich_measure_summary(
    records: Iterable[ResultRecord],
    *,
    higher_is_better_by_measure: dict[str, bool] | None = None,
    top: int = 5,
) -> dict:
    """Return annotation-free statistics and best/worst examples.

    This is intended for both CLI and GUI use. It works from per-image result
    records and does not require human annotations.
    """
    higher_is_better_by_measure = higher_is_better_by_measure or {}

    buckets: dict[str, list[dict]] = defaultdict(list)

    for rec in records:
        if rec.value is not None and np.isfinite(rec.value):
            buckets[rec.measure].append(
                {
                    "dataset": rec.dataset,
                    "image_id": rec.image_id,
                    "value": float(rec.value),
                }
            )

        if rec.sub_scores:
            for sub_name, sub_value in rec.sub_scores.items():
                if not np.isfinite(sub_value):
                    continue
                buckets[f"{rec.measure}.{sub_name}"].append(
                    {
                        "dataset": rec.dataset,
                        "image_id": rec.image_id,
                        "value": float(sub_value),
                    }
                )

    statistics: list[dict] = []
    examples: dict[str, dict] = {}

    for measure_name, rows in sorted(buckets.items()):
        values = np.asarray([row["value"] for row in rows], dtype=float)

        base_measure = measure_name.split(".", 1)[0]
        higher_is_better = higher_is_better_by_measure.get(base_measure, True)

        rows_sorted = sorted(rows, key=lambda row: row["value"])

        if higher_is_better:
            worst_rows = rows_sorted[:top]
            best_rows = list(reversed(rows_sorted[-top:]))
        else:
            worst_rows = list(reversed(rows_sorted[-top:]))
            best_rows = rows_sorted[:top]

        statistics.append(
            {
                "measure": measure_name,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "higher_is_better": bool(higher_is_better),
            }
        )

        examples[measure_name] = {
            "higher_is_better": bool(higher_is_better),
            "best": best_rows,
            "worst": worst_rows,
        }

    return {
        "statistics": statistics,
        "examples": examples,
    }