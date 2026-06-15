"""Per-image scatter data + rank-mean trend for one measure vs annotations.

The detailed-analysis view plots an IQA measure (y-axis) against the expert
human score (x-axis, typically z-scored MOS) for every image in a dataset, with
a smooth *rank-mean trend* line for visual guidance (as in Breger et al.'s
PhotIQA analysis).

The trend is computed by sorting the points by their human score and taking a
centred moving average of the measure values over a window of neighbouring
ranks — hence "rank-mean": the mean measure value among images of similar
human-rated quality. It is purely a visual aid and is not used for the
correlation coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ipasc_iqa.analysis.correlation import measure_values_by_image
from ipasc_iqa.core.result import RunResults


@dataclass
class ScatterData:
    measure: str
    image_ids: list[str]
    xs: list[float]  # human annotation score (z-MOS)
    ys: list[float]  # IQA measure value
    groups: list[str | None]  # per-point group label (e.g. algorithm), may be None
    trend_x: list[float]
    trend_y: list[float]


def rank_mean_trend(
    xs: list[float] | np.ndarray,
    ys: list[float] | np.ndarray,
    window: int | None = None,
) -> tuple[list[float], list[float]]:
    """Centred moving average of ``ys`` over points sorted by ``xs``.

    Returns ``(trend_x, trend_y)`` with one trend value per input point, in
    ascending-x order. ``window`` defaults to ~10% of the points (clamped to an
    odd value in [5, 151]); for very small samples it collapses to the mean.
    """
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    n = x.size
    if n < 2:
        return x.tolist(), y.tolist()

    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    y_sorted = y[order]

    if window is None:
        w = (n // 10) | 1  # ~10% of points, forced odd
        window = int(min(max(w, 5), 151))
    half = window // 2

    # Prefix sums → O(n) windowed means.
    csum = np.concatenate([[0.0], np.cumsum(y_sorted)])
    trend = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        trend[i] = (csum[hi] - csum[lo]) / (hi - lo)

    return x_sorted.tolist(), trend.tolist()


def build_scatter(
    run: RunResults,
    annotation_values: dict[str, float],
    measure: str,
    *,
    dataset: str | None = None,
    groups: dict[str, str] | None = None,
) -> ScatterData:
    """Pair a measure's per-image values with the annotation scores.

    ``x`` is the human score, ``y`` is the measure value (matching the
    requested plot orientation). Points present in only one of the two sources
    are dropped.
    """
    values = measure_values_by_image(run, measure, dataset)

    image_ids: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    grp: list[str | None] = []
    for image_id, measure_value in values.items():
        if image_id not in annotation_values:
            continue
        image_ids.append(image_id)
        xs.append(annotation_values[image_id])
        ys.append(measure_value)
        grp.append(groups.get(image_id) if groups else None)

    trend_x, trend_y = rank_mean_trend(xs, ys)
    return ScatterData(
        measure=measure,
        image_ids=image_ids,
        xs=xs,
        ys=ys,
        groups=grp,
        trend_x=trend_x,
        trend_y=trend_y,
    )
