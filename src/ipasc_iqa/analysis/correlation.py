"""Rank-correlation of IQA measure values against expert annotations.

Reproduces the core of the PhotIQA ``main.ipynb`` analysis: for each measure,
correlate its per-image values with the expert mean-opinion-score (typically the
z-scored MOS) using Spearman (SRCC) and Kendall (KRCC) rank correlation.

The function is deliberately decoupled from where the numbers come from: it takes
a :class:`~ipasc_iqa.core.result.RunResults` (measure values keyed by image id)
plus a plain ``{image_id: annotation_value}`` mapping. Building that mapping from a
dataset's annotations is the caller's job (see
:func:`ipasc_iqa.analysis.annotations` and the dataset adapters).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ipasc_iqa.core.result import RunResults

# Sentinel used by the runner / measures for non-finite scores (e.g. inf PSNR).
_INF_REPLACEMENT = 1000.0


@dataclass
class CorrelationRow:
    """One measure's correlation against the annotation scores."""

    measure: str
    n: int
    srcc: float | None  # Spearman rank correlation coefficient
    krcc: float | None  # Kendall rank correlation coefficient

    def as_dict(self) -> dict[str, object]:
        return {"measure": self.measure, "n": self.n, "srcc": self.srcc, "krcc": self.krcc}


def measure_values_by_image(
    run: RunResults, measure: str, dataset: str | None
) -> dict[str, float]:
    """Collect {image_id: value} for one measure (optionally one dataset)."""
    out: dict[str, float] = {}
    for rec in run.results:
        if rec.measure != measure:
            continue
        if dataset is not None and rec.dataset != dataset:
            continue
        if rec.value is None:
            continue
        v = rec.value
        if not np.isfinite(v):
            v = _INF_REPLACEMENT
        out[rec.image_id] = v
    return out


def correlate_measures_with_annotations(
    run: RunResults,
    annotation_values: dict[str, float],
    *,
    measures: list[str] | None = None,
    dataset: str | None = None,
) -> list[CorrelationRow]:
    """Compute SRCC/KRCC for each measure against ``annotation_values``.

    Parameters
    ----------
    run:
        Results of an evaluation run.
    annotation_values:
        ``{image_id: score}`` — the expert score to correlate against (e.g. the
        z-scored MOS of a quality task).
    measures:
        Restrict to these measure names; default = all measures in ``run``.
    dataset:
        Restrict to a single dataset name; default = all datasets in ``run``.

    Returns
    -------
    One :class:`CorrelationRow` per measure, sorted by ``|srcc|`` descending so the
    best-aligned measures float to the top of the table.
    """
    if measures is None:
        measures = sorted({r.measure for r in run.results})

    rows: list[CorrelationRow] = []
    for measure in measures:
        values = measure_values_by_image(run, measure, dataset)

        paired_x: list[float] = []
        paired_y: list[float] = []
        for image_id, mv in values.items():
            if image_id in annotation_values:
                paired_x.append(mv)
                paired_y.append(annotation_values[image_id])

        n = len(paired_x)
        if n < 2 or len(set(paired_x)) < 2 or len(set(paired_y)) < 2:
            rows.append(CorrelationRow(measure=measure, n=n, srcc=None, krcc=None))
            continue

        srcc = float(stats.spearmanr(paired_x, paired_y).statistic)
        krcc = float(stats.kendalltau(paired_x, paired_y).statistic)
        rows.append(CorrelationRow(measure=measure, n=n, srcc=srcc, krcc=krcc))

    rows.sort(key=lambda r: (-(abs(r.srcc) if r.srcc is not None else -1.0), r.measure))
    return rows
