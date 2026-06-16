"""Composite Mean Opinion Score from individual annotator ratings.

Absolute (1-4) ratings are combined the way PhotIQA does it: each annotator's
scores are z-normalised over the images *they* rated (removing per-rater scale
and offset bias), then the per-image average of those z-scores is the composite
MOS. Computing it from the individual ratings — rather than from a pre-averaged
column — lets contributed ratings be pooled with a dataset's published ratings
on equal footing.

Verified to reproduce PhotIQA's ``z-score mos`` column to ~1e-15 when using the
population standard deviation (``ddof=0``).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def composite_mos(per_image_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """Composite MOS per image from ``{image_id: {annotator_id: raw_score}}``.

    Each annotator is z-normalised (population std) over their own ratings; the
    composite for an image is the mean of the contributing annotators' z-scores.
    Annotators with zero variance (all-equal ratings) contribute 0.
    """
    # Gather each annotator's ratings across all images they scored.
    by_annotator: dict[str, dict[str, float]] = defaultdict(dict)
    for image_id, scores in per_image_scores.items():
        for annotator, value in scores.items():
            by_annotator[annotator][image_id] = value

    # Per-annotator z-normalisation.
    z_by_annotator: dict[str, dict[str, float]] = {}
    for annotator, image_scores in by_annotator.items():
        arr = np.asarray(list(image_scores.values()), dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0)) or 1.0
        z_by_annotator[annotator] = {
            img: (val - mean) / std for img, val in image_scores.items()
        }

    out: dict[str, float] = {}
    for image_id, scores in per_image_scores.items():
        zs = [z_by_annotator[a][image_id] for a in scores]
        if zs:
            out[image_id] = float(np.mean(zs))
    return out
