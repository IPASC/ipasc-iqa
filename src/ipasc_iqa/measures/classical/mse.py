"""Contrast-to-Noise Ratio."""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class MSEMeasure(BaseMeasure):
    metadata = MeasureMetadata(
        name="MSE",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=False,
        value_range=(0.0, float("inf")),
        description=(
            "Mean Squared Error: mean((signal - reference)**2). "
        ),
    )

    def evaluate(
        self,
        image: np.ndarray,
        *,
        reference: np.ndarray | None = None,
        reference_type: str | None = None,
        signal_roi: np.ndarray | None = None,
        noise_roi: np.ndarray | None = None,
    ) -> float:

        img = image.astype(np.float64)
        ref = reference.astype(np.float64)

        return float(np.square(img - ref).mean())
