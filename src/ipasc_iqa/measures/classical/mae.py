"""Mean absolute error."""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class MAEMeasure(BaseMeasure):
    metadata = MeasureMetadata(
        name="MAE",
        version="2.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=False,
        value_range=(0.0, float("inf")),
        description="Mean Absolute Error: mean(abs(image - reference)).",
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
        if reference is None:
            raise ValueError("MAE requires a reference image.")

        img = image.astype(np.float64)
        ref = reference.astype(np.float64)

        return float(np.mean(np.abs(img - ref)))
