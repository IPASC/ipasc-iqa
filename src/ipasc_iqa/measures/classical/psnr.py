"""Peak Signal-to-Noise Ratio."""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class PSNRMeasure(BaseMeasure):
    metadata = MeasureMetadata(
        name="psnr",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=True,
        value_range=(0.0, float("inf")),
        description="Peak Signal-to-Noise Ratio (dB). data_range inferred from reference.",
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
            raise ValueError("PSNR requires a reference image.")

        img = image.astype(np.float64)
        ref = reference.astype(np.float64)

        data_range = float(ref.max() - ref.min())
        if data_range == 0.0:
            data_range = 1.0

        mse = float(np.mean((img - ref) ** 2))
        if mse == 0.0:
            return float("inf")
        return float(10.0 * np.log10((data_range**2) / mse))
