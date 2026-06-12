"""Contrast-to-Noise Ratio."""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class CNRMeasure(BaseMeasure):
    metadata = MeasureMetadata(
        name="cnr",
        version="1.0.0",
        measure_type=MeasureType.REDUCED_REFERENCE,
        requires_signal_roi=True,
        requires_noise_roi=True,
        accepted_reference_types=None,
        higher_is_better=True,
        value_range=(0.0, float("inf")),
        description=(
            "Contrast-to-Noise Ratio: |mean(signal) - mean(noise)| / std(noise). "
            "Requires signal_roi and noise_roi masks."
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
        if signal_roi is None or noise_roi is None:
            raise ValueError("CNR requires both signal_roi and noise_roi.")

        img = image.astype(np.float64)
        signal_pixels = img[signal_roi.astype(bool)]
        noise_pixels = img[noise_roi.astype(bool)]

        noise_std = float(np.std(noise_pixels))
        if noise_std == 0.0:
            return float("inf")

        return float(abs(np.mean(signal_pixels) - np.mean(noise_pixels)) / noise_std)
