"""Structural Similarity Index (SSIM)."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata


class SSIMMeasure(BaseMeasure):
    metadata = MeasureMetadata(
        name="ssim",
        version="1.0.0",
        full_reference=True,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=True,
        value_range=(-1.0, 1.0),
        description="Structural Similarity Index Measure (skimage implementation).",
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
            raise ValueError("SSIM requires a reference image.")

        img = image.astype(np.float64)
        ref = reference.astype(np.float64)

        data_range = float(ref.max() - ref.min())
        if data_range == 0.0:
            data_range = 1.0

        # Handle channel axis: skimage expects (H, W) or (H, W, C)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
            ref = ref[:, :, 0]
            channel_axis = None
        elif img.ndim == 3:
            channel_axis = -1
        else:
            channel_axis = None

        score = structural_similarity(
            img, ref, data_range=data_range, channel_axis=channel_axis
        )
        return float(score)
