"""Total Variation (TV) — no-reference image smoothness measure.

Implemented with NumPy only (no PyTorch required), so it is available in
the base install without ``ipasc_iqa[deep]``.

TV measures the sum of local intensity differences across the image.  Low TV
indicates a smooth image; high TV indicates many sharp edges or noise.  As a
quality metric it is used to flag over-smoothing (reconstruction artefact) or
excessive noise.

This implementation uses the **anisotropic** (L1) definition:

    TV(I) = Σ |I[i+1,j] − I[i,j]| + Σ |I[i,j+1] − I[i,j]|

summed over all pixels and channels, then divided by ``H × W`` so the score
is independent of image resolution and comparable across datasets.
"""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class TVMeasure(BaseMeasure):
    """Total Variation (anisotropic / L1), normalised by pixel count.

    Measures the sum of absolute first-order intensity differences between
    adjacent pixels.  Dividing by ``H × W`` makes the score resolution-
    independent.

    Range: [0, ∞), lower = smoother image.  Available without
    ``ipasc_iqa[deep]``.
    """

    metadata = MeasureMetadata(
        name="TV",
        version="1.0.0",
        measure_type=MeasureType.NO_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=False,
        value_range=(0.0, float("inf")),
        citation=(
            "Rudin, Osher & Fatemi, 'Nonlinear total variation based noise removal "
            "algorithms', Physica D 60(1–4), 1992."
        ),
        bibtex=(
            "@article{rudin1992tv,\n"
            "  author  = {Rudin, Leonid I. and Osher, Stanley and Fatemi, Emad},\n"
            "  title   = {Nonlinear Total Variation Based Noise Removal Algorithms},\n"
            "  journal = {Physica D: Nonlinear Phenomena},\n"
            "  volume  = {60},\n"
            "  number  = {1--4},\n"
            "  pages   = {259--268},\n"
            "  year    = {1992}\n"
            "}"
        ),
        description=(
            "Total Variation (anisotropic L1), normalised by H×W. "
            "Measures image smoothness without a reference. "
            "Lower = smoother. Implemented with NumPy; no ipasc_iqa[deep] required."
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
        # Horizontal and vertical first-order differences summed over all channels
        tv = float(
            np.sum(np.abs(np.diff(img, axis=0)))   # vertical  (H-1, W, C)
            + np.sum(np.abs(np.diff(img, axis=1)))  # horizontal (H, W-1, C)
        )
        # Normalise by pixel count so the score is resolution-independent
        n_pixels = image.shape[0] * image.shape[1]
        return tv / n_pixels
