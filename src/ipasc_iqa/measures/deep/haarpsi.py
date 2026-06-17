"""Haar Wavelet-Based Perceptual Similarity Index (HaarPSI).

Shallow adapter over ``piq.haarpsi`` (Reisenhofer et al., 2018).

HaarPSI decomposes both images with a Haar wavelet, computes local
similarity of the wavelet coefficients, and weights the similarities by
relative wavelet coefficient magnitudes (a proxy for local saliency).
The final score is the logistic mean of the weighted local similarities,
which maps naturally to [0, 1].

Requires ``ipasc_iqa[deep]``.  All heavy imports (``torch``, ``piq``) are
deferred to first use so the base install stays lightweight.
"""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class HaarPSIMeasure(BaseMeasure):
    """Haar Wavelet-Based Perceptual Similarity Index (HaarPSI).

    Computes a wavelet-based weighted local similarity score.  The default
    3-scale Haar decomposition works well for images down to ~16×16; below
    that piq may raise an error.

    Range: [0, 1], higher = more similar.  Requires ``ipasc_iqa[deep]``.
    """

    metadata = MeasureMetadata(
        name="HAARPSI",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=True,
        value_range=(0.0, 1.0),
        citation=(
            "Reisenhofer et al., 'A Haar Wavelet-Based Perceptual Similarity Index "
            "for Image Quality Assessment', Signal Process. Image Commun. 61, 2018."
        ),
        bibtex=(
            "@article{reisenhofer2018haarpsi,\n"
            "  author  = {Reisenhofer, Rafael and Bosse, Sebastian and\n"
            "             Kutyniok, Gitta and Wiegand, Thomas},\n"
            "  title   = {A Haar Wavelet-Based Perceptual Similarity Index for\n"
            "             Image Quality Assessment},\n"
            "  journal = {Signal Processing: Image Communication},\n"
            "  volume  = {61},\n"
            "  pages   = {33--43},\n"
            "  year    = {2018}\n"
            "}"
        ),
        description=(
            "Haar Wavelet-Based Perceptual Similarity Index (HaarPSI). "
            "Computed via piq.haarpsi. "
            "Minimum recommended image size: ~16×16 for the default 3-scale decomposition. "
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
        if reference is None:
            raise ValueError("HaarPSI requires a reference image.")

        try:
            from piq import haarpsi
        except ImportError as exc:
            raise ImportError(
                "HaarPSI requires 'ipasc_iqa[deep]'. Install with:  pip install 'ipasc_iqa[deep]'"
            ) from exc
        import torch

        data_min = float(reference.min())
        data_max = float(reference.max())
        data_range = data_max - data_min if data_max > data_min else 1.0

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(arr.astype(np.float32))
            t = ((t - data_min) / data_range).clamp(0.0, 1.0)
            return t.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        with torch.no_grad():
            score = haarpsi(to_tensor(image), to_tensor(reference), data_range=1.0)

        return float(score.item())
