"""Gradient Magnitude Similarity Deviation (GMSD).

Shallow adapter over ``piq.gmsd`` (Xue et al., 2014).

GMSD improves on SSIM by computing the standard deviation of per-pixel
gradient magnitude similarity values, rather than the mean.  The standard
deviation captures the variation of local image distortions more precisely
than a single global pooling step and is highly correlated with perceived
quality.

Requires PyTorch-based metric dependencies.  All heavy imports (``torch``, ``piq``) are
deferred to first use so the base install stays lightweight.
"""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class GMSDMeasure(BaseMeasure):
    """Gradient Magnitude Similarity Deviation (GMSD).

    Computes the standard deviation of the per-pixel Gradient Magnitude
    Similarity (GMS) map.  Low deviation indicates consistently high
    similarity; high deviation indicates spatially variable distortions.

    Range: [0, ∞), lower = better quality.  Requires PyTorch-based metric dependencies.
    """

    metadata = MeasureMetadata(
        name="GMSD",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=False,
        value_range=(0.0, float("inf")),
        citation=(
            "Xue et al., 'Gradient Magnitude Similarity Deviation: A Highly Efficient "
            "Perceptual Image Quality Index', IEEE Trans. Image Process. 23(2), 2014."
        ),
        bibtex=(
            "@article{xue2014gmsd,\n"
            "  author  = {Xue, Wufeng and Zhang, Lei and Mou, Xuanqin and Bovik, Alan C.},\n"
            "  title   = {Gradient Magnitude Similarity Deviation: A Highly Efficient\n"
            "             Perceptual Image Quality Index},\n"
            "  journal = {IEEE Transactions on Image Processing},\n"
            "  volume  = {23},\n"
            "  number  = {2},\n"
            "  pages   = {684--695},\n"
            "  year    = {2014}\n"
            "}"
        ),
        description=(
            "Gradient Magnitude Similarity Deviation (GMSD). "
            "Computed via piq.gmsd. "
            "Lower deviation = better quality."
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
            raise ValueError("GMSD requires a reference image.")

        try:
            from piq import gmsd
        except ImportError as exc:
            raise ImportError(
                "GMSD requires PyTorch-based metric dependencies. Reinstall ipasc_iqa, or reinstall PyTorch following the official PyTorch selector."
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
            score = gmsd(to_tensor(image), to_tensor(reference), data_range=1.0)

        return float(score.item())
