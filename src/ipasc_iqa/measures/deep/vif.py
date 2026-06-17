"""Visual Information Fidelity in the Pixel Domain (VIF-P).

Shallow adapter over ``piq.vif_p`` (Sheikh & Bovik, 2006).

VIF measures how much visual information from the reference is preserved in
the distorted image, modelled via a Gaussian Scale Mixture (GSM) source and
an additive white Gaussian noise (AWGN) channel.  The pixel-domain variant
(VIF-P) replaces the wavelet decomposition with a Gaussian pyramid for
computational simplicity.

Requires ``ipasc_iqa[deep]``.  All heavy imports (``torch``, ``piq``) are
deferred to first use so the base install stays lightweight.
"""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class VIFMeasure(BaseMeasure):
    """Visual Information Fidelity in the Pixel Domain (VIF-P).

    Computes the ratio of mutual information between the reference and the
    distorted image, relative to the information in the reference itself.
    A perfect reconstruction scores 1.0; values below 1.0 indicate
    information loss.  Values above 1.0 can occur for enhanced images.

    Uses a 4-scale Gaussian pyramid internally.  Minimum recommended image
    size: 41×41 (the pyramid kernel is 11×11 at the finest scale).

    Range: [0, ∞), typical range [0, 1].  Higher = more similar.
    Requires ``ipasc_iqa[deep]``.
    """

    metadata = MeasureMetadata(
        name="VIF",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=True,
        value_range=(0.0, float("inf")),
        citation=(
            "Sheikh & Bovik, 'Image Information and Visual Quality', "
            "IEEE Trans. Image Process. 15(2), 2006."
        ),
        bibtex=(
            "@article{sheikh2006vif,\n"
            "  author  = {Sheikh, Hamid R. and Bovik, Alan C.},\n"
            "  title   = {Image Information and Visual Quality},\n"
            "  journal = {IEEE Transactions on Image Processing},\n"
            "  volume  = {15},\n"
            "  number  = {2},\n"
            "  pages   = {430--444},\n"
            "  year    = {2006}\n"
            "}"
        ),
        description=(
            "Visual Information Fidelity in the pixel domain (VIF-P). "
            "Computed via piq.vif_p. "
            "Typical range [0, 1]; values above 1 possible for enhanced images. "
            "Minimum recommended image size: 41×41. "
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
            raise ValueError("VIF requires a reference image.")

        try:
            from piq import vif_p
        except ImportError as exc:
            raise ImportError(
                "VIF requires 'ipasc_iqa[deep]'. Install with:  pip install 'ipasc_iqa[deep]'"
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
            score = vif_p(to_tensor(image), to_tensor(reference), data_range=1.0)

        return float(score.item())
