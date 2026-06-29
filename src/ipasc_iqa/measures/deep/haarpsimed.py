"""HaarPSI optimised for medical images.

Karner, Clemens, et al. "Parameter choices in HaarPSI for IQA with medical images."
2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI). IEEE, 2025.
"""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class HaarPSIMedMeasure(BaseMeasure):
    """Haar Wavelet-Based Perceptual Similarity Index (HaarPSI).

    Computes a wavelet-based weighted local similarity score.  The default
    3-scale Haar decomposition works well for images down to ~16×16; below
    that piq may raise an error.

    Range: [0, 1], higher = more similar.  Requires PyTorch-based metric dependencies.
    """

    metadata = MeasureMetadata(
        name="HAARPSImed",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=True,
        value_range=(0.0, 1.0),
        citation=(
            "Karner et al., 'Parameter Choices in HaarPSI for IQA with Medical Images', "
            "IEEE 22nd International Symposium on Biomedical Imaging (ISBI), 2025."
        ),
        bibtex=(
            "@inproceedings{karner2025haarpsimed,\n"
            "  author    = {Karner, Clemens and others},\n"
            "  title     = {Parameter Choices in {HaarPSI} for {IQA} with Medical Images},\n"
            "  booktitle = {IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},\n"
            "  year      = {2025}\n"
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
                "HaarPSI requires PyTorch-based metric dependencies. Reinstall ipasc_iqa, or reinstall PyTorch following the official PyTorch selector."
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
            score = haarpsi(to_tensor(image), to_tensor(reference), data_range=1.0, c=5, alpha=4.9)

        return float(score.item())
