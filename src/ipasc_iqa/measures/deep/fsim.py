"""Feature Similarity Index Measure (FSIM).

Shallow adapter over ``piq.fsim`` (Zhang et al., 2011).

FSIM computes similarity based on two low-level features that the human
visual system is sensitive to: Phase Congruency (PC, measuring the
significance of local structures) and Gradient Magnitude (GM, measuring
contrast).  PC is used as a weighting function so that perceptually
important regions contribute more to the final score.

The ``chromatic`` variant extends FSIM to colour images (FSIM_C) using the
IQ chrominance channels.  For single-channel photoacoustic images the
default ``chromatic=False`` is used; if you pass multi-channel images and
want to use colour information set ``chromatic=True``, which requires C=3.

Requires PyTorch-based metric dependencies.  All heavy imports (``torch``, ``piq``) are
deferred to first use so the base install stays lightweight.
"""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class FSIMMeasure(BaseMeasure):
    """Feature Similarity Index Measure (FSIM / FSIM_C).

    Computes image similarity via Phase Congruency and Gradient Magnitude.
    With ``chromatic=False`` (default) the measure is purely luminance-based
    and works with any number of channels.  With ``chromatic=True`` the
    IQ colour channels are also compared; C must equal 3 in that case.

    Range: [0, 1], higher = more similar.  Requires PyTorch-based metric dependencies.
    """

    metadata = MeasureMetadata(
        name="FSIM",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=True,
        value_range=(0.0, 1.0),
        citation=(
            "Zhang et al., 'FSIM: A Feature Similarity Index for Image Quality Assessment', "
            "IEEE Trans. Image Process. 20(8), 2011."
        ),
        bibtex=(
            "@article{zhang2011fsim,\n"
            "  author  = {Zhang, Lin and Zhang, Lei and Mou, Xuanqin and Zhang, David},\n"
            "  title   = {{FSIM}: A Feature Similarity Index for Image Quality Assessment},\n"
            "  journal = {IEEE Transactions on Image Processing},\n"
            "  volume  = {20},\n"
            "  number  = {8},\n"
            "  pages   = {2378--2386},\n"
            "  year    = {2011}\n"
            "}"
        ),
        description=(
            "Feature Similarity Index (FSIM). "
            "Computed via piq.fsim with chromatic=False (luminance only). "
            "Set chromatic=True for colour images (requires C=3). "
        ),
    )

    def __init__(self, chromatic: bool = False, device: str = "auto") -> None:
        # chromatic enters the cache key so changing it invalidates cached results.
        super().__init__(device=device, chromatic=chromatic)
        if self.device == "mps":
            self.device = "cpu"
        self._chromatic = chromatic

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
            raise ValueError("FSIM requires a reference image.")

        try:
            from piq import fsim
        except ImportError as exc:
            raise ImportError(
                "FSIM requires PyTorch-based metric dependencies. Reinstall ipasc_iqa, or reinstall PyTorch following the official PyTorch selector."
            ) from exc
        import torch

        data_min = float(reference.min())
        data_max = float(reference.max())
        data_range = data_max - data_min if data_max > data_min else 1.0

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(arr.astype(np.float32))
            t = ((t - data_min) / data_range).clamp(0.0, 1.0)
            return t.permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, C, H, W)

        img_t = to_tensor(image)
        ref_t = to_tensor(reference)

        # piq fsim defaults to chromatic=True, which requires C=3.  For
        # single-channel (grayscale) images the chromatic variant is
        # meaningless, so we always honour our own default of False.
        with torch.no_grad():
            score = fsim(img_t, ref_t, data_range=1.0, chromatic=self._chromatic)

        return float(score.item())
