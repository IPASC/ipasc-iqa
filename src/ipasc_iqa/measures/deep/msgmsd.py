"""Multi-Scale Gradient Magnitude Similarity Deviation (MS-GMSD).

Shallow adapter over ``piq.multi_scale_gmsd`` (Reisenhofer et al., 2018).

MS-GMSD extends GMSD to multiple spatial scales by averaging-pooling the
image before computing GMSD at each level.  The per-scale GMSD values are
combined with weights from the original paper:
``[0.096, 0.596, 0.289, 0.019]`` (finest → coarsest, 4 scales).

For small images the number of scales is automatically reduced so the
gradient kernel still fits at every level.  Scale weights are renormalised
when fewer than 4 scales are used.

Requires ``ipasc_iqa[deep]``.  All heavy imports (``torch``, ``piq``) are
deferred to first use so the base install stays lightweight.
"""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType

# Reisenhofer et al. (2018) scale weights: finest → coarsest, 4 levels.
# piq normalises these internally, but we renormalise when clipping scales.
_MS_GMSD_WEIGHTS: tuple[float, ...] = (0.096, 0.596, 0.289, 0.019)
# Gradient (Prewitt-like) kernel is 3×3; require at least 4 px after
# downsampling to leave headroom.
_MIN_DIM_PER_SCALE: int = 4


class MSGMSDMeasure(BaseMeasure):
    """Multi-Scale Gradient Magnitude Similarity Deviation (MS-GMSD).

    Combines GMSD scores computed at 4 spatial scales using the weights
    proposed by Reisenhofer et al.  More robust than single-scale GMSD
    for images with complex multi-scale structures.

    For small images the number of active scales is reduced automatically;
    scale weights are renormalised to sum to 1.

    Range: [0, ∞), lower = better quality.  Requires ``ipasc_iqa[deep]``.
    """

    metadata = MeasureMetadata(
        name="MSGMSD",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=False,
        value_range=(0.0, float("inf")),
        citation=(
            "Zhang et al., 'Gradient Magnitude Similarity Deviation on Multiple Scales "
            "for Color Image Quality Assessment', IEEE ICASSP 2017."
        ),
        bibtex=(
            "@inproceedings{zhang2017msgmsd,\n"
            "  author    = {Zhang, Bo and Sander, Pedro V. and Bermak, Amine},\n"
            "  title     = {Gradient Magnitude Similarity Deviation on Multiple Scales\n"
            "               for Color Image Quality Assessment},\n"
            "  booktitle = {IEEE International Conference on Acoustics, Speech and\n"
            "               Signal Processing (ICASSP)},\n"
            "  pages     = {1253--1257},\n"
            "  year      = {2017}\n"
            "}"
        ),
        description=(
            "Multi-Scale Gradient Magnitude Similarity Deviation (MS-GMSD). "
            "Computed via piq.multi_scale_gmsd with adaptive scale selection. "
            "Lower deviation = better quality. Requires ipasc_iqa[deep]."
        ),
    )

    def _adaptive_scale_weights(self, min_dim: int) -> "torch.Tensor":
        """Return weights for as many scales as the image can support.

        GMSD uses a 3×3 Prewitt-like gradient kernel; we require
        ``min_dim / 2^(s-1) >= _MIN_DIM_PER_SCALE`` at every scale ``s``.
        """
        import torch

        if min_dim < _MIN_DIM_PER_SCALE:
            n = 1
        else:
            n = min(4, int(np.floor(np.log2(min_dim / _MIN_DIM_PER_SCALE))) + 1)
            n = max(1, n)

        weights = torch.tensor(_MS_GMSD_WEIGHTS[:n], dtype=torch.float32)
        return weights / weights.sum()

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
            raise ValueError("MS-GMSD requires a reference image.")

        try:
            from piq import multi_scale_gmsd
        except ImportError as exc:
            raise ImportError(
                "MS-GMSD requires 'ipasc_iqa[deep]'. "
                "Install with:  pip install 'ipasc_iqa[deep]'"
            ) from exc
        import torch

        data_min = float(reference.min())
        data_max = float(reference.max())
        data_range = data_max - data_min if data_max > data_min else 1.0

        def to_tensor(arr: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(arr.astype(np.float32))
            t = ((t - data_min) / data_range).clamp(0.0, 1.0)
            return t.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        img_t = to_tensor(image)
        ref_t = to_tensor(reference)

        h, w = image.shape[0], image.shape[1]
        scale_weights = self._adaptive_scale_weights(min(h, w))

        with torch.no_grad():
            score = multi_scale_gmsd(
                img_t,
                ref_t,
                data_range=1.0,
                scale_weights=scale_weights,
                chromatic=False,
            )

        return float(score.item())
