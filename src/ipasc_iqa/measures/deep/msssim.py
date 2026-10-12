"""Multi-Scale Structural Similarity Index Measure (MS-SSIM).

Shallow adapter over ``piq.multi_scale_ssim`` (Wang et al., 2003).
Neither scipy nor scikit-image provide an MS-SSIM implementation;
``piq`` is the authoritative PyTorch reference.

Requires PyTorch-based metric dependencies.  All heavy imports (``torch``, ``piq``)
are deferred to first use so the base install stays lightweight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType

# Wang et al. (2003) scale weights, index 0 = finest scale.
# piq applies scale_weights[i] at iteration i, starting from the original
# (finest) resolution and downsampling by 2 at each step.
_WANG_WEIGHTS: tuple[float, ...] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


class MSSSIMMeasure(BaseMeasure):
    """Multi-Scale Structural Similarity Index Measure (MS-SSIM).

    Extends single-scale SSIM by computing the measure at multiple
    spatial resolutions and combining them with perceptually motivated
    weights (Wang et al., Asilomar 2003).

    For small images the number of scales is automatically reduced so
    the Gaussian kernel (``kernel_size × kernel_size``) still fits at
    every scale.  Scale weights are renormalised accordingly.

    Range: [0, 1], higher = more similar.  Requires PyTorch-based metric dependencies.
    """

    metadata = MeasureMetadata(
        name="MSSSIM",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=True,
        value_range=(0.0, 1.0),
        citation=(
            "Wang et al., 'Multiscale Structural Similarity for Image Quality Assessment', "
            "Proc. Asilomar Conf. Signals, Systems and Computers, 2003."
        ),
        bibtex=(
            "@inproceedings{wang2003msssim,\n"
            "  author    = {Wang, Zhou and Simoncelli, Eero P. and Bovik, Alan C.},\n"
            "  title     = {Multiscale Structural Similarity for Image Quality Assessment},\n"
            "  booktitle = {The Thirty-Seventh Asilomar Conference on Signals,\n"
            "               Systems and Computers},\n"
            "  pages     = {1398--1402},\n"
            "  year      = {2003}\n"
            "}"
        ),
        description=(
            "Multi-Scale Structural Similarity Index (MS-SSIM). "
            "Computed via piq.multi_scale_ssim — no reference implementation "
            "exists in scipy or scikit-image. "
        ),
    )

    def __init__(
        self,
        kernel_size: int = 11,
        kernel_sigma: float = 1.5,
        device: str = "auto",
    ) -> None:
        # Pass hyperparameters to super() so config_canonical_json() picks them
        # up and they contribute to the cache key.
        super().__init__(
            device=device,
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
        )
        self._kernel_size = kernel_size
        self._kernel_sigma = kernel_sigma

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, arr: np.ndarray, data_min: float, data_range: float) -> Any:
        """Convert an (H, W, C) float32 array to a (1, C, H, W) tensor in [0, 1].

        Both image and reference are normalised with the *same* ``data_min`` /
        ``data_range`` (derived from the reference) to preserve intensity
        differences across the pair.
        """
        import torch

        t = torch.from_numpy(arr.astype(np.float32))
        t = (t - data_min) / data_range
        t = t.clamp(0.0, 1.0)
        # (H, W, C) → (C, H, W) → (1, C, H, W)
        return t.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _adaptive_scale_weights(self, min_dim: int) -> Any:
        """Return a weight tensor with as many scales as the image can support.

        At scale s (1-indexed) the spatial dimension has been halved (s-1)
        times, so the minimum feasible dimension is
        ``min_dim / 2^(s-1) >= kernel_size``, giving
        ``s_max = floor(log2(min_dim / kernel_size)) + 1``.
        """
        import torch

        if min_dim < self._kernel_size:
            # Degenerate: image too small even for one scale — piq will likely
            # still run but warn; let it handle the edge case gracefully.
            n = 1
        else:
            n = min(5, int(np.floor(np.log2(min_dim / self._kernel_size))) + 1)
            n = max(1, n)

        weights = torch.tensor(
            _WANG_WEIGHTS[:n],
            dtype=torch.float32,
            device=self.device,
        )
        return weights / weights.sum()  # renormalise so scores stay in [0, 1]

    # ------------------------------------------------------------------
    # BaseMeasure interface
    # ------------------------------------------------------------------

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
            raise ValueError("MS-SSIM requires a reference image.")

        try:
            from piq import multi_scale_ssim
        except ImportError as exc:
            raise ImportError(
                "MS-SSIM requires PyTorch-based metric dependencies. Reinstall ipasc_iqa, or reinstall PyTorch following the official PyTorch selector."
            ) from exc
        import torch

        # Joint normalisation: map reference range → [0, 1].
        data_min = float(reference.min())
        data_max = float(reference.max())
        data_range = data_max - data_min if data_max > data_min else 1.0

        img_t = self._to_tensor(image, data_min, data_range)
        ref_t = self._to_tensor(reference, data_min, data_range)

        h, w = image.shape[0], image.shape[1]
        scale_weights = self._adaptive_scale_weights(min(h, w))

        with torch.no_grad():
            score = multi_scale_ssim(
                img_t,
                ref_t,
                kernel_size=self._kernel_size,
                kernel_sigma=self._kernel_sigma,
                data_range=1.0,
                scale_weights=scale_weights,
            )

        return float(score.item())
