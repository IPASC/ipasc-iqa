"""Information-Weighted Structural Similarity (IW-SSIM).

Shallow adapter over ``piq.information_weighted_ssim`` (Wang & Li, 2011).
Neither scipy nor scikit-image provide an IW-SSIM implementation;
``piq`` is the authoritative PyTorch reference.

IW-SSIM weights local SSIM patches by their information content, estimated
via a Gaussian scale-mixture model on steerable pyramid sub-bands.  This
makes it more sensitive to perceptually important fine-detail regions than
plain SSIM.

Requires PyTorch-based metric dependencies.  All heavy imports (``torch``, ``piq``)
are deferred to first use so the base install stays lightweight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class IWSSIMMeasure(BaseMeasure):
    """Information-Weighted Structural Similarity (IW-SSIM).

    Improves on SSIM by computing a local information-content map from a
    steerable pyramid decomposition and using it to weight the per-patch
    similarity scores.  Regions with high information content (edges,
    textures) contribute more to the final score than flat areas.

    Minimum recommended image size: 32×32 (the internal pyramid uses four
    decomposition levels).

    Range: [0, 1], higher = more similar.  Requires PyTorch-based metric dependencies.
    """

    metadata = MeasureMetadata(
        name="IWSSIM",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=True,
        value_range=(0.0, 1.0),
        citation=(
            "Wang & Li, 'Information Content Weighting for Perceptual Image Quality Assessment', "
            "IEEE Trans. Image Process. 20(5), 2011."
        ),
        bibtex=(
            "@article{wang2011iwssim,\n"
            "  author  = {Wang, Zhou and Li, Qiang},\n"
            "  title   = {Information Content Weighting for Perceptual Image Quality Assessment},\n"
            "  journal = {IEEE Transactions on Image Processing},\n"
            "  volume  = {20},\n"
            "  number  = {5},\n"
            "  pages   = {1185--1198},\n"
            "  year    = {2011}\n"
            "}"
        ),
        description=(
            "Information-Weighted Structural Similarity (IW-SSIM). "
            "Computed via piq.information_weighted_ssim — no reference implementation "
            "exists in scipy or scikit-image. "
            "With the default five scales and kernel_size=11, images must be at least 161×161. "
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
        if self.device == "mps":
                    self.device = "cpu"
                    
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
            raise ValueError("IW-SSIM requires a reference image.")

        try:
            from piq import information_weighted_ssim
        except ImportError as exc:
            raise ImportError(
                "IW-SSIM requires PyTorch-based metric dependencies. Reinstall ipasc_iqa, or reinstall PyTorch following the official PyTorch selector."
            ) from exc
        import torch

        # Joint normalisation: map reference range → [0, 1].
        data_min = float(reference.min())
        data_max = float(reference.max())
        data_range = data_max - data_min if data_max > data_min else 1.0

        img_t = self._to_tensor(image, data_min, data_range)
        ref_t = self._to_tensor(reference, data_min, data_range)

        with torch.no_grad():
            score = information_weighted_ssim(
                img_t,
                ref_t,
                data_range=1.0,
                kernel_size=self._kernel_size,
                kernel_sigma=self._kernel_sigma,
            )

        return float(score.item())
