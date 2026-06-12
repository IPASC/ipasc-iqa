"""Learned Perceptual Image Patch Similarity (LPIPS).

Shallow adapter over the ``lpips`` package
(Zhang et al., CVPR 2018 — https://github.com/richzhang/PerceptualSimilarity).

Requires ``ipasc_iqa[deep]``.  All heavy imports (``torch``, ``lpips``) are
deferred to first use so the base install stays lightweight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class LPIPSMeasure(BaseMeasure):
    metadata = MeasureMetadata(
        name="LPIPS",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=False,
        value_range=(0.0, float("inf")),
        citation=(
            "Zhang et al., 'The Unreasonable Effectiveness of Deep Features as a Perceptual "
            "Metric', CVPR 2018. https://github.com/richzhang/PerceptualSimilarity"
        ),
        description=(
            "Learned Perceptual Image Patch Similarity (AlexNet backbone). "
            "Lower distance = more similar. Requires ipasc_iqa[deep]."
        ),
    )

    def __init__(self, device: str = "auto", net: str = "alex") -> None:
        # net is passed as a kwarg so BaseMeasure captures it in _config and it
        # becomes part of config_canonical_json() — changing the backbone
        # automatically invalidates cached results.
        super().__init__(device=device, net=net)
        self._net = net
        self._model: Any = None  # lazy: loaded on first evaluate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Load and cache the LPIPS model (downloads weights on first call)."""
        if self._model is not None:
            return self._model
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPS requires 'ipasc_iqa[deep]'. "
                "Install with:  pip install 'ipasc_iqa[deep]'"
            ) from exc
        import torch

        model = lpips.LPIPS(net=self._net, verbose=False)
        self._model = model.to(torch.device(self.device)).eval()
        return self._model

    def _to_tensor(self, arr: np.ndarray, data_min: float, data_range: float) -> Any:
        """Convert an (H, W, C) float32 array to a (1, 3, H, W) tensor in [0, 1].

        Both image and reference are normalised with the *same* ``data_min`` /
        ``data_range`` (derived from the reference) so intensity differences
        between reconstruction and reference are preserved.
        """
        import torch

        t = torch.from_numpy(arr.astype(np.float32))
        # Normalise to [0, 1]
        t = (t - data_min) / data_range
        t = t.clamp(0.0, 1.0)
        # (H, W, C) → (C, H, W)
        t = t.permute(2, 0, 1)
        # Expand single-channel grayscale to 3 channels without copying data
        if t.shape[0] == 1:
            t = t.expand(3, -1, -1)
        # Add batch dim → (1, 3, H, W) and move to compute device
        return t.unsqueeze(0).to(torch.device(self.device))

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
            raise ValueError("LPIPS requires a reference image.")

        # Ensure torch+lpips are installed before touching tensors.
        model = self._get_model()

        import torch

        # Joint normalisation: map reference range → [0, 1].  Image pixels
        # outside this range are clamped, matching what would be visible on a
        # display calibrated to the reference.
        data_min = float(reference.min())
        data_max = float(reference.max())
        data_range = data_max - data_min if data_max > data_min else 1.0

        img_t = self._to_tensor(image, data_min, data_range)
        ref_t = self._to_tensor(reference, data_min, data_range)

        # normalize=True: lpips accepts [0, 1] and maps to [-1, 1] internally.
        with torch.no_grad():
            score = model(img_t, ref_t, normalize=True)

        return float(score.item())
