"""Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE).

No-reference measure implemented via ``piq``.  Requires ``ipasc_iqa[deep]``.
All heavy imports are deferred to first use so the base install stays lightweight.
"""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class BRISQUEMeasure(BaseMeasure):
    metadata = MeasureMetadata(
        name="BRISQUE",
        version="1.0.0",
        measure_type=MeasureType.NO_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=False,
        value_range=(0.0, float("inf")),
        citation=(
            "Mittal et al., 'No-Reference Image Quality Assessment in the Spatial Domain', "
            "IEEE TIP 2012."
        ),
        description=(
            "Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE). "
            "Lower score = better quality. Requires ipasc_iqa[deep]."
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
        try:
            from piq import brisque
        except ImportError as exc:
            raise ImportError(
                "BRISQUE requires 'ipasc_iqa[deep]'. "
                "Install with:  pip install 'ipasc_iqa[deep]'"
            ) from exc
        import torch

        # (H, W, C) → (1, C, H, W) float32
        t = torch.from_numpy(image.astype(np.float32))
        t = t.permute(2, 0, 1).unsqueeze(0)

        # Normalise to [0, 1] (piq default data_range=1.0)
        t_min, t_max = float(t.min()), float(t.max())
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)
        t = t.clamp(0.0, 1.0)

        # Adaptive kernel size: odd integer scaled to the shorter spatial dimension.
        # At 256×256 this matches the piq default of kernel_size=7, sigma=7/6.
        h, w = image.shape[0], image.shape[1]
        ks = max(1, int(np.ceil(min(h, w) / 41)))
        if ks % 2 == 0:
            ks += 1  # kernel_size must be odd
        kernel_sigma = float(ks) / 6.0

        return float(brisque(t, kernel_size=ks, kernel_sigma=kernel_sigma).item())
