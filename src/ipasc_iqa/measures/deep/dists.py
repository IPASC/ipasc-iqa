"""Deep Image Structure and Texture Similarity (DISTS).

Shallow adapter over the ``DISTS-pytorch`` package
(Ding et al., IEEE TPAMI 2022 — https://github.com/dingkeyan93/DISTS).

Requires ``ipasc_iqa[deep]``.  All heavy imports (``torch``, ``DISTS_pytorch``)
are deferred to first use so the base install stays lightweight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class DISTSMeasure(BaseMeasure):
    metadata = MeasureMetadata(
        name="DISTS",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=False,
        value_range=(0.0, 1.0),
        citation=(
            "Ding et al., 'Image Quality Assessment: Unifying Structure and Texture Similarity', "
            "IEEE TPAMI 2022. https://github.com/dingkeyan93/DISTS"
        ),
        bibtex=(
            "@article{ding2022dists,\n"
            "  author  = {Ding, Keyan and Ma, Kede and Wang, Shiqi and Simoncelli, Eero P.},\n"
            "  title   = {Image Quality Assessment: Unifying Structure and Texture Similarity},\n"
            "  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},\n"
            "  volume  = {44},\n"
            "  number  = {5},\n"
            "  pages   = {2567--2581},\n"
            "  year    = {2022}\n"
            "}"
        ),
        description=(
            "Deep Image Structure and Texture Similarity. "
            "Invariant to texture resampling; lower distance = more similar. "
            "Requires ipasc_iqa[deep]."
        ),
    )

    def __init__(self, device: str = "auto") -> None:
        # DISTS has no user-configurable backbone or hyperparameters, so no
        # extra kwargs are needed for config_canonical_json() / cache keying.
        super().__init__(device=device)
        self._model: Any = None  # lazy: loaded on first evaluate()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Load and cache the DISTS model (downloads weights on first call)."""
        if self._model is not None:
            return self._model
        try:
            from DISTS_pytorch import DISTS
        except ImportError as exc:
            raise ImportError(
                "DISTS requires 'ipasc_iqa[deep]'. "
                "Install with:  pip install 'ipasc_iqa[deep]'"
            ) from exc
        import torch

        self._model = DISTS().to(torch.device(self.device)).eval()
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
            raise ValueError("DISTS requires a reference image.")

        # Ensure torch+DISTS_pytorch are installed before touching tensors.
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

        # DISTS expects [0, 1] tensors directly — no normalize flag needed.
        with torch.no_grad():
            score = model(img_t, ref_t)

        return float(score.item())
