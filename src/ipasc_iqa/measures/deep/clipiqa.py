"""CLIP-IQA — No-reference image quality assessment via CLIP.

Shallow adapter over ``piq.CLIPIQA`` (Wang et al., 2023).

CLIP-IQA exploits the rich visual–language representations learned by CLIP
(Radford et al., 2021) to assess image quality without a reference.  The
model compares the image embedding against a set of quality-related text
prompts and outputs a quality score in [0, 1].

piq bundles its own CLIP weights; no separate ``open_clip_torch`` package
is required — just ``ipasc_iqa[deep]``.

The model is loaded lazily on the first ``evaluate()`` call so that
``ipasc_iqa list measures`` never triggers a download.

Requires ``ipasc_iqa[deep]``.  All heavy imports (``torch``, ``piq``) are
deferred to first use so the base install stays lightweight.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class CLIPIQAMeasure(BaseMeasure):
    """No-reference quality score via CLIP (CLIP-IQA).

    Compares CLIP image embeddings against quality-related text prompts.
    Designed for natural images; may generalise to photoacoustic
    reconstructions to some extent.

    Single-channel images are broadcast to 3 channels before passing to
    CLIP (zero-copy ``.expand``).

    Range: [0, 1], higher = better perceived quality.
    Requires ``ipasc_iqa[deep]`` (piq bundles CLIP weights).
    """

    metadata = MeasureMetadata(
        name="CLIPIQA",
        version="1.0.0",
        measure_type=MeasureType.NO_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=True,
        value_range=(0.0, 1.0),
        citation=(
            "Wang et al., 'Exploring CLIP for Assessing the Look and Feel of Images', "
            "AAAI 2023. "
            "Radford et al., 'Learning Transferable Visual Models from Natural Language "
            "Supervision', ICML 2021."
        ),
        bibtex=(
            "@inproceedings{wang2023clipiqa,\n"
            "  author    = {Wang, Jianyi and Chan, Kelvin C. K. and Loy, Chen Change},\n"
            "  title     = {Exploring {CLIP} for Assessing the Look and Feel of Images},\n"
            "  booktitle = {AAAI Conference on Artificial Intelligence},\n"
            "  year      = {2023}\n"
            "}"
        ),
        description=(
            "No-reference image quality via CLIP (CLIP-IQA). "
            "Computed via piq.CLIPIQA — piq bundles its own CLIP weights. "
            "Model downloaded on first evaluate() call. "
            "Requires ipasc_iqa[deep]."
        ),
    )

    def __init__(self, device: str = "auto") -> None:
        super().__init__(device=device)
        self._model: Any = None  # loaded lazily

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """Load and cache CLIP-IQA (downloads weights on first call)."""
        if self._model is not None:
            return self._model
        try:
            from piq import CLIPIQA
        except ImportError as exc:
            raise ImportError(
                "CLIP-IQA requires 'ipasc_iqa[deep]'. "
                "Install with:  pip install 'ipasc_iqa[deep]'"
            ) from exc
        import torch

        self._model = CLIPIQA().to(torch.device(self.device)).eval()
        return self._model

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
        model = self._get_model()

        import torch

        # No-reference: normalise per image to [0, 1].
        t = torch.from_numpy(image.astype(np.float32))
        t_min, t_max = float(t.min()), float(t.max())
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)
        t = t.clamp(0.0, 1.0)

        # (H, W, C) → (1, C, H, W)
        t = t.permute(2, 0, 1).unsqueeze(0)

        # CLIP expects 3-channel input; expand grayscale without copying data.
        if t.shape[1] == 1:
            t = t.expand(-1, 3, -1, -1).contiguous()

        t = t.to(torch.device(self.device))

        with torch.no_grad():
            score = model(t)

        return float(score.item())
