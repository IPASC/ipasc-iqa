"""BaseMeasure, MeasureMetadata, and EvaluationInput."""

from __future__ import annotations
from enum import Enum

from dataclasses import dataclass, field
from typing import Any

import numpy as np

class MeasureType(Enum):
    FULL_REFERENCE = 1
    REDUCED_REFERENCE = 2
    NO_REFERENCE = 3

@dataclass(frozen=True)
class MeasureMetadata:
    name: str
    version: str
    measure_type: MeasureType
    requires_signal_roi: bool
    requires_noise_roi: bool
    accepted_reference_types: tuple[str, ...] | None = None  # None = any
    higher_is_better: bool = True
    value_range: tuple[float, float] | None = None
    citation: str | None = None
    bibtex: str | None = None  # raw BibTeX entry, surfaced by get_bibtex_citation()
    description: str = ""


@dataclass
class EvaluationInput:
    image: np.ndarray
    reference: np.ndarray | None = None
    reference_type: str | None = None
    signal_roi: np.ndarray | None = None
    noise_roi: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "reference": self.reference,
            "reference_type": self.reference_type,
            "signal_roi": self.signal_roi,
            "noise_roi": self.noise_roi,
        }


class BaseMeasure:
    """Abstract base for all IQA measures.

    Subclasses must set a class-level ``metadata`` attribute.
    """

    metadata: MeasureMetadata

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Concrete subclasses (not abstract intermediaries) must declare metadata.
        # We skip the check for classes that leave metadata unset at definition time.

    def __init__(self, device: str = "auto", **kwargs: Any) -> None:
        from ipasc_iqa.core.device import resolve

        self.device = resolve(device)
        # Hyperparameters that affect the *result value* (device does not). Captured
        # so the cache key changes when a measure is reconfigured.
        self._config: dict[str, Any] = dict(kwargs)

    def config_canonical_json(self) -> str:
        """Canonical JSON of the measure's configuration (for cache keying)."""
        import json

        return json.dumps(self._config, sort_keys=True, separators=(",", ":"), default=str)

    def evaluate(
        self,
        image: np.ndarray,
        *,
        reference: np.ndarray | None = None,
        reference_type: str | None = None,
        signal_roi: np.ndarray | None = None,
        noise_roi: np.ndarray | None = None,
    ) -> float | dict[str, float]:
        raise NotImplementedError

    def batch_evaluate(
        self, items: list[EvaluationInput]
    ) -> list[float | dict[str, float]]:
        return [self.evaluate(**it.as_kwargs()) for it in items]

    def get_bibtex_citation(self) -> str:
        """Return a raw BibTeX entry for the measure's reference publication.

        Reads ``metadata.bibtex``. Measures without a canonical reference (e.g.
        PSNR, MSE) leave it unset, in which case this raises ``NotImplementedError``.
        """
        bibtex = self.metadata.bibtex
        if bibtex is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not provide a BibTeX citation."
            )
        return bibtex
