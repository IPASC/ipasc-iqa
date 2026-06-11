"""BaseMeasure, MeasureMetadata, and EvaluationInput."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeasureMetadata:
    name: str
    version: str
    full_reference: bool
    requires_signal_roi: bool
    requires_noise_roi: bool
    accepted_reference_types: tuple[str, ...] | None = None  # None = any
    higher_is_better: bool = True
    value_range: tuple[float, float] | None = None
    citation: str | None = None
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
