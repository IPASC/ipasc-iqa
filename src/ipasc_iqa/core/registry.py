"""Entry-point discovery and MeasureRegistry."""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ipasc_iqa.core.measure import BaseMeasure

_ENTRY_POINT_GROUP = "ipasc_iqa.measures"


class MeasureRegistry:
    """Lazy registry of all installed IQA measures."""

    def __init__(self) -> None:
        self._classes: dict[str, type[BaseMeasure]] = {}
        self._distributions: dict[str, str] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        from ipasc_iqa.core.measure import BaseMeasure

        eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
        for ep in eps:
            try:
                cls = ep.load()
            except Exception as exc:
                import warnings
                warnings.warn(f"Failed to load measure '{ep.name}': {exc}", stacklevel=2)
                continue

            if not (isinstance(cls, type) and issubclass(cls, BaseMeasure)):
                import warnings
                warnings.warn(
                    f"Entry point '{ep.name}' does not point to a BaseMeasure subclass.",
                    stacklevel=2,
                )
                continue

            if not hasattr(cls, "metadata"):
                import warnings
                warnings.warn(
                    f"Measure class '{cls.__name__}' has no metadata attribute, skipping.",
                    stacklevel=2,
                )
                continue

            self._classes[cls.metadata.name] = cls
            dist_name = ep.dist.name if ep.dist else "unknown"
            self._distributions[cls.metadata.name] = dist_name

        self._loaded = True

    def list_measures(self) -> list[str]:
        self._load()
        return sorted(self._classes)

    def get(self, name: str) -> type[BaseMeasure]:
        self._load()
        if name not in self._classes:
            raise KeyError(f"Unknown measure: '{name}'. Available: {self.list_measures()}")
        return self._classes[name]

    def distribution_for(self, name: str) -> str:
        self._load()
        return self._distributions.get(name, "unknown")

    def instantiate(self, name: str, device: str = "auto", **kwargs: object) -> BaseMeasure:
        return self.get(name)(device=device, **kwargs)


registry = MeasureRegistry()
