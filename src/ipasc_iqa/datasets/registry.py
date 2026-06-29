"""Discovery of dataset *adapters*.

Unlike measures, datasets are not arbitrary plugins of behaviour — but the set of
datasets the framework knows how to *fetch and convert* is defined by the adapter
classes in this package (and any registered via the ``ipasc_iqa.datasets`` entry
point group). Discovery is therefore adapter-driven, never disk-driven: only
datasets with an adapter are offered, and selecting one downloads/prepares it on
demand into a managed data directory.

Each adapter is a :class:`~ipasc_iqa.datasets.local.LocalDataset` subclass that
declares ``DATASET_NAME`` / ``DATASET_VERSION`` / ``DATASET_DESCRIPTION`` and
accepts a ``download`` keyword.
"""

from __future__ import annotations

import importlib.metadata
import warnings
from pathlib import Path

import platformdirs

from ipasc_iqa.datasets.local import LocalDataset

_ENTRY_POINT_GROUP = "ipasc_iqa.datasets"


def default_data_dir() -> Path:
    """Managed location where datasets are downloaded/converted."""
    return Path(platformdirs.user_data_dir("ipasc_iqa", "ipasc")) / "datasets"


class DatasetRegistry:
    """Registry of dataset adapter classes (built-in + entry-point)."""

    def __init__(self) -> None:
        self._classes: dict[str, type[LocalDataset]] = {}
        self._distributions: dict[str, str] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        # Built-in adapters.
        from ipasc_iqa.datasets.alladien import (
            DavoudiDataset,
            EFADataset,
            OADATDataset,
            PHIREDataset,
        )
        from ipasc_iqa.datasets.digital_twin import DigitalTwinDataset
        from ipasc_iqa.datasets.duke_pam import DukePAMDataset
        from ipasc_iqa.datasets.ear_vessels import EarVesselsDataset
        from ipasc_iqa.datasets.pammoth import PAMMOTHDataset
        from ipasc_iqa.datasets.phot_iqa import PhotIQADataset

        self.register(PhotIQADataset, distribution="ipasc_iqa")
        self.register(DigitalTwinDataset, distribution="ipasc_iqa")
        self.register(DukePAMDataset, distribution="ipasc_iqa")
        self.register(EarVesselsDataset, distribution="ipasc_iqa")
        self.register(PAMMOTHDataset, distribution="ipasc_iqa")
        self.register(DavoudiDataset, distribution="ipasc_iqa")
        self.register(OADATDataset, distribution="ipasc_iqa")
        self.register(EFADataset, distribution="ipasc_iqa")
        self.register(PHIREDataset, distribution="ipasc_iqa")

        # Third-party adapters via entry points (reserved group, forward-looking).
        for ep in importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP):
            try:
                cls = ep.load()
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"Failed to load dataset adapter '{ep.name}': {exc}", stacklevel=2)
                continue
            if not (isinstance(cls, type) and issubclass(cls, LocalDataset)):
                warnings.warn(
                    f"Dataset entry point '{ep.name}' is not a LocalDataset subclass.",
                    stacklevel=2,
                )
                continue
            dist = ep.dist.name if ep.dist else "unknown"
            self.register(cls, distribution=dist)

        self._loaded = True

    def register(self, cls: type[LocalDataset], *, distribution: str = "unknown") -> None:
        """Register an adapter class (also usable by tests / third parties)."""
        name = getattr(cls, "DATASET_NAME", None)
        if not name or name == "local":
            raise ValueError(f"Adapter {cls.__name__} must set a DATASET_NAME.")
        self._classes[name] = cls
        self._distributions[name] = distribution

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[dict]:
        self._load()
        out = []
        for name in sorted(self._classes):
            cls = self._classes[name]
            out.append(
                {
                    "name": name,
                    "version": cls.DATASET_VERSION,
                    "description": cls.DATASET_DESCRIPTION,
                    "bibtex": getattr(cls, "DATASET_BIBTEX", None),
                    "distribution": self._distributions.get(name, "unknown"),
                }
            )
        return out

    def get(self, name: str) -> type[LocalDataset]:
        self._load()
        if name not in self._classes:
            raise KeyError(f"Unknown dataset adapter: '{name}'. Available: {sorted(self._classes)}")
        return self._classes[name]

    def root_for(self, name: str, data_dir: Path | None = None) -> Path:
        """Managed root directory for a dataset's downloaded/converted data."""
        base = Path(data_dir) if data_dir is not None else default_data_dir()
        return base / name

    def is_prepared(self, name: str, data_dir: Path | None = None) -> bool:
        return self.get(name).is_prepared(self.root_for(name, data_dir))

    def manifest_path(self, name: str, data_dir: Path | None = None) -> Path:
        return self.get(name).prepared_manifest_path(self.root_for(name, data_dir))

    def instantiate(
        self, name: str, data_dir: Path | None = None, *, download: bool = True
    ) -> LocalDataset:
        """Instantiate the adapter, downloading/converting on demand."""
        cls = self.get(name)
        root = self.root_for(name, data_dir)
        root.mkdir(parents=True, exist_ok=True)
        return cls(root, download=download)


registry = DatasetRegistry()
