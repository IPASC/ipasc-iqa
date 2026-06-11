"""LocalDataset — manifest-driven dataset backed by files on disk."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ipasc_iqa.core.dataset import Dataset, ImageHashes, ImageRecord
from ipasc_iqa.core.roi import hash_array


class LocalDataset(Dataset):
    """A dataset whose images live in a local directory following the data contract.

    The directory must contain a ``manifest.json`` and an ``images/`` sub-directory.
    Subclasses may override ``_prepare`` to download and convert source data before
    the manifest is read.

    Registry contract
    -----------------
    Adapters that are discoverable via :class:`~ipasc_iqa.datasets.registry.DatasetRegistry`
    declare ``DATASET_NAME`` / ``DATASET_VERSION`` / ``DATASET_DESCRIPTION`` as class
    attributes, accept a ``download`` keyword in ``__init__``, and may override the
    ``is_prepared`` / ``prepared_manifest_path`` classmethods if their prepared
    manifest lives somewhere other than ``<root>/manifest.json``.
    """

    #: Registry metadata (overridden by concrete adapters).
    DATASET_NAME: str = "local"
    DATASET_VERSION: str = "unknown"
    DATASET_DESCRIPTION: str = ""

    @classmethod
    def prepared_manifest_path(cls, root: Path | str) -> Path:
        """Where the prepared manifest lands under *root* (before instantiation)."""
        return Path(root) / "manifest.json"

    @classmethod
    def is_prepared(cls, root: Path | str) -> bool:
        """True if the dataset is already downloaded/converted at *root*."""
        return cls.prepared_manifest_path(root).exists()

    def __init__(self, root: Path | str, *, download: bool = True) -> None:
        self.root = Path(root)
        self._download = download
        self._prepare()
        self._manifest: dict = self._load_manifest()
        self.name: str = self._manifest.get("dataset_name", self.root.name)
        self.version: str = self._manifest.get("dataset_version", "unknown")
        self._records: list[dict] = self._manifest.get("images", [])

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _prepare(self) -> None:
        """Called before the manifest is read.

        Override to download data, convert formats, and generate a manifest.
        The base implementation is a no-op.
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _load_manifest(self) -> dict:
        path = self._manifest_path()
        if not path.exists():
            raise FileNotFoundError(
                f"No manifest.json found in {self.root}. "
                "Run 'ipasc_iqa manifest generate <folder>' to create one, "
                "or call _prepare() in a subclass."
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_array(self, rel_path: str) -> np.ndarray:
        p = self.root / rel_path
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def _iter_records(self) -> Iterator[ImageRecord]:
        for entry in self._records:
            image = self._load_array(entry["path"])
            image_hash = hash_array(image)

            signal_roi: np.ndarray | None = None
            signal_hash: str | None = None
            if entry.get("signal_roi"):
                signal_roi = self._load_array(entry["signal_roi"]["path"]).astype(bool)
                signal_hash = hash_array(signal_roi)

            noise_roi: np.ndarray | None = None
            noise_hash: str | None = None
            if entry.get("noise_roi"):
                noise_roi = self._load_array(entry["noise_roi"]["path"]).astype(bool)
                noise_hash = hash_array(noise_roi)

            references: dict[str, np.ndarray] = {}
            ref_hashes: dict[str, str] = {}
            for ref_type, ref_entry in entry.get("references", {}).items():
                ref_arr = self._load_array(ref_entry["path"])
                references[ref_type] = ref_arr
                ref_hashes[ref_type] = hash_array(ref_arr)

            yield ImageRecord(
                id=entry["id"],
                image=image,
                signal_roi=signal_roi,
                noise_roi=noise_roi,
                references=references,
                metadata=entry.get("metadata", {}),
                hashes=ImageHashes(
                    image=image_hash,
                    signal_roi=signal_hash,
                    noise_roi=noise_hash,
                    references=ref_hashes,
                ),
            )
