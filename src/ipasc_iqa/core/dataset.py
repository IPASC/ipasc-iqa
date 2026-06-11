"""Dataset, DatasetManifest, ImageRecord, ImageHashes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass
class ImageHashes:
    image: str
    signal_roi: str | None = None
    noise_roi: str | None = None
    references: dict[str, str] = field(default_factory=dict)


@dataclass
class ImageRecord:
    id: str
    image: np.ndarray
    signal_roi: np.ndarray | None
    noise_roi: np.ndarray | None
    references: dict[str, np.ndarray]
    metadata: dict[str, Any]
    hashes: ImageHashes


class Dataset:
    """Abstract base for all datasets.

    Subclasses implement ``_iter_records`` and expose ``name``, ``version``.
    """

    name: str
    version: str

    @property
    def manifest_sha256(self) -> str:
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        h = hashlib.sha256()
        with open(manifest_path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def _manifest_path(self) -> Path:
        raise NotImplementedError

    def __iter__(self) -> Iterator[ImageRecord]:
        return self._iter_records()

    def _iter_records(self) -> Iterator[ImageRecord]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def verify(self) -> "VerificationReport":
        from ipasc_iqa.core.roi import hash_file

        errors: list[str] = []
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            return VerificationReport(ok=False, errors=["Manifest not found"])

        with open(manifest_path) as f:
            manifest = json.load(f)

        root = manifest_path.parent
        for entry in manifest.get("images", []):
            p = root / entry["path"]
            if not p.exists():
                errors.append(f"Missing: {entry['path']}")
                continue
            actual = hash_file(p)
            if actual != entry["sha256"]:
                errors.append(f"Hash mismatch: {entry['path']}")

            for key in ("signal_roi", "noise_roi"):
                if entry.get(key):
                    rp = root / entry[key]["path"]
                    if not rp.exists():
                        errors.append(f"Missing {key}: {entry[key]['path']}")
                    elif hash_file(rp) != entry[key]["sha256"]:
                        errors.append(f"Hash mismatch {key}: {entry[key]['path']}")

            for ref_type, ref_entry in entry.get("references", {}).items():
                rp = root / ref_entry["path"]
                if not rp.exists():
                    errors.append(f"Missing reference {ref_type}: {ref_entry['path']}")
                elif hash_file(rp) != ref_entry["sha256"]:
                    errors.append(f"Hash mismatch reference {ref_type}: {ref_entry['path']}")

        return VerificationReport(ok=len(errors) == 0, errors=errors)


@dataclass
class VerificationReport:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.ok:
            return "Verification passed."
        return "Verification FAILED:\n" + "\n".join(f"  - {e}" for e in self.errors)
