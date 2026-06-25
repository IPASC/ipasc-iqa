"""Unit tests for the PAMMOTH speed-of-sound dataset adapter (no network)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ipasc_iqa.datasets.pammoth import (
    LOG_FLOOR,
    REFERENCE_METHOD,
    UNPACKED_DIR,
    VARIANT_METHODS,
    PAMMOTHDataset,
)

ALL_METHODS = (REFERENCE_METHOD, *VARIANT_METHODS)


def _write_raw_zip(root: Path, slices: dict[str, np.ndarray]) -> None:
    """Create <root>/pammoth_soundspeed_iqa_subset.zip from archive-relative paths.

    Keys are paths *under* the top-level ``pammoth_soundspeed_iqa_subset/`` folder,
    e.g. ``volunteer1_X/02/PA/FWI/rec_800nm_sagittal_+00.npy``.
    """
    root.mkdir(parents=True, exist_ok=True)
    staging = root / "_npy_src"
    for name, arr in slices.items():
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
        # np.save appends .npy; the key already carries it, so fix the name.
        if path.with_suffix(".npy.npy").exists():
            path.with_suffix(".npy.npy").rename(path)
    with zipfile.ZipFile(root / "pammoth_soundspeed_iqa_subset.zip", "w") as z:
        for npy in staging.rglob("*.npy"):
            arcname = f"{UNPACKED_DIR}/{npy.relative_to(staging).as_posix()}"
            z.write(npy, arcname)


def _manifest(ds_root: Path) -> dict:
    return json.loads((ds_root / "converted" / "manifest.json").read_text())


def _slice_set(volunteer: str, scan: str, filename: str) -> dict[str, np.ndarray]:
    """One slice reconstructed by all four methods, with distinct values."""
    out = {}
    for i, method in enumerate(ALL_METHODS):
        arr = np.full((8, 10), 10.0 + i, dtype=np.float32)
        arr[0, 0] = 1.0  # below LOG_FLOOR → should be clamped
        out[f"{volunteer}/{scan}/PA/{method}/{filename}"] = arr
    return out


def test_fwi_reference_and_variants(tmp_path: Path) -> None:
    root = tmp_path / "pammoth"
    slices: dict[str, np.ndarray] = {}
    slices.update(_slice_set("volunteer1_X", "02", "rec_800nm_sagittal_+00.npy"))
    slices.update(_slice_set("volunteer1_X", "02", "rec_800nm_frontal_-50.npy"))
    _write_raw_zip(root, slices)

    PAMMOTHDataset(root, download=False)
    entries = _manifest(root)["images"]

    n_refs = 2
    assert len({e["metadata"]["stem"] for e in entries}) == n_refs
    assert len(entries) == n_refs * len(VARIANT_METHODS)
    assert len({e["id"] for e in entries}) == len(entries)  # ids unique

    for e in entries:
        assert e["shape"] == [8, 10, 1]
        assert e["dtype"] == "float32"
        assert e["metadata"]["algorithm"] in VARIANT_METHODS
        assert "ground_truth" in e["references"]
        # No segmentations for this dataset → no ROI fields.
        assert "signal_roi" not in e and "noise_roi" not in e

    # Each reference is paired with exactly the three SOS variant methods.
    by_stem: dict[str, set[str]] = {}
    for e in entries:
        by_stem.setdefault(e["metadata"]["stem"], set()).add(e["metadata"]["algorithm"])
    assert all(methods == set(VARIANT_METHODS) for methods in by_stem.values())


def test_log_compression_and_floor(tmp_path: Path) -> None:
    root = tmp_path / "pammoth"
    _write_raw_zip(root, _slice_set("volunteer1_X", "02", "rec_800nm_sagittal_+00.npy"))
    PAMMOTHDataset(root, download=False)

    conv = root / "converted"
    manifest = _manifest(root)
    ref = np.load(conv / manifest["images"][0]["references"]["ground_truth"]["path"])

    # FWI fill value was 10.0 → log10(10) == 1.0; clamped corner → log10(5).
    assert np.isclose(ref[1, 0, 0], 1.0)
    assert np.isclose(ref[0, 0, 0], np.log10(LOG_FLOOR))
    assert ref.min() >= np.log10(LOG_FLOOR)


def test_slice_without_fwi_is_skipped(tmp_path: Path) -> None:
    root = tmp_path / "pammoth"
    slices = {
        f"volunteer1_X/02/PA/{m}/rec_800nm_sagittal_+00.npy": np.full((8, 10), 10.0, np.float32)
        for m in VARIANT_METHODS  # no FWI reference present
    }
    _write_raw_zip(root, slices)
    PAMMOTHDataset(root, download=False)
    assert _manifest(root)["images"] == []


def test_download_false_without_data_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PAMMOTHDataset(tmp_path / "pammoth", download=False)
