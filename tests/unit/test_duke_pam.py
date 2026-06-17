"""Unit tests for the Duke PAM dataset adapter (no network)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from ipasc_iqa.datasets.duke_pam import (  # noqa: E402
    MAX_PATCHES_PER_IMAGE,
    PATCH,
    SUBSAMPLE_FACTORS,
    DukePAMDataset,
)


def _write_raw_zip(root: Path, images: dict[str, np.ndarray]) -> None:
    """Create <root>/raw_train.zip containing 16-bit PNGs from ``images``."""
    root.mkdir(parents=True, exist_ok=True)
    staging = root / "_png_src"
    staging.mkdir()
    for name, arr in images.items():
        Image.fromarray(arr.astype(np.uint16)).save(staging / name)
    with zipfile.ZipFile(root / "raw_train.zip", "w") as z:
        for png in staging.glob("*.png"):
            z.write(png, png.name)


def _manifest(ds_root: Path) -> dict:
    return json.loads((ds_root / "converted" / "manifest.json").read_text())


def test_tiling_variants_references_and_rois(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    images = {
        # A 1xN grid plus a 1xN grid with non-divisible remainder (dropped).
        "a.png": rng.integers(0, 65536, size=(PATCH, PATCH * 2), dtype=np.uint16),
        "b.png": rng.integers(0, 65536, size=(PATCH + 40, PATCH * 2 + 40), dtype=np.uint16),
    }
    root = tmp_path / "duke_pam"
    _write_raw_zip(root, images)

    DukePAMDataset(root, download=False)
    manifest = _manifest(root)
    entries = manifest["images"]

    # Expected patch count derives from the adapter's own tiling/cap logic, so the
    # test stays valid if PATCH / MAX_PATCHES_PER_IMAGE are retuned.
    expected_patches = 0
    for arr in images.values():
        h, w = arr.shape
        n_rows, n_cols = h // PATCH, w // PATCH
        if n_rows and n_cols:
            expected_patches += len(DukePAMDataset._patch_positions(n_rows, n_cols))

    stems = {e["metadata"]["stem"] for e in entries}
    assert len(stems) == expected_patches
    assert len(entries) == expected_patches * len(SUBSAMPLE_FACTORS)
    assert len({e["id"] for e in entries}) == len(entries)  # ids unique

    for e in entries:
        assert e["shape"] == [PATCH, PATCH, 1]
        assert e["dtype"] == "float32"
        assert "ground_truth" in e["references"]
        # Random patches are non-degenerate → ROI masks present and usable.
        assert "signal_roi" in e and "noise_roi" in e

    # Each stem has exactly the four factors.
    by_stem: dict[str, set[int]] = {}
    for e in entries:
        by_stem.setdefault(e["metadata"]["stem"], set()).add(e["metadata"]["subsample_factor"])
    assert all(factors == set(SUBSAMPLE_FACTORS) for factors in by_stem.values())


def test_values_normalised_and_nn_upscaling_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    root = tmp_path / "duke_pam"
    _write_raw_zip(root, {"x.png": rng.integers(0, 65536, size=(PATCH, PATCH), dtype=np.uint16)})
    DukePAMDataset(root, download=False)
    conv = root / "converted"
    manifest = _manifest(root)

    e4 = next(e for e in manifest["images"] if e["metadata"]["subsample_factor"] == 4)
    ref = np.load(conv / e4["references"]["ground_truth"]["path"])
    var = np.load(conv / e4["path"])

    assert ref.min() >= 0.0 and ref.max() <= 1.0
    assert var.shape == ref.shape == (PATCH, PATCH, 1)
    # 4x = keep every 4th pixel on a grid, then tile each into a 4x4 block.
    expected = np.repeat(np.repeat(ref[::4, ::4, :], 4, axis=0), 4, axis=1)
    assert np.array_equal(var, expected)


def test_flat_patch_gets_no_roi(tmp_path: Path) -> None:
    root = tmp_path / "duke_pam"
    _write_raw_zip(root, {"flat.png": np.full((PATCH, PATCH), 7, dtype=np.uint16)})
    DukePAMDataset(root, download=False)
    entries = _manifest(root)["images"]
    assert entries  # variants still produced
    assert all("signal_roi" not in e and "noise_roi" not in e for e in entries)


def test_patches_capped_per_image(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    side = PATCH * 5  # 5x5 = 25 candidate patches, above the cap
    root = tmp_path / "duke_pam"
    _write_raw_zip(root, {"big.png": rng.integers(0, 65536, size=(side, side), dtype=np.uint16)})
    DukePAMDataset(root, download=False)
    stems = {e["metadata"]["stem"] for e in _manifest(root)["images"]}
    assert len(stems) == MAX_PATCHES_PER_IMAGE


def test_download_false_without_data_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DukePAMDataset(tmp_path / "duke_pam", download=False)
