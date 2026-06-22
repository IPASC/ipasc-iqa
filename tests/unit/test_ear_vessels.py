"""Unit tests for the ear-vessel PAM dataset adapter (no network)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from ipasc_iqa.datasets.ear_vessels import (  # noqa: E402
    NOISE_PERCENTAGES,
    EarVesselsDataset,
)


def _write_raw_zip(root: Path, images: dict[str, np.ndarray]) -> None:
    """Create <root>/ear_vessel_dataset.zip with PNGs under ``<split>/image/``.

    Keys of ``images`` are archive-relative paths, e.g. ``train/image/a.png``.
    """
    root.mkdir(parents=True, exist_ok=True)
    staging = root / "_png_src"
    for name, arr in images.items():
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr.astype(np.uint8)).save(path)
    with zipfile.ZipFile(root / "ear_vessel_dataset.zip", "w") as z:
        for png in staging.rglob("*.png"):
            z.write(png, png.relative_to(staging).as_posix())


def _manifest(ds_root: Path) -> dict:
    return json.loads((ds_root / "converted" / "manifest.json").read_text())


def _vessel_mask() -> np.ndarray:
    """A non-degenerate binary vessel mask (some foreground, some background)."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[64:192, 64:192] = 255
    return mask


def test_references_variants_and_rois(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    images = {
        "train/image/a.png": rng.integers(0, 256, size=(256, 256), dtype=np.uint8),
        "train/groundtruth/a.png": _vessel_mask(),
        "test/image/b.png": rng.integers(0, 256, size=(256, 256), dtype=np.uint8),
        "test/groundtruth/b.png": _vessel_mask(),
    }
    root = tmp_path / "ear_vessels"
    _write_raw_zip(root, images)

    EarVesselsDataset(root, download=False)
    entries = _manifest(root)["images"]

    n_images = 2  # source images under <split>/image/ (masks are not counted)
    stems = {e["metadata"]["stem"] for e in entries}
    assert len(stems) == n_images
    assert len(entries) == n_images * len(NOISE_PERCENTAGES)
    assert len({e["id"] for e in entries}) == len(entries)  # ids unique

    for e in entries:
        assert e["shape"] == [256, 256, 1]
        assert e["dtype"] == "float32"
        assert "ground_truth" in e["references"]
        # Annotated masks supplied → ROI masks present.
        assert "signal_roi" in e and "noise_roi" in e
        assert e["metadata"]["split"] in {"train", "test"}

    # Each stem carries exactly the configured noise percentages.
    by_stem: dict[str, set[int]] = {}
    for e in entries:
        by_stem.setdefault(e["metadata"]["stem"], set()).add(e["metadata"]["noise_pct"])
    assert all(pcts == set(NOISE_PERCENTAGES) for pcts in by_stem.values())

    # The signal ROI is the annotated mask (non-zero pixels), not a threshold.
    conv = root / "converted"
    signal = np.load(conv / entries[0]["signal_roi"]["path"])
    noise = np.load(conv / entries[0]["noise_roi"]["path"])
    expected = (_vessel_mask() > 0)[:, :, np.newaxis]
    assert np.array_equal(signal, expected)
    assert np.array_equal(noise, ~expected)


def test_values_normalised_and_noise_increases_error(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    root = tmp_path / "ear_vessels"
    _write_raw_zip(
        root, {"train/image/x.png": rng.integers(0, 256, size=(256, 256), dtype=np.uint8)}
    )
    EarVesselsDataset(root, download=False)
    conv = root / "converted"
    manifest = _manifest(root)

    ref = np.load(conv / manifest["images"][0]["references"]["ground_truth"]["path"])
    assert ref.min() >= 0.0 and ref.max() <= 1.0

    # Higher noise percentage → larger mean absolute deviation from the reference.
    errs = []
    for pct in NOISE_PERCENTAGES:
        e = next(x for x in manifest["images"] if x["metadata"]["noise_pct"] == pct)
        var = np.load(conv / e["path"])
        assert var.shape == ref.shape == (256, 256, 1)
        assert var.min() >= 0.0 and var.max() <= 1.0
        errs.append(float(np.abs(var - ref).mean()))
    assert errs == sorted(errs)


def test_missing_mask_gets_no_roi(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    root = tmp_path / "ear_vessels"
    # Source image present, but no matching groundtruth segmentation.
    _write_raw_zip(
        root, {"train/image/x.png": rng.integers(0, 256, size=(256, 256), dtype=np.uint8)}
    )
    EarVesselsDataset(root, download=False)
    entries = _manifest(root)["images"]
    assert entries  # variants still produced
    assert all("signal_roi" not in e and "noise_roi" not in e for e in entries)


def test_download_false_without_data_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EarVesselsDataset(tmp_path / "ear_vessels", download=False)
