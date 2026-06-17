"""The PhotIQA adapter registers references + derived ROI masks in the manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from ipasc_iqa.datasets.phot_iqa import PhotIQADataset  # noqa: E402


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


@pytest.fixture()
def fake_source(tmp_path: Path) -> Path:
    """A minimal PhotIQA source layout (reference + 2 algorithm variants)."""
    root = tmp_path / "phot_iqa"
    ref_dir = root / "normalised_images" / "reference"
    alg_dir = root / "normalised_images" / "algorithms"

    # Reference: dark background (0) with a bright square object (200).
    ref = np.zeros((8, 8), dtype=np.uint8)
    ref[2:5, 2:5] = 200
    _write_png(ref_dir / "image0.png", ref)

    # Two "algorithm" reconstructions.
    _write_png(alg_dir / "image0_0.png", ref)
    _write_png(alg_dir / "image0_1.png", np.clip(ref + 10, 0, 255))
    return root


def test_manifest_registers_rois(fake_source: Path) -> None:
    PhotIQADataset(fake_source, download=False, force_convert=True)
    manifest = json.loads((fake_source / "converted" / "manifest.json").read_text(encoding="utf-8"))

    assert len(manifest["images"]) == 2  # two algorithm variants
    for entry in manifest["images"]:
        # ROI masks must be registered as {path, sha256}, like references.
        assert "signal_roi" in entry and entry["signal_roi"]["path"].endswith(".signal_roi.npy")
        assert "noise_roi" in entry and entry["noise_roi"]["path"].endswith(".noise_roi.npy")
        assert "references" in entry


def test_records_expose_complementary_masks(fake_source: Path) -> None:
    ds = PhotIQADataset(fake_source, download=False, force_convert=True)
    records = list(ds)
    assert records

    for rec in records:
        assert rec.signal_roi is not None
        assert rec.noise_roi is not None
        assert rec.signal_roi.dtype == bool
        # Background and signal partition the image (complementary, non-empty).
        assert np.array_equal(rec.signal_roi, ~rec.noise_roi)
        assert rec.signal_roi.any() and rec.noise_roi.any()


def test_cnr_snr_now_runnable(fake_source: Path) -> None:
    from ipasc_iqa.core.runner import run_evaluation
    from ipasc_iqa.measures.classical.cnr import CNRMeasure
    from ipasc_iqa.measures.classical.snr import SNRMeasure

    ds = PhotIQADataset(fake_source, download=False, force_convert=True)
    run = run_evaluation([ds], [CNRMeasure(), SNRMeasure()], progress=False)
    # With ROI masks present, the no-reference measures produce values (no skips).
    assert run.results
    assert not run.skipped
    assert {r.measure for r in run.results} == {"CNR", "SNR"}
