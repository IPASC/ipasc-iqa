"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a tiny synthetic dataset following the data contract."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    rng = np.random.default_rng(42)

    for i in range(3):
        stem = f"img_{i:04d}"
        img = rng.random((32, 32), dtype=np.float32)
        ref = rng.random((32, 32), dtype=np.float32)
        signal_roi = (img > 0.7).astype(np.float32)
        noise_roi = (img < 0.2).astype(np.float32)

        np.save(images_dir / f"{stem}.npy", img)
        np.save(images_dir / f"{stem}.ref.ground_truth.npy", ref)
        np.save(images_dir / f"{stem}.signal_roi.npy", signal_roi)
        np.save(images_dir / f"{stem}.noise_roi.npy", noise_roi)

    # Minimal manifest
    from ipasc_iqa.core.roi import hash_file

    entries = []
    for i in range(3):
        stem = f"img_{i:04d}"
        img_path = images_dir / f"{stem}.npy"
        ref_path = images_dir / f"{stem}.ref.ground_truth.npy"
        sig_path = images_dir / f"{stem}.signal_roi.npy"
        noise_path = images_dir / f"{stem}.noise_roi.npy"
        arr = np.load(img_path)
        entries.append(
            {
                "id": stem,
                "path": f"images/{stem}.npy",
                "sha256": hash_file(img_path),
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "signal_roi": {
                    "path": f"images/{stem}.signal_roi.npy",
                    "sha256": hash_file(sig_path),
                },
                "noise_roi": {
                    "path": f"images/{stem}.noise_roi.npy",
                    "sha256": hash_file(noise_path),
                },
                "references": {
                    "ground_truth": {
                        "path": f"images/{stem}.ref.ground_truth.npy",
                        "sha256": hash_file(ref_path),
                    }
                },
            }
        )

    manifest = {
        "schema_version": "1.0.0",
        "dataset_name": "synthetic_test",
        "dataset_version": "0.0.1",
        "images": entries,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return tmp_path
