"""Unit tests for LocalDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ipasc_iqa.core.dataset import ImageRecord
from ipasc_iqa.datasets.local import LocalDataset


def test_dataset_loads_records(synthetic_dataset: Path) -> None:
    ds = LocalDataset(synthetic_dataset)
    assert len(ds) == 3
    records = list(ds)
    assert len(records) == 3


def test_image_record_fields(synthetic_dataset: Path) -> None:
    ds = LocalDataset(synthetic_dataset)
    record: ImageRecord = next(iter(ds))
    assert isinstance(record.image, np.ndarray)
    assert record.image.ndim == 3
    assert "ground_truth" in record.references
    assert record.signal_roi is not None
    assert record.noise_roi is not None


def test_hashes_populated(synthetic_dataset: Path) -> None:
    ds = LocalDataset(synthetic_dataset)
    record = next(iter(ds))
    assert len(record.hashes.image) == 64  # sha256 hex
    assert "ground_truth" in record.hashes.references


def test_verify_passes(synthetic_dataset: Path) -> None:
    ds = LocalDataset(synthetic_dataset)
    report = ds.verify()
    assert report.ok
