"""PAM3 adapter tests using small synthetic MATLAB volumes (no network)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from ipasc_iqa.datasets.pammoth import PAMMOTHDataset, VARIANT_METHODS


def _write_volumes(root: Path) -> np.ndarray:
    vol = np.arange(320 * 320 * 165, dtype=np.float32).reshape(320, 320, 165)
    for method in ("FWI", *VARIANT_METHODS):
        folder = root / 'reconstructions' / 'volunteer1_X' / '02' / 'PA' / method
        folder.mkdir(parents=True)
        savemat(folder / 'rec_800nm.mat', {'p0_rec': vol})
    return vol


def test_extract_and_convert(tmp_path: Path) -> None:
    volume = _write_volumes(tmp_path)
    ds = PAMMOTHDataset(tmp_path, download=False)
    manifest = json.loads((tmp_path / 'converted' / 'manifest.json').read_text())
    assert len(ds) == 3 * len(VARIANT_METHODS)
    assert len(manifest['images']) == 9
    for plane, axis in (("sagittal", 0), ("frontal", 1), ("axial", 2)):
        source_mip = tmp_path / 'pammoth_soundspeed_iqa_subset' / 'volunteer1_X' / '02' / 'PA' / 'FWI' / f'rec_800nm_{plane}.npy'
        expected = np.max(volume, axis=axis)[150:-150, 150:].T
        np.testing.assert_array_equal(np.load(source_mip), expected)
        assert sum(e['metadata']['source_file'] == f'rec_800nm_{plane}.npy' for e in manifest['images']) == 3
    # Already converted: no attempt to regenerate (even if raw files disappear).
    import shutil
    shutil.rmtree(tmp_path / 'reconstructions')
    shutil.rmtree(tmp_path / 'pammoth_soundspeed_iqa_subset')
    assert len(PAMMOTHDataset(tmp_path, download=False)) == 9


def test_missing_raw_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PAMMOTHDataset(tmp_path, download=False)
