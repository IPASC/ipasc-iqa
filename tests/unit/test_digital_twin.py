"""Unit tests for the Digital Twin dataset adapter (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ipasc_iqa.datasets.digital_twin import (
    RECON_ALGORITHMS,
    RECON_SOURCE,
    SPLITS,
    DigitalTwinDataset,
)


def _build_source(root: Path, *, drop: tuple[str, str, str] | None = None) -> None:
    """Materialise a tiny copy of the source layout under ``<root>/data/``.

    ``drop`` optionally omits one (split, stem, algo) reconstruction. Both the
    calibration and testing splits are built (calibration is used to fit the
    per-algorithm calibration factors that are applied to the testing split).
    """
    rng = np.random.default_rng(0)
    data = root / "data"
    for split in SPLITS:
        p0 = data / split / "p0"
        p0.mkdir(parents=True)
        for stem in range(2):
            np.save(p0 / f"{stem}.npy", rng.random((8, 8)).astype(np.float32))
            for algo in RECON_ALGORITHMS:
                if drop == (split, str(stem), algo):
                    continue
                d = data / split / "recons" / algo / RECON_SOURCE
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / f"{stem}.npy", rng.random((8, 8)).astype(np.float32))


def test_converts_testing_split(tmp_path: Path) -> None:
    _build_source(tmp_path)
    DigitalTwinDataset(tmp_path, download=False)

    manifest = json.loads((tmp_path / "converted" / "manifest.json").read_text())
    ids = [e["id"] for e in manifest["images"]]

    # Only the (calibrated) testing split is emitted: 2 scenes × 6 algos = 12.
    assert len(ids) == 12
    assert len(set(ids)) == 12
    assert all(i.startswith("testing_") for i in ids)

    entry = manifest["images"][0]
    assert entry["shape"] == [8, 8, 1]  # channel axis added per the contract
    assert "ground_truth" in entry["references"]
    assert entry["metadata"]["algorithm"] in RECON_ALGORITHMS


def test_missing_reconstruction_is_skipped(tmp_path: Path) -> None:
    _build_source(tmp_path, drop=("testing", "1", "tr"))
    DigitalTwinDataset(tmp_path, download=False)
    manifest = json.loads((tmp_path / "converted" / "manifest.json").read_text())
    ids = {e["id"] for e in manifest["images"]}
    assert "testing_1_tr" not in ids
    assert "testing_1_bp" in ids  # other algos for that scene still present


def test_no_annotations(tmp_path: Path) -> None:
    _build_source(tmp_path)
    DigitalTwinDataset(tmp_path, download=False)
    manifest = json.loads((tmp_path / "converted" / "manifest.json").read_text())
    assert all("annotations" not in e["metadata"] for e in manifest["images"])


def test_download_false_without_data_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DigitalTwinDataset(tmp_path, download=False)


def test_converted_is_reused(tmp_path: Path) -> None:
    _build_source(tmp_path)
    DigitalTwinDataset(tmp_path, download=False)
    manifest_path = tmp_path / "converted" / "manifest.json"
    mtime = manifest_path.stat().st_mtime
    DigitalTwinDataset(tmp_path, download=False)  # second instantiation
    assert manifest_path.stat().st_mtime == mtime  # not re-converted


def test_bt_only_dataset_is_analysable(tmp_path: Path) -> None:
    """A dataset with no MOS becomes analysable once pairwise data is contributed.

    Regression: previously the analysis bailed with 'no expert annotations'
    before ever consulting contributed Bradley-Terry comparisons.
    """
    from ipasc_iqa.contrib.store import ROLE_ANNOTATOR, AnnotatorStore
    from ipasc_iqa.datasets.registry import registry
    from ipasc_iqa.gui import service

    data_dir = tmp_path / "dd"
    _build_source(data_dir / "digital_twin")
    if "digital_twin" not in registry._classes:
        registry.register(DigitalTwinDataset, distribution="t")

    ws = service.Workspace.resolve(tmp_path / "ws", data_dir)
    ws.ensure()
    service.run_benchmark(ws, ["digital_twin"], ["PSNR", "SSIM"])

    # No annotations of any kind yet → nothing to analyse.
    before = service.analysis_for_dataset(ws, "digital_twin")
    assert before["error"] is not None

    # Contribute pairwise comparisons (task=None) for the emitted testing scenes.
    with AnnotatorStore(ws.annotations_path) as store:
        user = store.create_user(
            email="u@x.com", name="U", role=ROLE_ANNOTATOR, password_hash=None, approved=True
        )
        for stem in range(2):
            scene = f"testing_{stem}"
            algos = [f"{scene}_{a}" for a in RECON_ALGORITHMS]
            ref = f"__ref__{scene}"
            comps = [{"winner": ref, "loser": a} for a in algos]
            comps += [
                {"winner": algos[i], "loser": algos[j]}
                for i in range(len(algos))
                for j in range(i + 1, len(algos))
            ]
            store.add_session(
                {
                    "user_uid": user["uid"],
                    "dataset": "digital_twin",
                    "task": None,
                    "reference": scene,
                    "order": [ref, *algos],
                    "comparisons": comps,
                    "n_comparisons": len(comps),
                }
            )

    after = service.analysis_for_dataset(ws, "digital_twin")
    assert after["error"] is None
    assert [s["key"] for s in after["sources"]] == ["bt"]
    assert after["active_task"] is None and after["all_tasks"] == []
    assert any(r["srcc"] is not None for r in after["tasks"][0]["rows"])
