"""Integration tests for the FastAPI GUI (via TestClient, no live server).

Datasets are adapter-driven, so the tests register a tiny adapter whose data is
pre-prepared on disk (download is a no-op).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ipasc_iqa.core.roi import hash_file  # noqa: E402
from ipasc_iqa.datasets.local import LocalDataset  # noqa: E402
from ipasc_iqa.datasets.registry import registry as dataset_registry  # noqa: E402
from ipasc_iqa.gui.app import create_app  # noqa: E402


class MiniDataset(LocalDataset):
    """A pre-prepared adapter for tests (no download)."""

    DATASET_NAME = "mini"
    DATASET_VERSION = "0.1.0"
    DATASET_DESCRIPTION = "Tiny synthetic test dataset."


def _write_mini_dataset(root: Path) -> None:
    """Materialise the 'mini' dataset under <root>/mini/ following the contract."""
    ds_root = root / "mini"
    images = ds_root / "images"
    images.mkdir(parents=True)

    rng = np.random.default_rng(7)
    entries = []
    for stem in range(3):
        ref = rng.random((16, 16), dtype=np.float32)
        ref_name = f"image{stem}.ref.pseudo_reference.npy"
        np.save(images / ref_name, ref)
        ref_sha = hash_file(images / ref_name)
        for alg in range(2):
            img = ref + rng.normal(0, 0.1 * (alg + 1), ref.shape).astype(np.float32)
            name = f"image{stem}.var.alg{alg}.npy"
            np.save(images / name, img)
            entries.append(
                {
                    "id": f"image{stem}_alg{alg}",
                    "path": f"images/{name}",
                    "sha256": hash_file(images / name),
                    "shape": [16, 16],
                    "dtype": "float32",
                    "references": {
                        "pseudo_reference": {"path": f"images/{ref_name}", "sha256": ref_sha}
                    },
                    "metadata": {
                        "algorithm": f"alg{alg}",
                        "annotations": {
                            "overall_quality": {"zscore_mos": float(stem) - alg},
                            "edge_visibility": {"zscore_mos": float(stem) + alg},
                        },
                    },
                }
            )
    manifest = {
        "schema_version": "1.0.0",
        "dataset_name": "mini",
        "dataset_version": "0.1.0",
        "images": entries,
    }
    (ds_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_mini_dataset(data_dir)
    if "mini" not in dataset_registry._classes:
        dataset_registry.register(MiniDataset, distribution="tests")
    workspace_root = tmp_path / "ws"
    return TestClient(create_app(workspace_root=workspace_root, data_dir=data_dir))


def test_pages_render(client: TestClient) -> None:
    for path in ("/", "/run", "/results", "/standings"):
        resp = client.get(path)
        assert resp.status_code == 200
        assert "IPASC" in resp.text or "IQA benchmarking" in resp.text


def test_api_datasets_adapter_driven(client: TestClient) -> None:
    ds = {d["name"]: d for d in client.get("/api/datasets").json()}
    assert "mini" in ds
    assert ds["mini"]["prepared"] is True
    assert ds["mini"]["has_annotations"] is True
    # The built-in phot_iqa adapter is always offered, even when not downloaded.
    assert "phot_iqa" in ds
    assert ds["phot_iqa"]["prepared"] is False


def test_measures_listed(client: TestClient) -> None:
    names = {m["name"] for m in client.get("/api/measures").json()}
    assert {"psnr", "ssim", "cnr", "snr"} <= names


def test_run_then_analysis_all_tasks(client: TestClient) -> None:
    resp = client.post("/run", data={"datasets": ["mini"], "measures": ["psnr", "ssim"]})
    assert resp.status_code == 200
    assert resp.headers.get("HX-Redirect") == "/results"

    frag = client.get("/api/analysis", params={"dataset": "mini"})
    assert frag.status_code == 200
    # Both tasks present (not selectable — all shown).
    assert "overall_quality" in frag.text
    assert "edge_visibility" in frag.text
    assert "SRCC" in frag.text and "psnr" in frag.text


def test_run_requires_selection(client: TestClient) -> None:
    resp = client.post("/run", data={"datasets": [], "measures": []})
    assert resp.status_code == 400
    assert "at least one" in resp.text.lower()


def test_standings_ranks_measures(client: TestClient) -> None:
    client.post("/run", data={"datasets": ["mini"], "measures": ["psnr", "ssim"]})
    frag = client.get("/api/standings")
    assert frag.status_code == 200
    # Measures (not algorithms) are ranked; both tasks appear.
    assert "psnr" in frag.text or "ssim" in frag.text
    assert "overall_quality" in frag.text
