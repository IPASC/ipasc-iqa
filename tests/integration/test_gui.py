"""Integration tests for the FastAPI GUI (via TestClient, no live server)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ipasc_iqa.core.roi import hash_file  # noqa: E402
from ipasc_iqa.gui.app import create_app  # noqa: E402


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """A workspace containing one annotated dataset following the data contract."""
    ds_root = tmp_path / "mini"
    images = ds_root / "images"
    images.mkdir(parents=True)

    rng = np.random.default_rng(7)
    entries = []
    # Two "algorithms" x three stems, each with a shared reference + annotation.
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
                        "pseudo_reference": {
                            "path": f"images/{ref_name}",
                            "sha256": ref_sha,
                        }
                    },
                    "metadata": {
                        "algorithm": f"alg{alg}",
                        # alg0 is "better" -> higher score
                        "annotations": {
                            "overall_quality": {"zscore_mos": float(stem) - alg}
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
    return tmp_path


@pytest.fixture()
def client(workspace: Path) -> TestClient:
    return TestClient(create_app(workspace_root=workspace))


def test_pages_render(client: TestClient) -> None:
    for path in ("/", "/run", "/results", "/standings"):
        resp = client.get(path)
        assert resp.status_code == 200
        assert "IQA benchmarking" in resp.text or "IPASC" in resp.text


def test_api_datasets_and_measures(client: TestClient) -> None:
    ds = client.get("/api/datasets").json()
    assert any(d["name"] == "mini" for d in ds)
    assert ds[0]["has_annotations"] is True

    measures = client.get("/api/measures").json()
    names = {m["name"] for m in measures}
    assert {"psnr", "ssim"} <= names


def test_run_then_correlate(client: TestClient) -> None:
    # Run a benchmark via the POST form endpoint.
    resp = client.post(
        "/run",
        data={"datasets": ["mini"], "measures": ["psnr", "ssim"]},
    )
    assert resp.status_code == 200
    assert resp.headers.get("HX-Redirect") == "/results"

    # Correlation fragment should now have data.
    frag = client.get("/api/correlation", params={"dataset": "mini", "task": "overall_quality"})
    assert frag.status_code == 200
    assert "SRCC" in frag.text
    assert "psnr" in frag.text


def test_run_requires_selection(client: TestClient) -> None:
    resp = client.post("/run", data={"datasets": [], "measures": []})
    assert resp.status_code == 400
    assert "at least one" in resp.text.lower()


def test_standings_fragment(client: TestClient) -> None:
    frag = client.get("/api/standings", params={"task": "overall_quality"})
    assert frag.status_code == 200
    # alg0 should outrank alg1 (higher z-MOS).
    assert "alg0" in frag.text
