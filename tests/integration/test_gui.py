"""Integration tests for the FastAPI GUI (via TestClient, no live server).

Datasets are adapter-driven, so the tests register a tiny adapter whose data is
pre-prepared on disk (download is a no-op).
"""

from __future__ import annotations

import json
import time
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


def _run_and_wait(
    client: TestClient,
    datasets: list[str],
    measures: list[str],
    *,
    timeout: float = 30.0,
):
    """POST /run (which launches a background job) and block until it finishes.

    Returns the final status-endpoint response, which carries the ``HX-Redirect``
    header once the run is done.
    """
    resp = client.post("/run", data={"datasets": datasets, "measures": measures})
    assert resp.status_code == 200
    jobs = client.app.state.jobs
    job_id = next(reversed(jobs))  # the most recently launched job
    deadline = time.time() + timeout
    while jobs[job_id].snapshot()["status"] == "running":
        if time.time() > deadline:
            raise AssertionError("benchmark job did not finish in time")
        time.sleep(0.02)
    return client.get(f"/api/run/{job_id}/status")


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
    assert {"PSNR", "SSIM", "CNR", "SNR"} <= names


def test_run_then_analysis_all_tasks(client: TestClient) -> None:
    # /run launches a background job; the HX-Redirect is emitted by the status
    # endpoint once the job completes.
    status = _run_and_wait(client, ["mini"], ["PSNR", "SSIM"])
    assert status.headers.get("HX-Redirect") == "/results"

    frag = client.get("/api/analysis", params={"dataset": "mini"})
    assert frag.status_code == 200
    # Both tasks present (not selectable — all shown).
    assert "overall_quality" in frag.text
    assert "edge_visibility" in frag.text
    assert "SRCC" in frag.text and "PSNR" in frag.text
    # 'mini' has references but no ROI masks -> cnr/snr are disabled, not shown.
    assert 'value="CNR"' in frag.text  # present as a disabled chip (oob)
    assert "disabled" in frag.text


def test_measure_options_reflect_capabilities(client: TestClient) -> None:
    from ipasc_iqa.gui import service

    ws = service.Workspace.resolve(
        client.app.state.workspace.root, client.app.state.workspace.data_dir
    )
    opts = {o["name"]: o for o in service.measure_options(ws, "mini")}
    # 'mini' has references (psnr/ssim usable) but no ROIs (cnr/snr unusable).
    assert opts["PSNR"]["usable"] is True
    assert opts["SSIM"]["usable"] is True
    assert opts["CNR"]["usable"] is False
    assert opts["SNR"]["usable"] is False


def test_partial_runs_merge_in_analysis(client: TestClient) -> None:
    # Run only psnr, then only ssim — analysis should show BOTH (merged store).
    _run_and_wait(client, ["mini"], ["PSNR"])
    _run_and_wait(client, ["mini"], ["SSIM"])

    frag = client.get("/api/analysis", params={"dataset": "mini"})
    assert frag.status_code == 200
    assert "PSNR" in frag.text
    assert "SSIM" in frag.text
    assert "merged results" in frag.text


def test_run_requires_selection(client: TestClient) -> None:
    resp = client.post("/run", data={"datasets": [], "measures": []})
    assert resp.status_code == 400
    assert "at least one" in resp.text.lower()


def test_standings_ranks_measures(client: TestClient) -> None:
    _run_and_wait(client, ["mini"], ["PSNR", "SSIM"])
    frag = client.get("/api/standings")
    assert frag.status_code == 200
    # Measures (not algorithms) are ranked; both tasks appear.
    assert "PSNR" in frag.text or "SSIM" in frag.text
    assert "overall_quality" in frag.text


def _seed_bt_comparisons(client: TestClient, task: str = "overall_quality") -> None:
    """Seed contributed pairwise comparisons for the 'mini' dataset (ref best)."""
    from ipasc_iqa.contrib.store import AnnotatorStore

    ws = client.app.state.workspace
    with AnnotatorStore(ws.annotations_path) as store:
        store.upsert_user("u@x.com", "U")
        for stem in range(3):
            a0, a1, ref = f"image{stem}_alg0", f"image{stem}_alg1", f"__ref__{stem}"
            comps = [
                {"winner": ref, "loser": a0},
                {"winner": ref, "loser": a1},
                {"winner": a0, "loser": a1},
            ] * 4
            store.add_session(
                {
                    "user_email": "u@x.com",
                    "dataset": "mini",
                    "task": task,
                    "reference": str(stem),
                    "order": [a1, a0, ref],
                    "comparisons": comps,
                    "n_comparisons": len(comps),
                }
            )


def test_bt_source_in_analysis(client: TestClient) -> None:
    _run_and_wait(client, ["mini"], ["PSNR", "SSIM"])
    _seed_bt_comparisons(client)

    # Both human-score sources are now offered; selecting BT yields correlations.
    frag = client.get(
        "/api/analysis",
        params={"dataset": "mini", "task": "overall_quality", "source": "bt"},
    )
    assert frag.status_code == 200
    assert "Pairwise (Bradley" in frag.text  # source dropdown shows BT option
    assert "PSNR" in frag.text and "SRCC" in frag.text

    # The scatter endpoint honours the BT source.
    scatter = client.get(
        "/api/scatter",
        params={"dataset": "mini", "task": "overall_quality", "measure": "PSNR", "source": "bt"},
    )
    assert scatter.status_code == 200
    assert "BT" in scatter.text  # BT axis label, not z-MOS


def test_bt_source_absent_without_contributions(client: TestClient) -> None:
    # With no contributed comparisons, only MOS is offered (no source dropdown).
    _run_and_wait(client, ["mini"], ["PSNR"])
    frag = client.get("/api/analysis", params={"dataset": "mini", "task": "overall_quality"})
    assert "Pairwise (Bradley" not in frag.text


def test_standings_endpoint_switch(client: TestClient) -> None:
    _run_and_wait(client, ["mini"], ["PSNR", "SSIM"])

    # The page exposes the MOS/BT endpoint dropdown.
    page = client.get("/standings")
    assert 'name="source"' in page.text and ">BT (pairwise)<" in page.text

    # MOS ranks measures against z-MOS by default.
    mos = client.get("/api/standings")
    assert "overall_quality" in mos.text and "z-MOS" in mos.text

    # BT has nothing until pairwise comparisons are contributed.
    bt_empty = client.get("/api/standings", params={"source": "bt"})
    assert "No rankings yet for the pairwise" in bt_empty.text

    _seed_bt_comparisons(client)
    bt = client.get("/api/standings", params={"source": "bt"})
    assert "overall_quality" in bt.text
    assert "Bradley" in bt.text and ("PSNR" in bt.text or "SSIM" in bt.text)


def test_download_page_and_export(client: TestClient) -> None:
    _run_and_wait(client, ["mini"], ["PSNR", "SSIM"])
    _seed_bt_comparisons(client)

    # Page renders with the dataset picker and the download form.
    page = client.get("/download")
    assert page.status_code == 200
    assert "/api/download" in page.text and ">Download<" in page.text

    # Export returns a JSON attachment with the expected sections.
    resp = client.get("/api/download", params={"dataset": "mini"})
    assert resp.status_code == 200
    assert "attachment" in resp.headers.get("content-disposition", "")
    assert resp.headers["content-type"].startswith("application/json")

    payload = resp.json()
    assert payload["anonymised"] is True
    ds = payload["datasets"][0]
    assert ds["name"] == "mini"
    assert ds["n_measure_results"] > 0
    assert "overall_quality" in ds["expert_annotations"]
    assert ds["pairwise_annotations"] and ds["bradley_terry"]
    # Annotators are pseudonymised — the seeded email never appears.
    assert ds["pairwise_annotations"][0]["annotator"].startswith("annotator_")
    assert "u@x.com" not in resp.text


def test_download_all_datasets(client: TestClient) -> None:
    _run_and_wait(client, ["mini"], ["PSNR"])
    payload = client.get("/api/download", params={"dataset": "all"}).json()
    assert "mini" in [d["name"] for d in payload["datasets"]]


def test_download_handles_non_finite_values(client: TestClient) -> None:
    # Non-finite cached values (NaN/Inf) must export as null, not crash the
    # JSON response (which disallows NaN/Inf).
    import math

    from ipasc_iqa.core.cache import ResultStore, make_cache_key
    from ipasc_iqa.core.result import ResultRecord

    ws = client.app.state.workspace
    with ResultStore(ws.store_path) as store:
        for img, val in [("image0_alg0", float("nan")), ("image1_alg0", float("inf"))]:
            key = make_cache_key(
                measure_name="PSNR",
                measure_version="1.0.0",
                image_sha256=img,
                reference_type="pseudo_reference",
                reference_sha256="r",
                signal_roi_sha256=None,
                noise_roi_sha256=None,
            )
            store.set(
                key,
                ResultRecord(
                    dataset="mini",
                    image_id=img,
                    image_sha256=img,
                    measure="PSNR",
                    measure_version="1.0.0",
                    reference_type="pseudo_reference",
                    signal_roi_sha256=None,
                    noise_roi_sha256=None,
                    value=val,
                    sub_scores=None,
                    wall_time_s=0.0,
                    device="cpu",
                ),
            )

    resp = client.get("/api/download", params={"dataset": "mini"})
    assert resp.status_code == 200
    values = [r["value"] for r in resp.json()["datasets"][0]["measure_results"]]
    assert all(v is None or math.isfinite(v) for v in values)
