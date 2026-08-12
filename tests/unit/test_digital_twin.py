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

    # No annotations of any kind yet → the metric overview still renders, but
    # there is no human-score analysis and the page says so (rather than erroring).
    before = service.analysis_for_dataset(ws, "digital_twin")
    assert before["error"] is None
    assert before["human_analysis_message"]
    assert before["tasks"] == [] and before["sources"] == []
    assert before["metric_summary"]

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


def test_no_repeat_annotations_and_done_message(tmp_path: Path) -> None:
    """A rater is never re-offered an item they've done; exhaustion reports done."""
    from ipasc_iqa.contrib.store import ROLE_ANNOTATOR, AnnotatorStore
    from ipasc_iqa.datasets.registry import registry
    from ipasc_iqa.gui import service

    data_dir = tmp_path / "dd"
    _build_source(data_dir / "digital_twin")
    if "digital_twin" not in registry._classes:
        registry.register(DigitalTwinDataset, distribution="t")
    ws = service.Workspace.resolve(tmp_path / "ws", data_dir)
    ws.ensure()
    service.run_benchmark(ws, ["digital_twin"], ["PSNR"])  # prepares the dataset
    with AnnotatorStore(ws.annotations_path) as store:
        uid = store.create_user(
            email="u@x.com", name="U", role=ROLE_ANNOTATOR, password_hash=None, approved=True
        )["uid"]

    # --- Comparative ranking: each reference set is offered at most once. ---
    sessions: dict = {}
    ranked: set = set()
    while True:
        res = service.start_ranking(
            ws, sessions, user_uid=uid, dataset_choice="random", task_choice="random"
        )
        if res.get("done"):
            assert res["scope"] == "all"
            break
        sess = sessions[res["session_id"]]["session"]
        key = (sess.dataset, sess.reference, sess.task)
        assert key not in ranked  # never re-offered
        ranked.add(key)
        with AnnotatorStore(ws.annotations_path) as store:
            store.add_session(sess.result())
    assert len(ranked) == 2  # the two testing scenes, each once

    # --- Quantitative rating: each variant is offered at most once. ---
    rated: set = set()
    while True:
        d = service.draw_rating_item(ws, "digital_twin", user_uid=uid)
        if d.get("done"):
            assert d["scope"] == "all"
            break
        key = (d["image_id"], d["task"])
        assert key not in rated  # never re-offered
        rated.add(key)
        service.record_rating(
            ws, user_uid=uid, dataset=d["dataset"], image_id=d["image_id"],
            task=d["task"], score=2,
        )
    assert len(rated) == 2 * len(RECON_ALGORITHMS)  # 2 scenes × all algorithm variants


def test_draws_advertise_the_images_to_prefetch(tmp_path: Path) -> None:
    """Each draw names what is coming next, so the page can warm the image cache."""
    from ipasc_iqa.contrib.store import ROLE_ANNOTATOR, AnnotatorStore
    from ipasc_iqa.datasets.registry import registry
    from ipasc_iqa.gui import service

    data_dir = tmp_path / "dd"
    _build_source(data_dir / "digital_twin")
    if "digital_twin" not in registry._classes:
        registry.register(DigitalTwinDataset, distribution="t")
    ws = service.Workspace.resolve(tmp_path / "ws", data_dir)
    ws.ensure()
    service.run_benchmark(ws, ["digital_twin"], ["PSNR"])  # prepares the dataset
    with AnnotatorStore(ws.annotations_path) as store:
        uid = store.create_user(
            email="pf@x.com", name="PF", role=ROLE_ANNOTATOR, password_hash=None, approved=True
        )["uid"]

    # Ratings: the look-ahead queue reports the variants queued behind this one.
    draw = service.draw_rating_item(ws, "digital_twin", user_uid=uid)
    assert draw["upcoming"], draw
    assert len(draw["upcoming"]) <= service.RATING_LOOKAHEAD - 1
    for entry in draw["upcoming"]:
        variant, reference = entry["items"]
        assert entry["dataset"] == "digital_twin"
        assert reference.startswith("__ref__")
        assert variant != draw["image_id"]

    # Rankings: the whole reference set is known up front, so everything not on
    # screen right now is offered for prefetching.
    sessions: dict = {}
    res = service.start_ranking(
        ws, sessions, user_uid=uid, dataset_choice="digital_twin", task_choice="random"
    )
    ctx = service.compare_view(sessions, res["session_id"])
    session = sessions[res["session_id"]]["session"]
    assert set(ctx["prefetch_items"]) == set(session.items) - {ctx["left_item"], ctx["right_item"]}
