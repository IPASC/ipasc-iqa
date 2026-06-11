"""Unit tests for the result cache / merged store."""

from __future__ import annotations

from pathlib import Path

from ipasc_iqa.core.cache import ResultStore, make_cache_key
from ipasc_iqa.core.result import ResultRecord


def _record(dataset="d", image="img", measure="psnr", value=1.0, version="1.0.0") -> ResultRecord:
    return ResultRecord(
        dataset=dataset,
        image_id=image,
        image_sha256="a" * 64,
        measure=measure,
        measure_version=version,
        reference_type="pseudo_reference",
        reference_sha256="b" * 64,
        signal_roi_sha256=None,
        noise_roi_sha256=None,
        value=value,
        sub_scores=None,
        wall_time_s=0.0,
        device="cpu",
    )


class TestCacheKey:
    def test_stable_for_same_inputs(self) -> None:
        kw = dict(
            measure_name="psnr",
            measure_version="1.0.0",
            image_sha256="x",
            reference_type="ground_truth",
            reference_sha256="y",
            signal_roi_sha256=None,
            noise_roi_sha256=None,
        )
        assert make_cache_key(**kw) == make_cache_key(**kw)

    def test_changes_with_version(self) -> None:
        base = dict(
            measure_name="psnr",
            image_sha256="x",
            reference_type=None,
            reference_sha256=None,
            signal_roi_sha256=None,
            noise_roi_sha256=None,
        )
        assert make_cache_key(measure_version="1.0.0", **base) != make_cache_key(
            measure_version="2.0.0", **base
        )


class TestResultStore:
    def test_get_set_roundtrip(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "store.json")
        rec = _record(value=42.0)
        store.set("k1", rec)
        got = store.get("k1")
        assert got is not None and got.value == 42.0
        assert store.get("missing") is None

    def test_newest_wins_per_dataset_image_measure(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "store.json")
        # Same (dataset, image, measure) recomputed with a new version -> new key.
        store.set("old", _record(value=1.0, version="1.0.0"))
        store.set("new", _record(value=2.0, version="2.0.0"))
        recs = store.records_for_dataset("d")
        assert len(recs) == 1  # the old one was superseded
        assert recs[0].value == 2.0
        assert store.get("old") is None

    def test_partial_runs_accumulate(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "store.json")
        store.set("kp", _record(measure="psnr", value=1.0))
        store.set("ks", _record(measure="ssim", value=0.5))
        measures = {r.measure for r in store.records_for_dataset("d")}
        assert measures == {"psnr", "ssim"}

    def test_persistence_round_trips_through_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "store.json"
        s = ResultStore(path)
        s.set("k", _record(value=7.0))
        s.save()
        # A fresh instance loads what was saved.
        reloaded = ResultStore(path)
        got = reloaded.get("k")
        assert got is not None and got.value == 7.0
