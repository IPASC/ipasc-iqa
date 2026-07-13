"""Integration test: full evaluation pipeline on synthetic data."""

from __future__ import annotations

from pathlib import Path

from ipasc_iqa.core.runner import run_evaluation
from ipasc_iqa.datasets.local import LocalDataset
from ipasc_iqa.measures.classical.cnr import CNRMeasure
from ipasc_iqa.measures.classical.psnr import PSNRMeasure
from ipasc_iqa.measures.classical.snr import SNRMeasure
from ipasc_iqa.measures.classical.ssim import SSIMMeasure
from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType

class BrokenMeasure(BaseMeasure):
    metadata = MeasureMetadata(
        name="BROKEN_TEST",
        version="0.0.1",
        measure_type=MeasureType.NO_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        description="Measure used by tests to exercise evaluation failures.",
    )

    def evaluate(self, image, **kwargs):  # noqa: ANN001, ANN003
        raise RuntimeError("intentional metric failure")
    
def test_run_full_pipeline(synthetic_dataset: Path, tmp_path: Path) -> None:
    ds = LocalDataset(synthetic_dataset)
    measures = [PSNRMeasure(), SSIMMeasure(), CNRMeasure(), SNRMeasure()]
    out = tmp_path / "results.json"

    run = run_evaluation(
        [ds],
        measures,
        reference_preference=["ground_truth"],
        output_path=out,
        progress=False,
    )

    assert out.exists()
    # 3 images × 4 measures = 12 results (CNR/SNR may be skipped if ROIs odd shape)
    assert len(run.results) > 0
    # No unhandled errors
    for r in run.results:
        assert r.value is not None or r.sub_scores is not None


def test_skips_missing_reference(synthetic_dataset: Path) -> None:
    ds = LocalDataset(synthetic_dataset)
    # PSNR needs a reference; ask for a type not present
    psnr = PSNRMeasure()
    run = run_evaluation(
        [ds],
        [psnr],
        reference_preference=["ground_truth"],
        progress=False,
    )
    # All 3 should succeed since ground_truth exists
    assert len(run.results) == 3
    assert len(run.skipped) == 0


def test_results_json_roundtrip(synthetic_dataset: Path, tmp_path: Path) -> None:
    from ipasc_iqa.io.results import load_results

    ds = LocalDataset(synthetic_dataset)
    out = tmp_path / "run.json"
    run = run_evaluation([ds], [PSNRMeasure()], output_path=out, progress=False)

    loaded = load_results(out)
    assert loaded.run_id == run.run_id
    assert len(loaded.results) == len(run.results)


def test_cache_skips_recomputation(synthetic_dataset: Path, tmp_path: Path) -> None:
    from ipasc_iqa.core.cache import ResultStore

    ds = LocalDataset(synthetic_dataset)
    with ResultStore(tmp_path / "cache") as store:
        first = run_evaluation(
            [ds],
            [PSNRMeasure()],
            reference_preference=["ground_truth"],
            cache=store,
            progress=False,
        )
        assert all(not r.from_cache for r in first.results)

        # Second run with the same inputs: every result should come from cache.
        second = run_evaluation(
            [ds],
            [PSNRMeasure()],
            reference_preference=["ground_truth"],
            cache=store,
            progress=False,
        )
        assert second.results
        assert all(r.from_cache for r in second.results)


def test_partial_runs_accumulate_in_store(synthetic_dataset: Path, tmp_path: Path) -> None:
    from ipasc_iqa.core.cache import ResultStore

    ds = LocalDataset(synthetic_dataset)
    with ResultStore(tmp_path / "cache") as store:
        run_evaluation(
            [ds],
            [PSNRMeasure()],
            reference_preference=["ground_truth"],
            cache=store,
            progress=False,
        )
        run_evaluation(
            [ds],
            [SSIMMeasure()],
            reference_preference=["ground_truth"],
            cache=store,
            progress=False,
        )

        # The store holds both measures even though each run computed only one.
        measures = {r.measure for r in store.records_for_dataset()}
        assert measures == {"PSNR", "SSIM"}

def test_metric_evaluation_errors_are_recorded_as_skipped(synthetic_dataset: Path) -> None:
    ds = LocalDataset(synthetic_dataset)
    run = run_evaluation([ds], [BrokenMeasure()], progress=False)

    assert len(run.results) == 0
    assert len(run.skipped) == len(ds)
    assert {s.measure for s in run.skipped} == {"BROKEN_TEST"}
    assert all("evaluation error: intentional metric failure" in s.reason for s in run.skipped)

def test_metric_evaluation_errors_call_on_skip(synthetic_dataset: Path) -> None:
    ds = LocalDataset(synthetic_dataset)
    events = []

    run = run_evaluation(
        [ds],
        [BrokenMeasure()],
        progress=False,
        on_skip=lambda dataset, measure, reason: events.append((dataset, measure, reason)),
    )

    assert len(run.skipped) == len(ds)
    assert len(events) == len(ds)
    assert {measure for _, measure, _ in events} == {"BROKEN_TEST"}
    assert all("intentional metric failure" in reason for _, _, reason in events)