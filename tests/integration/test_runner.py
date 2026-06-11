"""Integration test: full evaluation pipeline on synthetic data."""

from __future__ import annotations

from pathlib import Path

from ipasc_iqa.core.runner import run_evaluation
from ipasc_iqa.datasets.local import LocalDataset
from ipasc_iqa.measures.classical.psnr import PSNRMeasure
from ipasc_iqa.measures.classical.ssim import SSIMMeasure
from ipasc_iqa.measures.classical.cnr import CNRMeasure
from ipasc_iqa.measures.classical.snr import SNRMeasure


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
    from ipasc_iqa.io.results import load_results, save_results

    ds = LocalDataset(synthetic_dataset)
    out = tmp_path / "run.json"
    run = run_evaluation([ds], [PSNRMeasure()], output_path=out, progress=False)

    loaded = load_results(out)
    assert loaded.run_id == run.run_id
    assert len(loaded.results) == len(run.results)
