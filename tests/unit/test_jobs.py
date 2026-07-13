"""Unit tests for the benchmark job progress state (per-dataset percentages)."""

from __future__ import annotations

from ipasc_iqa.gui.jobs import _JobState


def _by_dataset(snap: dict) -> dict[str, dict]:
    return {d["dataset"]: d for d in snap["progress"]}


def test_progress_starts_at_zero_per_dataset() -> None:
    job = _JobState("j")
    job.set_plan({"mini": 4, "duke": 2}, ["PSNR", "SSIM"])
    progress = _by_dataset(job.snapshot())
    assert progress["mini"]["percent"] == 0
    assert progress["duke"]["percent"] == 0
    # Every selected measure is listed (so chips render before any result lands).
    assert [m["name"] for m in progress["mini"]["measures"]] == ["PSNR", "SSIM"]


def test_progress_is_percentage_per_dataset_and_measure() -> None:
    job = _JobState("j")
    job.set_plan({"mini": 4, "duke": 2}, ["PSNR", "SSIM"])
    for _ in range(4):  # PSNR done on all 4 mini images
        job.record_progress("mini", "PSNR")

    mini = _by_dataset(job.snapshot())["mini"]
    assert mini["percent"] == 50  # 4 of 8 (4 images × 2 measures) pairs
    psnr = next(m for m in mini["measures"] if m["name"] == "PSNR")
    assert psnr["done"] == 4 and psnr["percent"] == 100
    ssim = next(m for m in mini["measures"] if m["name"] == "SSIM")
    assert ssim["percent"] == 0


def test_progress_caps_at_100() -> None:
    job = _JobState("j")
    job.set_plan({"duke": 2}, ["SSIM"])
    for _ in range(10):  # more callbacks than images (defensive)
        job.record_progress("duke", "SSIM")
    duke = _by_dataset(job.snapshot())["duke"]
    assert duke["percent"] == 100
    assert duke["measures"][0]["done"] == 2  # capped at the image count

def test_done_snapshot_includes_skipped_measure_issues() -> None:
    job = _JobState("j")
    job.set_plan({"mini": 3}, ["PSNR", "BROKEN"])
    for _ in range(3):
        job.record_progress("mini", "PSNR")
        job.record_progress("mini", "BROKEN")

    job.finish(
        "run-1",
        [
            {
                "dataset": "mini",
                "measure": "BROKEN",
                "count": 3,
                "reasons": [{"reason": "evaluation error: boom", "count": 3}],
            }
        ],
    )

    snap = job.snapshot()
    broken = next(m for m in _by_dataset(snap)["mini"]["measures"] if m["name"] == "BROKEN")
    psnr = next(m for m in _by_dataset(snap)["mini"]["measures"] if m["name"] == "PSNR")

    assert snap["run_issues"][0]["measure"] == "BROKEN"
    assert broken["issues"] == 3
    assert psnr["issues"] == 0

def test_running_snapshot_marks_measure_with_live_issues() -> None:
    job = _JobState("j")
    job.set_plan({"mini": 3}, ["PSNR", "BROKEN"])
    job.record_progress("mini", "PSNR")
    job.record_skip("mini", "BROKEN", "evaluation error: boom")
    job.record_progress("mini", "BROKEN")

    snap = job.snapshot()
    mini = _by_dataset(snap)["mini"]
    broken = next(m for m in mini["measures"] if m["name"] == "BROKEN")

    assert snap["status"] == "running"
    assert snap["run_issues"][0]["measure"] == "BROKEN"
    assert broken["issues"] == 1