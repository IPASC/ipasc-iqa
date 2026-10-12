"""Metric progress reports real outcomes after evaluation/cache lookup."""
from ipasc_iqa.core.runner import run_evaluation
from ipasc_iqa.datasets.local import LocalDataset
from ipasc_iqa.gui.jobs import _JobState
from ipasc_iqa.measures.classical.psnr import PSNRMeasure
from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class _MemoryStore:
    def __init__(self):
        self.values = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value


class _FailingMeasure(BaseMeasure):
    metadata = MeasureMetadata(
        name="PROGRESS_FAILURE_TEST", version="0.0.1",
        measure_type=MeasureType.NO_REFERENCE,
        requires_signal_roi=False, requires_noise_roi=False,
        description="Raises so progress must mark this as skipped.",
    )

    def evaluate(self, image, **kwargs):
        raise RuntimeError("expected failure")


def test_progress_tracks_computed_cached_and_skipped(synthetic_dataset):
    ds = LocalDataset(synthetic_dataset)
    cache = _MemoryStore()
    job = _JobState("test")
    job.set_plan({ds.name: len(ds)}, ["PSNR", "PROGRESS_FAILURE_TEST"])
    run_evaluation([ds], [PSNRMeasure(), _FailingMeasure()], cache=cache,
                   progress=False, on_progress=job.record_progress)
    first = job.snapshot()["progress"][0]
    assert (first["computed"], first["cached"], first["skipped"]) == (len(ds), 0, len(ds))
    assert first["percent"] == 100

    other_job = _JobState("second")
    other_job.set_plan({ds.name: len(ds)}, ["PSNR"])
    run_evaluation([ds], [PSNRMeasure()], cache=cache,
                   progress=False, on_progress=other_job.record_progress)
    second = other_job.snapshot()["progress"][0]
    assert (second["computed"], second["cached"], second["skipped"]) == (0, len(ds), 0)
    assert second["percent"] == 100
