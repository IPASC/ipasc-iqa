"""Unit tests for the analysis layer (annotations, correlation, standings)."""

from __future__ import annotations

from datetime import UTC, datetime

from ipasc_iqa.analysis.correlation import correlate_measures_with_annotations
from ipasc_iqa.analysis.standings import compute_standings
from ipasc_iqa.core.result import (
    EnvironmentInfo,
    ResultRecord,
    RunConfig,
    RunResults,
)


def _make_run(values: dict[str, float], measure: str = "psnr") -> RunResults:
    records = [
        ResultRecord(
            dataset="d",
            image_id=img,
            image_sha256="0" * 64,
            measure=measure,
            measure_version="1.0.0",
            reference_type="pseudo_reference",
            signal_roi_sha256=None,
            noise_roi_sha256=None,
            value=val,
            sub_scores=None,
            wall_time_s=0.0,
            device="cpu",
        )
        for img, val in values.items()
    ]
    return RunResults(
        run_id="test",
        environment=EnvironmentInfo(python="3.13", platform="test"),
        started=datetime.now(UTC),
        config=RunConfig(),
        results=records,
    )


class TestCorrelation:
    def test_perfect_positive_correlation(self) -> None:
        run = _make_run({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
        annotations = {"a": 10.0, "b": 20.0, "c": 30.0, "d": 40.0}
        rows = correlate_measures_with_annotations(run, annotations)
        assert len(rows) == 1
        assert rows[0].srcc == 1.0
        assert rows[0].krcc == 1.0
        assert rows[0].n == 4

    def test_perfect_negative_correlation(self) -> None:
        run = _make_run({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0})
        annotations = {"a": 40.0, "b": 30.0, "c": 20.0, "d": 10.0}
        rows = correlate_measures_with_annotations(run, annotations)
        assert rows[0].srcc == -1.0

    def test_missing_annotations_reduce_n(self) -> None:
        run = _make_run({"a": 1.0, "b": 2.0, "c": 3.0})
        annotations = {"a": 1.0, "b": 2.0}  # 'c' has no annotation
        rows = correlate_measures_with_annotations(run, annotations)
        assert rows[0].n == 2

    def test_too_few_pairs_yields_none(self) -> None:
        run = _make_run({"a": 1.0})
        rows = correlate_measures_with_annotations(run, {"a": 1.0})
        assert rows[0].srcc is None


class TestStandings:
    def test_ranks_groups_per_dataset(self) -> None:
        records = [
            ("d1", "alg0", 1.0),
            ("d1", "alg0", 1.0),
            ("d1", "alg1", 5.0),
            ("d1", "alg2", 3.0),
        ]
        rows = compute_standings(records)
        ranked = {r.rank: r.group for r in rows if r.dataset == "d1"}
        assert ranked[1] == "alg1"  # highest mean
        assert ranked[2] == "alg2"
        assert ranked[3] == "alg0"

    def test_lower_is_better(self) -> None:
        records = [("d", "a", 1.0), ("d", "b", 5.0)]
        rows = compute_standings(records, higher_is_better=False)
        assert rows[0].group == "a"
        assert rows[0].rank == 1
