"""Shared dataset preparation progress and callback isolation."""

from ipasc_iqa.datasets.progress import preparation_progress, report_preparation


def test_optional_notifications():
    observed = []
    report_preparation("downloading", "no listener")
    with preparation_progress(lambda stage, detail: observed.append((stage, detail))):
        report_preparation("downloading", "archive")
        report_preparation("extracting", "archive")
    report_preparation("converting", "no listener")
    assert observed == [("downloading", "archive"), ("extracting", "archive")]


def test_nested_callbacks_do_not_leak():
    first, second = [], []
    with preparation_progress(lambda stage, detail: first.append(stage)):
        report_preparation("downloading", "")
        with preparation_progress(lambda stage, detail: second.append(stage)):
            report_preparation("extracting", "")
        report_preparation("converting", "")
    assert first == ["downloading", "converting"]
    assert second == ["extracting"]
