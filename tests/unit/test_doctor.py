from __future__ import annotations

import json

from typer.testing import CliRunner

from ipasc_iqa.cli.main import app
from ipasc_iqa.core.doctor import CheckResult, has_errors, run_doctor


def test_has_errors() -> None:
    results = [
        CheckResult(name="ok", group="core", status="ok", message="OK"),
        CheckResult(name="bad", group="core", status="error", message="Bad"),
    ]

    assert has_errors(results)


def test_run_doctor_reports_core_dependencies() -> None:
    results = run_doctor()
    names = {(result.group, result.name) for result in results}

    assert ("core", "numpy") in names
    assert ("deep", "torch-numpy interop") in names


def test_doctor_json_cli() -> None:
    result = CliRunner().invoke(app, ["doctor", "--json"])

    # The command may return 1 in an environment with broken optional torch/numpy
    # interop, but it should always emit valid JSON for automated diagnostics.
    assert result.exit_code in {0, 1}
    payload = json.loads(result.output)
    assert isinstance(payload, list)
    assert any(item["name"] == "numpy" and item["group"] == "core" for item in payload)
