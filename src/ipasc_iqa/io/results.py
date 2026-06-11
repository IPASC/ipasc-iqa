"""Results JSON I/O."""

from __future__ import annotations

import json
from pathlib import Path

from ipasc_iqa.core.result import RunResults


def save_results(run: RunResults, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")


def load_results(path: Path) -> RunResults:
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunResults.model_validate(data)
