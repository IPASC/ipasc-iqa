"""Function layer for the GUI — pure logic, no web framework.

Everything the UI needs is expressed here as functions over a :class:`Workspace`
returning plain dicts / dataclasses. Routes are thin adapters over these; tests
exercise these directly without spinning up a server.

A *workspace* is a directory containing:

- one or more dataset folders (each with a ``manifest.json``, possibly under a
  ``converted/`` sub-folder), and
- a ``results/`` folder where run outputs (``<run_id>.json``) are written.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ipasc_iqa.analysis.correlation import correlate_measures_with_annotations
from ipasc_iqa.analysis.manifest_query import (
    annotation_tasks_in_manifest,
    annotation_values_from_manifest,
    groups_from_manifest,
)
from ipasc_iqa.analysis.standings import compute_standings
from ipasc_iqa.core.registry import registry
from ipasc_iqa.core.runner import run_evaluation
from ipasc_iqa.datasets.local import LocalDataset
from ipasc_iqa.io.manifest import load_manifest
from ipasc_iqa.io.results import load_results

DEFAULT_TASK = "overall_quality"


@dataclass(frozen=True)
class Workspace:
    root: Path

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def ensure(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


def list_measures() -> list[dict]:
    """All registered measures with display metadata."""
    out = []
    for name in registry.list_measures():
        cls = registry.get(name)
        m = cls.metadata
        out.append(
            {
                "name": name,
                "version": m.version,
                "full_reference": m.full_reference,
                "requires_signal_roi": m.requires_signal_roi,
                "requires_noise_roi": m.requires_noise_roi,
                "higher_is_better": m.higher_is_better,
                "description": m.description,
                "distribution": registry.distribution_for(name),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def _find_manifest(folder: Path) -> Path | None:
    direct = folder / "manifest.json"
    if direct.exists():
        return direct
    converted = folder / "converted" / "manifest.json"
    if converted.exists():
        return converted
    return None


def discover_datasets(workspace: Workspace) -> list[dict]:
    """Scan the workspace for datasets (folders with a manifest)."""
    out: list[dict] = []
    seen: set[Path] = set()
    if not workspace.root.exists():
        return out

    for manifest_path in sorted(workspace.root.rglob("manifest.json")):
        # Skip results/ and avoid double-counting converted/ under a parent.
        if workspace.results_dir in manifest_path.parents:
            continue
        manifest_path = manifest_path.resolve()
        if manifest_path in seen:
            continue
        seen.add(manifest_path)

        try:
            manifest = load_manifest(manifest_path)
        except Exception:
            continue

        tasks = annotation_tasks_in_manifest(manifest)
        groups = sorted(set(groups_from_manifest(manifest, "algorithm").values()))
        out.append(
            {
                "name": manifest.get("dataset_name", manifest_path.parent.name),
                "version": manifest.get("dataset_version", "unknown"),
                "root": str(manifest_path.parent),
                "manifest_path": str(manifest_path),
                "n_images": len(manifest.get("images", [])),
                "annotation_tasks": tasks,
                "has_annotations": bool(tasks),
                "groups": groups,
            }
        )
    return out


def get_dataset(workspace: Workspace, name: str) -> dict | None:
    for ds in discover_datasets(workspace):
        if ds["name"] == name:
            return ds
    return None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def run_benchmark(
    workspace: Workspace,
    dataset_names: list[str],
    measure_names: list[str],
    *,
    reference_preference: list[str] | None = None,
    device: str = "auto",
) -> str:
    """Run an evaluation and persist results. Returns the run_id."""
    workspace.ensure()
    available = {d["name"]: d for d in discover_datasets(workspace)}
    datasets = []
    for name in dataset_names:
        if name not in available:
            raise ValueError(f"Unknown dataset: {name}")
        datasets.append(LocalDataset(Path(available[name]["root"])))

    measures = [registry.instantiate(m, device=device) for m in measure_names]
    if reference_preference is None:
        reference_preference = ["ground_truth", "pseudo_reference"]

    run = run_evaluation(
        datasets,
        measures,
        reference_preference=reference_preference,
        progress=False,
    )
    out_path = workspace.results_dir / f"{run.run_id}.json"
    out_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return run.run_id


def list_runs(workspace: Workspace) -> list[dict]:
    if not workspace.results_dir.exists():
        return []
    runs = []
    for p in sorted(workspace.results_dir.glob("*.json"), reverse=True):
        try:
            run = load_results(p)
        except Exception:
            continue
        runs.append(
            {
                "run_id": run.run_id,
                "finished": run.finished.isoformat() if run.finished else None,
                "datasets": [d.name for d in run.datasets],
                "measures": [m.name for m in run.measures],
                "n_results": len(run.results),
            }
        )
    return runs


def _latest_run_for_dataset(workspace: Workspace, dataset_name: str):
    """Return the most recent RunResults that contains the given dataset."""
    if not workspace.results_dir.exists():
        return None
    for p in sorted(workspace.results_dir.glob("*.json"), reverse=True):
        try:
            run = load_results(p)
        except Exception:
            continue
        if any(d.name == dataset_name for d in run.datasets):
            return run
    return None


# ---------------------------------------------------------------------------
# Detailed analysis (correlation table)
# ---------------------------------------------------------------------------


def correlation_table(
    workspace: Workspace,
    dataset_name: str,
    *,
    measures: list[str] | None = None,
    task: str = DEFAULT_TASK,
    run_id: str | None = None,
) -> dict:
    """Build the Detailed Analysis table: measures x {SRCC, KRCC} vs annotation MOS."""
    ds = get_dataset(workspace, dataset_name)
    if ds is None:
        return {"dataset": dataset_name, "task": task, "rows": [], "error": "dataset not found"}

    if run_id is not None:
        run = load_results(workspace.results_dir / f"{run_id}.json")
    else:
        run = _latest_run_for_dataset(workspace, dataset_name)

    if run is None:
        return {
            "dataset": dataset_name,
            "task": task,
            "rows": [],
            "error": "no benchmark run found for this dataset — run a benchmark first",
        }

    manifest = load_manifest(Path(ds["manifest_path"]))
    annotation_values = annotation_values_from_manifest(manifest, task, "zscore_mos")
    if not annotation_values:
        return {
            "dataset": dataset_name,
            "task": task,
            "rows": [],
            "error": f"dataset has no '{task}' annotations",
        }

    rows = correlate_measures_with_annotations(
        run, annotation_values, measures=measures, dataset=dataset_name
    )
    return {
        "dataset": dataset_name,
        "task": task,
        "run_id": run.run_id,
        "available_tasks": ds["annotation_tasks"],
        "rows": [r.as_dict() for r in rows],
        "error": None,
    }


# ---------------------------------------------------------------------------
# Standings
# ---------------------------------------------------------------------------


def standings_table(workspace: Workspace, *, task: str = DEFAULT_TASK) -> dict:
    """Rank algorithm groups within each dataset by mean expert score (z-MOS)."""
    records: list[tuple[str, str, float]] = []
    datasets_present: list[str] = []
    for ds in discover_datasets(workspace):
        if not ds["has_annotations"]:
            continue
        manifest = load_manifest(Path(ds["manifest_path"]))
        values = annotation_values_from_manifest(manifest, task, "zscore_mos")
        groups = groups_from_manifest(manifest, "algorithm")
        if not values or not groups:
            continue
        datasets_present.append(ds["name"])
        for image_id, score in values.items():
            if image_id in groups:
                records.append((ds["name"], groups[image_id], score))

    rows = [s.as_dict() for s in compute_standings(records)]
    return {"task": task, "datasets": datasets_present, "rows": rows}
