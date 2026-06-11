"""Function layer for the GUI — pure logic, no web framework.

Everything the UI needs is expressed here as functions returning plain dicts /
dataclasses. Routes are thin adapters over these; tests exercise these directly.

Datasets are *adapter-driven* (see :mod:`ipasc_iqa.datasets.registry`): only
datasets with a registered adapter are offered, and they are downloaded/converted
on demand into a managed ``data_dir``. The workspace's ``results/`` folder stores
run outputs (``<run_id>.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ipasc_iqa.analysis.correlation import correlate_measures_with_annotations
from ipasc_iqa.analysis.manifest_query import (
    annotation_tasks_in_manifest,
    annotation_values_from_manifest,
)
from ipasc_iqa.core.registry import registry as measure_registry
from ipasc_iqa.core.runner import run_evaluation
from ipasc_iqa.datasets.registry import default_data_dir
from ipasc_iqa.datasets.registry import registry as dataset_registry
from ipasc_iqa.io.manifest import load_manifest
from ipasc_iqa.io.results import load_results


@dataclass(frozen=True)
class Workspace:
    """Where results are written (``root``) and datasets are managed (``data_dir``)."""

    root: Path
    data_dir: Path

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    def ensure(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def resolve(root: Path | str, data_dir: Path | str | None = None) -> Workspace:
        return Workspace(
            root=Path(root).resolve(),
            data_dir=Path(data_dir).resolve() if data_dir else default_data_dir(),
        )


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


def list_measures() -> list[dict]:
    out = []
    for name in measure_registry.list_measures():
        m = measure_registry.get(name).metadata
        out.append(
            {
                "name": name,
                "version": m.version,
                "full_reference": m.full_reference,
                "requires_signal_roi": m.requires_signal_roi,
                "requires_noise_roi": m.requires_noise_roi,
                "higher_is_better": m.higher_is_better,
                "description": m.description,
                "distribution": measure_registry.distribution_for(name),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Datasets (adapter-driven)
# ---------------------------------------------------------------------------


def _capabilities_from_manifest(manifest: dict) -> dict:
    """What sibling data the dataset provides (drives which measures are usable)."""
    has_references = has_signal_roi = has_noise_roi = False
    for entry in manifest.get("images", []):
        if entry.get("references"):
            has_references = True
        if entry.get("signal_roi"):
            has_signal_roi = True
        if entry.get("noise_roi"):
            has_noise_roi = True
        if has_references and has_signal_roi and has_noise_roi:
            break
    return {
        "has_references": has_references,
        "has_signal_roi": has_signal_roi,
        "has_noise_roi": has_noise_roi,
    }


def _measure_usable(measure_meta: dict, caps: dict) -> bool:
    """True if the dataset can satisfy a measure's requirements."""
    if measure_meta["full_reference"] and not caps.get("has_references"):
        return False
    if measure_meta["requires_signal_roi"] and not caps.get("has_signal_roi"):
        return False
    return not (measure_meta["requires_noise_roi"] and not caps.get("has_noise_roi"))


def discover_datasets(workspace: Workspace) -> list[dict]:
    """List datasets that have a registered adapter.

    Datasets already downloaded/converted report their image count, annotation
    tasks, and capabilities (references / ROI masks); others are still listed (so
    they can be selected to trigger a download).
    """
    out: list[dict] = []
    for spec in dataset_registry.list_datasets():
        name = spec["name"]
        prepared = dataset_registry.is_prepared(name, workspace.data_dir)
        tasks: list[str] = []
        n_images: int | None = None
        caps = {"has_references": False, "has_signal_roi": False, "has_noise_roi": False}
        if prepared:
            try:
                manifest = load_manifest(dataset_registry.manifest_path(name, workspace.data_dir))
                tasks = annotation_tasks_in_manifest(manifest)
                n_images = len(manifest.get("images", []))
                caps = _capabilities_from_manifest(manifest)
            except Exception:
                prepared = False
        out.append(
            {
                **spec,
                "prepared": prepared,
                "n_images": n_images,
                "annotation_tasks": tasks,
                "has_annotations": bool(tasks),
                **caps,
            }
        )
    return out


def measure_options(
    workspace: Workspace,
    dataset_name: str | None,
    *,
    selected: list[str] | None = None,
) -> list[dict]:
    """Measures annotated with ``usable``/``checked`` for a given dataset.

    A measure is *unusable* when the dataset cannot satisfy its requirements
    (no reference, or missing ROI mask). On first load (``selected is None``) all
    usable measures are checked; unusable ones are never checked.
    """
    ds = get_dataset(workspace, dataset_name) if dataset_name else None
    options = []
    for m in list_measures():
        usable = _measure_usable(m, ds) if ds and ds["prepared"] else True
        checked = usable if selected is None else (usable and m["name"] in selected)
        options.append({"name": m["name"], "usable": usable, "checked": checked})
    return options


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
    """Download/prepare datasets as needed, run evaluation, persist results."""
    workspace.ensure()
    datasets = [
        dataset_registry.instantiate(name, workspace.data_dir, download=True)
        for name in dataset_names
    ]
    measures = [measure_registry.instantiate(m, device=device) for m in measure_names]
    if reference_preference is None:
        reference_preference = ["ground_truth", "pseudo_reference"]

    run = run_evaluation(
        datasets, measures, reference_preference=reference_preference, progress=False
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
# Detailed analysis — one correlation table per available task
# ---------------------------------------------------------------------------


def analysis_for_dataset(
    workspace: Workspace,
    dataset_name: str,
    *,
    measures: list[str] | None = None,
    run_id: str | None = None,
) -> dict:
    """Correlate measures against expert MOS, one table per annotation task.

    Tasks are not selectable — every task the dataset provides is shown, since
    different datasets expose different sub-tasks.
    """
    ds = get_dataset(workspace, dataset_name)
    if ds is None:
        return {"dataset": dataset_name, "tasks": [], "error": "dataset not found"}
    if not ds["prepared"]:
        return {
            "dataset": dataset_name,
            "tasks": [],
            "error": "dataset not downloaded yet — run a benchmark on it first",
        }

    run = (
        load_results(workspace.results_dir / f"{run_id}.json")
        if run_id is not None
        else _latest_run_for_dataset(workspace, dataset_name)
    )
    if run is None:
        return {
            "dataset": dataset_name,
            "tasks": [],
            "error": "no benchmark run found for this dataset — run a benchmark first",
        }

    manifest = load_manifest(dataset_registry.manifest_path(dataset_name, workspace.data_dir))
    task_names = annotation_tasks_in_manifest(manifest)
    if not task_names:
        return {
            "dataset": dataset_name,
            "tasks": [],
            "error": "dataset has no expert annotations",
        }

    # Only correlate measures the dataset can actually support, intersected with
    # the user's selection — so unusable measures never produce empty rows.
    caps = _capabilities_from_manifest(manifest)
    usable = [m["name"] for m in list_measures() if _measure_usable(m, caps)]
    chosen = [m for m in (measures or usable) if m in usable]

    tasks = []
    for task in task_names:
        values = annotation_values_from_manifest(manifest, task, "zscore_mos")
        rows = correlate_measures_with_annotations(
            run, values, measures=chosen, dataset=dataset_name
        )
        tasks.append({"task": task, "rows": [r.as_dict() for r in rows]})

    return {"dataset": dataset_name, "run_id": run.run_id, "tasks": tasks, "error": None}


# ---------------------------------------------------------------------------
# Standings — IQA measures ranked by correlation, per dataset, per task
# ---------------------------------------------------------------------------


def standings(workspace: Workspace) -> dict:
    """For each prepared+annotated dataset with a run, rank measures by |SRCC|.

    Produces one ranked leaderboard per (dataset, task). Correlation rows arrive
    already sorted by ``|srcc|`` descending, so list position is the rank.
    """
    entries: list[dict] = []
    for ds in discover_datasets(workspace):
        if not ds["prepared"] or not ds["has_annotations"]:
            continue
        run = _latest_run_for_dataset(workspace, ds["name"])
        if run is None:
            continue
        manifest = load_manifest(dataset_registry.manifest_path(ds["name"], workspace.data_dir))
        for task in ds["annotation_tasks"]:
            values = annotation_values_from_manifest(manifest, task, "zscore_mos")
            rows = correlate_measures_with_annotations(run, values, dataset=ds["name"])
            ranked = [
                {"rank": i + 1, **r.as_dict()}
                for i, r in enumerate(rows)
                if r.srcc is not None
            ]
            if ranked:
                entries.append({"dataset": ds["name"], "task": task, "rows": ranked})

    return {"entries": entries}
