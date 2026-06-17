"""Function layer for the GUI — pure logic, no web framework.

Everything the UI needs is expressed here as functions returning plain dicts /
dataclasses. Routes are thin adapters over these; tests exercise these directly.

Datasets are *adapter-driven* (see :mod:`ipasc_iqa.datasets.registry`): only
datasets with a registered adapter are offered, and they are downloaded/converted
on demand into a managed ``data_dir``. The workspace's ``results/`` folder stores
run outputs (``<run_id>.json``).
"""

from __future__ import annotations

import io
import json
import math
import random
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ipasc_iqa import __version__
from ipasc_iqa.analysis.aggregate import measure_stats
from ipasc_iqa.analysis.correlation import correlate_measures_with_annotations
from ipasc_iqa.analysis.manifest_query import (
    annotation_tasks_in_manifest,
    annotation_values_from_manifest,
    annotator_ratings_from_manifest,
    groups_from_manifest,
)
from ipasc_iqa.analysis.mos import composite_mos
from ipasc_iqa.analysis.scatter import build_scatter
from ipasc_iqa.contrib.bradley_terry import fit_bradley_terry
from ipasc_iqa.contrib.ranking import RankingSession
from ipasc_iqa.contrib.store import AnnotatorStore, normalize_email
from ipasc_iqa.core.cache import ResultStore
from ipasc_iqa.core.measure import MeasureType
from ipasc_iqa.core.registry import registry as measure_registry
from ipasc_iqa.core.result import EnvironmentInfo, RunConfig, RunResults
from ipasc_iqa.core.runner import run_evaluation
from ipasc_iqa.datasets.registry import default_data_dir
from ipasc_iqa.datasets.registry import registry as dataset_registry
from ipasc_iqa.gui.charts import scatter_chart
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

    @property
    def store_path(self) -> Path:
        """Persistent, newest-wins result store (also the run cache).

        A diskcache directory (SQLite-backed), not a single file.
        """
        return self.results_dir / "cache"

    @property
    def annotations_path(self) -> Path:
        """diskcache directory for contributed annotators + their rankings."""
        return self.root / "annotations"

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
                "measure_type": m.measure_type.name,
                "requires_signal_roi": m.requires_signal_roi,
                "requires_noise_roi": m.requires_noise_roi,
                "higher_is_better": m.higher_is_better,
                "description": m.description,
                "bibtex": m.bibtex,
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
    if measure_meta["measure_type"] == MeasureType.FULL_REFERENCE and not caps.get(
        "has_references"
    ):
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
        options.append(
            {
                "name": m["name"],
                "measure_type": m["measure_type"],
                "usable": usable,
                "checked": checked,
            }
        )
    return options


# ---------------------------------------------------------------------------
# Last benchmark selection — remembered so the analysis page can default to the
# measures/dataset the user just ran, instead of "all measures, first dataset".
# ---------------------------------------------------------------------------


def _last_selection_path(workspace: Workspace) -> Path:
    return workspace.results_dir / "last_selection.json"


def save_last_selection(
    workspace: Workspace, datasets: list[str], measures: list[str]
) -> None:
    """Persist the dataset/measure selection from the most recent benchmark run."""
    workspace.ensure()
    _last_selection_path(workspace).write_text(
        json.dumps({"datasets": list(datasets), "measures": list(measures)}),
        encoding="utf-8",
    )


def load_last_selection(workspace: Workspace) -> dict:
    """Most recent benchmark selection, or empty lists if none was recorded."""
    path = _last_selection_path(workspace)
    if not path.exists():
        return {"datasets": [], "measures": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"datasets": [], "measures": []}
    return {
        "datasets": list(data.get("datasets", [])),
        "measures": list(data.get("measures", [])),
    }


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
    on_result: Callable[[str], None] | None = None,
) -> str:
    """Download/prepare datasets as needed, run evaluation, persist results.

    Results are written through a persistent, newest-wins store that also serves
    as the cache: already-computed (dataset, image, measure) pairs are reused
    rather than recomputed, and partial runs accumulate instead of overwriting.
    """
    workspace.ensure()
    datasets = [
        dataset_registry.instantiate(name, workspace.data_dir, download=True)
        for name in dataset_names
    ]
    measures = [measure_registry.instantiate(m, device=device) for m in measure_names]
    if reference_preference is None:
        reference_preference = ["ground_truth", "pseudo_reference"]

    with ResultStore(workspace.store_path) as store:
        run = run_evaluation(
            datasets,
            measures,
            reference_preference=reference_preference,
            cache=store,
            progress=False,
            on_result=on_result,
        )
        # Pre-compute per-dataset mean/std over the merged per-image results and
        # persist them, so reporting doesn't have to recompute from scratch.
        for name in dataset_names:
            _write_dataset_stats(workspace, name, store.records_for_dataset(name))

    # Keep a per-run audit file alongside the merged store.
    out_path = workspace.results_dir / f"{run.run_id}.json"
    out_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return run.run_id


def _dataset_stats_path(workspace: Workspace, dataset_name: str) -> Path:
    return workspace.results_dir / f"{dataset_name}.stats.json"


def _write_dataset_stats(workspace: Workspace, dataset_name: str, records: list) -> None:
    """Persist ``{measure: {n, mean, std}}`` for one dataset to a sidecar JSON."""
    stats = measure_stats(records)
    payload = {
        "dataset": dataset_name,
        "computed": datetime.now(UTC).isoformat(),
        "measures": stats,
    }
    path = _dataset_stats_path(workspace, dataset_name)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dataset_measure_stats(workspace: Workspace, dataset_name: str) -> dict[str, dict[str, float]]:
    """Per-measure mean/std for a dataset.

    Prefers the pre-computed sidecar written at run time; falls back to computing
    live from the merged store if the sidecar is missing or unreadable.
    """
    path = _dataset_stats_path(workspace, dataset_name)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("measures", {})
        except (OSError, ValueError):
            pass
    if not workspace.store_path.exists():
        return {}
    with ResultStore(workspace.store_path) as store:
        return measure_stats(store.records_for_dataset(dataset_name))


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


def _merged_run(workspace: Workspace, dataset_name: str) -> RunResults | None:
    """Build a RunResults view from the merged store (newest result per key).

    This is what the analysis views read, so running only a subset of measures
    still shows the union of everything computed for the dataset so far.
    """
    if not workspace.store_path.exists():
        return None
    with ResultStore(workspace.store_path) as store:
        records = store.records_for_dataset(dataset_name)
    if not records:
        return None
    return RunResults(
        run_id="merged",
        environment=EnvironmentInfo(python="", platform="merged-store"),
        started=datetime.now(UTC),
        finished=datetime.now(UTC),
        config=RunConfig(cache_enabled=True),
        results=records,
    )


# ---------------------------------------------------------------------------
# Detailed analysis — one correlation table per available task
# ---------------------------------------------------------------------------


def analysis_for_dataset(
    workspace: Workspace,
    dataset_name: str,
    *,
    measures: list[str] | None = None,
    run_id: str | None = None,
    task: str | None = None,
    source: str = "mos",
) -> dict:
    """Correlate measures against a human-score source for one annotation task.

    ``source`` selects which human scores to correlate against: ``"mos"`` (the
    dataset's absolute expert ratings) or ``"bt"`` (continuous Bradley-Terry
    scores derived from contributed pairwise comparisons). ``all_tasks`` /
    ``active_task`` / ``sources`` / ``active_source`` are always present so the
    template can render the dropdowns regardless of error state.
    """
    _empty: dict = {"all_tasks": [], "active_task": None, "sources": [], "active_source": source}

    ds = get_dataset(workspace, dataset_name)
    if ds is None:
        return {"dataset": dataset_name, "tasks": [], "error": "dataset not found", **_empty}
    if not ds["prepared"]:
        return {
            "dataset": dataset_name,
            "tasks": [],
            "error": "dataset not downloaded yet — run a benchmark on it first",
            **_empty,
        }

    run = (
        load_results(workspace.results_dir / f"{run_id}.json")
        if run_id is not None
        else _merged_run(workspace, dataset_name)
    )
    if run is None:
        return {
            "dataset": dataset_name,
            "tasks": [],
            "error": "no benchmark run found for this dataset — run a benchmark first",
            **_empty,
        }

    manifest = load_manifest(dataset_registry.manifest_path(dataset_name, workspace.data_dir))
    # Tasks can come from manifest MOS *or* from contributed pairwise comparisons,
    # so a dataset with no expert ratings is still analysable once BT data exists.
    task = task or None
    task_names, has_unnamed = _available_tasks(workspace, dataset_name, manifest)
    if not task_names and not has_unnamed:
        return {
            "dataset": dataset_name,
            "tasks": [],
            "error": "no expert annotations or contributed comparisons to analyse yet",
            **_empty,
        }

    # Resolve the active task. A dataset with no sub-tasks but contributed
    # comparisons has a single unnamed task (active_task=None, no dropdown).
    active_task = _resolve_active_task(task, task_names)

    # Only correlate measures the dataset can actually support, intersected with
    # the user's selection — so unusable measures never produce empty rows.
    caps = _capabilities_from_manifest(manifest)
    usable = [m["name"] for m in list_measures() if _measure_usable(m, caps)]
    chosen = [m for m in (measures or usable) if m in usable]

    # Resolve the human-score source (MOS vs contributed Bradley-Terry).
    mos_values = mos_values_for_dataset_task(workspace, dataset_name, active_task, manifest)
    bt = bt_scores_for_dataset_task(workspace, dataset_name, active_task)
    sources: list[dict] = []
    if mos_values:
        sources.append({"key": "mos", "label": _SOURCE_LABELS["mos"]})
    if bt["scores"]:
        sources.append({"key": "bt", "label": _SOURCE_LABELS["bt"]})
    source_keys = {s["key"] for s in sources}
    active_source = source if source in source_keys else (sources[0]["key"] if sources else "mos")
    values = bt["scores"] if active_source == "bt" else mos_values

    rows = correlate_measures_with_annotations(run, values, measures=chosen, dataset=dataset_name)

    # Annotate each row with its measure's BibTeX (for the citation icon) and the
    # pre-computed per-dataset mean/std of its values.
    bibtex_by_measure = {m["name"]: m["bibtex"] for m in list_measures()}
    stats_by_measure = dataset_measure_stats(workspace, dataset_name)
    row_dicts = []
    for r in rows:
        d = r.as_dict()
        d["bibtex"] = bibtex_by_measure.get(r.measure)
        stat = stats_by_measure.get(r.measure, {})
        d["mean"] = stat.get("mean")
        d["std"] = stat.get("std")
        row_dicts.append(d)

    return {
        "dataset": dataset_name,
        "description": ds.get("description"),
        "bibtex": ds.get("bibtex"),
        "run_id": run.run_id,
        "tasks": [{"task": active_task, "rows": row_dicts}],
        "all_tasks": task_names,
        "active_task": active_task,
        "meta": task_annotation_summary(workspace, dataset_name, active_task, manifest),
        "sources": sources,
        "active_source": active_source,
        "error": None,
    }


def task_annotation_summary(
    workspace: Workspace, dataset_name: str, task: str | None, manifest: dict
) -> dict:
    """Annotation-density metadata for one (dataset, task), for methods reporting.

    Averages are over the *whole dataset* (``n_images``, same count the run page
    reports) so they read as a coverage fraction — e.g. 36 rankings over 471
    images is 0.08, not 1. Ranking/rating observations exclude reference
    (ground-truth) pseudo-items, since those aren't assessed. Returns:

    - ``n_images``: total images in the dataset.
    - ``n_ratings`` / ``n_rankings``: total rating / ranking observations for this
      task (a ranking session contributes one observation per variant it ranks).
    - ``avg_ratings`` / ``avg_rankings``: those totals divided by ``n_images``;
      ``None`` if the dataset has no images.
    """
    n_images = len(manifest.get("images", []))

    # Ratings: per-image annotator counts (manifest per-rater + contributed).
    rated = _mos_annotator_scores(workspace, dataset_name, task, manifest)
    n_ratings = sum(
        len(raters) for img, raters in rated.items() if not img.startswith(_REF_PREFIX)
    )

    # Rankings: total variant placements across sessions for this task.
    with AnnotatorStore(workspace.annotations_path) as store:
        sessions = store.sessions()
    n_rankings = 0
    for s in sessions:
        if s.get("dataset") != dataset_name or s.get("task") != task:
            continue
        n_rankings += sum(
            1 for i in s.get("order", []) if not i.startswith(_REF_PREFIX)
        )

    return {
        "n_images": n_images,
        "n_ratings": n_ratings,
        "n_rankings": n_rankings,
        "avg_ratings": (n_ratings / n_images) if n_images else None,
        "avg_rankings": (n_rankings / n_images) if n_images else None,
    }


def scatter_for_measure(
    workspace: Workspace,
    dataset_name: str,
    measure: str,
    *,
    task: str | None = None,
    run_id: str | None = None,
    source: str = "mos",
) -> dict:
    """Build the scatter-plot model for one measure vs a human-score source.

    ``source`` is ``"mos"`` (absolute ratings) or ``"bt"`` (Bradley-Terry scores
    from contributed comparisons). Returns ``{"error": ...}`` when the plot can't
    be produced (dataset not ready, no run, no annotations, or < 2 paired points).
    """
    ds = get_dataset(workspace, dataset_name)
    if ds is None or not ds["prepared"]:
        return {"error": "dataset not available"}

    run = (
        load_results(workspace.results_dir / f"{run_id}.json")
        if run_id is not None
        else _merged_run(workspace, dataset_name)
    )
    if run is None:
        return {"error": "no benchmark run found for this dataset"}

    manifest = load_manifest(dataset_registry.manifest_path(dataset_name, workspace.data_dir))
    task = task or None
    task_names, has_unnamed = _available_tasks(workspace, dataset_name, manifest)
    if not task_names and not has_unnamed:
        return {"error": "no expert annotations or contributed comparisons to analyse yet"}
    active_task = _resolve_active_task(task, task_names)

    task_label = active_task or "all"
    if source == "bt":
        human_values = bt_scores_for_dataset_task(workspace, dataset_name, active_task)["scores"]
        x_label = f"Human score (BT · {task_label})"
    else:
        human_values = mos_values_for_dataset_task(workspace, dataset_name, active_task, manifest)
        x_label = f"Human score (z-MOS · {task_label})"
    if not human_values:
        return {"error": f"no '{source}' human scores available for this dataset/task"}

    groups = groups_from_manifest(manifest, "algorithm")
    data = build_scatter(run, human_values, measure, dataset=dataset_name, groups=groups)
    if len(data.xs) < 2:
        return {"error": f"not enough paired data points for {measure}"}

    rows = correlate_measures_with_annotations(
        run, human_values, measures=[measure], dataset=dataset_name
    )
    srcc = rows[0].srcc if rows else None
    krcc = rows[0].krcc if rows else None

    meta = measure_registry.get(measure).metadata
    chart = scatter_chart(
        data,
        srcc=srcc,
        krcc=krcc,
        x_label=x_label,
        y_label=f"{measure}{' ↑' if meta.higher_is_better else ' ↓'}",
    )

    stats = dataset_measure_stats(workspace, dataset_name).get(measure, {})
    return {
        "error": None,
        "measure": measure,
        "dataset": dataset_name,
        "task": active_task,
        "chart": chart,
        "srcc": srcc,
        "krcc": krcc,
        "n": len(data.xs),
        "mean": stats.get("mean"),
        "std": stats.get("std"),
        "higher_is_better": meta.higher_is_better,
        "bibtex": meta.bibtex,
    }


# ---------------------------------------------------------------------------
# Standings — IQA measures ranked by correlation, per dataset, per task
# ---------------------------------------------------------------------------


def standings(workspace: Workspace, source: str = "mos") -> dict:
    """Rank measures by |SRCC| against a human-score endpoint, per dataset+task.

    ``source`` selects the human scores to rank against: ``"mos"`` (expert
    absolute ratings) or ``"bt"`` (continuous Bradley-Terry scores from
    contributed pairwise comparisons). Correlation rows arrive already sorted by
    ``|srcc|`` descending, so list position is the rank.
    """
    source = "bt" if source == "bt" else "mos"
    entries: list[dict] = []
    for ds in discover_datasets(workspace):
        if not ds["prepared"]:
            continue
        run = _merged_run(workspace, ds["name"])
        if run is None:
            continue
        manifest = load_manifest(dataset_registry.manifest_path(ds["name"], workspace.data_dir))

        # Enumerate tasks from both manifest MOS *and* contributed annotations
        # (ratings/comparisons), so a dataset whose only human scores are
        # contributed — e.g. ratings recorded under the unnamed task=None —
        # still appears for either source.
        named, has_unnamed = _available_tasks(workspace, ds["name"], manifest)
        tasks: list[str | None] = [*named]
        if has_unnamed:
            tasks.append(None)

        for task in tasks:
            if source == "bt":
                values = bt_scores_for_dataset_task(workspace, ds["name"], task)["scores"]
            else:
                values = mos_values_for_dataset_task(workspace, ds["name"], task, manifest)
            if not values:
                continue
            rows = correlate_measures_with_annotations(run, values, dataset=ds["name"])
            ranked = [
                {"rank": i + 1, **r.as_dict()} for i, r in enumerate(rows) if r.srcc is not None
            ]
            if ranked:
                entries.append({"dataset": ds["name"], "task": task or "overall", "rows": ranked})

    return {"entries": entries, "source": source}


# ---------------------------------------------------------------------------
# Contributed annotations — lightweight login + 2AFC pairwise ranking
# ---------------------------------------------------------------------------

_REF_PREFIX = "__ref__"


def annotator_login(workspace: Workspace, name: str, email: str) -> dict:
    """Recognise/create an annotator by email. Returns user + is_new + count."""
    with AnnotatorStore(workspace.annotations_path) as store:
        user, is_new = store.upsert_user(email, name)
        count = store.count_sessions(user_email=email)
    return {"user": user, "is_new": is_new, "count": count}


def annotator_for(workspace: Workspace, email: str | None) -> dict | None:
    if not email:
        return None
    with AnnotatorStore(workspace.annotations_path) as store:
        user = store.get_user(email)
        if user is None:
            return None
        return {"user": user, "count": store.count_sessions(user_email=email)}


def annotators_leaderboard(workspace: Workspace) -> list[dict]:
    """Per-annotator ranking/comparison totals for the contribute screen."""
    with AnnotatorStore(workspace.annotations_path) as store:
        return store.leaderboard()


def _mos_annotator_scores(
    workspace: Workspace, dataset_name: str, task: str | None, manifest: dict
) -> dict[str, dict[str, float]]:
    """Per-image, per-annotator raw ratings feeding the MOS endpoint.

    Pools the dataset's published per-rater ratings (from the manifest) with
    contributed quantitative ratings, so a composite MOS can be computed over all
    annotators. (Contributed ratings are merged once the rating-collection tool
    is in place.)
    """
    scores: dict[str, dict[str, float]] = {}
    if task is not None:
        for image_id, raters in annotator_ratings_from_manifest(manifest, task).items():
            scores.setdefault(image_id, {}).update(raters)
    # Pool in contributed quantitative ratings (one annotator per email).
    with AnnotatorStore(workspace.annotations_path) as store:
        for r in store.ratings(dataset=dataset_name):
            if r.get("task") != task or r.get("image_id") is None or r.get("score") is None:
                continue
            scores.setdefault(r["image_id"], {})[f"user:{r['user_email']}"] = float(r["score"])
    return scores


def mos_values_for_dataset_task(
    workspace: Workspace,
    dataset_name: str,
    task: str | None,
    manifest: dict | None = None,
) -> dict[str, float]:
    """Composite MOS per image for (dataset, task); ``{}`` if no ratings exist.

    Built from individual annotator ratings (z-normalise each annotator, then
    average). Falls back to the manifest's pre-computed ``zscore_mos`` for older
    conversions that don't carry per-rater data.
    """
    if manifest is None:
        manifest = load_manifest(dataset_registry.manifest_path(dataset_name, workspace.data_dir))
    per_annotator = _mos_annotator_scores(workspace, dataset_name, task, manifest)
    if per_annotator:
        return composite_mos(per_annotator)
    if task is not None:
        return annotation_values_from_manifest(manifest, task, "zscore_mos")
    return {}


def bt_scores_for_dataset_task(workspace: Workspace, dataset_name: str, task: str | None) -> dict:
    """Per-image Bradley-Terry quality scores from contributed comparisons.

    Comparisons are aggregated per reference set (across all annotators), and each
    set is fit independently and anchored at its reference image (beta = 0) — so a
    variant's score is its *deficit below ground truth*, a scale that is
    comparable across reference sets. Reference (GT) items are excluded from the
    returned per-image mapping: full-reference IQMs are computed against the GT,
    so including it would trivially inflate their correlation (per Breger et al.).

    Returns ``{"scores": {image_id: beta}, "se": {...}, "n_comparisons", "n_sets"}``.
    """
    with AnnotatorStore(workspace.annotations_path) as store:
        sessions = store.sessions()

    by_ref: dict[str, list[tuple[str, str]]] = defaultdict(list)
    items_by_ref: dict[str, set[str]] = defaultdict(set)
    for s in sessions:
        if s.get("dataset") != dataset_name or s.get("task") != task:
            continue
        ref = s.get("reference")
        for c in s.get("comparisons", []):
            by_ref[ref].append((c["winner"], c["loser"]))
            items_by_ref[ref].update((c["winner"], c["loser"]))

    scores: dict[str, float] = {}
    ses: dict[str, float] = {}
    n_comparisons = 0
    for ref, comps in by_ref.items():
        items = sorted(items_by_ref[ref])
        anchor = f"{_REF_PREFIX}{ref}"
        result = fit_bradley_terry(
            items, comps, anchor=anchor if anchor in items_by_ref[ref] else None
        )
        n_comparisons += len(comps)
        for item in items:
            if item.startswith(_REF_PREFIX):
                continue  # GT excluded from the correlation mapping
            scores[item] = result.beta[item]
            ses[item] = result.se[item]

    return {
        "scores": scores,
        "se": ses,
        "n_comparisons": n_comparisons,
        "n_sets": len(by_ref),
    }


# Human-score sources the analysis can correlate measures against.
_SOURCE_LABELS = {
    "mos": "Absolute rating (MOS)",
    "bt": "Pairwise (Bradley–Terry)",
}


def _sources_for(
    manifest: dict, workspace: Workspace, dataset_name: str, task: str | None
) -> list[dict]:
    """Available human-score sources for a (dataset, task): MOS and/or BT."""
    out: list[dict] = []
    if task and mos_values_for_dataset_task(workspace, dataset_name, task, manifest):
        out.append({"key": "mos", "label": _SOURCE_LABELS["mos"]})
    if bt_scores_for_dataset_task(workspace, dataset_name, task)["scores"]:
        out.append({"key": "bt", "label": _SOURCE_LABELS["bt"]})
    return out


def human_score_sources(workspace: Workspace, dataset_name: str) -> list[dict]:
    """Sources for a dataset's first annotation task (for the initial page load)."""
    ds = get_dataset(workspace, dataset_name)
    if ds is None or not ds["prepared"]:
        return []
    manifest = load_manifest(dataset_registry.manifest_path(dataset_name, workspace.data_dir))
    tasks = annotation_tasks_in_manifest(manifest)
    return _sources_for(manifest, workspace, dataset_name, tasks[0] if tasks else None)


def _available_tasks(
    workspace: Workspace, dataset_name: str, manifest: dict
) -> tuple[list[str], bool]:
    """Annotation tasks available from *either* MOS or contributed comparisons.

    Returns ``(named_tasks, has_unnamed)`` where ``named_tasks`` are the named
    tasks (manifest MOS tasks ∪ contributed tasks) and ``has_unnamed`` is True
    when contributed annotations (pairwise or ratings) were collected for a
    dataset with no sub-tasks (task=None).
    """
    mos_tasks = annotation_tasks_in_manifest(manifest)
    with AnnotatorStore(workspace.annotations_path) as store:
        contrib_tasks = store.dataset_tasks(dataset_name) | store.rating_tasks(dataset_name)
    named = sorted(set(mos_tasks) | {t for t in contrib_tasks if t})
    has_unnamed = None in contrib_tasks
    return named, has_unnamed


def _resolve_active_task(task: str | None, named_tasks: list[str]) -> str | None:
    """The task to analyse: caller's choice if valid, else the first named task.

    ``None`` when there are no named tasks (a dataset annotated without sub-tasks).
    """
    if not named_tasks:
        return None
    return task if (task and task in named_tasks) else named_tasks[0]


def _dataset_root(workspace: Workspace, dataset_name: str) -> Path:
    return dataset_registry.manifest_path(dataset_name, workspace.data_dir).parent


def _reference_groups(manifest: dict) -> dict[str, list[dict]]:
    """Group manifest image entries by their shared reference (``metadata.stem``)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest.get("images", []):
        stem = entry.get("metadata", {}).get("stem")
        if stem is not None:
            groups[str(stem)].append(entry)
    return groups


def _items_for_stem(entries: list[dict]) -> list[str]:
    """Rankable item ids for one reference group: each variant + the reference."""
    items = [e["id"] for e in entries]
    for e in entries:
        if e.get("references"):
            items.append(f"{_REF_PREFIX}{e['metadata']['stem']}")
            break
    return items


def eligible_annotation_datasets(workspace: Workspace) -> list[dict]:
    """Prepared datasets that have at least one reference group of ≥2 items."""
    out: list[dict] = []
    for ds in discover_datasets(workspace):
        if not ds["prepared"]:
            continue
        try:
            manifest = load_manifest(dataset_registry.manifest_path(ds["name"], workspace.data_dir))
        except Exception:
            continue
        groups = _reference_groups(manifest)
        if any(len(_items_for_stem(v)) >= 2 for v in groups.values()):
            out.append({"name": ds["name"], "tasks": ds["annotation_tasks"]})
    return out


def annotation_tasks_for(workspace: Workspace, dataset_name: str | None) -> list[str]:
    """Sub-tasks offered for a dataset ('random'/empty → none, decided per draw)."""
    if not dataset_name or dataset_name == "random":
        return []
    try:
        manifest = load_manifest(dataset_registry.manifest_path(dataset_name, workspace.data_dir))
    except Exception:
        return []
    return annotation_tasks_in_manifest(manifest)


def start_ranking(
    workspace: Workspace,
    sessions: dict[str, dict],
    *,
    user_email: str,
    dataset_choice: str,
    task_choice: str,
    rng: random.Random | None = None,
) -> dict:
    """Draw a random reference set and create a new :class:`RankingSession`.

    ``dataset_choice`` / ``task_choice`` may be ``"random"``; they are remembered
    so the *next* set after this one is drawn the same way. Returns
    ``{"error": ...}`` on failure or ``{"session_id": ...}`` on success.
    """
    rng = rng or random.Random()
    eligible = {e["name"] for e in eligible_annotation_datasets(workspace)}
    if not eligible:
        return {"error": "No datasets with rankable variants are prepared yet."}

    dataset_name = dataset_choice
    if dataset_name in (None, "", "random"):
        dataset_name = rng.choice(sorted(eligible))
    elif dataset_name not in eligible:
        return {"error": "That dataset isn't available for annotation."}

    manifest = load_manifest(dataset_registry.manifest_path(dataset_name, workspace.data_dir))
    groups = _reference_groups(manifest)
    rankable = {
        stem: items
        for stem, entries in groups.items()
        if len(items := _items_for_stem(entries)) >= 2
    }
    if not rankable:
        return {"error": "That dataset has no rankable reference sets."}

    stem = rng.choice(sorted(rankable))
    item_ids = rankable[stem]

    tasks = annotation_tasks_in_manifest(manifest)
    chosen_task: str | None = None
    if tasks:
        chosen_task = task_choice if task_choice in tasks else rng.choice(sorted(tasks))

    session = RankingSession(
        items=item_ids,
        dataset=dataset_name,
        reference=stem,
        task=chosen_task,
        user_email=normalize_email(user_email),
    )
    sid = uuid.uuid4().hex
    sessions[sid] = {
        "session": session,
        "dataset_choice": dataset_choice,
        "task_choice": task_choice,
    }
    return {"error": None, "session_id": sid}


def compare_view(sessions: dict[str, dict], session_id: str) -> dict | None:
    """Build the template context for the current comparison, or ``None``."""
    wrapper = sessions.get(session_id)
    if wrapper is None:
        return None
    session: RankingSession = wrapper["session"]
    pair = session.current_pair()
    if pair is None:
        return None
    prompt = "Click on the image that has higher quality"
    if session.task:
        prompt += f" for the task: {session.task.replace('_', ' ')}"
    placed = len(session.sorted_ids)
    return {
        "session_id": session_id,
        "dataset": session.dataset,
        "prompt": prompt,
        "left_item": pair["left"],
        "right_item": pair["right"],
        "placed": placed,
        "n_items": session.n_items,
        "n_comparisons": len(session.comparisons),
    }


def submit_choice(
    workspace: Workspace,
    sessions: dict[str, dict],
    session_id: str,
    winner: str,
) -> dict:
    """Record a choice. Returns ``{"status": "continue"|"completed"|"missing", ...}``.

    On completion the ranking is persisted, a fresh set is drawn with the same
    dataset/task choice, and the new ``session_id`` is returned.
    """
    wrapper = sessions.get(session_id)
    if wrapper is None:
        return {"status": "missing"}
    session: RankingSession = wrapper["session"]
    try:
        session.submit(winner)
    except ValueError:
        # Stale/duplicate click that no longer matches the current pair — ignore.
        return {"status": "continue", "session_id": session_id}

    if not session.is_complete:
        return {"status": "continue", "session_id": session_id}

    # Persist the finished ranking and immediately draw the next set.
    with AnnotatorStore(workspace.annotations_path) as store:
        store.add_session(session.result())
        count = store.count_sessions(user_email=session.user_email)
    sessions.pop(session_id, None)

    nxt = start_ranking(
        workspace,
        sessions,
        user_email=session.user_email,
        dataset_choice=wrapper["dataset_choice"],
        task_choice=wrapper["task_choice"],
    )
    return {
        "status": "completed",
        "count": count,
        "next_session_id": nxt.get("session_id"),
        "error": nxt.get("error"),
    }


# ---------------------------------------------------------------------------
# Single-stimulus MOS rating (quantitative 1-4)
# ---------------------------------------------------------------------------


def draw_rating_item(
    workspace: Workspace,
    dataset_choice: str,
    *,
    rng: random.Random | None = None,
) -> dict:
    """Pick a random reconstruction + its reference + a random task to rate.

    The task is always randomised (no sub-task selection) so the annotator must
    read the prompt each time. Returns ``{"error": ...}`` or a draw dict with
    ``dataset``, ``image_id`` (variant), ``reference_item`` and ``task``.
    """
    rng = rng or random.Random()
    eligible = {e["name"] for e in eligible_annotation_datasets(workspace)}
    if not eligible:
        return {"error": "No datasets with rankable variants are prepared yet."}

    dataset_name = dataset_choice
    if dataset_name in (None, "", "random"):
        dataset_name = rng.choice(sorted(eligible))
    elif dataset_name not in eligible:
        return {"error": "That dataset isn't available for rating."}

    manifest = load_manifest(dataset_registry.manifest_path(dataset_name, workspace.data_dir))
    groups = _reference_groups(manifest)
    # Variants only (drop the reference pseudo-item); a stem needs a reference too.
    rateable: dict[str, list[str]] = {}
    for stem, entries in groups.items():
        items = _items_for_stem(entries)
        variants = [i for i in items if not i.startswith(_REF_PREFIX)]
        has_ref = any(i.startswith(_REF_PREFIX) for i in items)
        if variants and has_ref:
            rateable[stem] = variants
    if not rateable:
        return {"error": "That dataset has no reconstructions with a reference to rate."}

    stem = rng.choice(sorted(rateable))
    image_id = rng.choice(rateable[stem])
    tasks = annotation_tasks_in_manifest(manifest)
    task = rng.choice(sorted(tasks)) if tasks else None

    return {
        "error": None,
        "dataset": dataset_name,
        "image_id": image_id,
        "reference_item": f"{_REF_PREFIX}{stem}",
        "task": task,
    }


def record_rating(
    workspace: Workspace,
    *,
    user_email: str,
    dataset: str,
    image_id: str,
    task: str | None,
    score: int,
) -> int:
    """Persist one 1-4 rating; returns the annotator's total rating count."""
    score = max(1, min(4, int(score)))
    email = normalize_email(user_email)
    with AnnotatorStore(workspace.annotations_path) as store:
        store.add_rating(
            {
                "user_email": email,
                "dataset": dataset,
                "image_id": image_id,
                "task": task,
                "score": score,
            }
        )
        return sum(1 for r in store.ratings() if r.get("user_email") == email)


def resolve_item_path(workspace: Workspace, dataset_name: str, item_id: str) -> Path | None:
    """Map an annotation item id back to its on-disk ``.npy`` (via the manifest)."""
    try:
        manifest = load_manifest(dataset_registry.manifest_path(dataset_name, workspace.data_dir))
    except Exception:
        return None
    root = _dataset_root(workspace, dataset_name)

    rel: str | None = None
    if item_id.startswith(_REF_PREFIX):
        stem = item_id[len(_REF_PREFIX) :]
        for e in manifest.get("images", []):
            if str(e.get("metadata", {}).get("stem")) == stem and e.get("references"):
                refs = e["references"]
                ref = refs.get("ground_truth") or next(iter(refs.values()))
                rel = ref["path"]
                break
    else:
        for e in manifest.get("images", []):
            if e["id"] == item_id:
                rel = e["path"]
                break

    if rel is None:
        return None
    # Defensive: the resolved path must stay within the dataset root.
    path = (root / rel).resolve()
    if root.resolve() not in path.parents:
        return None
    return path


def render_item_png(path: Path) -> bytes:
    """Load a stored ``.npy`` and encode it as a display PNG (min-max to 8-bit)."""
    from PIL import Image

    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    lo, hi = float(arr.min()), float(arr.max())
    scaled = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    img = (scaled * 255.0).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Data export — bundle cached metric results + annotations as downloadable JSON
# ---------------------------------------------------------------------------

# Runtime-only fields not useful to downstream consumers of an export.
_EXPORT_DROP_FIELDS = {"wall_time_s", "device", "from_cache"}


def _anonymise(emails: set[str]) -> dict[str, str]:
    """Stable pseudonyms for annotators (sorted), so exports carry no PII."""
    return {email: f"annotator_{i + 1:03d}" for i, email in enumerate(sorted(emails))}


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/Inf) with None for JSON output.

    Cached metric values can be NaN/Inf (e.g. a measure on degenerate input, or an
    infinite PSNR), which ``json.dumps(allow_nan=False)`` — used by the response —
    rejects. Mapping them to null keeps the export serialisable; consumers read it
    back as a missing value (e.g. NaN in pandas).
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _manifest_annotations(manifest: dict) -> dict[str, dict[str, dict]]:
    """Expert ratings embedded in the manifest: ``{task: {image_id: {kind: value}}}``."""
    out: dict[str, dict[str, dict]] = {}
    for entry in manifest.get("images", []):
        ann = entry.get("metadata", {}).get("annotations", {})
        for task, kinds in ann.items():
            out.setdefault(task, {})[entry["id"]] = kinds
    return out


def _bt_blocks(workspace: Workspace, dataset_name: str, manifest: dict) -> list[dict]:
    """Bradley-Terry scores per task (named tasks + the unnamed one, if any)."""
    named, has_unnamed = _available_tasks(workspace, dataset_name, manifest or {"images": []})
    tasks: list[str | None] = [*named]
    if has_unnamed:
        tasks.append(None)

    blocks: list[dict] = []
    for task in tasks:
        bt = bt_scores_for_dataset_task(workspace, dataset_name, task)
        if not bt["scores"]:
            continue
        blocks.append(
            {
                "task": task,
                "n_comparisons": bt["n_comparisons"],
                "n_sets": bt["n_sets"],
                "scores": {
                    img: {"beta": round(beta, 6), "se": round(bt["se"][img], 6)}
                    for img, beta in bt["scores"].items()
                },
            }
        )
    return blocks


def export_payload(workspace: Workspace, dataset_name: str | None = None) -> dict:
    """Bundle cached metric results + annotations into a self-contained dict.

    With ``dataset_name`` None (or "all") every dataset that has any data is
    included; otherwise only the named one. Annotators are anonymised so the
    bundle is safe to publish (e.g. to Zenodo).
    """
    all_names = [d["name"] for d in discover_datasets(workspace)]
    if dataset_name and dataset_name != "all":
        names = [dataset_name] if dataset_name in all_names else []
    else:
        names = all_names

    with AnnotatorStore(workspace.annotations_path) as astore:
        all_sessions = astore.sessions()
    anon = _anonymise({s.get("user_email") for s in all_sessions if s.get("user_email")})

    records_by_ds: dict[str, list] = {}
    if workspace.store_path.exists():
        with ResultStore(workspace.store_path) as rstore:
            for name in names:
                records_by_ds[name] = rstore.records_for_dataset(name)

    datasets_out: list[dict] = []
    for name in names:
        ds = get_dataset(workspace, name)
        manifest: dict = {}
        if ds and ds["prepared"]:
            manifest = load_manifest(dataset_registry.manifest_path(name, workspace.data_dir))

        measure_results = [
            {k: v for k, v in r.model_dump(mode="json").items() if k not in _EXPORT_DROP_FIELDS}
            for r in records_by_ds.get(name, [])
        ]
        expert_annotations = _manifest_annotations(manifest)
        pairwise_annotations = [
            {
                "annotator": anon.get(s.get("user_email")),
                "task": s.get("task"),
                "reference": s.get("reference"),
                "order": s.get("order"),
                "comparisons": s.get("comparisons"),
                "n_comparisons": s.get("n_comparisons"),
                "created": s.get("created"),
            }
            for s in all_sessions
            if s.get("dataset") == name
        ]
        bradley_terry = _bt_blocks(workspace, name, manifest)

        # Skip datasets with nothing to export (e.g. not-yet-downloaded ones).
        if not (measure_results or expert_annotations or pairwise_annotations or bradley_terry):
            continue

        datasets_out.append(
            {
                "name": name,
                "version": ds["version"] if ds else None,
                "n_measure_results": len(measure_results),
                "measure_results": measure_results,
                "expert_annotations": expert_annotations,
                "pairwise_annotations": pairwise_annotations,
                "bradley_terry": bradley_terry,
            }
        )

    return _json_safe(
        {
            "tool": {"name": "ipasc_iqa", "version": __version__},
            "exported": datetime.now(UTC).isoformat(),
            "anonymised": True,
            "n_annotators": len(anon),
            "datasets": datasets_out,
        }
    )
