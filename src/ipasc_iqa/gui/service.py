"""Function layer for the GUI — pure logic, no web framework.

Everything the UI needs is expressed here as functions returning plain dicts /
dataclasses. Routes are thin adapters over these; tests exercise these directly.

Datasets are *adapter-driven* (see :mod:`ipasc_iqa.datasets.registry`): only
datasets with a registered adapter are offered, and they are downloaded/converted
on demand into a managed ``data_dir``. The workspace's ``results/`` folder stores
run outputs (``<run_id>.json``).
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import threading
import time
import uuid
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from collections import defaultdict
from urllib.parse import urlencode
import os

import numpy as np

from ipasc_iqa import __version__
from ipasc_iqa.analysis.aggregate import (
    measure_directions,
    measure_stats,
    rich_measure_summary,
)
from ipasc_iqa.analysis.correlation import correlate_measures_with_annotations
from ipasc_iqa.analysis.manifest_query import (
    annotation_tasks_in_manifest,
    annotation_values_from_manifest,
    annotator_ratings_from_manifest,
    comparison_tasks_in_manifest,
    comparisons_from_manifest,
    groups_from_manifest,
    ranking_observations_from_manifest,
)
from ipasc_iqa.analysis.image_explorer import image_group_detail, image_groups_from_manifest
from ipasc_iqa.analysis.mos import composite_mos
from ipasc_iqa.analysis.scatter import build_scatter
from ipasc_iqa.contrib.bradley_terry import fit_bradley_terry
from ipasc_iqa.contrib.ranking import RankingSession
from ipasc_iqa.contrib.store import AnnotatorStore
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
    def model_cache_dir(self) -> Path:
        """Writable cache for third-party model weights used by deep metrics.

        Some libraries default to ~/.cache or /opt/.cache in web deployments.
        The GUI should instead use a workspace-owned directory.
        """
        return self.results_dir / "model_cache"

    @property
    def annotations_path(self) -> Path:
        """diskcache directory for contributed annotations (rankings + ratings)."""
        return self.root / "annotations"

    @property
    def users_path(self) -> Path:
        """diskcache directory for accounts (kept separate from annotations).

        Separated so deleting/exporting the annotations data leaves the user
        accounts untouched.
        """
        return self.root / "users"

    def ensure(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def resolve(root: Path | str, data_dir: Path | str | None = None) -> Workspace:
        return Workspace(
            root=Path(root).resolve(),
            data_dir=Path(data_dir).resolve() if data_dir else default_data_dir(),
        )

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Manifest cache
# ---------------------------------------------------------------------------
# Manifests are read on nearly every request (drawing an annotation item, serving
# an image, listing datasets) and are large enough that re-parsing the JSON each
# time dominated some endpoints. They only change when a dataset is (re)prepared,
# so cache the parsed dict keyed by path + mtime + size: a stale entry is
# impossible, and a fresh conversion invalidates itself.

_MANIFEST_CACHE: dict[str, tuple[tuple[int, int], dict]] = {}
_MANIFEST_LOCK = threading.Lock()


def load_manifest_cached(path: Path | str) -> dict:
    """:func:`load_manifest`, memoised on the file's (mtime_ns, size)."""
    path = Path(path)
    stat = path.stat()
    stamp = (stat.st_mtime_ns, stat.st_size)
    key = str(path)
    with _MANIFEST_LOCK:
        hit = _MANIFEST_CACHE.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
    manifest = load_manifest(path)
    with _MANIFEST_LOCK:
        _MANIFEST_CACHE[key] = (stamp, manifest)
    return manifest


def dataset_manifest(workspace: Workspace, dataset_name: str) -> dict:
    """The (cached) manifest of a prepared dataset in this workspace."""
    return load_manifest_cached(dataset_registry.manifest_path(dataset_name, workspace.data_dir))


def open_store(workspace: Workspace) -> AnnotatorStore:
    """The process-wide annotator store for a workspace (see AnnotatorStore)."""
    return AnnotatorStore(workspace.annotations_path, workspace.users_path)


def torch_device_status() -> dict:
    """Return the PyTorch execution backend visible to the GUI server."""
    try:
        import torch
    except Exception as exc:
        return {
            "available": False,
            "backend": "none",
            "label": "PyTorch not available",
            "message": (
                "PyTorch is not available in this environment. "
                "PyTorch-based metrics cannot run until PyTorch is installed."
            ),
            "detail": str(exc),
        }

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )

    if cuda_available:
        name = torch.cuda.get_device_name(0)
        return {
            "available": True,
            "backend": "cuda",
            "label": f"GPU available: CUDA ({name})",
            "message": "PyTorch-based metrics can use CUDA GPU acceleration.",
            "detail": f"torch {torch.__version__}, CUDA {torch.version.cuda}",
        }

    if mps_available:
        return {
            "available": True,
            "backend": "mps",
            "label": "GPU available: Apple MPS",
            "message": "PyTorch-based metrics can use Apple Silicon GPU acceleration.",
            "detail": f"torch {torch.__version__}",
        }

    return {
        "available": False,
        "backend": "cpu",
        "label": "No compatible GPU backend detected",
        "message": (
            "PyTorch-based metrics will run on CPU and may be slower on large datasets."
        ),
        "detail": f"torch {torch.__version__}",
    }

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


def _selected_measure_records(
    run: RunResults,
    manifest: dict,
    dataset_name: str,
    measures: list[str] | None,
) -> tuple[list, list[str]]:
    """Filter run records by dataset capabilities and the selected measures.

    This is the shared measure-selection logic for all analysis panels. The
    Metric overview, Human agreement, and Image explorer should all see the
    same selected/usable metric set.
    """
    measure_meta = list_measures()
    caps = _capabilities_from_manifest(manifest)
    usable = [m["name"] for m in measure_meta if _measure_usable(m, caps)]
    chosen = [m for m in (measures or usable) if m in usable]

    selected_records = [
        rec
        for rec in run.results
        if rec.dataset == dataset_name and rec.measure in chosen
    ]

    return selected_records, chosen


# Human-readable labels for the measure-type enum (used in the analysis UI).
_MEASURE_TYPE_LABELS = {
    MeasureType.FULL_REFERENCE: "full-reference",
    MeasureType.REDUCED_REFERENCE: "reduced-reference",
    MeasureType.NO_REFERENCE: "no-reference",
}


def _format_value_range(value_range: tuple[float, float] | None) -> str | None:
    """Render a measure's value range as ``[lo, hi]`` (∞ for infinities), or None."""
    if not value_range:
        return None

    def _fmt(x: float) -> str:
        if x == float("inf"):
            return "∞"
        if x == float("-inf"):
            return "−∞"
        return f"{x:g}"

    lo, hi = value_range
    return f"[{_fmt(lo)}, {_fmt(hi)}]"


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
                manifest = dataset_manifest(workspace, name)
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
    workspace: Workspace,
    datasets: list[str],
    measures: list[str],
    *,
    run_id: str | None = None,
) -> None:
    """Persist the dataset/measure selection from the most recent benchmark run."""
    workspace.ensure()
    payload = {"datasets": list(datasets), "measures": list(measures)}
    if run_id is not None:
        payload["run_id"] = run_id
    _last_selection_path(workspace).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def load_last_selection(workspace: Workspace) -> dict:
    """Most recent benchmark selection, or empty lists if none was recorded."""
    path = _last_selection_path(workspace)
    if not path.exists():
        return {"datasets": [], "measures": [], "run_id": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"datasets": [], "measures": [], "run_id": None}
    return {
        "datasets": list(data.get("datasets", [])),
        "measures": list(data.get("measures", [])),
        "run_id": data.get("run_id"),
    }


def get_dataset(workspace: Workspace, name: str) -> dict | None:
    for ds in discover_datasets(workspace):
        if ds["name"] == name:
            return ds
    return None


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def _can_write_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".ipasc_iqa_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def ensure_writable_model_cache(workspace: Workspace) -> None:
    """Use a workspace-owned cache for third-party metric model files.

    This avoids deployments where libraries try to write model weights under
    non-writable locations such as /opt/.cache.
    """
    cache_dir = workspace.model_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    defaults = {
        "XDG_CACHE_HOME": cache_dir / "xdg",
        "TORCH_HOME": cache_dir / "torch",
        "HF_HOME": cache_dir / "hf",
    }

    for key, fallback in defaults.items():
        current = os.environ.get(key)
        if current is None or not _can_write_dir(Path(current)):
            fallback.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(fallback)

def run_benchmark(
    workspace: Workspace,
    dataset_names: list[str],
    measure_names: list[str],
    *,
    reference_preference: list[str] | None = None,
    device: str = "auto",
    on_progress: Callable[[str, str], None] | None = None,
    on_plan: Callable[[dict[str, int], list[str]], None] | None = None,
    on_skip: Callable[[str, str, str], None] | None = None,
) -> str:
    """Download/prepare datasets as needed, run evaluation, persist results.

    Results are written through a persistent, newest-wins store that also serves
    as the cache: already-computed (dataset, image, measure) pairs are reused
    rather than recomputed, and partial runs accumulate instead of overwriting.

    ``on_plan`` is called once the datasets are prepared with
    ``({dataset: n_images}, [measure_name, ...])`` so a caller can size a progress
    display; ``on_progress(dataset, measure)`` then fires once per evaluated pair.
    """
    workspace.ensure()
    ensure_writable_model_cache(workspace)

    datasets = [
        dataset_registry.instantiate(name, workspace.data_dir, download=True)
        for name in dataset_names
    ]

    measures = [measure_registry.instantiate(m, device=device) for m in measure_names]

    if reference_preference is None:
        reference_preference = ["ground_truth", "pseudo_reference"]
    if on_plan is not None:
        on_plan({ds.name: len(ds) for ds in datasets}, [m.metadata.name for m in measures])

    with ResultStore(workspace.store_path) as store:
        run = run_evaluation(
            datasets,
            measures,
            reference_preference=reference_preference,
            cache=store,
            progress=False,
            on_progress=on_progress,
            on_skip=on_skip,
        )
        # Pre-compute per-dataset mean/std over the merged per-image results and
        # persist them, so reporting doesn't have to recompute from scratch.
        for name in dataset_names:
            _write_dataset_stats(workspace, name, store.records_for_dataset(name))

    # Keep a per-run audit file alongside the merged store.
    out_path = workspace.results_dir / f"{run.run_id}.json"
    out_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    save_last_selection(workspace, dataset_names, measure_names, run_id=run.run_id)

    # New measure values → invalidate cached analysis/standings/scatter views.
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        store.bump_results_epoch()
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


def _latest_selected_run(workspace: Workspace, dataset_name: str) -> RunResults | None:
    """Return the latest run associated with the saved GUI selection, if relevant."""
    last = load_last_selection(workspace)
    run_id = last.get("run_id")
    if not run_id or dataset_name not in last.get("datasets", []):
        return None

    path = workspace.results_dir / f"{run_id}.json"
    if not path.exists():
        return None

    try:
        return load_results(path)
    except Exception:
        return None


def _analysis_run_issues(
    workspace: Workspace,
    dataset_name: str,
    selected_measures: list[str],
    selected_records: list,
) -> list[dict]:
    """Explain selected metrics that are missing or incomplete in analysis.

    The analysis tables read from the merged result store, which only contains
    successful metric values. This helper re-attaches the skipped/failure audit
    from the latest run so failed metrics do not disappear silently.
    """
    produced = {rec.measure for rec in selected_records if rec.dataset == dataset_name}
    selected = set(selected_measures)

    latest = _latest_selected_run(workspace, dataset_name)

    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if latest is not None:
        for skipped in latest.skipped:
            if skipped.dataset != dataset_name:
                continue
            if skipped.measure not in selected:
                continue
            grouped[skipped.measure][skipped.reason] += 1

    issues: list[dict] = []
    for measure in sorted(selected):
        reasons = grouped.get(measure, {})
        missing_completely = measure not in produced

        if not reasons and not missing_completely:
            continue

        if reasons:
            reason_items = [
                {"reason": reason, "count": count}
                for reason, count in sorted(
                    reasons.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
            count = sum(reasons.values())
        else:
            reason_items = [
                {
                    "reason": (
                        "Selected for the latest run, but no cached values are available "
                        "for this dataset/measure. The run may have been interrupted, "
                        "or these values may come from an older cache state."
                    ),
                    "count": 1,
                }
            ]
            count = 0

        issues.append(
            {
                "dataset": dataset_name,
                "measure": measure,
                "count": count,
                "missing_completely": missing_completely,
                "reasons": reason_items,
            }
        )

    return issues

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
    dataset_display_name = ds["display_name"] if ds is not None else dataset_name
    if ds is None:
        return {
            "dataset": dataset_name,
            "dataset_display_name": dataset_display_name,
            "tasks": [],
            "error": "dataset not found",
            **_empty,
        }
    if not ds["prepared"]:
        return {
            "dataset": dataset_name,
            "dataset_display_name": dataset_display_name,
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
            "dataset_display_name": dataset_display_name,
            "tasks": [],
            "error": "no benchmark run found for this dataset — run a benchmark first",
            **_empty,
        }
    
    measures_meta = list_measures()

    directions = {
        measure["name"]: bool(measure["higher_is_better"])
        for measure in measures_meta
    }

    bibtex_by_measure = {
        measure["name"]: measure.get("bibtex")
        for measure in measures_meta
    }

    manifest = dataset_manifest(workspace, dataset_name)

    # Tasks can come from manifest MOS *or* from contributed pairwise comparisons,
    # so a dataset with no expert ratings is still analysable once BT data exists.
    task = task or None
    task_names, has_unnamed = _available_tasks(workspace, dataset_name, manifest)

    # Apply the same usable/selected measure filter to every analysis panel.
    selected_records, chosen = _selected_measure_records(
        run,
        manifest,
        dataset_name,
        measures,
    )

    run_issues = _analysis_run_issues(
        workspace,
        dataset_name,
        chosen,
        selected_records,
    )

    metric_summary = _annotate_metric_summary_bibtex(
        rich_measure_summary(
            selected_records,
            higher_is_better_by_measure=directions,
            top=5,
        ),
        measures_meta,
    )

    if not task_names and not has_unnamed:
        return {
            "dataset": dataset_name,
            "dataset_display_name": dataset_display_name,
            "description": ds.get("description"),
            "bibtex": ds.get("bibtex"),
            "run_id": run.run_id,
            "tasks": [],
            "all_tasks": [],
            "active_task": None,
            "meta": None,
            "sources": [],
            "active_source": source,
            "metric_summary": metric_summary,
            "run_issues": run_issues,
            "human_analysis_message": (
                "No expert annotations or contributed comparisons are available for this dataset."
            ),
            "error": None,
        }

    # Resolve the active task. A dataset with no sub-tasks but contributed
    # comparisons has a single unnamed task (active_task=None, no dropdown).
    active_task = _resolve_active_task(task, task_names)

    # Resolve the human-score source (MOS vs contributed Bradley-Terry).
    mos_values = mos_values_for_dataset_task(workspace, dataset_name, active_task, manifest)
    bt = bt_scores_for_dataset_task(workspace, dataset_name, active_task, manifest)
    sources: list[dict] = []
    if mos_values:
        sources.append({"key": "mos", "label": _SOURCE_LABELS["mos"]})
    if bt["scores"]:
        sources.append({"key": "bt", "label": _SOURCE_LABELS["bt"]})
    source_keys = {s["key"] for s in sources}
    active_source = source if source in source_keys else (sources[0]["key"] if sources else "mos")
    values = bt["scores"] if active_source == "bt" else mos_values

    rows = correlate_measures_with_annotations(run, values, measures=chosen, dataset=dataset_name)

    # Keep the raw correlations untouched, but add direction-adjusted versions
    # for the Human agreement view. Human scores are oriented as higher = better;
    # lower-is-better IQA measures therefore require a sign flip.
    stats_by_measure = dataset_measure_stats(workspace, dataset_name)
    row_dicts = []
    for r in rows:
        d = r.as_dict()
        higher_is_better = directions.get(r.measure, True)
        direction = 1.0 if higher_is_better else -1.0

        d["higher_is_better"] = higher_is_better
        d["srcc_agreement"] = None if r.srcc is None else direction * r.srcc
        d["krcc_agreement"] = None if r.krcc is None else direction * r.krcc
        d["bibtex"] = bibtex_by_measure.get(r.measure)

        stat = stats_by_measure.get(r.measure, {})
        d["mean"] = stat.get("mean")
        d["std"] = stat.get("std")
        row_dicts.append(d)

    # Human agreement is ranked by the signed, direction-adjusted SRCC: strong
    # positive agreement first, genuine inverse rankings last. Missing values
    # are kept at the bottom.
    row_dicts.sort(
        key=lambda d: (
            d["srcc_agreement"] is None,
            -d["srcc_agreement"] if d["srcc_agreement"] is not None else 0.0,
            d["measure"],
        )
    )

    return {
        "dataset": dataset_name,
        "dataset_display_name": dataset_display_name,
        "description": ds.get("description"),
        "bibtex": ds.get("bibtex"),
        "run_id": run.run_id,
        "tasks": [{"task": active_task, "rows": row_dicts}],
        "all_tasks": task_names,
        "active_task": active_task,
        "meta": task_annotation_summary(workspace, dataset_name, active_task, manifest),
        "sources": sources,
        "active_source": active_source,
        "metric_summary": metric_summary,
        "run_issues": run_issues,
        "human_analysis_message": None,
        "error": None,
    }

def _annotate_metric_summary_bibtex(metric_summary: dict, measures_meta: list[dict]) -> dict:
    """Attach BibTeX citations to Metric overview statistics and examples."""
    bibtex_by_measure = {
        measure["name"]: measure.get("bibtex")
        for measure in measures_meta
    }

    for stat in metric_summary.get("statistics", []):
        measure_name = stat.get("measure")
        if not measure_name:
            continue
        base_measure = measure_name.split(".", 1)[0]
        stat["bibtex"] = (
            bibtex_by_measure.get(measure_name)
            or bibtex_by_measure.get(base_measure)
        )

    for measure_name, groups in metric_summary.get("examples", {}).items():
        base_measure = measure_name.split(".", 1)[0]
        groups["bibtex"] = (
            bibtex_by_measure.get(measure_name)
            or bibtex_by_measure.get(base_measure)
        )

    return metric_summary

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
      task. A contributed ranking session contributes one observation per variant
      it ranks; each pairwise comparison (published in the manifest or
      contributed) counts as one ranking observation.
    - ``avg_ratings`` / ``avg_rankings``: those totals divided by ``n_images``;
      ``None`` if the dataset has no images.
    """
    n_images = len(manifest.get("images", []))

    # Ratings: per-image annotator counts (manifest per-rater + contributed).
    rated = _mos_annotator_scores(workspace, dataset_name, task, manifest)
    n_ratings = sum(len(raters) for img, raters in rated.items() if not img.startswith(_REF_PREFIX))

    # Rankings: published ranking placements embedded in the manifest (one
    # observation per non-reference variant per annotator set), plus contributed
    # variant placements (approved contributors only, to match the filtered
    # result aggregates).
    n_rankings = ranking_observations_from_manifest(manifest, task)
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        sessions = store.sessions()
        approved = store.approved_uids()
    for s in sessions:
        if s.get("dataset") != dataset_name or s.get("task") != task:
            continue
        if s.get("user_uid") not in approved:
            continue
        n_rankings += sum(1 for i in s.get("order", []) if not i.startswith(_REF_PREFIX))

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

    manifest = dataset_manifest(workspace, dataset_name)
    task = task or None
    task_names, has_unnamed = _available_tasks(workspace, dataset_name, manifest)
    if not task_names and not has_unnamed:
        return {"error": "no expert annotations or contributed comparisons to analyse yet"}
    active_task = _resolve_active_task(task, task_names)

    task_name = str(active_task).strip().lower() if active_task is not None else None
    task_label = (
        "overall quality"
        if task_name in {None, "all", "overall", "overall_quality", "overall quality", "quality"}
        else str(active_task).replace("_", " ")
    )
    if source == "bt":
        human_values = bt_scores_for_dataset_task(
            workspace, dataset_name, active_task, manifest
        )["scores"]
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
    direction = 1.0 if meta.higher_is_better else -1.0
    srcc_agreement = None if srcc is None else direction * srcc
    krcc_agreement = None if krcc is None else direction * krcc

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
        "srcc_agreement": srcc_agreement,
        "krcc_agreement": krcc_agreement,
        "n": len(data.xs),
        "mean": stats.get("mean"),
        "std": stats.get("std"),
        "higher_is_better": meta.higher_is_better,
        "bibtex": meta.bibtex,
        "description": meta.description,
        "measure_type": _MEASURE_TYPE_LABELS.get(meta.measure_type, meta.measure_type.name.lower()),
        "value_range": _format_value_range(meta.value_range),
    }

def image_detail_for_dataset(
    workspace: Workspace,
    dataset_name: str,
    image_id: str,
) -> dict:
    """Return image-level metric scores for the Metric overview detail dialog."""

    ds = get_dataset(workspace, dataset_name)
    if ds is None or not ds["prepared"]:
        return {
            "error": "dataset not available",
            "dataset": dataset_name,
            "image_id": image_id,
            "scores": [],
        }

    run = _merged_run(workspace, dataset_name)
    if run is None:
        return {
            "error": "no benchmark run found for this dataset",
            "dataset": dataset_name,
            "image_id": image_id,
            "scores": [],
        }

    records = [
        rec
        for rec in run.results
        if rec.dataset == dataset_name and rec.image_id == image_id
    ]

    if not records:
        return {
            "error": f"no metric results found for image '{image_id}'",
            "dataset": dataset_name,
            "image_id": image_id,
            "scores": [],
        }

    directions = {
        measure["name"]: bool(measure["higher_is_better"])
        for measure in list_measures()
    }

    distributions: dict[str, list[float]] = defaultdict(list)

    for rec in run.results:
        if rec.dataset != dataset_name:
            continue

        if rec.value is not None:
            distributions[rec.measure].append(float(rec.value))

        if rec.sub_scores:
            for sub_name, sub_value in rec.sub_scores.items():
                if sub_value is not None:
                    distributions[f"{rec.measure}.{sub_name}"].append(float(sub_value))

    def relative_position(key: str, value: float, *, higher_is_better: bool) -> dict:
        """Return dataset-relative quality position for a score.

        quality_position = 0 means worst value for this metric in the dataset.
        quality_position = 1 means best value for this metric in the dataset.

        For higher-is-better metrics, high numeric values are better.
        For lower-is-better metrics, low numeric values are better.
        """

        values = distributions.get(key, [])
        if not values:
            return {
                "relative_position": None,
                "relative_percent": None,
                "quality_position": None,
                "quality_percent": None,
                "color_hue": None,
            }

        lo = min(values)
        hi = max(values)

        if hi == lo:
            numeric_rel = 0.5
        else:
            numeric_rel = (value - lo) / (hi - lo)

        numeric_rel = max(0.0, min(1.0, numeric_rel))

        if higher_is_better:
            quality_rel = numeric_rel
        else:
            quality_rel = 1.0 - numeric_rel

        return {
            "relative_position": numeric_rel,
            "relative_percent": round(numeric_rel * 100),
            "quality_position": quality_rel,
            "quality_percent": round(quality_rel * 100),
            "color_hue": round(quality_rel * 120),
        }

    scores: list[dict] = []

    for rec in sorted(records, key=lambda r: r.measure):
        higher_is_better = directions.get(rec.measure, True)

        if rec.value is not None:
            value = float(rec.value)
            scores.append(
                {
                    "measure": rec.measure,
                    "value": value,
                    "higher_is_better": higher_is_better,
                    "kind": "main",
                    **relative_position(rec.measure, value, higher_is_better=higher_is_better),
                }
            )

        if rec.sub_scores:
            for sub_name, sub_value in sorted(rec.sub_scores.items()):
                if sub_value is None:
                    continue

                key = f"{rec.measure}.{sub_name}"
                value = float(sub_value)

                scores.append(
                    {
                        "measure": key,
                        "value": value,
                        "higher_is_better": higher_is_better,
                        "kind": "sub_score",
                        **relative_position(key, value, higher_is_better=higher_is_better),
                    }
                )

    manifest = dataset_manifest(workspace, dataset_name)
    groups = image_groups_from_manifest(manifest)

    active_stem: str | None = None
    for group in groups:
        items = group.get("items", [])
        if any(item.get("image_id") == image_id for item in items):
            stem = group.get("stem")
            active_stem = str(stem) if stem is not None else None
            break

    image_explorer_url = None
    if active_stem is not None:
        image_explorer_url = "/api/analysis/image-explorer?" + urlencode(
            {
                "dataset": dataset_name,
                "stem": active_stem,
                "focus_image_id": image_id,
            }
        )

    return {
        "error": None,
        "dataset": dataset_name,
        "image_id": image_id,
        "stem": active_stem,
        "image_explorer_url": image_explorer_url,
        "scores": scores,
    }

def image_explorer_for_dataset(
    workspace: Workspace,
    dataset_name: str,
    *,
    stem: str | None = None,
    measures: list[str] | None = None,
    focus_image_id: str | None = None,
) -> dict:
    """Return grouped image variants and metric scores for the Image explorer tab."""

    ds = get_dataset(workspace, dataset_name)
    if ds is None:
        return {
            "error": "dataset not found",
            "dataset": dataset_name,
            "groups": [],
            "active_stem": None,
            "items": [],
        }

    if not ds["prepared"]:
        return {
            "error": "dataset not downloaded yet — run a benchmark on it first",
            "dataset": dataset_name,
            "groups": [],
            "active_stem": None,
            "items": [],
        }

    run = _merged_run(workspace, dataset_name)
    if run is None:
        return {
            "error": "no benchmark run found for this dataset — run a benchmark first",
            "dataset": dataset_name,
            "groups": [],
            "active_stem": None,
            "items": [],
        }

    manifest = dataset_manifest(workspace, dataset_name)

    measures_meta = list_measures()
    directions = {
        measure["name"]: bool(measure["higher_is_better"])
        for measure in measures_meta
    }
    bibtex_by_measure = {
        measure["name"]: measure.get("bibtex")
        for measure in measures_meta
    }

    selected_records, chosen = _selected_measure_records(
        run,
        manifest,
        dataset_name,
        measures,
    )

    detail = image_group_detail(
        manifest,
        selected_records,
        dataset_name,
        stem=stem,
        higher_is_better_by_measure=directions,
    )

    for group in detail.get("groups", []):
        items = group.get("items", [])
        preview_item = items[0] if items else None
        preview_image_id = preview_item.get("image_id") if preview_item else None

        group["preview_image_id"] = preview_image_id
        group["preview_image_url"] = (
            "/api/annotate/image?"
            + urlencode(
                {
                    "dataset": dataset_name,
                    "item": preview_image_id,
                }
            )
            if preview_image_id
            else None
        )
        group["is_active"] = group.get("stem") == detail.get("active_stem")

    for item in detail["items"]:
        item["image_url"] = "/api/annotate/image?" + urlencode(
            {
                "dataset": dataset_name,
                "item": item["image_id"],
            }
        )

    for row in detail.get("score_table", []):
        measure_name = row.get("measure")
        if not measure_name:
            continue
        base_measure = measure_name.split(".", 1)[0]
        row["bibtex"] = (
            bibtex_by_measure.get(measure_name)
            or bibtex_by_measure.get(base_measure)
        )

    return {
        "dataset": dataset_name,
        "selected_measures": chosen,
        "focus_image_id": focus_image_id,
        **detail,
    }


# ---------------------------------------------------------------------------
# Standings — IQA measures ranked by correlation, per dataset, per task
# ---------------------------------------------------------------------------


def standings(workspace: Workspace, source: str = "mos") -> dict:
    """Rank measures by direction-adjusted SRCC, per dataset+task.

    ``source`` selects the human scores to rank against: ``"mos"`` (expert
    absolute ratings) or ``"bt"`` (continuous Bradley-Terry scores from
    contributed pairwise comparisons). Raw SRCC/KRCC are preserved, while the
    ranking uses correlations oriented so that positive values always indicate
    agreement with increasing human quality.
    """
    source = "bt" if source == "bt" else "mos"
    # Measure metadata for the per-row info/citation popups (same fields the
    # detailed-analysis description block uses).
    meta_by_measure: dict[str, dict] = {}
    for name in measure_registry.list_measures():
        m = measure_registry.get(name).metadata
        meta_by_measure[name] = {
            "bibtex": m.bibtex,
            "description": m.description,
            "measure_type": _MEASURE_TYPE_LABELS.get(m.measure_type, m.measure_type.name.lower()),
            "value_range": _format_value_range(m.value_range),
            "higher_is_better": m.higher_is_better,
        }

    entries: list[dict] = []
    for ds in discover_datasets(workspace):
        if not ds["prepared"]:
            continue
        run = _merged_run(workspace, ds["name"])
        if run is None:
            continue
        manifest = dataset_manifest(workspace, ds["name"])

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
                values = bt_scores_for_dataset_task(workspace, ds["name"], task, manifest)["scores"]
            else:
                values = mos_values_for_dataset_task(workspace, ds["name"], task, manifest)
            if not values:
                continue
            rows = correlate_measures_with_annotations(run, values, dataset=ds["name"])
            agreement_rows: list[dict] = []
            for r in rows:
                if r.srcc is None:
                    continue
                meta = meta_by_measure.get(r.measure, {})
                higher_is_better = bool(meta.get("higher_is_better", True))
                direction = 1.0 if higher_is_better else -1.0
                agreement_rows.append(
                    {
                        **r.as_dict(),
                        **meta,
                        "srcc_agreement": direction * r.srcc,
                        "krcc_agreement": (
                            None if r.krcc is None else direction * r.krcc
                        ),
                    }
                )

            agreement_rows.sort(
                key=lambda d: (-d["srcc_agreement"], d["measure"])
            )
            ranked = [
                {"rank": i + 1, **row}
                for i, row in enumerate(agreement_rows)
            ]
            if ranked:
                entries.append(
                    {
                        "dataset": ds["name"],
                        "dataset_display_name": ds["display_name"],
                        "dataset_bibtex": ds.get("bibtex"),
                        "task": task or "overall",
                        "rows": ranked,
                    }
                )

    return {"entries": entries, "source": source}


# ---------------------------------------------------------------------------
# Contributed annotations — lightweight login + 2AFC pairwise ranking
# ---------------------------------------------------------------------------

_REF_PREFIX = "__ref__"


def results_epoch(workspace: Workspace) -> int:
    """Token that bumps whenever anything affecting computed results changes.

    The GUI keys its analysis/standings/scatter cache on this, so an unchanged
    view is served without recomputation.
    """
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        return store.results_epoch()


def account_for(workspace: Workspace, user: dict | None) -> dict | None:
    """Wrap the logged-in user with their contribution count for the templates."""
    if not user:
        return None
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        return {"user": user, "count": store.count_sessions(user_uid=user["uid"])}


def annotators_leaderboard(workspace: Workspace) -> list[dict]:
    """Per-annotator ranking/comparison totals for the contribute screen."""
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
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
    # Pool in contributed quantitative ratings (one annotator per uid, so a
    # rater's identity — and their MOS contribution — survives a profile edit).
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        approved = store.approved_uids()
        for r in store.ratings(dataset=dataset_name):
            uid = r.get("user_uid")
            if r.get("task") != task or r.get("image_id") is None or r.get("score") is None:
                continue
            # Hard-exclude pending/unapproved/deleted contributors from results.
            if not uid or uid not in approved:
                continue
            scores.setdefault(r["image_id"], {})[f"user:{uid}"] = float(r["score"])
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
        manifest = dataset_manifest(workspace, dataset_name)
    per_annotator = _mos_annotator_scores(workspace, dataset_name, task, manifest)
    if per_annotator:
        return composite_mos(per_annotator)
    if task is not None:
        return annotation_values_from_manifest(manifest, task, "zscore_mos")
    return {}


def _pooled_comparisons(
    workspace: Workspace, dataset_name: str, task: str | None, manifest: dict
) -> tuple[dict[str, list[tuple[str, str]]], dict[str, set[str]]]:
    """Pool a dataset/task's pairwise comparisons from both sources.

    Published comparisons embedded in the manifest (``human_comparisons``) and
    approved contributed comparisons from the annotator store are merged into
    ``(by_ref, items_by_ref)`` keyed by reference-set id — exactly how the MOS
    path pools manifest ratings with contributed ratings.
    """
    by_ref: dict[str, list[tuple[str, str]]] = defaultdict(list)
    items_by_ref: dict[str, set[str]] = defaultdict(set)

    # Published comparisons that ship with the dataset (always included).
    for ref, winner, loser in comparisons_from_manifest(manifest, task):
        by_ref[ref].append((winner, loser))
        items_by_ref[ref].update((winner, loser))

    # Contributed comparisons, restricted to approved contributors.
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        sessions = store.sessions()
        approved = store.approved_uids()
    for s in sessions:
        if s.get("dataset") != dataset_name or s.get("task") != task:
            continue
        if s.get("user_uid") not in approved:
            continue
        ref = s.get("reference")
        for c in s.get("comparisons", []):
            by_ref[ref].append((c["winner"], c["loser"]))
            items_by_ref[ref].update((c["winner"], c["loser"]))

    return by_ref, items_by_ref


def bt_scores_for_dataset_task(
    workspace: Workspace,
    dataset_name: str,
    task: str | None,
    manifest: dict | None = None,
) -> dict:
    """Per-image Bradley-Terry quality scores from pairwise comparisons.

    Comparisons — published (manifest) and approved contributed, pooled — are
    aggregated per reference set (across all annotators), and each set is fit
    independently and anchored at its reference image (beta = 0), so a variant's
    score is its *deficit below ground truth*, a scale comparable across
    reference sets. Reference (GT) items are excluded from the returned per-image
    mapping: full-reference IQMs are computed against the GT, so including it
    would trivially inflate their correlation (per Breger et al.).

    Returns ``{"scores": {image_id: beta}, "se": {...}, "n_comparisons", "n_sets"}``.
    """
    if manifest is None:
        manifest = dataset_manifest(workspace, dataset_name)
    by_ref, items_by_ref = _pooled_comparisons(workspace, dataset_name, task, manifest)

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
    if bt_scores_for_dataset_task(workspace, dataset_name, task, manifest)["scores"]:
        out.append({"key": "bt", "label": _SOURCE_LABELS["bt"]})
    return out


def human_score_sources(workspace: Workspace, dataset_name: str) -> list[dict]:
    """Sources for a dataset's first annotation task (for the initial page load)."""
    ds = get_dataset(workspace, dataset_name)
    if ds is None or not ds["prepared"]:
        return []
    manifest = dataset_manifest(workspace, dataset_name)
    # Consider MOS *and* pairwise tasks so a BT-only dataset still offers a source.
    named, _ = _available_tasks(workspace, dataset_name, manifest)
    first_task = named[0] if named else None
    return _sources_for(manifest, workspace, dataset_name, first_task)


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
    comparison_tasks = comparison_tasks_in_manifest(manifest)
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        contrib_tasks = store.dataset_tasks(dataset_name) | store.rating_tasks(dataset_name)
    named = sorted(set(mos_tasks) | set(comparison_tasks) | {t for t in contrib_tasks if t})
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
            manifest = dataset_manifest(workspace, ds["name"])
        except Exception:
            continue
        groups = _reference_groups(manifest)
        if any(len(_items_for_stem(v)) >= 2 for v in groups.values()):
            out.append(
                {
                    "name": ds["name"],
                    "display_name": ds["display_name"],
                    "tasks": ds["annotation_tasks"],
                }
            )
    return out


def annotation_tasks_for(workspace: Workspace, dataset_name: str | None) -> list[str]:
    """Sub-tasks offered for a dataset ('random'/empty → none, decided per draw)."""
    if not dataset_name or dataset_name == "random":
        return []
    try:
        manifest = dataset_manifest(workspace, dataset_name)
    except Exception:
        return []
    return annotation_tasks_in_manifest(manifest)


def manifest_task_names(manifest: dict) -> list[str]:
    """Every task a manifest names, from *both* annotation sources.

    A dataset can name its quality task in per-image ``metadata.annotations``
    (MOS studies) or in the ``human_comparisons`` block (published 2AFC studies,
    e.g. the Alladien adapters) — or both. Reading only the first source meant a
    comparison-only dataset looked task-less, so contributions to it were stored
    with ``task=None`` and then displayed as a separate "overall" bucket beside
    the manifest's own "overall_quality". Always consider both.
    """
    mos_tasks = annotation_tasks_in_manifest(manifest)
    return sorted(set(mos_tasks) | set(comparison_tasks_in_manifest(manifest)))


#: Task names that denote the same question and must never form separate buckets.
#: Which one wins is decided per dataset by the manifest (the published name).
TASK_SYNONYMS: tuple[frozenset[str], ...] = (
    frozenset({"overall", "overall_quality", "overall quality", "quality"}),
)


def canonical_task_map(manifest: dict) -> dict[str | None, str]:
    """``{stored task → canonical task}`` for one dataset's contributions.

    Two things split a single quality question into two buckets in the analyses:

    * annotations stored with ``task=None`` on a dataset that *does* name a task
      (what happened to comparison-only datasets before ``manifest_task_names``
      considered ``human_comparisons``), and
    * a synonym of the manifest's own task name (``overall`` vs
      ``overall_quality``) creeping in from an adapter or an import.

    Both are resolved *towards the manifest*: it is the published name. Nothing
    is merged when the dataset names several tasks and the intent is ambiguous.
    """
    names = manifest_task_names(manifest)
    mapping: dict[str | None, str] = {}
    if len(names) == 1:
        # Unnamed contributions can only have meant the single declared task.
        mapping[None] = names[0]
    for group in TASK_SYNONYMS:
        declared = [n for n in names if n in group]
        if len(declared) != 1:
            continue  # ambiguous (or unrelated) — leave well alone
        for alias in group - {declared[0]}:
            mapping[alias] = declared[0]
    return mapping


def merge_duplicate_tasks(workspace: Workspace) -> dict[str, int]:
    """Fold split task buckets into the canonical name; ``{dataset: n_changed}``.

    Runs at start-up so a deployment that already accumulated split annotations
    heals itself, and so any future adapter that names a task differently from
    the contributions it attracts cannot leave two half-populated leaderboards.
    """
    store = open_store(workspace)
    changed: dict[str, int] = {}
    for ds in discover_datasets(workspace):
        if not ds["prepared"]:
            continue
        try:
            manifest = dataset_manifest(workspace, ds["name"])
        except Exception:  # noqa: BLE001,PERF203 — a broken manifest must not stop boot
            continue
        n = store.retag_tasks(ds["name"], canonical_task_map(manifest))
        if n:
            changed[ds["name"]] = n
    return changed


def _tasks_or_unnamed(manifest: dict) -> list[str | None]:
    """Annotation tasks for a manifest, or ``[None]`` when it declares none."""
    tasks = manifest_task_names(manifest)
    return list(tasks) if tasks else [None]


def _remaining_ranking_pool(
    workspace: Workspace, user_uid: str, dataset_filter: str | None
) -> dict[str, list[tuple[str, str | None]]]:
    """``{dataset: [(stem, task), ...]}`` of reference sets the user hasn't ranked.

    Each rater ranks a given reference set (per task) at most once; completed
    sessions are keyed by ``(dataset, reference, task)``.
    """
    done = set(open_store(workspace).user_index(user_uid)["sessions"])
    pool: dict[str, list[tuple[str, str | None]]] = {}
    for e in eligible_annotation_datasets(workspace):
        name = e["name"]
        if dataset_filter and name != dataset_filter:
            continue
        manifest = dataset_manifest(workspace, name)
        groups = _reference_groups(manifest)
        rankable = [s for s, entries in groups.items() if len(_items_for_stem(entries)) >= 2]
        cands = [
            (stem, task)
            for stem in rankable
            for task in _tasks_or_unnamed(manifest)
            if (name, stem, task) not in done
        ]
        if cands:
            pool[name] = cands
    return pool


def _remaining_rating_pool(
    workspace: Workspace, user_uid: str, dataset_filter: str | None
) -> dict[str, list[tuple[str, str, str | None]]]:
    """``{dataset: [(image_id, stem, task), ...]}`` of variants the user hasn't rated.

    Each rater rates a given variant (per task) at most once; ratings are keyed by
    ``(dataset, image_id, task)``.
    """
    done = set(open_store(workspace).user_index(user_uid)["ratings"])
    pool: dict[str, list[tuple[str, str, str | None]]] = {}
    for e in eligible_annotation_datasets(workspace):
        name = e["name"]
        if dataset_filter and name != dataset_filter:
            continue
        manifest = dataset_manifest(workspace, name)
        cands: list[tuple[str, str, str | None]] = []
        for stem, entries in _reference_groups(manifest).items():
            items = _items_for_stem(entries)
            variants = [i for i in items if not i.startswith(_REF_PREFIX)]
            has_ref = any(i.startswith(_REF_PREFIX) for i in items)
            if not (variants and has_ref):
                continue
            for image_id in variants:
                for task in _tasks_or_unnamed(manifest):
                    if (name, image_id, task) not in done:
                        cands.append((image_id, stem, task))
        if cands:
            pool[name] = cands
    return pool


def _exhausted_signal(
    workspace: Workspace,
    user_uid: str,
    dataset_filter: str | None,
    pool_fn,
) -> dict:
    """Build the ``done`` result when a draw has no remaining candidates.

    ``scope`` is ``"all"`` when nothing remains in *any* dataset (show the grand
    thank-you), else ``"dataset"`` (the chosen dataset is finished, others remain).
    """
    if dataset_filter is None or not pool_fn(workspace, user_uid, None):
        return {"done": True, "scope": "all"}
    return {"done": True, "scope": "dataset", "dataset": dataset_filter}


def start_ranking(
    workspace: Workspace,
    sessions: dict[str, dict],
    *,
    user_uid: str,
    dataset_choice: str,
    task_choice: str,
    rng: random.Random | None = None,
) -> dict:
    """Draw a not-yet-ranked reference set and create a :class:`RankingSession`.

    Only reference sets this user hasn't ranked are offered. Returns
    ``{"error": ...}`` on failure, ``{"done": True, "scope": ...}`` when the user
    has ranked everything (in the chosen scope), or ``{"session_id": ...}``.
    """
    rng = rng or random.Random()
    eligible = {e["name"] for e in eligible_annotation_datasets(workspace)}
    if not eligible:
        return {"error": "No datasets with rankable variants are prepared yet."}

    dataset_filter = None if dataset_choice in (None, "", "random") else dataset_choice
    if dataset_filter and dataset_filter not in eligible:
        return {"error": "That dataset isn't available for annotation."}

    pool = _remaining_ranking_pool(workspace, user_uid, dataset_filter)
    if not pool:
        return _exhausted_signal(workspace, user_uid, dataset_filter, _remaining_ranking_pool)

    dataset_name = rng.choice(sorted(pool))
    stem, chosen_task = rng.choice(pool[dataset_name])
    manifest = dataset_manifest(workspace, dataset_name)
    item_ids = _items_for_stem(_reference_groups(manifest)[stem])

    session = RankingSession(
        items=item_ids,
        dataset=dataset_name,
        reference=stem,
        task=chosen_task,
        user_uid=user_uid,
    )
    sid = uuid.uuid4().hex
    sessions[sid] = {
        "session": session,
        "dataset_choice": dataset_choice,
        "task_choice": task_choice,
        "created": time.monotonic(),
    }
    _evict_stale_sessions(sessions)
    # Every image in this reference set will be shown during the session, so warm
    # them all now rather than paying the render cost pair by pair.
    prefetch_annotation_images(workspace, dataset_name, item_ids)
    return {"error": None, "session_id": sid}


#: Ranking sessions live in memory; drop ones abandoned this long ago (seconds).
SESSION_TTL = 12 * 3600


def _evict_stale_sessions(sessions: dict[str, dict]) -> None:
    """Forget abandoned ranking sessions so the process doesn't grow unbounded."""
    cutoff = time.monotonic() - SESSION_TTL
    for sid in [s for s, w in sessions.items() if w.get("created", 0) < cutoff]:
        sessions.pop(sid, None)


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
    shown = {pair["left"], pair["right"]}
    return {
        "session_id": session_id,
        "dataset": session.dataset,
        "prompt": prompt,
        "left_item": pair["left"],
        "right_item": pair["right"],
        "placed": placed,
        "n_items": session.n_items,
        "n_comparisons": len(session.comparisons),
        # The rest of this reference set — any of them can come up next, and the
        # set is small, so let the browser fetch them while the rater decides.
        "prefetch_items": [i for i in session.items if i not in shown],
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
    store = open_store(workspace)
    store.add_session(session.result())
    count = store.count_sessions(user_uid=session.user_uid)
    sessions.pop(session_id, None)

    nxt = start_ranking(
        workspace,
        sessions,
        user_uid=session.user_uid,
        dataset_choice=wrapper["dataset_choice"],
        task_choice=wrapper["task_choice"],
    )
    return {"status": "completed", "count": count, "next": nxt}


# ---------------------------------------------------------------------------
# Single-stimulus MOS rating (quantitative 1-4)
# ---------------------------------------------------------------------------


# Ratings are drawn one at a time, but the browser can only prefetch an image if
# it knows which one comes next. Each rater therefore gets a small look-ahead
# queue: a draw serves the head and reports the rest as ``upcoming`` so the page
# can warm both the server-side render cache and the browser cache while the
# current image is being judged.
RATING_LOOKAHEAD = 3

_RATING_QUEUES: dict[tuple[str, str], list[tuple[str, str, str, str | None]]] = {}
_RATING_QUEUE_LOCK = threading.Lock()


def _rating_queue(
    workspace: Workspace,
    dataset_choice: str,
    user_uid: str,
    dataset_filter: str | None,
    rng: random.Random,
) -> list[tuple[str, str, str, str | None]]:
    """Refill and return the rater's look-ahead queue of ``(dataset, image, stem, task)``.

    Entries the rater has since completed are dropped, so the queue can never
    re-offer something already rated.
    """
    key = (user_uid, dataset_choice)
    done = set(open_store(workspace).user_index(user_uid)["ratings"])
    with _RATING_QUEUE_LOCK:
        queue = [q for q in _RATING_QUEUES.get(key, []) if (q[0], q[1], q[3]) not in done]

    if not queue:
        # Deliberately outside the lock: building the pool walks every reference
        # group, and one rater refilling their queue must not stall everyone
        # else's draw. Two raters refilling at once is harmless duplicate work.
        pool = _remaining_rating_pool(workspace, user_uid, dataset_filter)
        flat = [
            (ds, img, stem, task)
            for ds, cands in pool.items()
            for img, stem, task in cands
        ]
        rng.shuffle(flat)
        queue = flat[:RATING_LOOKAHEAD]

    with _RATING_QUEUE_LOCK:
        _RATING_QUEUES[key] = queue
    return list(queue)


def draw_rating_item(
    workspace: Workspace,
    dataset_choice: str,
    *,
    user_uid: str,
    rng: random.Random | None = None,
) -> dict:
    """Pick a not-yet-rated reconstruction + its reference + task to rate.

    Only variants this user hasn't rated are offered. Returns ``{"error": ...}``,
    ``{"done": True, "scope": ...}`` when the user has rated everything (in the
    chosen scope), or a draw dict with ``dataset``, ``image_id`` (variant),
    ``reference_item``, ``task`` and ``upcoming`` (items to prefetch).
    """
    rng = rng or random.Random()
    eligible = {e["name"] for e in eligible_annotation_datasets(workspace)}
    if not eligible:
        return {"error": "No datasets with rankable variants are prepared yet."}

    dataset_filter = None if dataset_choice in (None, "", "random") else dataset_choice
    if dataset_filter and dataset_filter not in eligible:
        return {"error": "That dataset isn't available for rating."}

    queue = _rating_queue(workspace, dataset_choice, user_uid, dataset_filter, rng)
    if not queue:
        return _exhausted_signal(workspace, user_uid, dataset_filter, _remaining_rating_pool)

    dataset_name, image_id, stem, task = queue[0]
    upcoming = [
        {"dataset": ds, "items": [img, f"{_REF_PREFIX}{st}"]}
        for ds, img, st, _ in queue[1:]
    ]
    for entry in upcoming:
        prefetch_annotation_images(workspace, entry["dataset"], entry["items"])
    return {
        "error": None,
        "dataset": dataset_name,
        "image_id": image_id,
        "reference_item": f"{_REF_PREFIX}{stem}",
        "task": task,
        "upcoming": upcoming,
    }


def record_rating(
    workspace: Workspace,
    *,
    user_uid: str,
    dataset: str,
    image_id: str,
    task: str | None,
    score: int,
) -> int:
    """Persist one 1-4 rating; returns the annotator's total rating count."""
    score = max(1, min(4, int(score)))
    store = open_store(workspace)
    store.add_rating(
        {
            "user_uid": user_uid,
            "dataset": dataset,
            "image_id": image_id,
            "task": task,
            "score": score,
        }
    )
    return store.count_ratings(user_uid=user_uid)


def resolve_item_path(workspace: Workspace, dataset_name: str, item_id: str) -> Path | None:
    """Map an annotation item id back to its on-disk ``.npy`` (via the manifest)."""
    try:
        manifest = dataset_manifest(workspace, dataset_name)
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


def display_window_for(
    workspace: Workspace, dataset_name: str, item_id: str
) -> tuple[float, float] | None:
    """Shared display intensity window (lo, hi) for an annotation item.

    Returns the reference (ground-truth) array's min/max for the item's reference
    group, so every item shown together — the two variants in a 2AFC comparison,
    or a variant and its reference in a rating — is rendered on the *same* dynamic
    range. Without this, per-image min-max stretching would hide systematic
    brightness differences and bias subjective quality judgements.

    ``None`` if no reference can be resolved (caller falls back to per-image).
    """
    try:
        manifest = dataset_manifest(workspace, dataset_name)
    except Exception:
        return None
    root = _dataset_root(workspace, dataset_name)

    ref_rel: str | None = None
    if item_id.startswith(_REF_PREFIX):
        stem = item_id[len(_REF_PREFIX) :]
        for e in manifest.get("images", []):
            if str(e.get("metadata", {}).get("stem")) == stem and e.get("references"):
                refs = e["references"]
                ref_rel = (refs.get("ground_truth") or next(iter(refs.values())))["path"]
                break
    else:
        for e in manifest.get("images", []):
            if e["id"] == item_id and e.get("references"):
                refs = e["references"]
                ref_rel = (refs.get("ground_truth") or next(iter(refs.values())))["path"]
                break

    if ref_rel is None:
        return None
    ref_path = (root / ref_rel).resolve()
    if root.resolve() not in ref_path.parents or not ref_path.exists():
        return None
    return _reference_window(ref_path)


# Display windows are a property of the reference array, which every variant in a
# group shares — so compute it once per reference file rather than re-loading the
# whole array for each image request.
_WINDOW_CACHE: dict[str, tuple[float, float] | None] = {}


def _reference_window(ref_path: Path) -> tuple[float, float] | None:
    key = str(ref_path)
    with _MANIFEST_LOCK:
        if key in _WINDOW_CACHE:
            return _WINDOW_CACHE[key]
    arr = np.load(ref_path).astype(np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    window = (lo, hi) if hi > lo else None
    with _MANIFEST_LOCK:
        _WINDOW_CACHE[key] = window
    return window


def render_item_png(path: Path, window: tuple[float, float] | None = None) -> bytes:
    """Load a stored ``.npy`` and encode it as a display PNG (scaled to 8-bit).

    ``window`` is an explicit ``(lo, hi)`` intensity range to map to [0, 255]
    (so paired images share one dynamic range); when ``None`` the image's own
    min/max is used.
    """
    from PIL import Image

    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    lo, hi = window if window is not None else (float(arr.min()), float(arr.max()))
    scaled = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    img = (scaled * 255.0).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Annotation image cache
# ---------------------------------------------------------------------------
# Serving one annotation image used to parse the manifest, load the reference
# array to derive the display window, load the item array, rescale and PNG-encode
# it — on *every* request, with no HTTP validators, so revisiting an image redid
# all of it. The rendered bytes are a pure function of (dataset, item) for a given
# manifest, so they are memoised here (bounded LRU) and tagged with an ETag the
# browser can revalidate against.

#: Byte budget for the in-memory rendered-PNG cache (override for large studies).
IMAGE_CACHE_BYTES = int(os.environ.get("IPASC_IQA_IMAGE_CACHE_BYTES", 256 * 1024 * 1024))

_IMAGE_CACHE: OrderedDict[tuple[str, str], tuple[bytes, str]] = OrderedDict()
_IMAGE_CACHE_SIZE = 0
_IMAGE_LOCK = threading.Lock()


def _image_cache_get(key: tuple[str, str]) -> tuple[bytes, str] | None:
    with _IMAGE_LOCK:
        hit = _IMAGE_CACHE.get(key)
        if hit is not None:
            _IMAGE_CACHE.move_to_end(key)
        return hit


def _image_cache_put(key: tuple[str, str], value: tuple[bytes, str]) -> None:
    global _IMAGE_CACHE_SIZE
    with _IMAGE_LOCK:
        if key in _IMAGE_CACHE:
            _IMAGE_CACHE_SIZE -= len(_IMAGE_CACHE[key][0])
        _IMAGE_CACHE[key] = value
        _IMAGE_CACHE.move_to_end(key)
        _IMAGE_CACHE_SIZE += len(value[0])
        while _IMAGE_CACHE_SIZE > IMAGE_CACHE_BYTES and len(_IMAGE_CACHE) > 1:
            _, evicted = _IMAGE_CACHE.popitem(last=False)
            _IMAGE_CACHE_SIZE -= len(evicted[0])


def annotation_image(workspace: Workspace, dataset: str, item: str) -> tuple[bytes, str] | None:
    """Rendered PNG + ETag for one annotation item, or ``None`` if unresolvable."""
    key = (dataset, item)
    hit = _image_cache_get(key)
    if hit is not None:
        return hit

    path = resolve_item_path(workspace, dataset, item)
    if path is None or not path.exists():
        return None
    # Window both images of a comparison/rating to their shared reference range so
    # they are displayed on the same dynamic range (fair subjective assessment).
    png = render_item_png(path, window=display_window_for(workspace, dataset, item))
    value = (png, f'"{hashlib.sha256(png).hexdigest()[:32]}"')
    _image_cache_put(key, value)
    return value


# Warming runs on a small shared pool rather than a thread per draw: with tens of
# concurrent annotators, thread-per-request would spawn unboundedly, and the work
# is deliberately low priority — it must never compete with serving a rater the
# image they are waiting for.
_PREFETCH_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="img-warm")


def prefetch_annotation_images(workspace: Workspace, dataset: str, items: list[str]) -> None:
    """Render *items* into the image cache in the background (best effort).

    Called when an annotation session starts, and for the items queued behind the
    current rating, so the images the rater is about to be shown are already
    encoded by the time the browser asks for them.
    """

    def _warm() -> None:
        for item in items:
            try:
                annotation_image(workspace, dataset, item)
            except Exception:  # noqa: BLE001,PERF203 — warming is best effort
                continue

    _PREFETCH_POOL.submit(_warm)


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

    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as astore:
        all_sessions = astore.sessions()
        expertise_by_uid = {u["uid"]: u.get("expertise") for u in astore.all_users()}
        approved_uids = astore.approved_uids()
    anon = _anonymise({s.get("user_uid") for s in all_sessions if s.get("user_uid")})
    # Pseudonym → rater metadata (no PII). Contributions are kept in full and
    # *tagged* with expertise + approval, so downstream analysis can filter by
    # either. (The live platform aggregates hard-exclude unapproved raters; the
    # derived ``bradley_terry`` block below mirrors that approved-only view.)
    annotators_meta = {
        pseudonym: {
            "expertise": expertise_by_uid.get(uid),
            "approved": uid in approved_uids,
        }
        for uid, pseudonym in anon.items()
    }

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
            manifest = dataset_manifest(workspace, name)

        measure_results = [
            {k: v for k, v in r.model_dump(mode="json").items() if k not in _EXPORT_DROP_FIELDS}
            for r in records_by_ds.get(name, [])
        ]
        expert_annotations = _manifest_annotations(manifest)
        pairwise_annotations = [
            {
                "annotator": anon.get(s.get("user_uid")),
                "expertise": expertise_by_uid.get(s.get("user_uid")),
                "approved": s.get("user_uid") in approved_uids,
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
            "annotators": annotators_meta,
            "datasets": datasets_out,
        }
    )
