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
import random
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ipasc_iqa.analysis.aggregate import measure_stats
from ipasc_iqa.analysis.correlation import correlate_measures_with_annotations
from ipasc_iqa.analysis.manifest_query import (
    annotation_tasks_in_manifest,
    annotation_values_from_manifest,
    groups_from_manifest,
)
from ipasc_iqa.analysis.scatter import build_scatter
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
    if measure_meta["measure_type"] == MeasureType.FULL_REFERENCE and not caps.get("has_references"):
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
) -> dict:
    """Correlate measures against expert MOS for a single annotation task.

    When ``task`` is provided and valid it is used as the active task;
    otherwise the first available task is shown.  ``all_tasks`` and
    ``active_task`` are always present in the return value so that the
    template can render the sub-task dropdown (or omit it for single-task
    datasets) regardless of error state.
    """
    _empty: dict = {"all_tasks": [], "active_task": None}

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
    task_names = annotation_tasks_in_manifest(manifest)
    if not task_names:
        return {
            "dataset": dataset_name,
            "tasks": [],
            "error": "dataset has no expert annotations",
            **_empty,
        }

    # Resolve the active task: honour the caller's choice when valid, else first.
    active_task = task if (task and task in task_names) else task_names[0]

    # Only correlate measures the dataset can actually support, intersected with
    # the user's selection — so unusable measures never produce empty rows.
    caps = _capabilities_from_manifest(manifest)
    usable = [m["name"] for m in list_measures() if _measure_usable(m, caps)]
    chosen = [m for m in (measures or usable) if m in usable]

    values = annotation_values_from_manifest(manifest, active_task, "zscore_mos")
    rows = correlate_measures_with_annotations(
        run, values, measures=chosen, dataset=dataset_name
    )

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
        "run_id": run.run_id,
        "tasks": [{"task": active_task, "rows": row_dicts}],
        "all_tasks": task_names,
        "active_task": active_task,
        "error": None,
    }


def scatter_for_measure(
    workspace: Workspace,
    dataset_name: str,
    measure: str,
    *,
    task: str | None = None,
    run_id: str | None = None,
) -> dict:
    """Build the scatter-plot model for one measure vs the human scores.

    Returns ``{"error": ...}`` when the plot can't be produced (dataset not
    ready, no run, no annotations, or fewer than two paired points).
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
    task_names = annotation_tasks_in_manifest(manifest)
    if not task_names:
        return {"error": "dataset has no expert annotations"}
    active_task = task if (task and task in task_names) else task_names[0]

    annotation_values = annotation_values_from_manifest(manifest, active_task, "zscore_mos")
    groups = groups_from_manifest(manifest, "algorithm")
    data = build_scatter(run, annotation_values, measure, dataset=dataset_name, groups=groups)
    if len(data.xs) < 2:
        return {"error": f"not enough paired data points for {measure}"}

    rows = correlate_measures_with_annotations(
        run, annotation_values, measures=[measure], dataset=dataset_name
    )
    srcc = rows[0].srcc if rows else None
    krcc = rows[0].krcc if rows else None

    meta = measure_registry.get(measure).metadata
    chart = scatter_chart(
        data,
        srcc=srcc,
        krcc=krcc,
        x_label=f"Human score (z-MOS · {active_task})",
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


def standings(workspace: Workspace) -> dict:
    """For each prepared+annotated dataset with a run, rank measures by |SRCC|.

    Produces one ranked leaderboard per (dataset, task). Correlation rows arrive
    already sorted by ``|srcc|`` descending, so list position is the rank.
    """
    entries: list[dict] = []
    for ds in discover_datasets(workspace):
        if not ds["prepared"] or not ds["has_annotations"]:
            continue
        run = _merged_run(workspace, ds["name"])
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


def resolve_item_path(workspace: Workspace, dataset_name: str, item_id: str) -> Path | None:
    """Map an annotation item id back to its on-disk ``.npy`` (via the manifest)."""
    try:
        manifest = load_manifest(dataset_registry.manifest_path(dataset_name, workspace.data_dir))
    except Exception:
        return None
    root = _dataset_root(workspace, dataset_name)

    rel: str | None = None
    if item_id.startswith(_REF_PREFIX):
        stem = item_id[len(_REF_PREFIX):]
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
