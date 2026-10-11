"""Background benchmark jobs for the GUI.

Each call to :func:`launch_run` starts one benchmark in a daemon thread and
returns a *job_id* immediately. The caller is responsible for storing the
returned :class:`_JobState` in ``app.state.jobs`` so that status-polling
routes can find it.
"""

from __future__ import annotations

import threading
import uuid
from collections import Counter, defaultdict

from ipasc_iqa.datasets.registry import registry as dataset_registry


def _dataset_display_name(dataset: str) -> str:
    """Return the user-facing name while preserving the internal dataset ID."""
    try:
        cls = dataset_registry.get(dataset)
    except KeyError:
        return dataset
    return getattr(cls, "DATASET_DISPLAY_NAME", None) or dataset


class _JobState:
    """Thread-safe state for one background benchmark run.

    Progress is tracked per (dataset, measure) pair against each dataset's image
    count, so the UI can show a percentage per dataset (and per measure within it).
    """

    __slots__ = (
        "job_id",
        "status",
        "run_id",
        "error",
        "message",
        "run_issues",
        "dataset_errors",
        "preparation",
        "_n_images",
        "_measures",
        "_done",
        "_outcomes",
        "_last_outcome",
        "_lock",
    )

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.status = "running"
        self.run_id: str | None = None
        self.error: str | None = None
        self._n_images: dict[str, int] = {}  # dataset -> image count (the denominator)
        self.message: str = "Preparing/downloading datasets..."
        self.run_issues: list[dict] = []
        self.dataset_errors: list[dict] = []
        self.preparation: dict[str, dict] = {}
        self._measures: list[str] = []  # measure names, in selection order
        self._done: dict[tuple[str, str], int] = {}  # (dataset, measure) -> pairs done
        self._outcomes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self._last_outcome: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutators (called from the background thread)
    # ------------------------------------------------------------------

    def set_plan(self, n_images_by_dataset: dict[str, int], measure_names: list[str]) -> None:
        with self._lock:
            self._n_images = dict(n_images_by_dataset)
            self._measures = list(measure_names)
            self.message = "Computing metrics..."

    def record_progress(self, dataset: str, measure: str, outcome: str) -> None:
        """Record a finished image/measure pair and its actual cache outcome."""
        if outcome not in {"computed", "cached", "skipped"}:
            raise ValueError(f"Unknown metric outcome: {outcome}")
        with self._lock:
            key = (dataset, measure)
            self._done[key] = self._done.get(key, 0) + 1
            self._outcomes[key][outcome] += 1
            self._last_outcome[dataset] = (measure, outcome)

    def finish(self, run_id: str, run_issues: list[dict] | None = None) -> None:
        with self._lock:
            self.run_id = run_id
            self.run_issues = list(run_issues or [])
            self.status = "done"

    def fail(self, error: str) -> None:
        with self._lock:
            self.error = error
            self.status = "error"

    def init_preparation(self, dataset_names: list[str]) -> None:
        with self._lock:
            self.preparation = {
                name: {"dataset": name, "dataset_display_name": _dataset_display_name(name),
                       "state": "queued", "detail": "Waiting", "stage": None,
                       "attempt": None, "max_attempts": None, "downloaded": None, "total": None}
                for name in dataset_names
            }

    def update_preparation(self, dataset: str, state: str, detail: str, **kwargs) -> None:
        with self._lock:
            item = self.preparation.get(dataset)
            if item is None:
                return
            item.update(state=state, detail=detail, **kwargs)

    def record_dataset_error(self, dataset: str, reason: str) -> None:
        with self._lock:
            self.dataset_errors.append({"dataset": dataset, "reason": reason})

    # ------------------------------------------------------------------
    # Readers (called from route handlers — any thread)
    # ------------------------------------------------------------------

    @staticmethod
    def _pct(done: int, total: int) -> int:
        return 100 if total <= 0 else min(100, round(100 * done / total))

    def snapshot(self) -> dict:
        """Return a plain-dict copy of the current state (safe to read from any thread).

        ``progress`` is a per-dataset list, each with an overall percentage and a
        per-measure breakdown.
        """
        with self._lock:
            issue_counts = {
                (issue["dataset"], issue["measure"]): issue["count"]
                for issue in self.run_issues
            }
            progress = []
            for dataset, n_images in self._n_images.items():
                measures = [
                    {
                        "name": m,
                        "done": min(self._done.get((dataset, m), 0), n_images),
                        "total": n_images,
                        "percent": self._pct(self._done.get((dataset, m), 0), n_images),
                        "issues": issue_counts.get((dataset, m), 0),
                        "computed": self._outcomes[(dataset, m)]["computed"],
                        "cached": self._outcomes[(dataset, m)]["cached"],
                        "skipped": self._outcomes[(dataset, m)]["skipped"],
                    }
                    for m in self._measures
                ]
                ds_done = sum(mp["done"] for mp in measures)
                progress.append(
                    {
                        "dataset": dataset,
                        "dataset_display_name": _dataset_display_name(dataset),
                        "percent": self._pct(ds_done, n_images * len(self._measures)),
                        "measures": measures,
                        "computed": sum(m["computed"] for m in measures),
                        "cached": sum(m["cached"] for m in measures),
                        "skipped": sum(m["skipped"] for m in measures),
                        "last_measure": self._last_outcome.get(dataset, (None, None))[0],
                        "last_outcome": self._last_outcome.get(dataset, (None, None))[1],
                    }
                )
            return {
                "job_id": self.job_id,
                "status": self.status,
                "run_id": self.run_id,
                "error": self.error,
                "message": self.message,
                "progress": progress,
                "preparation": [dict(item) for item in self.preparation.values()],
                "run_issues": [
                    {
                        **issue,
                        "dataset_display_name": _dataset_display_name(issue["dataset"]),
                    }
                    for issue in self.run_issues
                ],
                "dataset_errors": [
                    {
                        **issue,
                        "dataset_display_name": _dataset_display_name(issue["dataset"]),
                    }
                    for issue in self.dataset_errors
                ],
            }
        
    def record_skip(self, dataset: str, measure: str, reason: str) -> None:
        with self._lock:
            for issue in self.run_issues:
                if issue["dataset"] == dataset and issue["measure"] == measure:
                    issue["count"] += 1
                    for item in issue["reasons"]:
                        if item["reason"] == reason:
                            item["count"] += 1
                            break
                    else:
                        issue["reasons"].append({"reason": reason, "count": 1})
                    return

            self.run_issues.append(
                {
                    "dataset": dataset,
                    "measure": measure,
                    "count": 1,
                    "reasons": [{"reason": reason, "count": 1}],
                }
            )

def _summarize_run_issues(workspace: object, run_id: str) -> list[dict]:
    """Summarize skipped/failed calculations from the saved run JSON.

    The runner already records both expected skips, such as missing references
    or ROIs, and unexpected metric exceptions in run.skipped. The GUI only needs
    a compact per-dataset/per-measure summary for the progress panel.
    """
    from ipasc_iqa.gui.service import Workspace
    from ipasc_iqa.io.results import load_results

    ws: Workspace = workspace  # type: ignore[assignment]
    run = load_results(ws.results_dir / f"{run_id}.json")

    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for skipped in run.skipped:
        grouped[(skipped.dataset, skipped.measure)][skipped.reason] += 1

    out = []
    for (dataset, measure), reasons in sorted(grouped.items()):
        out.append(
            {
                "dataset": dataset,
                "measure": measure,
                "count": sum(reasons.values()),
                "reasons": [
                    {"reason": reason, "count": count}
                    for reason, count in reasons.most_common()
                ],
            }
        )
    return out

def launch_run(
    workspace: object,
    jobs: dict[str, _JobState],
    dataset_names: list[str],
    measure_names: list[str],
    *,
    reference_preference: list[str] | None = None,
    device: str = "auto",
) -> str:
    """Start a benchmark run in a daemon thread and return the job_id.

    The :class:`_JobState` is stored in *jobs* before the thread starts, so
    the route handler can immediately render a progress fragment.
    """
    # Import here to avoid circular imports (jobs → service, not vice-versa).
    from ipasc_iqa.gui.service import Workspace, run_benchmark

    ws: Workspace = workspace  # type: ignore[assignment]

    job_id = uuid.uuid4().hex[:12]
    job = _JobState(job_id)
    jobs[job_id] = job
    job.init_preparation(dataset_names)

    def _run() -> None:
        try:
            run_id = run_benchmark(
                ws,
                dataset_names,
                measure_names,
                reference_preference=reference_preference,
                device=device,
                on_progress=job.record_progress,
                on_plan=job.set_plan,
                on_skip=job.record_skip,
                on_dataset_error=job.record_dataset_error,
                on_dataset_preparation=job.update_preparation,
            )
            job.finish(run_id, _summarize_run_issues(ws, run_id))
        except Exception as exc:  # noqa: BLE001
            job.fail(str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return job_id