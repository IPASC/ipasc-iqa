"""Background benchmark jobs for the GUI.

Each call to :func:`launch_run` starts one benchmark in a daemon thread and
returns a *job_id* immediately. The caller is responsible for storing the
returned :class:`_JobState` in ``app.state.jobs`` so that status-polling
routes can find it.
"""

from __future__ import annotations

import threading
import uuid


class _JobState:
    """Thread-safe state for one background benchmark run."""

    __slots__ = ("job_id", "status", "run_id", "error", "_done_per_measure", "_lock")

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.status = "running"
        self.run_id: str | None = None
        self.error: str | None = None
        self._done_per_measure: dict[str, int] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutators (called from the background thread)
    # ------------------------------------------------------------------

    def record_result(self, measure_name: str) -> None:
        with self._lock:
            self._done_per_measure[measure_name] = (
                self._done_per_measure.get(measure_name, 0) + 1
            )

    def finish(self, run_id: str) -> None:
        with self._lock:
            self.run_id = run_id
            self.status = "done"

    def fail(self, error: str) -> None:
        with self._lock:
            self.error = error
            self.status = "error"

    # ------------------------------------------------------------------
    # Readers (called from route handlers — any thread)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a plain-dict copy of the current state (safe to read from any thread)."""
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "run_id": self.run_id,
                "error": self.error,
                "done_per_measure": dict(self._done_per_measure),
            }


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

    def _run() -> None:
        try:
            run_id = run_benchmark(
                ws,
                dataset_names,
                measure_names,
                reference_preference=reference_preference,
                device=device,
                on_result=job.record_result,
            )
            job.finish(run_id)
        except Exception as exc:  # noqa: BLE001
            job.fail(str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return job_id
