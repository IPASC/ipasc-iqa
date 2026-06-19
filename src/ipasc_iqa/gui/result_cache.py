"""Persistent cache for expensive GUI computations (analysis / standings / scatter).

Repeatedly "just looking at results" should not re-run the correlation/aggregation
work on every page view — especially on a small server (e.g. a Raspberry Pi).
Each computed output is stored in a persistent (disk-backed) cache keyed by the
inputs *and* a ``results_epoch`` token that bumps whenever anything affecting
results changes (new annotation, approval/role change, benchmark run). So an
unchanged view is a dict lookup; only the first view after a change recomputes.

Recompute is **single-flight**: at most one background recompute runs per
(kind, params) at a time, so concurrent viewers don't each kick off the same
heavy job — they wait for the in-flight one. A short grace period lets fast
recomputes finish inline (no UI flicker); slower ones return ``None`` so the
caller can show a "recomputing…" placeholder that polls until the value lands.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

# How long a request will wait for an in-progress recompute before giving up and
# returning None (caller then shows the polling "recomputing…" placeholder).
GRACE_SECONDS = 0.5
_POLL_STEP = 0.02


class ResultCache:
    def __init__(self, path: Path | str) -> None:
        import diskcache

        self.path = Path(path)
        self._cache = diskcache.Cache(str(self.path))
        self._running: dict[str, int] = {}  # signature -> epoch currently computing
        self._lock = threading.Lock()

    def close(self) -> None:
        self._cache.close()

    def __enter__(self) -> ResultCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def _sig(kind: str, params: dict) -> str:
        return f"{kind}:{json.dumps(params, sort_keys=True, default=str)}"

    @classmethod
    def _key(cls, kind: str, params: dict, epoch: int) -> str:
        raw = f"{cls._sig(kind, params)}:{epoch}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, kind: str, params: dict, epoch: int) -> dict | None:
        return self._cache.get(self._key(kind, params, epoch))

    def _schedule(
        self, kind: str, params: dict, epoch: int, compute_fn: Callable[[], dict]
    ) -> None:
        """Start a background recompute unless one is already running for this sig."""
        sig = self._sig(kind, params)
        with self._lock:
            if sig in self._running:
                return  # in-flight already — viewers will re-poll once it lands
            self._running[sig] = epoch

        def _run() -> None:
            try:
                value = compute_fn()
            except Exception as exc:  # noqa: BLE001 — cache an error so polling stops
                value = {"error": f"Could not compute results: {exc}"}
            try:
                self._cache.set(self._key(kind, params, epoch), value)
            finally:
                with self._lock:
                    self._running.pop(sig, None)

        threading.Thread(target=_run, daemon=True).start()

    def get_or_schedule(
        self,
        kind: str,
        params: dict,
        epoch: int,
        compute_fn: Callable[[], dict],
        *,
        grace: float = GRACE_SECONDS,
    ) -> dict | None:
        """Return the cached value, or schedule a recompute and wait briefly.

        Returns the value if it is (or quickly becomes) available, else ``None``
        — signalling the caller to show a polling "recomputing…" placeholder.
        """
        value = self.get(kind, params, epoch)
        if value is not None:
            return value
        self._schedule(kind, params, epoch, compute_fn)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            time.sleep(_POLL_STEP)
            value = self.get(kind, params, epoch)
            if value is not None:
                return value
        return None
