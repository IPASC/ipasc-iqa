"""Persistent cache for expensive GUI computations (analysis / standings / scatter).

Repeatedly "just looking at results" should not re-run the correlation/aggregation
work on every page view — especially on a small server (e.g. a Raspberry Pi).
Each computed output is stored in a persistent (disk-backed) cache keyed by the
inputs *and* a ``results_epoch`` token that bumps whenever anything affecting
results changes (new annotation, approval/role change, benchmark run). So an
unchanged view is a dict lookup; only the first view after a change recomputes.

**Staleness is tolerated on purpose.** Every single contributed annotation bumps
the epoch, so during an active annotation session a strictly epoch-keyed cache
would recompute everything every few seconds — while the numbers barely move.
Instead the most recent value for a view is kept under a stable signature and
served for up to :data:`STALE_TTL` seconds even when the epoch has moved on. Past
that age the stale value is *still* returned immediately, with a background
recompute scheduled; the fresh numbers land on a later view. Only a view that has
never been computed makes the caller wait.

Recompute is **single-flight**: at most one background recompute runs per
(kind, params) at a time, so concurrent viewers don't each kick off the same
heavy job — they wait for (or read past) the in-flight one. A short grace period
lets fast first computes finish inline (no UI flicker); slower ones return
``None`` so the caller can show a "recomputing…" placeholder that polls.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

# How long a request will wait for an in-progress recompute before giving up and
# returning None (caller then shows the polling "recomputing…" placeholder).
GRACE_SECONDS = 0.5
_POLL_STEP = 0.02

#: How long a cached value keeps being served after the data behind it changed.
#: One hour by default; set ``IPASC_IQA_RESULT_STALE_TTL=0`` to always recompute
#: on change (what the tests do, so they can assert on fresh numbers).
STALE_TTL = float(os.environ.get("IPASC_IQA_RESULT_STALE_TTL", 3600))

#: Key added to every returned value describing its freshness (read by templates).
META_KEY = "cache_meta"

# Bump when cached GUI model fields change.
CACHE_SCHEMA_VERSION = 2


class ResultCache:
    def __init__(self, path: Path | str, stale_ttl: float | None = None) -> None:
        import diskcache

        self.path = Path(path)
        # Read the env var per instance (not at import) so a deployment or a test
        # can set it before building the app.
        self.stale_ttl = (
            float(os.environ.get("IPASC_IQA_RESULT_STALE_TTL", STALE_TTL))
            if stale_ttl is None
            else stale_ttl
        )
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
        return (
            f"v{CACHE_SCHEMA_VERSION}:{kind}:"
            f"{json.dumps(params, sort_keys=True, default=str)}"
        )

    @classmethod
    def _key(cls, kind: str, params: dict, epoch: int) -> str:
        raw = f"{cls._sig(kind, params)}:{epoch}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @classmethod
    def _latest_key(cls, kind: str, params: dict) -> str:
        raw = f"latest:{cls._sig(kind, params)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, kind: str, params: dict, epoch: int) -> dict | None:
        """The value computed for exactly this epoch, if it is cached."""
        return self._cache.get(self._key(kind, params, epoch))

    def _store(self, kind: str, params: dict, epoch: int, value: dict) -> None:
        self._cache.set(self._key(kind, params, epoch), value)
        self._cache.set(
            self._latest_key(kind, params),
            {"value": value, "epoch": epoch, "ts": time.time()},
        )

    def is_recomputing(self, kind: str, params: dict) -> bool:
        with self._lock:
            return self._sig(kind, params) in self._running

    def _schedule(
        self, kind: str, params: dict, epoch: int, compute_fn: Callable[[], dict]
    ) -> bool:
        """Start a background recompute unless one is already running for this sig.

        Returns whether a job is now in flight (started here or already running) —
        i.e. whether a "recomputing" state is worth showing.
        """
        sig = self._sig(kind, params)
        with self._lock:
            if sig in self._running:
                return True  # in-flight already — viewers read the stale value meanwhile
            self._running[sig] = epoch

        def _run() -> None:
            try:
                value = compute_fn()
            except Exception as exc:  # noqa: BLE001 — cache an error so polling stops
                value = {"error": f"Could not compute results: {exc}"}
            try:
                self._store(kind, params, epoch, value)
            finally:
                with self._lock:
                    self._running.pop(sig, None)

        threading.Thread(target=_run, daemon=True).start()
        return True

    @staticmethod
    def _with_meta(value: dict, *, ts: float, **meta: object) -> dict:
        """Tag a value with freshness metadata without mutating the cached dict.

        ``computed`` is when the numbers were produced — shown to viewers so a
        leaderboard always says how current it is.
        """
        stamp = datetime.fromtimestamp(ts, UTC)
        return {
            **value,
            META_KEY: {
                **meta,
                "computed_at": ts,
                "computed": stamp.strftime("%Y-%m-%d %H:%M UTC"),
            },
        }

    def _computed_at(self, kind: str, params: dict, epoch: int) -> float:
        """When the value for *epoch* was computed (falls back to 'just now')."""
        latest = self._cache.get(self._latest_key(kind, params))
        if latest is not None and latest.get("epoch") == epoch:
            return float(latest.get("ts", time.time()))
        return time.time()

    def get_or_schedule(
        self,
        kind: str,
        params: dict,
        epoch: int,
        compute_fn: Callable[[], dict],
        *,
        grace: float = GRACE_SECONDS,
    ) -> dict | None:
        """Return the value to display, or ``None`` if there is nothing yet.

        In order: the value for this exact epoch; a recent-enough previous value
        (no recompute); an older previous value (returned now, recompute queued);
        otherwise compute — waiting up to *grace* — and signal the caller to show
        a polling placeholder if that takes too long.
        """
        value = self.get(kind, params, epoch)
        if value is not None:
            ts = self._computed_at(kind, params, epoch)
            return self._with_meta(
                value, ts=ts, stale=False, age=max(0.0, time.time() - ts), refreshing=False
            )

        latest = self._cache.get(self._latest_key(kind, params))
        stale: dict | None = None
        if latest is not None and isinstance(latest.get("value"), dict):
            ts = float(latest.get("ts", 0))
            age = max(0.0, time.time() - ts)
            if age < self.stale_ttl:
                # Recent enough: serve it and do *not* recompute. This is what
                # keeps a busy annotation session from re-running the analyses
                # after every single contribution.
                return self._with_meta(
                    latest["value"], ts=ts, stale=True, age=age, refreshing=False
                )
            stale = self._with_meta(latest["value"], ts=ts, stale=True, age=age, refreshing=True)

        # Either nothing has ever been computed, or what we have is past the TTL.
        self._schedule(kind, params, epoch, compute_fn)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            time.sleep(_POLL_STEP)
            value = self.get(kind, params, epoch)
            if value is not None:
                return self._with_meta(
                    value, ts=time.time(), stale=False, age=0.0, refreshing=False
                )
        # Too slow to wait for: show the previous numbers if we have any (flagged
        # as refreshing), else tell the caller to render the polling placeholder.
        return stale
