"""Unit tests for the persistent results view cache (caching + single-flight)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from ipasc_iqa.gui.result_cache import META_KEY, ResultCache


def _value(result: dict | None) -> dict | None:
    """Drop the freshness metadata the cache attaches to what it returns."""
    return None if result is None else {k: v for k, v in result.items() if k != META_KEY}


def test_caches_per_epoch_and_recomputes_on_change(tmp_path: Path) -> None:
    calls = {"n": 0}

    def compute() -> dict:
        calls["n"] += 1
        return {"v": calls["n"]}

    # stale_ttl=0 → never serve a value once the data behind it changed.
    with ResultCache(tmp_path / "vc", stale_ttl=0) as rc:
        # Miss → compute once and return.
        assert _value(rc.get_or_schedule("k", {"a": 1}, 1, compute, grace=2.0)) == {"v": 1}
        # Hit (same epoch) → served from cache, no recompute.
        assert _value(rc.get_or_schedule("k", {"a": 1}, 1, compute, grace=2.0)) == {"v": 1}
        assert calls["n"] == 1
        # Different params → its own entry.
        assert _value(rc.get_or_schedule("k", {"a": 2}, 1, compute, grace=2.0)) == {"v": 2}
        # New epoch (data changed) → recompute.
        assert _value(rc.get_or_schedule("k", {"a": 1}, 2, compute, grace=2.0)) == {"v": 3}


def test_recent_value_is_served_without_recomputing(tmp_path: Path) -> None:
    """A changed epoch does not recompute while the last value is still young."""
    calls = {"n": 0}

    def compute() -> dict:
        calls["n"] += 1
        return {"v": calls["n"]}

    with ResultCache(tmp_path / "vc", stale_ttl=3600) as rc:
        assert _value(rc.get_or_schedule("k", {}, 1, compute, grace=2.0)) == {"v": 1}
        # Epochs 2 and 3 (e.g. two fresh annotations) reuse the recent value.
        for epoch in (2, 3):
            out = rc.get_or_schedule("k", {}, epoch, compute, grace=2.0)
            assert _value(out) == {"v": 1}
            assert out[META_KEY]["stale"] is True
            assert out[META_KEY]["refreshing"] is False
        assert calls["n"] == 1
        assert not rc.is_recomputing("k", {})


def test_expired_value_is_served_while_refreshing(tmp_path: Path) -> None:
    """Past the TTL the old value still comes back — the refresh happens behind it."""

    def compute() -> dict:
        return {"v": 1}

    def compute2() -> dict:
        time.sleep(0.2)
        return {"v": 2}

    with ResultCache(tmp_path / "vc", stale_ttl=0.05) as rc:
        assert _value(rc.get_or_schedule("k", {}, 1, compute, grace=2.0)) == {"v": 1}
        time.sleep(0.1)  # let it age past the TTL
        # Grace shorter than the recompute → the caller must not be left waiting.
        out = rc.get_or_schedule("k", {}, 2, compute2, grace=0.02)
        # The caller is never blocked: it gets the old numbers, flagged as refreshing.
        assert _value(out) == {"v": 1}
        assert out[META_KEY]["refreshing"] is True
        time.sleep(0.5)
        assert rc.get("k", {}, 2) == {"v": 2}


def test_single_flight_no_parallel_recompute(tmp_path: Path) -> None:
    starts = []
    lock = threading.Lock()

    def compute() -> dict:
        with lock:
            starts.append(1)
        time.sleep(0.2)  # hold the compute so concurrent callers pile up
        return {"v": 1}

    with ResultCache(tmp_path / "vc", stale_ttl=0) as rc:
        results: list[dict | None] = []
        rlock = threading.Lock()

        def worker() -> None:
            r = _value(rc.get_or_schedule("k", {}, 1, compute, grace=2.0))
            with rlock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # All five concurrent viewers got the value, but it was computed only once.
    assert results == [{"v": 1}] * 5
    assert len(starts) == 1


def test_returns_none_when_recompute_exceeds_grace(tmp_path: Path) -> None:
    def slow() -> dict:
        time.sleep(0.5)
        return {"v": 1}

    with ResultCache(tmp_path / "vc", stale_ttl=0) as rc:
        # Grace shorter than the compute → caller gets None (shows "recomputing").
        assert rc.get_or_schedule("k", {}, 1, slow, grace=0.05) is None
        # Once the background compute finishes, the value is cached.
        time.sleep(0.7)
        assert rc.get("k", {}, 1) == {"v": 1}
