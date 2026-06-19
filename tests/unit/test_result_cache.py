"""Unit tests for the persistent results view cache (caching + single-flight)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from ipasc_iqa.gui.result_cache import ResultCache


def test_caches_per_epoch_and_recomputes_on_change(tmp_path: Path) -> None:
    calls = {"n": 0}

    def compute() -> dict:
        calls["n"] += 1
        return {"v": calls["n"]}

    with ResultCache(tmp_path / "vc") as rc:
        # Miss → compute once and return.
        assert rc.get_or_schedule("k", {"a": 1}, 1, compute, grace=2.0) == {"v": 1}
        # Hit (same epoch) → served from cache, no recompute.
        assert rc.get_or_schedule("k", {"a": 1}, 1, compute, grace=2.0) == {"v": 1}
        assert calls["n"] == 1
        # Different params → its own entry.
        assert rc.get_or_schedule("k", {"a": 2}, 1, compute, grace=2.0) == {"v": 2}
        # New epoch (data changed) → recompute.
        assert rc.get_or_schedule("k", {"a": 1}, 2, compute, grace=2.0) == {"v": 3}


def test_single_flight_no_parallel_recompute(tmp_path: Path) -> None:
    starts = []
    lock = threading.Lock()

    def compute() -> dict:
        with lock:
            starts.append(1)
        time.sleep(0.2)  # hold the compute so concurrent callers pile up
        return {"v": 1}

    with ResultCache(tmp_path / "vc") as rc:
        results: list[dict | None] = []
        rlock = threading.Lock()

        def worker() -> None:
            r = rc.get_or_schedule("k", {}, 1, compute, grace=2.0)
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

    with ResultCache(tmp_path / "vc") as rc:
        # Grace shorter than the compute → caller gets None (shows "recomputing").
        assert rc.get_or_schedule("k", {}, 1, slow, grace=0.05) is None
        # Once the background compute finishes, the value is cached.
        time.sleep(0.7)
        assert rc.get("k", {}, 1) == {"v": 1}
