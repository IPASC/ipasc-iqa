"""Local store for contributed annotators and their completed rankings.

Backed by the same ``diskcache`` engine as the result store: SQLite-backed,
process- and thread-safe, so it tolerates concurrent annotators in a public
deployment. Two kinds of entries share the cache, distinguished by a tag tuple:

- ``("user", email)``        → ``{email, name, created}``
- ``("session", sid)``       → a completed ranking (see :meth:`add_session`)

Identity is intentionally lightweight: there is no password. An annotator is
recognised by (lower-cased) email; their rankings accumulate under that key.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


def normalize_email(email: str) -> str:
    return email.strip().lower()


class AnnotatorStore:
    def __init__(self, path: Path | str) -> None:
        import diskcache

        self.path = Path(path)
        self._cache = diskcache.Cache(str(self.path))

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._cache.close()

    def __enter__(self) -> AnnotatorStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- users -------------------------------------------------------------

    def get_user(self, email: str) -> dict | None:
        return self._cache.get(("user", normalize_email(email)))

    def upsert_user(self, email: str, name: str) -> tuple[dict, bool]:
        """Return ``(user, is_new)``. Existing users keep their stored name."""
        key = ("user", normalize_email(email))
        existing = self._cache.get(key)
        if existing is not None:
            return existing, False
        user = {
            "email": normalize_email(email),
            "name": name.strip(),
            "created": datetime.now(UTC).isoformat(),
        }
        self._cache.set(key, user)
        return user, True

    # -- sessions ----------------------------------------------------------

    def add_session(self, result: dict) -> str:
        """Persist one completed ranking; returns the generated session id."""
        sid = uuid.uuid4().hex
        record = {**result, "session_id": sid, "created": datetime.now(UTC).isoformat()}
        self._cache.set(("session", sid), record)
        return sid

    def _session_keys(self) -> Iterator[tuple]:
        for k in self._cache:
            if isinstance(k, tuple) and k and k[0] == "session":
                yield k

    # -- ratings (single-stimulus MOS) ------------------------------------

    def add_rating(self, record: dict) -> str:
        """Persist one quantitative (1-4) rating; returns the generated id."""
        rid = uuid.uuid4().hex
        stored = {**record, "rating_id": rid, "created": datetime.now(UTC).isoformat()}
        self._cache.set(("rating", rid), stored)
        return rid

    def _rating_keys(self) -> Iterator[tuple]:
        for k in self._cache:
            if isinstance(k, tuple) and k and k[0] == "rating":
                yield k

    def ratings(self, *, dataset: str | None = None) -> list[dict]:
        out: list[dict] = []
        for key in self._rating_keys():
            rec = self._cache.get(key)
            if rec is None:
                continue
            if dataset is not None and rec.get("dataset") != dataset:
                continue
            out.append(rec)
        return out

    def rating_tasks(self, dataset: str) -> set[str | None]:
        """Distinct task values among contributed ratings for a dataset."""
        return {r.get("task") for r in self.ratings(dataset=dataset)}

    def _user_keys(self) -> Iterator[tuple]:
        for k in self._cache:
            if isinstance(k, tuple) and k and k[0] == "user":
                yield k

    def sessions(self, *, user_email: str | None = None) -> list[dict]:
        out: list[dict] = []
        target = normalize_email(user_email) if user_email else None
        for key in self._session_keys():
            rec = self._cache.get(key)
            if rec is None:
                continue
            if target is not None and rec.get("user_email") != target:
                continue
            out.append(rec)
        return out

    def count_sessions(self, *, user_email: str | None = None) -> int:
        return len(self.sessions(user_email=user_email))

    def dataset_tasks(self, dataset: str) -> set[str | None]:
        """Distinct task values among completed sessions for a dataset.

        May include ``None`` (a dataset annotated without sub-tasks).
        """
        out: set[str | None] = set()
        for key in self._session_keys():
            rec = self._cache.get(key)
            if rec is not None and rec.get("dataset") == dataset:
                out.add(rec.get("task"))
        return out

    def leaderboard(self) -> list[dict]:
        """Per-annotator totals: ``[{name, rankings, comparisons, ratings}]``.

        Includes registered annotators with no activity. Sorted by total
        contributions (rankings + ratings) desc, then name.
        """
        users = {k[1]: self._cache.get(k) for k in self._user_keys()}

        def _blank() -> dict[str, int]:
            return {"rankings": 0, "comparisons": 0, "ratings": 0}

        agg: dict[str, dict[str, int]] = {email: _blank() for email in users}
        for key in self._session_keys():
            rec = self._cache.get(key)
            if rec is None:
                continue
            stats = agg.setdefault(rec.get("user_email"), _blank())
            stats["rankings"] += 1
            stats["comparisons"] += int(rec.get("n_comparisons", 0))
        for key in self._rating_keys():
            rec = self._cache.get(key)
            if rec is None:
                continue
            agg.setdefault(rec.get("user_email"), _blank())["ratings"] += 1

        rows = [
            {
                "name": (users.get(email) or {}).get("name") or email,
                "rankings": stats["rankings"],
                "comparisons": stats["comparisons"],
                "ratings": stats["ratings"],
            }
            for email, stats in agg.items()
        ]
        rows.sort(key=lambda r: (-(r["rankings"] + r["ratings"]), r["name"].lower()))
        return rows
