"""Local store for users (annotators/admins) and their contributed annotations.

Backed by the same ``diskcache`` engine as the result store: SQLite-backed,
process- and thread-safe, so it tolerates concurrent users in a public
deployment.

The data lives in **two separate** diskcache directories so accounts and
contributions have independent lifecycles — deleting the annotations directory
must not wipe the user accounts, and bundling/"downloading" the annotation data
must not drag the accounts along. Each entry kind is distinguished by a tag
tuple:

*Users cache* (``users/``) — the accounts and everything that authenticates them:

- ``("user", uid)``            → an account (see :meth:`create_user`)
- ``("email", email)``         → ``uid`` (unique index for login / uniqueness)
- ``("authsession", token)``   → ``{"uid": uid}`` (login session, TTL'd)
- ``("reset", token)``         → ``{"uid": uid}`` (password reset, short TTL)
- ``("setting", key)``         → instance-wide admin flag

*Annotations cache* (``annotations/``) — the contributed data:

- ``("session", sid)``         → a completed ranking (see :meth:`add_session`)
- ``("rating", rid)``          → a single-stimulus rating (see :meth:`add_rating`)
- ``("uidx", uid)``            → per-user contribution index (see :meth:`user_index`)
- ``("meta", "results_epoch")``→ cache-invalidation counter
- ``("schema", "version")``    → migration marker

Accounts are keyed by a stable **uid** (uuid hex). Annotations reference that
uid (``user_uid``), so a user can change their name/email without breaking the
attribution of work they already did. Login is by (lower-cased) email, resolved
to a uid through the ``("email", …)`` index.

**Instances are shared per directory pair.** Opening a ``diskcache.Cache`` is not
free (SQLite connection + pragmas), and the web app touches the store on *every*
request — including the auth middleware. Constructing an :class:`AnnotatorStore`
for a pair of paths therefore returns the process-wide instance for those paths,
and :meth:`close` is a no-op; the underlying caches are thread-safe and live for
the process. Use :meth:`close_all` to genuinely release them (tests, CLI exit).
"""

from __future__ import annotations

import secrets
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# Roles.
ROLE_ADMIN = "admin"
ROLE_ANNOTATOR = "annotator"
ROLE_DELETED = "DELETED_USER"

# Bumped when the on-disk layout changes; gates the one-time migration.
SCHEMA_VERSION = 4

# Tag namespaces that belong in the *users* cache rather than the annotations one.
# Used by the v4 migration to relocate legacy entries out of a combined store.
_USER_DOMAIN_TAGS = frozenset({"user", "email", "authsession", "reset", "setting"})


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AnnotatorStore:
    """Two-cache store for accounts (users cache) and annotations (annotations cache).

    Parameters
    ----------
    path:
        The annotations cache directory (e.g. ``<workspace>/annotations``).
    users_path:
        The users cache directory. Defaults to the sibling ``users/`` folder
        next to ``path`` — i.e. ``<workspace>/users`` — so the two stores live
        side by side under the workspace root with independent lifecycles.
    """

    #: Process-wide instances, keyed by (annotations path, users path).
    _instances: dict[tuple[str, str], AnnotatorStore] = {}
    _instances_lock = threading.Lock()

    @staticmethod
    def _paths(path: Path | str, users_path: Path | str | None) -> tuple[Path, Path]:
        p = Path(path)
        return p, (Path(users_path) if users_path is not None else p.parent / "users")

    def __new__(cls, path: Path | str, users_path: Path | str | None = None) -> AnnotatorStore:
        p, up = cls._paths(path, users_path)
        key = (str(p.resolve()), str(up.resolve()))
        with cls._instances_lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = super().__new__(cls)
                inst._opened = False
                cls._instances[key] = inst
            return inst

    def __init__(self, path: Path | str, users_path: Path | str | None = None) -> None:
        import diskcache

        # __new__ hands back a shared instance; only the first call opens it.
        if self._opened:
            return
        self.path, self.users_path = self._paths(path, users_path)
        self._cache = diskcache.Cache(str(self.path))
        self._users = diskcache.Cache(str(self.users_path))
        self._index_lock = threading.Lock()
        self._opened = True
        self._migrate()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """No-op: the instance is shared process-wide (see :meth:`close_all`)."""

    @classmethod
    def close_all(cls) -> None:
        """Really close every shared instance and forget them."""
        with cls._instances_lock:
            instances = list(cls._instances.values())
            cls._instances.clear()
        for inst in instances:
            if getattr(inst, "_opened", False):
                inst._cache.close()
                inst._users.close()
                inst._opened = False

    def __enter__(self) -> AnnotatorStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- migration ---------------------------------------------------------

    def _migrate(self) -> None:
        """One-time upgrade of legacy data layouts.

        v1 keyed users by email and stamped annotations with ``user_email``.
        v2 keys users by uid and stamps annotations with ``user_uid``. Legacy
        accounts are migrated with ``role=annotator`` and no password, so they
        must use the reset flow to set one before they can log in.

        v3 adds an ``approved`` gate for contributing. Accounts that predate it
        are grandfathered in (``approved=True``) so the upgrade locks nobody out.

        v4 splits accounts out of the combined cache into a separate *users*
        cache, so deleting/exporting the annotations data leaves accounts intact.
        Any user-domain entries still sitting in the annotations cache are moved
        across (see :data:`_USER_DOMAIN_TAGS`).
        """
        version = self._cache.get(("schema", "version"))
        if version == SCHEMA_VERSION:
            return
        if version is None:
            # Distinguish a brand-new store from an un-migrated v1 store: a v1
            # store has ("user", <email-string>) entries.
            legacy_user_keys = [
                k
                for k in self._cache
                if isinstance(k, tuple) and len(k) == 2 and k[0] == "user" and "@" in str(k[1])
            ]
            email_to_uid: dict[str, str] = {}
            for key in legacy_user_keys:
                rec = self._cache.get(key)
                if rec is None:
                    continue
                email = normalize_email(rec.get("email", key[1]))
                uid = uuid.uuid4().hex
                email_to_uid[email] = uid
                self._cache.set(
                    ("user", uid),
                    {
                        "uid": uid,
                        "email": email,
                        "name": rec.get("name", ""),
                        "role": ROLE_ANNOTATOR,
                        "password_hash": None,
                        "created": rec.get("created", _now()),
                        "updated": _now(),
                    },
                )
                self._cache.set(("email", email), uid)
                self._cache.delete(key)

            # Stamp existing annotations with user_uid (derived from user_email).
            for key in list(self._session_keys()) + list(self._rating_keys()):
                rec = self._cache.get(key)
                if rec is None or rec.get("user_uid"):
                    continue
                uid = email_to_uid.get(normalize_email(rec.get("user_email", "")))
                if uid:
                    self._cache.set(key, {**rec, "user_uid": uid})

        # → v4: relocate any user-domain entries that still live in the
        # annotations cache (legacy combined store) into the users cache.
        for key in list(self._cache):
            if isinstance(key, tuple) and key and key[0] in _USER_DOMAIN_TAGS:
                self._users.set(key, self._cache.get(key))
                self._cache.delete(key)

        # → v3: grandfather every existing account past the approval gate.
        for key in list(self._user_keys()):
            rec = self._users.get(key)
            if rec is not None and "approved" not in rec:
                self._users.set(key, {**rec, "approved": True})

        self._cache.set(("schema", "version"), SCHEMA_VERSION)

    # -- users -------------------------------------------------------------

    def get_user(self, uid: str) -> dict | None:
        return self._users.get(("user", uid))

    def get_user_by_email(self, email: str) -> dict | None:
        uid = self._users.get(("email", normalize_email(email)))
        return self.get_user(uid) if uid else None

    def email_taken(self, email: str) -> bool:
        return self._users.get(("email", normalize_email(email))) is not None

    def create_user(
        self,
        *,
        email: str,
        name: str,
        role: str,
        password_hash: str | None,
        approved: bool = False,
        expertise: str | None = None,
    ) -> dict:
        """Create an account. Raises ``ValueError`` if the email is in use."""
        email = normalize_email(email)
        if self.email_taken(email):
            raise ValueError("email already registered")
        uid = uuid.uuid4().hex
        now = _now()
        user = {
            "uid": uid,
            "email": email,
            "name": name.strip(),
            "role": role,
            "password_hash": password_hash,
            "approved": approved,
            "expertise": expertise,
            "created": now,
            "updated": now,
        }
        self._users.set(("user", uid), user)
        self._users.set(("email", email), uid)
        return user

    def update_user(self, uid: str, **fields: object) -> dict | None:
        """Patch a user record, maintaining the email index. Returns the record.

        The read-modify-write runs inside a cache transaction so two concurrent
        edits of the same account cannot lose one another's changes.
        """
        with self._users.transact():
            user = self.get_user(uid)
            if user is None:
                return None
            new_email = fields.get("email")
            if new_email is not None:
                new_email = normalize_email(str(new_email))
                old_email = normalize_email(user.get("email", ""))
                if new_email != old_email:
                    if old_email:
                        self._users.delete(("email", old_email))
                    if new_email:
                        self._users.set(("email", new_email), uid)
                fields["email"] = new_email
            affects_results = "role" in fields or "approved" in fields
            user = {**user, **fields, "updated": _now()}
            self._users.set(("user", uid), user)
        if affects_results:
            self.bump_results_epoch()  # approval/role change alters which data counts
        return user

    def delete_user(self, uid: str) -> None:
        """Soft-delete: blank metadata, mark DELETED_USER, drop login means.

        The ``("user", uid)`` record is kept (so an admin could recover it), but
        the email index, password, and any live sessions are removed, so the
        account can no longer authenticate.
        """
        user = self.get_user(uid)
        if user is None:
            return
        old_email = normalize_email(user.get("email", ""))
        if old_email:
            self._users.delete(("email", old_email))
        self._users.set(
            ("user", uid),
            {
                "uid": uid,
                "email": "",
                "name": "",
                "role": ROLE_DELETED,
                "password_hash": None,
                "created": user.get("created", _now()),
                "updated": _now(),
            },
        )
        self.delete_sessions_for_uid(uid)
        self.bump_results_epoch()  # the user's contributions no longer count

    def _user_keys(self) -> Iterator[tuple]:
        for k in self._users:
            if isinstance(k, tuple) and k and k[0] == "user":
                yield k

    def all_users(self) -> list[dict]:
        out = []
        for k in self._user_keys():
            rec = self._users.get(k)
            if rec is not None:
                out.append(rec)
        return out

    def approved_uids(self) -> set[str]:
        """UIDs allowed to contribute to results: admins + approved annotators.

        Excludes pending (unapproved) and ``DELETED_USER`` accounts, so their
        annotations can be filtered out of the live aggregates.
        """
        return {
            u["uid"]
            for u in self.all_users()
            if u.get("role") == ROLE_ADMIN or bool(u.get("approved"))
        }

    # -- auth sessions -----------------------------------------------------

    def create_auth_session(self, uid: str, ttl_seconds: int) -> str:
        token = secrets.token_urlsafe(32)
        self._users.set(("authsession", token), {"uid": uid}, expire=ttl_seconds)
        return token

    def auth_session_uid(self, token: str) -> str | None:
        rec = self._users.get(("authsession", token))
        return rec.get("uid") if rec else None

    def delete_auth_session(self, token: str) -> None:
        self._users.delete(("authsession", token))

    def delete_sessions_for_uid(self, uid: str) -> None:
        for k in list(self._users):
            if isinstance(k, tuple) and k and k[0] == "authsession":
                rec = self._users.get(k)
                if rec is not None and rec.get("uid") == uid:
                    self._users.delete(k)

    # -- password-reset tokens --------------------------------------------

    def create_reset_token(self, uid: str, ttl_seconds: int) -> str:
        token = secrets.token_urlsafe(32)
        self._users.set(("reset", token), {"uid": uid}, expire=ttl_seconds)
        return token

    def pop_reset_uid(self, token: str) -> str | None:
        """Return the uid for a reset token and invalidate it (single use)."""
        rec = self._users.get(("reset", token))
        if rec is None:
            return None
        self._users.delete(("reset", token))
        return rec.get("uid")

    # -- instance settings (small, admin-toggled flags) --------------------

    def get_setting(self, key: str, default: object = None) -> object:
        return self._users.get(("setting", key), default)

    def set_setting(self, key: str, value: object) -> None:
        self._users.set(("setting", key), value)

    # -- results epoch -----------------------------------------------------
    # A monotonically increasing token bumped whenever something that affects
    # computed results changes (contributed annotations, approval/role changes,
    # account deletion, benchmark runs). The GUI caches analysis/standings/scatter
    # outputs keyed by this token, so unchanged views are served without recompute.

    def results_epoch(self) -> int:
        return int(self._cache.get(("meta", "results_epoch"), 0))

    def bump_results_epoch(self) -> None:
        self._cache.incr(("meta", "results_epoch"), 1)

    # -- per-user contribution index --------------------------------------
    # Drawing the next item and showing a contribution count used to scan every
    # session/rating in the store on each submit — O(n) SQLite reads that grew
    # with the study. Each user instead gets one index record summarising what
    # they have already done; writes keep it up to date, and it is rebuilt from
    # a scan on first use (so existing deployments need no migration).

    @staticmethod
    def _blank_index() -> dict:
        return {"sessions": {}, "ratings": {}, "n_comparisons": 0}

    def _scan_user_index(self, uid: str) -> dict:
        idx = self._blank_index()
        for key in self._session_keys():
            rec = self._cache.get(key)
            if rec is None or rec.get("user_uid") != uid:
                continue
            idx["sessions"][(rec.get("dataset"), rec.get("reference"), rec.get("task"))] = key[1]
            idx["n_comparisons"] += int(rec.get("n_comparisons", 0))
        for key in self._rating_keys():
            rec = self._cache.get(key)
            if rec is None or rec.get("user_uid") != uid:
                continue
            idx["ratings"][(rec.get("dataset"), rec.get("image_id"), rec.get("task"))] = key[1]
        return idx

    def user_index(self, uid: str) -> dict:
        """What *uid* has already contributed, as an O(1) lookup.

        ``{"sessions": {(dataset, reference, task): sid},
        "ratings": {(dataset, image_id, task): rid}, "n_comparisons": int}``.
        """
        idx = self._cache.get(("uidx", uid))
        if idx is None:
            idx = self._scan_user_index(uid)
            self._cache.set(("uidx", uid), idx)
        return idx

    def invalidate_user_indexes(self) -> None:
        """Drop every cached per-user index (after a bulk rewrite of records)."""
        for key in [k for k in self._cache if isinstance(k, tuple) and k and k[0] == "uidx"]:
            self._cache.delete(key)

    # -- sessions (ranking) ------------------------------------------------

    def add_session(self, result: dict) -> str:
        """Persist one completed ranking; returns the session id.

        Idempotent per ``(user, dataset, reference, task)``: re-submitting a set
        the user already ranked (double-click, duplicate tab) returns the
        existing id instead of storing a second, aggregate-skewing copy.
        """
        uid = result.get("user_uid")
        entry = (result.get("dataset"), result.get("reference"), result.get("task"))
        with self._index_lock:
            idx = self.user_index(uid) if uid else None
            if idx is not None and entry in idx["sessions"]:
                return str(idx["sessions"][entry])
            sid = uuid.uuid4().hex
            record = {**result, "session_id": sid, "created": _now()}
            self._cache.set(("session", sid), record)
            if idx is not None:
                idx["sessions"][entry] = sid
                idx["n_comparisons"] += int(result.get("n_comparisons", 0))
                self._cache.set(("uidx", uid), idx)
        self.bump_results_epoch()
        return sid

    def _session_keys(self) -> Iterator[tuple]:
        for k in self._cache:
            if isinstance(k, tuple) and k and k[0] == "session":
                yield k

    def sessions(self, *, user_uid: str | None = None) -> list[dict]:
        out: list[dict] = []
        for key in self._session_keys():
            rec = self._cache.get(key)
            if rec is None:
                continue
            if user_uid is not None and rec.get("user_uid") != user_uid:
                continue
            out.append(rec)
        return out

    def count_sessions(self, *, user_uid: str | None = None) -> int:
        if user_uid is not None:
            return len(self.user_index(user_uid)["sessions"])
        return len(self.sessions())

    def count_ratings(self, *, user_uid: str | None = None) -> int:
        if user_uid is not None:
            return len(self.user_index(user_uid)["ratings"])
        return len(self.ratings())

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

    # -- ratings (single-stimulus MOS) ------------------------------------

    def add_rating(self, record: dict) -> str:
        """Persist one quantitative (1-4) rating; returns the id.

        Idempotent per ``(user, dataset, image_id, task)`` — see
        :meth:`add_session`.
        """
        uid = record.get("user_uid")
        entry = (record.get("dataset"), record.get("image_id"), record.get("task"))
        with self._index_lock:
            idx = self.user_index(uid) if uid else None
            if idx is not None and entry in idx["ratings"]:
                return str(idx["ratings"][entry])
            rid = uuid.uuid4().hex
            stored = {**record, "rating_id": rid, "created": _now()}
            self._cache.set(("rating", rid), stored)
            if idx is not None:
                idx["ratings"][entry] = rid
                self._cache.set(("uidx", uid), idx)
        self.bump_results_epoch()
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

    # -- task maintenance --------------------------------------------------

    def retag_tasks(self, dataset: str, mapping: dict[str | None, str]) -> int:
        """Rewrite the ``task`` of a dataset's contributions; returns how many.

        Used to fold annotations that were recorded under a different (or
        absent) task name into the dataset's canonical task, so the same
        question does not end up split across two buckets in the analyses.
        """
        if not mapping:
            return 0
        changed = 0
        for key in list(self._session_keys()) + list(self._rating_keys()):
            rec = self._cache.get(key)
            if rec is None or rec.get("dataset") != dataset:
                continue
            old = rec.get("task")
            new = mapping.get(old)
            if new is None or new == old:
                continue
            self._cache.set(key, {**rec, "task": new})
            changed += 1
        if changed:
            self.invalidate_user_indexes()  # index keys embed the task
            self.bump_results_epoch()
        return changed

    # -- leaderboard -------------------------------------------------------

    def leaderboard(self) -> list[dict]:
        """Per-user totals: ``[{name, rankings, comparisons, ratings}]``.

        Keyed by uid so renames/email changes don't split a user's tally.
        Includes registered users with no activity; deleted users are omitted.
        Sorted by total contributions (rankings + ratings) desc, then name.
        """
        rows = []
        for user in self.all_users():
            if user.get("role") == ROLE_DELETED:
                continue
            idx = self.user_index(user["uid"])
            rows.append(
                {
                    "name": user.get("name") or "(unknown)",
                    "rankings": len(idx["sessions"]),
                    "comparisons": int(idx["n_comparisons"]),
                    "ratings": len(idx["ratings"]),
                }
            )
        rows.sort(key=lambda r: (-(r["rankings"] + r["ratings"]), r["name"].lower()))
        return rows
