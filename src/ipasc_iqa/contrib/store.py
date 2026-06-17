"""Local store for users (annotators/admins) and their contributed annotations.

Backed by the same ``diskcache`` engine as the result store: SQLite-backed,
process- and thread-safe, so it tolerates concurrent users in a public
deployment. Several kinds of entries share the cache, distinguished by a tag
tuple:

- ``("user", uid)``            → an account (see :meth:`create_user`)
- ``("email", email)``         → ``uid`` (unique index for login / uniqueness)
- ``("authsession", token)``   → ``{"uid": uid}`` (login session, TTL'd)
- ``("reset", token)``         → ``{"uid": uid}`` (password reset, short TTL)
- ``("session", sid)``         → a completed ranking (see :meth:`add_session`)
- ``("rating", rid)``          → a single-stimulus rating (see :meth:`add_rating`)
- ``("schema", "version")``    → migration marker

Accounts are keyed by a stable **uid** (uuid hex). Annotations reference that
uid (``user_uid``), so a user can change their name/email without breaking the
attribution of work they already did. Login is by (lower-cased) email, resolved
to a uid through the ``("email", …)`` index.
"""

from __future__ import annotations

import secrets
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
SCHEMA_VERSION = 3


def normalize_email(email: str) -> str:
    return email.strip().lower()


def _now() -> str:
    return datetime.now(UTC).isoformat()


class AnnotatorStore:
    def __init__(self, path: Path | str) -> None:
        import diskcache

        self.path = Path(path)
        self._cache = diskcache.Cache(str(self.path))
        self._migrate()

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

    # -- migration ---------------------------------------------------------

    def _migrate(self) -> None:
        """One-time upgrade of legacy (email-keyed, password-less) data.

        v1 keyed users by email and stamped annotations with ``user_email``.
        v2 keys users by uid and stamps annotations with ``user_uid``. Legacy
        accounts are migrated with ``role=annotator`` and no password, so they
        must use the reset flow to set one before they can log in.

        v3 adds an ``approved`` gate for contributing. Accounts that predate it
        are grandfathered in (``approved=True``) so the upgrade locks nobody out.
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

        # → v3: grandfather every existing account past the new approval gate.
        for key in list(self._user_keys()):
            rec = self._cache.get(key)
            if rec is not None and "approved" not in rec:
                self._cache.set(key, {**rec, "approved": True})

        self._cache.set(("schema", "version"), SCHEMA_VERSION)

    # -- users -------------------------------------------------------------

    def get_user(self, uid: str) -> dict | None:
        return self._cache.get(("user", uid))

    def get_user_by_email(self, email: str) -> dict | None:
        uid = self._cache.get(("email", normalize_email(email)))
        return self.get_user(uid) if uid else None

    def email_taken(self, email: str) -> bool:
        return self._cache.get(("email", normalize_email(email))) is not None

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
        self._cache.set(("user", uid), user)
        self._cache.set(("email", email), uid)
        return user

    def update_user(self, uid: str, **fields: object) -> dict | None:
        """Patch a user record, maintaining the email index. Returns the record."""
        user = self.get_user(uid)
        if user is None:
            return None
        new_email = fields.get("email")
        if new_email is not None:
            new_email = normalize_email(str(new_email))
            old_email = normalize_email(user.get("email", ""))
            if new_email != old_email:
                if old_email:
                    self._cache.delete(("email", old_email))
                if new_email:
                    self._cache.set(("email", new_email), uid)
            fields["email"] = new_email
        user = {**user, **fields, "updated": _now()}
        self._cache.set(("user", uid), user)
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
            self._cache.delete(("email", old_email))
        self._cache.set(
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

    def _user_keys(self) -> Iterator[tuple]:
        for k in self._cache:
            if isinstance(k, tuple) and k and k[0] == "user":
                yield k

    def all_users(self) -> list[dict]:
        out = []
        for k in self._user_keys():
            rec = self._cache.get(k)
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
        self._cache.set(("authsession", token), {"uid": uid}, expire=ttl_seconds)
        return token

    def auth_session_uid(self, token: str) -> str | None:
        rec = self._cache.get(("authsession", token))
        return rec.get("uid") if rec else None

    def delete_auth_session(self, token: str) -> None:
        self._cache.delete(("authsession", token))

    def delete_sessions_for_uid(self, uid: str) -> None:
        for k in list(self._cache):
            if isinstance(k, tuple) and k and k[0] == "authsession":
                rec = self._cache.get(k)
                if rec is not None and rec.get("uid") == uid:
                    self._cache.delete(k)

    # -- password-reset tokens --------------------------------------------

    def create_reset_token(self, uid: str, ttl_seconds: int) -> str:
        token = secrets.token_urlsafe(32)
        self._cache.set(("reset", token), {"uid": uid}, expire=ttl_seconds)
        return token

    def pop_reset_uid(self, token: str) -> str | None:
        """Return the uid for a reset token and invalidate it (single use)."""
        rec = self._cache.get(("reset", token))
        if rec is None:
            return None
        self._cache.delete(("reset", token))
        return rec.get("uid")

    # -- instance settings (small, admin-toggled flags) --------------------

    def get_setting(self, key: str, default: object = None) -> object:
        return self._cache.get(("setting", key), default)

    def set_setting(self, key: str, value: object) -> None:
        self._cache.set(("setting", key), value)

    # -- sessions (ranking) ------------------------------------------------

    def add_session(self, result: dict) -> str:
        """Persist one completed ranking; returns the generated session id."""
        sid = uuid.uuid4().hex
        record = {**result, "session_id": sid, "created": _now()}
        self._cache.set(("session", sid), record)
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
        return len(self.sessions(user_uid=user_uid))

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
        """Persist one quantitative (1-4) rating; returns the generated id."""
        rid = uuid.uuid4().hex
        stored = {**record, "rating_id": rid, "created": _now()}
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

    # -- leaderboard -------------------------------------------------------

    def leaderboard(self) -> list[dict]:
        """Per-user totals: ``[{name, rankings, comparisons, ratings}]``.

        Keyed by uid so renames/email changes don't split a user's tally.
        Includes registered users with no activity; deleted users are omitted.
        Sorted by total contributions (rankings + ratings) desc, then name.
        """
        users = {u["uid"]: u for u in self.all_users()}

        def _blank() -> dict[str, int]:
            return {"rankings": 0, "comparisons": 0, "ratings": 0}

        agg: dict[str, dict[str, int]] = {
            uid: _blank() for uid, u in users.items() if u.get("role") != ROLE_DELETED
        }
        for key in self._session_keys():
            rec = self._cache.get(key)
            if rec is None or not rec.get("user_uid"):
                continue
            stats = agg.setdefault(rec["user_uid"], _blank())
            stats["rankings"] += 1
            stats["comparisons"] += int(rec.get("n_comparisons", 0))
        for key in self._rating_keys():
            rec = self._cache.get(key)
            if rec is None or not rec.get("user_uid"):
                continue
            agg.setdefault(rec["user_uid"], _blank())["ratings"] += 1

        rows = []
        for uid, stats in agg.items():
            user = users.get(uid) or {}
            if user.get("role") == ROLE_DELETED:
                continue
            name = user.get("name") or "(unknown)"
            rows.append({"name": name, **stats})
        rows.sort(key=lambda r: (-(r["rankings"] + r["ratings"]), r["name"].lower()))
        return rows
