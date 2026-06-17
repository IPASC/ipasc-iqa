"""Unit tests for the auth layer: accounts, sessions, reset, profile, migration."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("bcrypt")

from ipasc_iqa.contrib.store import (  # noqa: E402
    ROLE_ADMIN,
    ROLE_ANNOTATOR,
    ROLE_DELETED,
    SCHEMA_VERSION,
    AnnotatorStore,
)
from ipasc_iqa.gui import auth, service  # noqa: E402


def _ws(tmp_path: Path) -> service.Workspace:
    ws = service.Workspace.resolve(tmp_path / "ws", tmp_path / "data")
    ws.ensure()
    return ws


def test_register_and_authenticate(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    user = auth.register_user(ws, "Ann", "  Ann@Example.com ", "password123")
    assert user["role"] == ROLE_ANNOTATOR
    assert user["email"] == "ann@example.com"  # normalised
    assert auth.authenticate(ws, "ann@example.com", "password123")["uid"] == user["uid"]
    assert auth.authenticate(ws, "ann@example.com", "wrong") is None
    with pytest.raises(ValueError):
        auth.register_user(ws, "Other", "ann@example.com", "password123")


def test_password_minimum_length(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        auth.register_user(_ws(tmp_path), "X", "x@y.com", "short")


def test_sessions(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    u = auth.register_user(ws, "A", "a@b.com", "password123")
    token = auth.create_session(ws, u["uid"])
    assert auth.get_session_user(ws, token)["uid"] == u["uid"]
    auth.destroy_session(ws, token)
    assert auth.get_session_user(ws, token) is None
    assert auth.get_session_user(ws, None) is None


def test_reset_flow_single_use(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    u = auth.register_user(ws, "A", "a@b.com", "password123")
    token = auth.issue_reset_token(ws, "a@b.com")
    assert token
    assert auth.consume_reset_token(ws, token, "newpassword1") is True
    assert auth.consume_reset_token(ws, token, "newpassword2") is False  # single use
    assert auth.authenticate(ws, "a@b.com", "newpassword1")["uid"] == u["uid"]
    # No token (and no enumeration) for an unknown email.
    assert auth.issue_reset_token(ws, "nobody@b.com") is None


def test_profile_update_keeps_uid_and_frees_email(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    u = auth.register_user(ws, "A", "a@b.com", "password123")
    updated = auth.update_profile(
        ws, u["uid"], name="A2", email="a2@b.com", new_password="password999"
    )
    assert updated["uid"] == u["uid"]  # uid is stable across edits
    assert updated["email"] == "a2@b.com"
    assert auth.authenticate(ws, "a2@b.com", "password999")["uid"] == u["uid"]
    assert auth.authenticate(ws, "a@b.com", "password123") is None  # old email freed


def test_profile_email_conflict(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    auth.register_user(ws, "A", "a@b.com", "password123")
    b = auth.register_user(ws, "B", "b@b.com", "password123")
    with pytest.raises(ValueError):
        auth.update_profile(ws, b["uid"], name="B", email="a@b.com")


def test_delete_account_blocks_login(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    u = auth.register_user(ws, "A", "a@b.com", "password123")
    auth.delete_account(ws, u["uid"])
    assert auth.authenticate(ws, "a@b.com", "password123") is None
    with AnnotatorStore(ws.annotations_path) as store:
        rec = store.get_user(u["uid"])
        assert rec["role"] == ROLE_DELETED and rec["email"] == ""
    # The freed email can be reused for a fresh account.
    auth.register_user(ws, "C", "a@b.com", "password123")


def test_promote_admin_creates_and_promotes(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    admin = auth.promote_admin(ws, "boss@b.com", name="Boss", password="password123")
    assert admin["role"] == ROLE_ADMIN
    assert auth.authenticate(ws, "boss@b.com", "password123")["role"] == ROLE_ADMIN
    auth.register_user(ws, "A", "a@b.com", "password123")
    auth.promote_admin(ws, "a@b.com")  # promote existing, keep password
    assert auth.authenticate(ws, "a@b.com", "password123")["role"] == ROLE_ADMIN


def test_registration_pending_until_auto_approve(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    assert auth.auto_approve_enabled(ws) is False
    pending = auth.register_user(ws, "A", "a@b.com", "password123")
    assert auth.is_approved(pending) is False

    auth.set_auto_approve(ws, True)
    assert auth.auto_approve_enabled(ws) is True
    approved = auth.register_user(ws, "B", "b@b.com", "password123")
    assert auth.is_approved(approved) is True

    auth.set_auto_approve(ws, False)
    assert auth.register_user(ws, "C", "c@b.com", "password123")["approved"] is False


def test_set_approved_and_admin_always_approved(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    u = auth.register_user(ws, "A", "a@b.com", "password123")
    assert auth.is_approved(u) is False
    auth.set_user_approved(ws, u["uid"], True)
    assert auth.is_approved(auth.authenticate(ws, "a@b.com", "password123")) is True
    assert auth.approve_user_by_email(ws, "nobody@b.com") is False
    admin = auth.promote_admin(ws, "boss@b.com", name="Boss", password="password123")
    assert admin["approved"] is True and auth.is_approved(admin) is True


def test_expertise_captured_validated_and_editable(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    u = auth.register_user(ws, "A", "a@b.com", "password123", "Expert")
    assert u["expertise"] == "expert"  # normalised to canonical key
    with pytest.raises(ValueError):
        auth.register_user(ws, "B", "b@b.com", "password123", "wizard")
    updated = auth.update_profile(ws, u["uid"], name="A", email="a@b.com", expertise="novice")
    assert updated["expertise"] == "novice"
    by_email = {a["email"]: a["expertise"] for a in auth.list_accounts(ws)}
    assert by_email["a@b.com"] == "novice"


def test_migration_v2_to_v3_grandfathers_approval(tmp_path: Path) -> None:
    import diskcache

    ws = _ws(tmp_path)
    cache = diskcache.Cache(str(ws.annotations_path))
    cache.set(("schema", "version"), 2)
    cache.set(
        ("user", "abc"),
        {
            "uid": "abc",
            "email": "old@b.com",
            "name": "Old",
            "role": "annotator",
            "password_hash": None,
        },
    )
    cache.set(("email", "old@b.com"), "abc")
    cache.close()
    with AnnotatorStore(ws.annotations_path) as store:
        assert store._cache.get(("schema", "version")) == SCHEMA_VERSION
        assert store.get_user("abc")["approved"] is True  # grandfathered


def test_migration_from_legacy_v1(tmp_path: Path) -> None:
    import diskcache

    ws = _ws(tmp_path)
    # Simulate a v1 store: email-keyed user + a session stamped with user_email.
    cache = diskcache.Cache(str(ws.annotations_path))
    cache.set(("user", "legacy@b.com"), {"email": "legacy@b.com", "name": "Legacy"})
    cache.set(
        ("session", "s1"),
        {"user_email": "legacy@b.com", "dataset": "d", "task": None, "n_comparisons": 0},
    )
    cache.close()

    with AnnotatorStore(ws.annotations_path) as store:
        assert store._cache.get(("schema", "version")) == SCHEMA_VERSION
        assert store._cache.get(("user", "legacy@b.com")) is None  # old key removed
        user = store.get_user_by_email("legacy@b.com")
        assert user is not None
        assert user["role"] == ROLE_ANNOTATOR and user["password_hash"] is None
        assert user["approved"] is True  # legacy contributors grandfathered in
        sessions = store.sessions()
        assert sessions and sessions[0]["user_uid"] == user["uid"]
