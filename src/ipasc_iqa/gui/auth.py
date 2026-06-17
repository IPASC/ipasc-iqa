"""Authentication & account management — pure logic, no web framework.

Sits alongside :mod:`ipasc_iqa.gui.service`. Passwords are hashed with bcrypt
(salt is embedded in the hash). Login sessions and password-reset tokens are
stored in the same :class:`~ipasc_iqa.contrib.store.AnnotatorStore` as the
accounts they reference.

Accounts are keyed by a stable uid; annotations reference that uid, so changing
a name or email never re-attributes past work.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import bcrypt

from ipasc_iqa.contrib.store import (
    ROLE_ADMIN,
    ROLE_ANNOTATOR,
    ROLE_DELETED,
    AnnotatorStore,
    normalize_email,
)

if TYPE_CHECKING:
    from ipasc_iqa.gui.service import Workspace

logger = logging.getLogger("ipasc_iqa.gui.auth")

COOKIE_NAME = "ipasc_session"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 days
RESET_TTL_SECONDS = 60 * 60  # 1 hour


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str | None) -> bool:
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Registration / authentication
# ---------------------------------------------------------------------------


def register_user(workspace: Workspace, name: str, email: str, password: str) -> dict:
    """Self-registration: create an annotator account. Raises ``ValueError``."""
    name, email = name.strip(), normalize_email(email)
    if not name or "@" not in email:
        raise ValueError("A name and a valid email are required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    with AnnotatorStore(workspace.annotations_path) as store:
        # email_taken covers DELETED_USER too (their index entry is removed, so a
        # freed email can be reused — which is the desired behaviour).
        return store.create_user(
            email=email,
            name=name,
            role=ROLE_ANNOTATOR,
            password_hash=hash_password(password),
        )


def authenticate(workspace: Workspace, email: str, password: str) -> dict | None:
    """Return the user for valid credentials, else ``None``.

    Rejects deleted accounts and accounts without a password (e.g. migrated
    legacy users who must reset first).
    """
    with AnnotatorStore(workspace.annotations_path) as store:
        user = store.get_user_by_email(email)
    if user is None or user.get("role") == ROLE_DELETED:
        return None
    if not verify_password(password, user.get("password_hash")):
        return None
    return user


# ---------------------------------------------------------------------------
# Login sessions
# ---------------------------------------------------------------------------


def create_session(workspace: Workspace, uid: str) -> str:
    with AnnotatorStore(workspace.annotations_path) as store:
        return store.create_auth_session(uid, SESSION_TTL_SECONDS)


def get_session_user(workspace: Workspace, token: str | None) -> dict | None:
    """Resolve a session-cookie token to its (non-deleted) user, or ``None``."""
    if not token:
        return None
    with AnnotatorStore(workspace.annotations_path) as store:
        uid = store.auth_session_uid(token)
        if uid is None:
            return None
        user = store.get_user(uid)
    if user is None or user.get("role") == ROLE_DELETED:
        return None
    return user


def destroy_session(workspace: Workspace, token: str | None) -> None:
    if not token:
        return
    with AnnotatorStore(workspace.annotations_path) as store:
        store.delete_auth_session(token)


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------


def issue_reset_token(workspace: Workspace, email: str) -> str | None:
    """Create a single-use reset token for ``email``, or ``None`` if no account.

    Callers must respond identically whether or not this returns a token, to
    avoid leaking which emails are registered.
    """
    with AnnotatorStore(workspace.annotations_path) as store:
        user = store.get_user_by_email(email)
        if user is None or user.get("role") == ROLE_DELETED:
            return None
        return store.create_reset_token(user["uid"], RESET_TTL_SECONDS)


def consume_reset_token(workspace: Workspace, token: str, new_password: str) -> bool:
    """Set a new password from a valid reset token (single use). Returns success."""
    if len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    with AnnotatorStore(workspace.annotations_path) as store:
        uid = store.pop_reset_uid(token)
        if uid is None:
            return False
        user = store.get_user(uid)
        if user is None or user.get("role") == ROLE_DELETED:
            return False
        store.update_user(uid, password_hash=hash_password(new_password))
        return True


# ---------------------------------------------------------------------------
# Profile / account management
# ---------------------------------------------------------------------------


def update_profile(
    workspace: Workspace,
    uid: str,
    *,
    name: str,
    email: str,
    new_password: str | None = None,
) -> dict:
    """Update a user's name/email/password. Raises ``ValueError`` on conflict."""
    name, email = name.strip(), normalize_email(email)
    if not name or "@" not in email:
        raise ValueError("A name and a valid email are required.")
    with AnnotatorStore(workspace.annotations_path) as store:
        user = store.get_user(uid)
        if user is None:
            raise ValueError("Account not found.")
        if email != normalize_email(user.get("email", "")) and store.email_taken(email):
            raise ValueError("That email is already in use.")
        fields: dict[str, object] = {"name": name, "email": email}
        if new_password:
            if len(new_password) < 8:
                raise ValueError("Password must be at least 8 characters.")
            fields["password_hash"] = hash_password(new_password)
        updated = store.update_user(uid, **fields)
    assert updated is not None
    return updated


def delete_account(workspace: Workspace, uid: str) -> None:
    """Honour an account-deletion request (soft delete; see store.delete_user)."""
    with AnnotatorStore(workspace.annotations_path) as store:
        store.delete_user(uid)


# ---------------------------------------------------------------------------
# Admin bootstrap (used by the CLI)
# ---------------------------------------------------------------------------


def promote_admin(
    workspace: Workspace,
    email: str,
    *,
    name: str | None = None,
    password: str | None = None,
) -> dict:
    """Create an admin, or promote an existing account to admin.

    If ``password`` is given it is (re)set. Used by ``ipasc_iqa-gui create-admin``.
    """
    email = normalize_email(email)
    with AnnotatorStore(workspace.annotations_path) as store:
        user = store.get_user_by_email(email)
        if user is None:
            if not password:
                raise ValueError("A password is required to create a new admin.")
            return store.create_user(
                email=email,
                name=(name or email).strip(),
                role=ROLE_ADMIN,
                password_hash=hash_password(password),
            )
        fields: dict[str, object] = {"role": ROLE_ADMIN}
        if name:
            fields["name"] = name.strip()
        if password:
            fields["password_hash"] = hash_password(password)
        updated = store.update_user(user["uid"], **fields)
    assert updated is not None
    return updated


__all__ = [
    "ROLE_ADMIN",
    "ROLE_ANNOTATOR",
    "ROLE_DELETED",
    "authenticate",
    "consume_reset_token",
    "create_session",
    "delete_account",
    "destroy_session",
    "get_session_user",
    "hash_password",
    "issue_reset_token",
    "promote_admin",
    "register_user",
    "update_profile",
    "verify_password",
]
