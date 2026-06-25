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

# Instance setting: when on, newly self-registered accounts are approved
# immediately (handy for testing or a conference sign-up spike). Off by default.
AUTO_APPROVE_KEY = "auto_approve_registrations"

# Self-assessed photoacoustic-imaging (PAI) expertise, captured at sign-up so the
# contributed data can later be filtered by rater expertise for a publication.
EXPERTISE_LEVELS = ("expert", "knowledgeable", "novice")
EXPERTISE_LABELS = {"expert": "Expert", "knowledgeable": "Knowledgeable", "novice": "Novice"}


def normalize_expertise(value: str | None) -> str:
    """Validate a self-assessed expertise choice, returning the canonical key."""
    key = (value or "").strip().lower()
    if key not in EXPERTISE_LEVELS:
        raise ValueError("Please select your PAI expertise.")
    return key


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


def register_user(
    workspace: Workspace,
    name: str,
    email: str,
    password: str,
    expertise: str | None = None,
) -> dict:
    """Self-registration: create an annotator account. Raises ``ValueError``.

    ``expertise`` is the rater's self-assessed PAI expertise (required by the
    sign-up form; ``None`` only for programmatic callers). The account is
    approved immediately only when the instance's auto-approve flag is on (see
    :func:`auto_approve_enabled`); otherwise it starts pending and an admin must
    approve it before it can contribute annotations.
    """
    name, email = name.strip(), normalize_email(email)
    if not name or "@" not in email:
        raise ValueError("A name and a valid email are required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    expertise = normalize_expertise(expertise) if expertise is not None else None
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        approved = bool(store.get_setting(AUTO_APPROVE_KEY, False))
        # email_taken covers DELETED_USER too (their index entry is removed, so a
        # freed email can be reused — which is the desired behaviour).
        return store.create_user(
            email=email,
            name=name,
            role=ROLE_ANNOTATOR,
            password_hash=hash_password(password),
            approved=approved,
            expertise=expertise,
        )


def authenticate(workspace: Workspace, email: str, password: str) -> dict | None:
    """Return the user for valid credentials, else ``None``.

    Rejects deleted accounts and accounts without a password (e.g. migrated
    legacy users who must reset first).
    """
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
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
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        return store.create_auth_session(uid, SESSION_TTL_SECONDS)


def get_session_user(workspace: Workspace, token: str | None) -> dict | None:
    """Resolve a session-cookie token to its (non-deleted) user, or ``None``."""
    if not token:
        return None
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
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
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        store.delete_auth_session(token)


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------


def issue_reset_token(workspace: Workspace, email: str) -> str | None:
    """Create a single-use reset token for ``email``, or ``None`` if no account.

    Callers must respond identically whether or not this returns a token, to
    avoid leaking which emails are registered.
    """
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        user = store.get_user_by_email(email)
        if user is None or user.get("role") == ROLE_DELETED:
            return None
        return store.create_reset_token(user["uid"], RESET_TTL_SECONDS)


def consume_reset_token(workspace: Workspace, token: str, new_password: str) -> bool:
    """Set a new password from a valid reset token (single use). Returns success."""
    if len(new_password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
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
    expertise: str | None = None,
) -> dict:
    """Update a user's name/email/password/expertise. Raises ``ValueError``."""
    name, email = name.strip(), normalize_email(email)
    if not name or "@" not in email:
        raise ValueError("A name and a valid email are required.")
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        user = store.get_user(uid)
        if user is None:
            raise ValueError("Account not found.")
        if email != normalize_email(user.get("email", "")) and store.email_taken(email):
            raise ValueError("That email is already in use.")
        fields: dict[str, object] = {"name": name, "email": email}
        if expertise is not None:
            fields["expertise"] = normalize_expertise(expertise)
        if new_password:
            if len(new_password) < 8:
                raise ValueError("Password must be at least 8 characters.")
            fields["password_hash"] = hash_password(new_password)
        updated = store.update_user(uid, **fields)
    assert updated is not None
    return updated


def delete_account(workspace: Workspace, uid: str) -> None:
    """Honour an account-deletion request (soft delete; see store.delete_user)."""
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        store.delete_user(uid)


# ---------------------------------------------------------------------------
# Approval gate + instance settings
# ---------------------------------------------------------------------------


def is_approved(user: dict | None) -> bool:
    """Whether a user may contribute annotations. Admins are always approved."""
    if not user:
        return False
    if user.get("role") == ROLE_ADMIN:
        return True
    return bool(user.get("approved"))


def auto_approve_enabled(workspace: Workspace) -> bool:
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        return bool(store.get_setting(AUTO_APPROVE_KEY, False))


def set_auto_approve(workspace: Workspace, enabled: bool) -> None:
    """Toggle the instance-wide auto-approval flag for new registrations."""
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        store.set_setting(AUTO_APPROVE_KEY, bool(enabled))


def set_user_approved(workspace: Workspace, uid: str, approved: bool) -> dict | None:
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        return store.update_user(uid, approved=bool(approved))


def set_user_role(workspace: Workspace, uid: str, role: str) -> dict:
    """Change an account's role (admin ↔ annotator). Raises ``ValueError``.

    Refuses to demote the last remaining admin (which would lock everyone out of
    admin functions) and won't touch deleted accounts. Promoting to admin also
    marks the account approved, since admins are always approved.
    """
    if role not in (ROLE_ADMIN, ROLE_ANNOTATOR):
        raise ValueError("Invalid role.")
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        user = store.get_user(uid)
        if user is None or user.get("role") == ROLE_DELETED:
            raise ValueError("Account not found.")
        if user.get("role") == ROLE_ADMIN and role != ROLE_ADMIN:
            admins = [u for u in store.all_users() if u.get("role") == ROLE_ADMIN]
            if len(admins) <= 1:
                raise ValueError("Cannot demote the last remaining admin.")
        fields: dict[str, object] = {"role": role}
        if role == ROLE_ADMIN:
            fields["approved"] = True
        updated = store.update_user(uid, **fields)
    assert updated is not None
    return updated


def approve_user_by_email(workspace: Workspace, email: str) -> bool:
    """Approve the account with this email (used by the CLI). Returns success."""
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        user = store.get_user_by_email(email)
        if user is None:
            return False
        store.update_user(user["uid"], approved=True)
        return True


def list_accounts(workspace: Workspace) -> list[dict]:
    """All accounts (without password hashes) for the admin users page.

    Pending (unapproved, non-deleted) accounts are surfaced first, then by name.
    """
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        users = store.all_users()
    out = [
        {
            "uid": u["uid"],
            "name": u.get("name", ""),
            "email": u.get("email", ""),
            "role": u.get("role"),
            "approved": is_approved(u),
            "deleted": u.get("role") == ROLE_DELETED,
            "expertise": u.get("expertise"),
        }
        for u in users
    ]
    out.sort(key=lambda u: (u["deleted"], u["approved"], u["name"].lower()))
    return out


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
    with AnnotatorStore(workspace.annotations_path, workspace.users_path) as store:
        user = store.get_user_by_email(email)
        if user is None:
            if not password:
                raise ValueError("A password is required to create a new admin.")
            return store.create_user(
                email=email,
                name=(name or email).strip(),
                role=ROLE_ADMIN,
                password_hash=hash_password(password),
                approved=True,
            )
        fields: dict[str, object] = {"role": ROLE_ADMIN, "approved": True}
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
    "approve_user_by_email",
    "authenticate",
    "auto_approve_enabled",
    "consume_reset_token",
    "create_session",
    "delete_account",
    "destroy_session",
    "get_session_user",
    "hash_password",
    "is_approved",
    "issue_reset_token",
    "list_accounts",
    "promote_admin",
    "register_user",
    "set_auto_approve",
    "set_user_approved",
    "set_user_role",
    "update_profile",
    "verify_password",
]
