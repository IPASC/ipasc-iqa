"""Authentication, profile, and account-management routes (full-page forms)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from ipasc_iqa.gui import auth, service
from ipasc_iqa.gui.app import templates

router = APIRouter()
logger = logging.getLogger("ipasc_iqa.gui.auth")


def _ws(request: Request) -> service.Workspace:
    return request.app.state.workspace


def _set_session_cookie(request: Request, resp: Response, token: str) -> None:
    resp.set_cookie(
        auth.COOKIE_NAME,
        token,
        max_age=auth.SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=request.app.state.cookie_secure,
    )


def _redirect_with_session(request: Request, uid: str, to: str = "/") -> Response:
    token = auth.create_session(_ws(request), uid)
    resp = RedirectResponse(to, status_code=303)
    _set_session_cookie(request, resp, token)
    return resp


# ---------------------------------------------------------------------------
# Login / logout
# ---------------------------------------------------------------------------


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request) -> Response:
    if request.state.user is not None:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {})


@router.post("/login")
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
) -> Response:
    user = auth.authenticate(_ws(request), email, password)
    if user is None:
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid email or password.", "email": email},
            status_code=401,
        )
    return _redirect_with_session(request, user["uid"])


@router.post("/logout")
def logout(request: Request) -> Response:
    auth.destroy_session(_ws(request), request.cookies.get(auth.COOKIE_NAME))
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(auth.COOKIE_NAME)
    return resp


# ---------------------------------------------------------------------------
# Self-registration
# ---------------------------------------------------------------------------


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request) -> Response:
    if request.state.user is not None:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "register.html", {})


@router.post("/register")
def register_submit(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    expertise: str = Form(...),
) -> Response:
    try:
        user = auth.register_user(_ws(request), name, email, password, expertise)
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "register.html",
            {"error": str(exc), "name": name, "email": email, "expertise": expertise},
            status_code=400,
        )
    return _redirect_with_session(request, user["uid"])


# ---------------------------------------------------------------------------
# Password reset
# ---------------------------------------------------------------------------


@router.get("/forgot", response_class=HTMLResponse)
def forgot_page(request: Request) -> Response:
    return templates.TemplateResponse(request, "forgot.html", {})


@router.post("/forgot")
def forgot_submit(request: Request, email: str = Form(...)) -> Response:
    token = auth.issue_reset_token(_ws(request), email)
    if token is not None:
        # No email backend: log the link so an operator can relay it.
        reset_url = f"{str(request.base_url).rstrip('/')}/reset?token={token}"
        logger.warning("Password reset requested for %s — reset link: %s", email, reset_url)
    # Always identical response (don't reveal whether the email is registered).
    return templates.TemplateResponse(
        request,
        "forgot.html",
        {"sent": True},
    )


@router.get("/reset", response_class=HTMLResponse)
def reset_page(request: Request, token: str = "") -> Response:
    return templates.TemplateResponse(request, "reset.html", {"token": token})


@router.post("/reset")
def reset_submit(
    request: Request,
    token: str = Form(...),
    password: str = Form(...),
    confirm: str = Form(...),
) -> Response:
    if password != confirm:
        return templates.TemplateResponse(
            request,
            "reset.html",
            {"token": token, "error": "Passwords do not match."},
            status_code=400,
        )
    try:
        ok = auth.consume_reset_token(_ws(request), token, password)
    except ValueError as exc:
        return templates.TemplateResponse(
            request, "reset.html", {"token": token, "error": str(exc)}, status_code=400
        )
    if not ok:
        return templates.TemplateResponse(
            request,
            "reset.html",
            {"error": "This reset link is invalid or has expired."},
            status_code=400,
        )
    return templates.TemplateResponse(
        request, "login.html", {"message": "Password updated — please sign in."}
    )


# ---------------------------------------------------------------------------
# Profile / account deletion
# ---------------------------------------------------------------------------


@router.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request) -> Response:
    return templates.TemplateResponse(request, "profile.html", {"profile": request.state.user})


@router.post("/profile")
def profile_submit(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    expertise: str = Form(...),
    new_password: str = Form(default=""),
) -> Response:
    user = request.state.user
    try:
        updated = auth.update_profile(
            _ws(request),
            user["uid"],
            name=name,
            email=email,
            new_password=new_password or None,
            expertise=expertise,
        )
    except ValueError as exc:
        return templates.TemplateResponse(
            request,
            "profile.html",
            {
                "profile": {**user, "name": name, "email": email, "expertise": expertise},
                "error": str(exc),
            },
            status_code=400,
        )
    return templates.TemplateResponse(
        request,
        "profile.html",
        {"profile": updated, "message": "Profile updated."},
    )


@router.post("/profile/delete")
def profile_delete(request: Request) -> Response:
    user = request.state.user
    auth.delete_account(_ws(request), user["uid"])
    auth.destroy_session(_ws(request), request.cookies.get(auth.COOKIE_NAME))
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(auth.COOKIE_NAME)
    return resp


# ---------------------------------------------------------------------------
# Admin: user management (the /admin/* prefix is admin-gated by the middleware)
# ---------------------------------------------------------------------------


@router.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request) -> Response:
    ws = _ws(request)
    return templates.TemplateResponse(
        request,
        "admin_users.html",
        {
            "active": "admin",
            "users": auth.list_accounts(ws),
            "auto_approve": auth.auto_approve_enabled(ws),
        },
    )


@router.post("/admin/auto-approve")
def admin_set_auto_approve(request: Request, enabled: str = Form(default="off")) -> Response:
    auth.set_auto_approve(_ws(request), enabled == "on")
    return RedirectResponse("/admin/users", status_code=303)


@router.post("/admin/users/{uid}/approve")
def admin_approve(request: Request, uid: str) -> Response:
    auth.set_user_approved(_ws(request), uid, True)
    return RedirectResponse("/admin/users", status_code=303)


@router.post("/admin/users/{uid}/revoke")
def admin_revoke(request: Request, uid: str) -> Response:
    auth.set_user_approved(_ws(request), uid, False)
    return RedirectResponse("/admin/users", status_code=303)
