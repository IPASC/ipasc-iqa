"""FastAPI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ipasc_iqa import __version__
from ipasc_iqa.gui import auth
from ipasc_iqa.gui.service import Workspace

_HERE = Path(__file__).parent
TEMPLATES_DIR = _HERE / "templates"
STATIC_DIR = _HERE / "static"

# Shared template engine (imported by routes).
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --- Access-control policy (prefix-based, default-deny) --------------------
# Anyone (no login): auth flow, the public Standings page, and static assets.
PUBLIC_PREFIXES = (
    "/login",
    "/register",
    "/forgot",
    "/reset",
    "/logout",
    "/standings",
    "/api/standings",
    "/static",
    "/favicon.ico",
)
# Admin-only: running benchmarks and downloading/exporting data.
ADMIN_PREFIXES = (
    "/run",
    "/api/run",
    "/api/runs",
    "/download",
    "/api/download",
)


def _matches(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == p or path.startswith(p + "/") for p in prefixes)


def create_app(
    workspace_root: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``workspace_root`` (results) resolves from: the explicit argument, the
    ``IPASC_IQA_WORKSPACE`` env var, or the current directory. ``data_dir``
    (managed dataset downloads) resolves from: the argument, the
    ``IPASC_IQA_DATA_DIR`` env var, or the platform default data directory.
    """
    if workspace_root is None:
        workspace_root = os.environ.get("IPASC_IQA_WORKSPACE", ".")
    if data_dir is None:
        data_dir = os.environ.get("IPASC_IQA_DATA_DIR") or None
    workspace = Workspace.resolve(workspace_root, data_dir)
    workspace.ensure()

    app = FastAPI(title="ipasc_iqa", version=__version__)
    app.state.workspace = workspace
    # Send the session cookie only over HTTPS when deployed behind TLS.
    app.state.cookie_secure = os.environ.get("IPASC_IQA_COOKIE_SECURE", "0") == "1"
    app.state.jobs = {}  # job_id → _JobState; populated by gui.jobs.launch_run
    app.state.ranking_sessions = {}  # session_id → {session, dataset_choice, task_choice}
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Resolve the current user and enforce the access policy (default-deny)."""
        token = request.cookies.get(auth.COOKIE_NAME)
        user = auth.get_session_user(workspace, token)
        request.state.user = user

        path = request.url.path
        if not _matches(path, PUBLIC_PREFIXES):
            is_hx = request.headers.get("HX-Request") == "true"
            if user is None:
                # Not logged in → send to the login page.
                if is_hx:
                    return Response(status_code=401, headers={"HX-Redirect": "/login"})
                return RedirectResponse("/login", status_code=303)
            if _matches(path, ADMIN_PREFIXES) and user.get("role") != auth.ROLE_ADMIN:
                return templates.TemplateResponse(
                    request, "403.html", {"user": user}, status_code=403
                )
        return await call_next(request)

    # Routers (imported here to avoid circulars).
    from ipasc_iqa.gui.routes import api, pages
    from ipasc_iqa.gui.routes import auth as auth_routes

    app.include_router(auth_routes.router)
    app.include_router(pages.router)
    app.include_router(api.router)
    return app
