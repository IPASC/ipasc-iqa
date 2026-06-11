"""FastAPI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ipasc_iqa import __version__
from ipasc_iqa.gui.service import Workspace

_HERE = Path(__file__).parent
TEMPLATES_DIR = _HERE / "templates"
STATIC_DIR = _HERE / "static"

# Shared template engine (imported by routes).
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def create_app(workspace_root: Path | str | None = None) -> FastAPI:
    """Build the FastAPI app.

    The workspace root resolves from (in order): the explicit argument, the
    ``IPASC_IQA_WORKSPACE`` environment variable, or the current directory.
    """
    if workspace_root is None:
        workspace_root = os.environ.get("IPASC_IQA_WORKSPACE", ".")
    workspace = Workspace(root=Path(workspace_root).resolve())
    workspace.ensure()

    app = FastAPI(title="ipasc_iqa", version=__version__)
    app.state.workspace = workspace
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Routers (imported here to avoid circulars).
    from ipasc_iqa.gui.routes import api, pages

    app.include_router(pages.router)
    app.include_router(api.router)
    return app
