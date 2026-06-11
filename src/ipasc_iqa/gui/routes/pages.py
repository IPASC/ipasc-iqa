"""HTML page routes. Thin adapters: fetch data via service, render a template."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ipasc_iqa.gui import service
from ipasc_iqa.gui.app import templates

router = APIRouter()


def _ws(request: Request) -> service.Workspace:
    return request.app.state.workspace


@router.get("/", response_class=HTMLResponse)
def welcome(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "welcome.html", {"active": "welcome"})


@router.get("/run", response_class=HTMLResponse)
def run_page(request: Request) -> HTMLResponse:
    ws = _ws(request)
    return templates.TemplateResponse(
        request,
        "run.html",
        {
            "active": "run",
            "datasets": service.discover_datasets(ws),
            "measures": service.list_measures(),
        },
    )


@router.get("/results", response_class=HTMLResponse)
def results_page(request: Request) -> HTMLResponse:
    ws = _ws(request)
    datasets = service.discover_datasets(ws)
    # Prefer a prepared dataset as the initial selection.
    prepared = [d for d in datasets if d["prepared"]]
    selected = (prepared or datasets or [{"name": None}])[0]["name"]
    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "active": "results",
            "datasets": datasets,
            "measure_options": service.measure_options(ws, selected),
            "selected_dataset": selected,
        },
    )


@router.get("/standings", response_class=HTMLResponse)
def standings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "standings.html", {"active": "standings"}
    )
