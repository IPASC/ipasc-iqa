"""HTML page routes. Thin adapters: fetch data via service, render a template."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ipasc_iqa.gui import auth, service
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
    # If a job is already running, surface it so the page shows progress
    # instead of letting the user accidentally queue a second run.
    active_job = next(
        (j.snapshot() for j in request.app.state.jobs.values() if j.status == "running"),
        None,
    )
    return templates.TemplateResponse(
        request,
        "run.html",
        {
            "active": "run",
            "datasets": service.discover_datasets(ws),
            "measures": service.list_measures(),
            "active_job": active_job,
        },
    )


@router.get("/results", response_class=HTMLResponse)
def results_page(request: Request) -> HTMLResponse:
    ws = _ws(request)
    datasets = service.discover_datasets(ws)
    # Default to what the user last ran a benchmark on: its first dataset and the
    # measures it covered. Fall back to a prepared dataset / all usable measures.
    last = service.load_last_selection(ws)
    available = {d["name"] for d in datasets}
    last_datasets = [d for d in last["datasets"] if d in available]
    prepared = [d for d in datasets if d["prepared"]]
    selected = (
        last_datasets[0] if last_datasets else (prepared or datasets or [{"name": None}])[0]["name"]
    )
    selected_measures = last["measures"] or None
    # Provide the annotation tasks for the initially selected dataset so the
    # sub-task dropdown can be rendered on first load without an extra request.
    sel_ds = next((d for d in datasets if d["name"] == selected), None)
    initial_tasks: list[str] = sel_ds["annotation_tasks"] if sel_ds else []
    initial_sources = service.human_score_sources(ws, selected) if selected else []
    return templates.TemplateResponse(
        request,
        "results.html",
        {
            "active": "results",
            "datasets": datasets,
            "measure_options": service.measure_options(ws, selected, selected=selected_measures),
            "selected_dataset": selected,
            "initial_tasks": initial_tasks,
            "initial_sources": initial_sources,
        },
    )


@router.get("/standings", response_class=HTMLResponse)
def standings_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "standings.html", {"active": "standings"})


@router.get("/download", response_class=HTMLResponse)
def download_page(request: Request) -> HTMLResponse:
    ws = _ws(request)
    return templates.TemplateResponse(
        request,
        "download.html",
        {"active": "download", "datasets": service.discover_datasets(ws)},
    )


@router.get("/contribute", response_class=HTMLResponse)
def contribute_page(request: Request) -> HTMLResponse:
    ws = _ws(request)
    # The auth middleware guarantees a logged-in user on this protected route.
    account = service.account_for(ws, request.state.user)
    return templates.TemplateResponse(
        request,
        "contribute.html",
        {
            "active": "contribute",
            "account": account,
            "pending": not auth.is_approved(request.state.user),
            "eligible_datasets": service.eligible_annotation_datasets(ws),
            "annotators": service.annotators_leaderboard(ws),
        },
    )
