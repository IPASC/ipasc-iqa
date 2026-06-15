"""Action + HTMX-fragment routes. Thin adapters over the service layer.

GET endpoints under ``/api`` return HTML *fragments* (for HTMX swaps) or JSON.
The POST ``/run`` triggers a benchmark and redirects to the results page.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ipasc_iqa.gui import service
from ipasc_iqa.gui.app import templates
from ipasc_iqa.gui.jobs import launch_run

router = APIRouter()


def _ws(request: Request) -> service.Workspace:
    return request.app.state.workspace


# ---------------------------------------------------------------------------
# Run a benchmark
# ---------------------------------------------------------------------------


@router.post("/run")
def run_benchmark(
    request: Request,
    datasets: list[str] = Form(default=[]),
    measures: list[str] = Form(default=[]),
) -> Response:
    ws = _ws(request)
    if not datasets or not measures:
        return templates.TemplateResponse(
            request,
            "fragments/run_status.html",
            {"error": "Select at least one dataset and one measure.", "run_id": None},
            status_code=400,
        )
    job_id = launch_run(ws, request.app.state.jobs, datasets, measures)
    snap = request.app.state.jobs[job_id].snapshot()
    return templates.TemplateResponse(request, "fragments/job_progress.html", snap)


@router.get("/api/run/{job_id}/status", response_class=HTMLResponse)
def run_job_status(request: Request, job_id: str) -> Response:
    """Polling endpoint — returns a progress fragment; redirects when done."""
    jobs = request.app.state.jobs
    if job_id not in jobs:
        return HTMLResponse("<p class='notice notice--warn'>Job not found.</p>")
    snap = jobs[job_id].snapshot()
    resp = templates.TemplateResponse(request, "fragments/job_progress.html", snap)
    if snap["status"] == "done":
        resp.headers["HX-Redirect"] = "/results"
    return resp


# ---------------------------------------------------------------------------
# Detailed analysis fragment — one correlation table per task
# ---------------------------------------------------------------------------


@router.get("/api/scatter", response_class=HTMLResponse)
def scatter_fragment(
    request: Request,
    dataset: str,
    measure: str,
    task: str | None = Query(default=None),
) -> HTMLResponse:
    """Scatter plot (SVG) of one measure's per-image values vs the human scores."""
    ws = _ws(request)
    model = service.scatter_for_measure(ws, dataset, measure, task=task)
    return templates.TemplateResponse(request, "fragments/scatter.html", {"s": model})


@router.get("/api/analysis", response_class=HTMLResponse)
def analysis_fragment(
    request: Request,
    dataset: str,
    measures: list[str] | None = Query(default=None),
    task: str | None = Query(default=None),
) -> HTMLResponse:
    ws = _ws(request)
    result = service.analysis_for_dataset(ws, dataset, measures=measures, task=task)
    # Re-render the measure checkboxes (disabled/checked for this dataset) and
    # swap them out-of-band, so changing the dataset updates which are usable.
    options = service.measure_options(ws, dataset, selected=measures)
    return templates.TemplateResponse(
        request,
        "fragments/analysis_response.html",
        {"result": result, "measure_options": options},
    )


# ---------------------------------------------------------------------------
# Standings fragment — measures ranked by correlation, per dataset, per task
# ---------------------------------------------------------------------------


@router.get("/api/standings", response_class=HTMLResponse)
def standings_fragment(request: Request) -> HTMLResponse:
    ws = _ws(request)
    result = service.standings(ws)
    return templates.TemplateResponse(
        request, "fragments/standings_table.html", {"result": result}
    )


# ---------------------------------------------------------------------------
# JSON endpoints (for scripting / debugging)
# ---------------------------------------------------------------------------


@router.get("/api/measures")
def api_measures(request: Request) -> JSONResponse:
    return JSONResponse(service.list_measures())


@router.get("/api/datasets")
def api_datasets(request: Request) -> JSONResponse:
    return JSONResponse(service.discover_datasets(_ws(request)))


@router.get("/api/runs")
def api_runs(request: Request) -> JSONResponse:
    return JSONResponse(service.list_runs(_ws(request)))
