"""Action + HTMX-fragment routes. Thin adapters over the service layer.

GET endpoints under ``/api`` return HTML *fragments* (for HTMX swaps) or JSON.
The POST ``/run`` triggers a benchmark and redirects to the results page.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ipasc_iqa.gui import service
from ipasc_iqa.gui.app import templates

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
    run_id = service.run_benchmark(ws, datasets, measures)
    # HTMX-aware redirect to the results page.
    resp = templates.TemplateResponse(
        request,
        "fragments/run_status.html",
        {"error": None, "run_id": run_id},
    )
    resp.headers["HX-Redirect"] = "/results"
    return resp


# ---------------------------------------------------------------------------
# Detailed analysis (correlation) fragment
# ---------------------------------------------------------------------------


@router.get("/api/correlation", response_class=HTMLResponse)
def correlation_fragment(
    request: Request,
    dataset: str,
    task: str = service.DEFAULT_TASK,
    measures: list[str] | None = None,
) -> HTMLResponse:
    ws = _ws(request)
    table = service.correlation_table(ws, dataset, measures=measures, task=task)
    return templates.TemplateResponse(
        request, "fragments/correlation_table.html", {"table": table}
    )


# ---------------------------------------------------------------------------
# Standings fragment
# ---------------------------------------------------------------------------


@router.get("/api/standings", response_class=HTMLResponse)
def standings_fragment(
    request: Request, task: str = service.DEFAULT_TASK
) -> HTMLResponse:
    ws = _ws(request)
    table = service.standings_table(ws, task=task)
    return templates.TemplateResponse(
        request, "fragments/standings_table.html", {"table": table}
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
