"""Action + HTMX-fragment routes. Thin adapters over the service layer.

GET endpoints under ``/api`` return HTML *fragments* (for HTMX swaps) or JSON.
The POST ``/run`` triggers a benchmark and redirects to the results page.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Query, Request
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
# Detailed analysis fragment — one correlation table per task
# ---------------------------------------------------------------------------


@router.get("/api/analysis", response_class=HTMLResponse)
def analysis_fragment(
    request: Request,
    dataset: str,
    measures: list[str] | None = Query(default=None),
) -> HTMLResponse:
    ws = _ws(request)
    result = service.analysis_for_dataset(ws, dataset, measures=measures)
    return templates.TemplateResponse(
        request, "fragments/analysis_tables.html", {"result": result}
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
