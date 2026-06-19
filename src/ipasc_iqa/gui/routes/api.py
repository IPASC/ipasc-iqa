"""Action + HTMX-fragment routes. Thin adapters over the service layer.

GET endpoints under ``/api`` return HTML *fragments* (for HTMX swaps) or JSON.
The POST ``/run`` triggers a benchmark and redirects to the results page.
"""

from __future__ import annotations

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ipasc_iqa.gui import auth, service
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
    # Remember this selection so the analysis page can default to it.
    service.save_last_selection(ws, datasets, measures)
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


def _recomputing(request: Request) -> HTMLResponse:
    """Placeholder that polls this same URL until the cached value is ready."""
    poll_url = request.url.path + (f"?{request.url.query}" if request.url.query else "")
    return templates.TemplateResponse(request, "fragments/recomputing.html", {"poll_url": poll_url})


@router.get("/api/scatter", response_class=HTMLResponse)
def scatter_fragment(
    request: Request,
    dataset: str,
    measure: str,
    task: str | None = Query(default=None),
    source: str = Query(default="mos"),
) -> HTMLResponse:
    """Scatter plot (SVG) of one measure's per-image values vs the human scores."""
    ws = _ws(request)
    params = {"dataset": dataset, "measure": measure, "task": task, "source": source}
    model = request.app.state.result_cache.get_or_schedule(
        "scatter",
        params,
        service.results_epoch(ws),
        lambda: service.scatter_for_measure(ws, dataset, measure, task=task, source=source),
    )
    if model is None:
        return _recomputing(request)
    return templates.TemplateResponse(request, "fragments/scatter.html", {"s": model})


@router.get("/api/analysis", response_class=HTMLResponse)
def analysis_fragment(
    request: Request,
    dataset: str,
    measures: list[str] | None = Query(default=None),
    task: str | None = Query(default=None),
    source: str = Query(default="mos"),
) -> HTMLResponse:
    ws = _ws(request)
    params = {
        "dataset": dataset,
        "measures": sorted(measures) if measures else None,
        "task": task,
        "source": source,
    }

    def _compute() -> dict:
        return service.analysis_for_dataset(
            ws, dataset, measures=measures, task=task, source=source
        )

    result = request.app.state.result_cache.get_or_schedule(
        "analysis", params, service.results_epoch(ws), _compute
    )
    if result is None:
        return _recomputing(request)
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
def standings_fragment(request: Request, source: str = Query(default="mos")) -> HTMLResponse:
    ws = _ws(request)
    result = request.app.state.result_cache.get_or_schedule(
        "standings",
        {"source": source},
        service.results_epoch(ws),
        lambda: service.standings(ws, source=source),
    )
    if result is None:
        return _recomputing(request)
    return templates.TemplateResponse(request, "fragments/standings_table.html", {"result": result})


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


@router.get("/api/download")
def api_download(request: Request, dataset: str = Query(default="all")) -> JSONResponse:
    """Download cached metric results + annotations as a JSON file (attachment)."""
    from datetime import UTC, datetime

    ws = _ws(request)
    payload = service.export_payload(ws, None if dataset == "all" else dataset)
    label = "all" if dataset == "all" else dataset
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    filename = f"ipasc_iqa_export_{label}_{stamp}.json"
    return JSONResponse(
        payload,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Contributed annotations — login, 2AFC ranking, image rendering
# ---------------------------------------------------------------------------


def _pending_response(request: Request) -> Response:
    """Block annotation for signed-in but unapproved accounts (HTMX fragment)."""
    return templates.TemplateResponse(request, "fragments/annotate_pending.html", {})


def _picker_response(request: Request, account: dict, *, message: str | None = None) -> Response:
    """Render the dataset/sub-task picker fragment for a logged-in annotator."""
    return templates.TemplateResponse(
        request,
        "fragments/annotate_picker.html",
        {
            "account": account,
            "eligible_datasets": service.eligible_annotation_datasets(_ws(request)),
            "annotators": service.annotators_leaderboard(_ws(request)),
            "message": message,
        },
    )


def _compare_or_picker(request: Request, account: dict, session_id: str | None) -> Response:
    """Render the comparison fragment, or fall back to the picker on failure."""
    sessions = request.app.state.ranking_sessions
    ctx = service.compare_view(sessions, session_id) if session_id else None
    if ctx is None:
        return _picker_response(
            request, account, message="That annotation session has expired — start a new one."
        )
    return templates.TemplateResponse(request, "fragments/annotate_compare.html", ctx)


@router.get("/api/annotate/tasks", response_class=HTMLResponse)
def contribute_tasks(request: Request, dataset: str = Query(default="random")) -> HTMLResponse:
    """Sub-task <select> for the chosen dataset (empty for 'random')."""
    return templates.TemplateResponse(
        request,
        "fragments/annotate_task_field.html",
        {"tasks": service.annotation_tasks_for(_ws(request), dataset)},
    )


def _rate_fragment(
    request: Request,
    account: dict,
    draw: dict,
    dataset_choice: str,
    *,
    toast: str | None = None,
) -> Response:
    """Render the single-stimulus rating fragment, or the picker on failure."""
    if draw.get("error"):
        return _picker_response(request, account, message=draw["error"])
    task = draw["task"]
    if task:
        prompt = (
            f"Rate the quality of the right image for the task: "
            f"{task.replace('_', ' ')}  (1 = low, 4 = best)"
        )
    else:
        prompt = "Rate the quality of the right image  (1 = low, 4 = best)"
    return templates.TemplateResponse(
        request,
        "fragments/annotate_rate.html",
        {
            "dataset": draw["dataset"],
            "image_id": draw["image_id"],
            "reference_item": draw["reference_item"],
            "task": task or "",
            "dataset_choice": dataset_choice,
            "prompt": prompt,
            "toast": toast,
        },
    )


@router.post("/contribute/start")
def contribute_start(
    request: Request,
    dataset: str = Form(default="random"),
    rating_type: str = Form(default="comparative"),
) -> Response:
    ws = _ws(request)
    if not auth.is_approved(request.state.user):
        return _pending_response(request)
    account = service.account_for(ws, request.state.user)

    if rating_type == "quantitative":
        draw = service.draw_rating_item(ws, dataset)
        return _rate_fragment(request, account, draw, dataset)

    # Comparative (2AFC): the sub-task is always randomised per session.
    res = service.start_ranking(
        ws,
        request.app.state.ranking_sessions,
        user_uid=request.state.user["uid"],
        dataset_choice=dataset,
        task_choice="random",
    )
    if res.get("error"):
        return _picker_response(request, account, message=res["error"])
    return _compare_or_picker(request, account, res["session_id"])


@router.post("/contribute/rate")
def contribute_rate(
    request: Request,
    dataset: str = Form(...),
    image_id: str = Form(...),
    score: int = Form(...),
    task: str = Form(default=""),
    dataset_choice: str = Form(default="random"),
) -> Response:
    ws = _ws(request)
    if not auth.is_approved(request.state.user):
        return _pending_response(request)
    account = service.account_for(ws, request.state.user)

    count = service.record_rating(
        ws,
        user_uid=request.state.user["uid"],
        dataset=dataset,
        image_id=image_id,
        task=task or None,
        score=score,
    )
    draw = service.draw_rating_item(ws, dataset_choice)
    toast = f"Rating saved — thank you! ({count} total). Next:"
    return _rate_fragment(request, account, draw, dataset_choice, toast=toast)


@router.post("/contribute/choose")
def contribute_choose(
    request: Request,
    session_id: str = Form(...),
    winner: str = Form(...),
) -> Response:
    ws = _ws(request)
    if not auth.is_approved(request.state.user):
        return _pending_response(request)
    account = service.account_for(ws, request.state.user)

    outcome = service.submit_choice(ws, request.app.state.ranking_sessions, session_id, winner)
    if outcome["status"] == "missing":
        return _picker_response(
            request, account, message="That annotation session has expired — start a new one."
        )
    if outcome["status"] == "continue":
        return _compare_or_picker(request, account, outcome["session_id"])

    # Completed one full ranking → reflect the new count and continue.
    account = {**account, "count": outcome["count"]}
    if outcome.get("error") or not outcome.get("next_session_id"):
        return _picker_response(
            request,
            account,
            message=f"Ranking saved — thank you! ({outcome['count']} total). "
            + (outcome.get("error") or "No more sets available right now."),
        )
    ctx = service.compare_view(request.app.state.ranking_sessions, outcome["next_session_id"])
    ctx["toast"] = f"Ranking saved — thank you! ({outcome['count']} total). Next set:"
    return templates.TemplateResponse(request, "fragments/annotate_compare.html", ctx)


@router.post("/contribute/finish")
def contribute_finish(request: Request) -> Response:
    ws = _ws(request)
    account = service.account_for(ws, request.state.user)
    return _picker_response(
        request, account, message="Session ended. Start another whenever you like."
    )


@router.get("/api/annotate/image")
def contribute_image(request: Request, dataset: str, item: str) -> Response:
    ws = _ws(request)
    path = service.resolve_item_path(ws, dataset, item)
    if path is None or not path.exists():
        return Response(status_code=404)
    # Window both images of a comparison/rating to their shared reference range so
    # they are displayed on the same dynamic range (fair subjective assessment).
    window = service.display_window_for(ws, dataset, item)
    return Response(content=service.render_item_png(path, window=window), media_type="image/png")
