"""Render checks for the annotation fragments.

The GUI integration tests never reach the comparison/rating stage (their tiny
fixture dataset has no reference groups), so these render the fragments directly
— enough to catch a broken template before a rater does.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from ipasc_iqa.gui.app import templates  # noqa: E402

env = templates.env


def test_compare_fragment_prefetches_the_rest_of_the_set() -> None:
    out = env.get_template("fragments/annotate_compare.html").render(
        session_id="s1",
        dataset="mini",
        prompt="Pick one",
        left_item="a",
        right_item="b",
        placed=1,
        n_items=4,
        n_comparisons=2,
        prefetch_items=["c", "__ref__0"],
    )
    assert out.count('rel="prefetch"') == 2
    # Submitting must dim the stage and disable the buttons (no double-submits).
    assert 'hx-indicator="#contribute-root"' in out
    assert 'hx-disabled-elt="find button"' in out


def test_rate_fragment_prefetches_the_queued_items() -> None:
    out = env.get_template("fragments/annotate_rate.html").render(
        dataset="mini",
        image_id="img0",
        reference_item="__ref__0",
        task="overall_quality",
        dataset_choice="random",
        prompt="Rate it",
        upcoming=[{"dataset": "mini", "items": ["img1", "__ref__1"]}],
    )
    assert out.count('rel="prefetch"') == 2
    assert 'hx-indicator="#contribute-root"' in out


def test_fragments_render_without_the_optional_context() -> None:
    """Older/edge contexts (no prefetch list) must not blow up the page."""
    assert env.get_template("fragments/annotate_compare.html").render(
        session_id="s", dataset="d", prompt="p", left_item="a", right_item="b",
        placed=1, n_items=2, n_comparisons=0,
    )
    assert env.get_template("fragments/annotate_rate.html").render(
        dataset="d", image_id="i", reference_item="__ref__0", task="", dataset_choice="random",
        prompt="p",
    )


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"stale": False, "age": 0.0, "refreshing": False, "computed": "2026-01-01 09:00 UTC"},
         "just now"),
        ({"stale": True, "age": 400.0, "refreshing": False, "computed": "2026-01-01 09:00 UTC"},
         "6 minutes ago"),
        ({"stale": True, "age": 4000.0, "refreshing": True, "computed": "2026-01-01 09:00 UTC"},
         "recomputing in the background"),
    ],
)
def test_cache_status_always_states_when_results_were_computed(meta: dict, expected: str) -> None:
    out = env.get_template("fragments/cache_status.html").render(
        meta=meta, poll_url="/api/standings"
    )
    assert "Last updated:" in out
    assert meta["computed"] in out
    assert expected in out
    # Only a running refresh polls for the newer numbers.
    assert ("hx-get" in out) is meta["refreshing"]


def test_cache_status_is_silent_without_metadata() -> None:
    out = env.get_template("fragments/cache_status.html").render(meta=None, poll_url="/x")
    assert out.strip() == ""
