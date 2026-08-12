"""Per-user contribution index: counts, idempotent writes, shared instances."""

from __future__ import annotations

from pathlib import Path

from ipasc_iqa.contrib.store import ROLE_ANNOTATOR, AnnotatorStore


def _store(tmp_path: Path) -> AnnotatorStore:
    return AnnotatorStore(tmp_path / "annotations", tmp_path / "users")


def _user(store: AnnotatorStore, email: str = "a@x.com") -> str:
    return store.create_user(
        email=email, name="A", role=ROLE_ANNOTATOR, password_hash=None, approved=True
    )["uid"]


def test_same_paths_yield_the_same_instance(tmp_path: Path) -> None:
    """Opening the store per request must not open a new SQLite connection."""
    try:
        first = _store(tmp_path)
        assert _store(tmp_path) is first
        # A different workspace is still its own store.
        other = AnnotatorStore(tmp_path / "other" / "annotations", tmp_path / "other" / "users")
        assert other is not first
    finally:
        AnnotatorStore.close_all()


def test_counts_come_from_the_index(tmp_path: Path) -> None:
    try:
        store = _store(tmp_path)
        uid = _user(store)
        for i in range(3):
            store.add_rating(
                {"user_uid": uid, "dataset": "d", "image_id": f"img{i}", "task": "t", "score": 2}
            )
        store.add_session(
            {"user_uid": uid, "dataset": "d", "reference": "s0", "task": "t", "n_comparisons": 4}
        )

        assert store.count_ratings(user_uid=uid) == 3
        assert store.count_sessions(user_uid=uid) == 1
        idx = store.user_index(uid)
        assert ("d", "img1", "t") in idx["ratings"]
        assert ("d", "s0", "t") in idx["sessions"]
        assert idx["n_comparisons"] == 4

        row = next(r for r in store.leaderboard() if r["name"] == "A")
        assert (row["ratings"], row["rankings"], row["comparisons"]) == (3, 1, 4)
    finally:
        AnnotatorStore.close_all()


def test_index_is_rebuilt_from_existing_records(tmp_path: Path) -> None:
    """Deployments that predate the index get it back from a one-off scan."""
    try:
        store = _store(tmp_path)
        uid = _user(store)
        store.add_rating(
            {"user_uid": uid, "dataset": "d", "image_id": "img0", "task": "t", "score": 1}
        )
        store.invalidate_user_indexes()
        assert store.count_ratings(user_uid=uid) == 1
        assert ("d", "img0", "t") in store.user_index(uid)["ratings"]
    finally:
        AnnotatorStore.close_all()


def test_resubmitting_the_same_annotation_is_ignored(tmp_path: Path) -> None:
    """A double-click must not store a second copy that skews the aggregates."""
    try:
        store = _store(tmp_path)
        uid = _user(store)
        rating = {"user_uid": uid, "dataset": "d", "image_id": "img0", "task": "t", "score": 3}
        first = store.add_rating(rating)
        assert store.add_rating({**rating, "score": 4}) == first
        assert len(store.ratings()) == 1

        session = {
            "user_uid": uid, "dataset": "d", "reference": "s0", "task": "t", "n_comparisons": 2,
        }
        sid = store.add_session(session)
        assert store.add_session(session) == sid
        assert len(store.sessions()) == 1
    finally:
        AnnotatorStore.close_all()


def test_retag_tasks_rewrites_and_refreshes_the_index(tmp_path: Path) -> None:
    try:
        store = _store(tmp_path)
        uid = _user(store)
        store.add_rating(
            {"user_uid": uid, "dataset": "d", "image_id": "img0", "task": None, "score": 2}
        )
        store.add_session(
            {"user_uid": uid, "dataset": "d", "reference": "s0", "task": "overall",
             "n_comparisons": 1}
        )
        # A dataset the mapping doesn't mention is untouched.
        store.add_rating(
            {"user_uid": uid, "dataset": "other", "image_id": "img0", "task": None, "score": 2}
        )
        before = store.results_epoch()

        assert store.retag_tasks("d", {None: "overall_quality", "overall": "overall_quality"}) == 2

        assert store.rating_tasks("d") == {"overall_quality"}
        assert store.dataset_tasks("d") == {"overall_quality"}
        assert store.rating_tasks("other") == {None}
        assert store.results_epoch() > before
        # The index keys embed the task, so they must have been rebuilt.
        assert ("d", "img0", "overall_quality") in store.user_index(uid)["ratings"]
    finally:
        AnnotatorStore.close_all()
