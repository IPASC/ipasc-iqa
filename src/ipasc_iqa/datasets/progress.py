"""Lightweight preparation notifications shared by dataset adapters.

Callbacks are scoped to the current adapter instantiation (including worker threads),
so concurrent jobs do not receive each other's updates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_ProgressCallback = Callable[..., None]
_current_callback: ContextVar[_ProgressCallback | None] = ContextVar(
    "ipasc_iqa_dataset_preparation_progress", default=None
)


@contextmanager
def preparation_progress(callback: _ProgressCallback | None) -> Iterator[None]:
    """Bind a progress handler for one adapter instantiation."""
    token = _current_callback.set(callback)
    try:
        yield
    finally:
        _current_callback.reset(token)


def report_preparation(stage: str, detail: str) -> None:
    """Report the current stage if the caller requested notifications."""
    callback = _current_callback.get()
    if callback is not None:
        callback(stage, detail)
