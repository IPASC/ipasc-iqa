"""Result caching and a persistent, newest-wins result store.

Two responsibilities, one object:

1. **Cache** — during a run, a ``(dataset, image, measure)`` whose inputs are
   unchanged is not recomputed; the previously stored value is reused. This is
   what makes repeated/partial runs cheap (important for a public deployment).

2. **Merge** — the store keeps exactly one record per ``(dataset, image, measure)``,
   always the most recently computed one. Running only a subset of measures
   therefore *adds to* the accumulated results rather than replacing them, so the
   analysis views show the union of everything computed so far.

The backend is [`diskcache`](https://grantjenks.com/docs/diskcache/): an
SQLite-backed, transactional, process- and thread-safe key/value store. This
scales to a public multi-user deployment without the load-modify-save races a
single JSON file would suffer.

The cache key follows the roadmap composition: it hashes the measure name +
version, the image hash, the reference type + hash, the ROI hashes, and the
measure's canonical config JSON. Any change to those invalidates the entry.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from ipasc_iqa.core.result import ResultRecord

if TYPE_CHECKING:
    from collections.abc import Iterator


def make_cache_key(
    *,
    measure_name: str,
    measure_version: str,
    image_sha256: str,
    reference_type: str | None,
    reference_sha256: str | None,
    signal_roi_sha256: str | None,
    noise_roi_sha256: str | None,
    measure_config_json: str = "{}",
) -> str:
    parts = [
        measure_name,
        measure_version,
        image_sha256,
        reference_type or "",
        reference_sha256 or "",
        signal_roi_sha256 or "",
        noise_roi_sha256 or "",
        measure_config_json,
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class ResultStore:
    """diskcache-backed, newest-wins store keyed by content hash.

    Two kinds of entries live in one cache, distinguished by a tuple-key tag:

    - ``("r", cache_key)`` → the serialized :class:`ResultRecord` (the cache).
    - ``("i", dataset, image_id, measure)`` → the ``cache_key`` currently held for
      that triple (the merge index). Lets a newer computation supersede an older
      one and lets the analysis views enumerate one record per triple.

    Use as a context manager, or remember to :meth:`close` it.
    """

    def __init__(self, path: Path | str) -> None:
        import diskcache

        self.path = Path(path)
        self._cache = diskcache.Cache(str(self.path))

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._cache.close()

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- runner cache interface -------------------------------------------

    def get(self, key: str) -> ResultRecord | None:
        rec = self._cache.get(("r", key))
        return ResultRecord.model_validate(rec) if rec is not None else None

    def set(self, key: str, record: ResultRecord) -> None:
        index_key = ("i", record.dataset, record.image_id, record.measure)
        # Atomic supersede-and-store: each op is transactional, and transact()
        # batches them so concurrent writers never interleave a partial update.
        with self._cache.transact():
            previous = self._cache.get(index_key)
            if previous is not None and previous != key:
                # A newer computation for this (dataset, image, measure) replaces
                # the old one (e.g. measure version or an input hash changed).
                self._cache.delete(("r", previous))
            self._cache.set(("r", key), record.model_dump(mode="json"))
            self._cache.set(index_key, key)

    # -- merged query for the analysis views ------------------------------

    def _index_keys(self) -> Iterator[tuple]:
        for k in self._cache:
            if isinstance(k, tuple) and k and k[0] == "i":
                yield k

    def records_for_dataset(self, dataset: str | None = None) -> list[ResultRecord]:
        out: list[ResultRecord] = []
        for index_key in self._index_keys():
            _, ds, _image, _measure = index_key
            if dataset is not None and ds != dataset:
                continue
            cache_key = self._cache.get(index_key)
            rec = self._cache.get(("r", cache_key)) if cache_key is not None else None
            if rec is not None:
                out.append(ResultRecord.model_validate(rec))
        return out

    def __len__(self) -> int:
        return sum(1 for _ in self._index_keys())
