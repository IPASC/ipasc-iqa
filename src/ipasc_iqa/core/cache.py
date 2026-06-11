"""Result caching and a persistent, newest-wins result store.

Two responsibilities, one object:

1. **Cache** — during a run, a ``(dataset, image, measure)`` whose inputs are
   unchanged is not recomputed; the previously stored value is reused. This is
   what makes repeated/partial runs cheap (important for a public deployment).

2. **Merge** — the store keeps exactly one record per ``(dataset, image, measure)``,
   always the most recently computed one. Running only a subset of measures
   therefore *adds to* the accumulated results rather than replacing them, so the
   analysis views show the union of everything computed so far.

The cache key follows the roadmap composition: it hashes the measure name +
version, the image hash, the reference type + hash, the ROI hashes, and the
measure's canonical config JSON. Any change to those invalidates the entry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ipasc_iqa.core.result import ResultRecord


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


def _index_key(dataset: str, image_id: str, measure: str) -> str:
    return f"{dataset}\x00{image_id}\x00{measure}"


class ResultStore:
    """JSON-backed, newest-wins store keyed by content hash.

    Also exposes the ``get``/``set`` pair the runner uses as a cache.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        # cache_key -> serialized ResultRecord
        self._by_key: dict[str, dict] = {}
        # (dataset, image, measure) -> cache_key currently held for it
        self._index: dict[str, str] = {}
        self._load()

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        self._by_key = data.get("by_key", {})
        self._index = data.get("index", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"by_key": self._by_key, "index": self._index}),
            encoding="utf-8",
        )

    # -- runner cache interface -------------------------------------------

    def get(self, key: str) -> ResultRecord | None:
        rec = self._by_key.get(key)
        return ResultRecord.model_validate(rec) if rec is not None else None

    def set(self, key: str, record: ResultRecord) -> None:
        idx = _index_key(record.dataset, record.image_id, record.measure)
        previous = self._index.get(idx)
        if previous is not None and previous != key:
            # A newer computation for this (dataset, image, measure) supersedes the
            # old one (e.g. the measure version or an input hash changed).
            self._by_key.pop(previous, None)
        self._by_key[key] = record.model_dump(mode="json")
        self._index[idx] = key

    # -- merged query for the analysis views ------------------------------

    def records_for_dataset(self, dataset: str | None = None) -> list[ResultRecord]:
        out: list[ResultRecord] = []
        for rec in self._by_key.values():
            if dataset is None or rec.get("dataset") == dataset:
                out.append(ResultRecord.model_validate(rec))
        return out

    def __len__(self) -> int:
        return len(self._by_key)
