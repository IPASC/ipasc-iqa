"""Read derived analysis inputs (annotations, groups) from a dataset manifest.

These operate on the manifest's per-image ``metadata`` without loading the image
arrays, so the GUI can build correlation/standings inputs cheaply.

The PhotIQA converter embeds annotations into each image entry like::

    "metadata": {
        "algorithm": "alg0",
        "annotations": {
            "overall_quality": {"mos": 2.8, "zscore_mos": 0.149},
            ...
        }
    }
"""

from __future__ import annotations

from pathlib import Path

from ipasc_iqa.io.manifest import load_manifest


def annotation_tasks_in_manifest(manifest: dict) -> list[str]:
    tasks: set[str] = set()
    for entry in manifest.get("images", []):
        ann = entry.get("metadata", {}).get("annotations", {})
        tasks.update(ann.keys())
    return sorted(tasks)


def annotation_values_from_manifest(
    manifest: dict,
    task: str,
    kind: str = "zscore_mos",
) -> dict[str, float]:
    """Return ``{image_id: score}`` for one annotation *task* and *kind*."""
    out: dict[str, float] = {}
    for entry in manifest.get("images", []):
        ann = entry.get("metadata", {}).get("annotations", {})
        if task in ann and kind in ann[task] and ann[task][kind] is not None:
            out[entry["id"]] = float(ann[task][kind])
    return out


def annotator_ratings_from_manifest(manifest: dict, task: str) -> dict[str, dict[str, float]]:
    """Return ``{image_id: {annotator_id: raw_score}}`` for one task.

    Reads the per-rater scores embedded under each image's
    ``metadata.annotations[task]["ratings"]``. Annotator ids are namespaced
    (``manifest:<rater>``) so they never collide with contributed annotators.
    """
    out: dict[str, dict[str, float]] = {}
    for entry in manifest.get("images", []):
        ann = entry.get("metadata", {}).get("annotations", {}).get(task)
        if not ann:
            continue
        ratings = ann.get("ratings")
        if not ratings:
            continue
        out[entry["id"]] = {
            f"manifest:{rater}": float(value)
            for rater, value in ratings.items()
            if value is not None
        }
    return out


def groups_from_manifest(manifest: dict, key: str = "algorithm") -> dict[str, str]:
    """Return ``{image_id: group}`` using a metadata *key* (default 'algorithm')."""
    out: dict[str, str] = {}
    for entry in manifest.get("images", []):
        group = entry.get("metadata", {}).get(key)
        if group is not None:
            out[entry["id"]] = str(group)
    return out


def load_manifest_for_dataset(dataset_root: Path) -> dict:
    """Load manifest.json from a dataset root (handles PhotIQA's converted/ layout)."""
    dataset_root = Path(dataset_root)
    direct = dataset_root / "manifest.json"
    if direct.exists():
        return load_manifest(direct)
    converted = dataset_root / "converted" / "manifest.json"
    if converted.exists():
        return load_manifest(converted)
    raise FileNotFoundError(f"No manifest.json found under {dataset_root}")
