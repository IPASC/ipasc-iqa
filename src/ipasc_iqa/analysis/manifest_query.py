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


def comparison_tasks_in_manifest(manifest: dict) -> list[str]:
    """Tasks that carry embedded pairwise comparisons (published 2AFC datasets).

    Mirrors :func:`annotation_tasks_in_manifest` but for the ``human_comparisons``
    block written by pairwise-study adapters (e.g. the Alladien datasets).
    """
    tasks = {c.get("task") for c in manifest.get("human_comparisons", []) if c.get("task")}
    return sorted(tasks)


#: Anchor pseudo-item prefix for a reference image (see gui.service._REF_PREFIX).
_REF_ITEM_PREFIX = "__ref__"


def ranking_observations_from_manifest(manifest: dict, task: str | None) -> int:
    """Count published ranking *placements* for one *task*.

    Each comparison set is one annotator ranking one reference set; it contributes
    one observation per distinct **non-reference** variant it ranks (mirroring how
    a contributed ranking session counts ``order`` placements). So an annotator's
    many pairwise comparisons within a set count once per variant, and the
    reference (``__ref__`` anchor) is excluded.
    """
    total = 0
    for cset in manifest.get("human_comparisons", []):
        if cset.get("task") != task:
            continue
        items = {
            item
            for pair in cset.get("comparisons", [])
            for item in pair
            if not str(item).startswith(_REF_ITEM_PREFIX)
        }
        total += len(items)
    return total


def comparisons_from_manifest(manifest: dict, task: str | None) -> list[tuple[str, str, str]]:
    """Return ``(reference, winner, loser)`` triples embedded for one *task*.

    The manifest stores published 2AFC comparisons in the same shape the
    contributed pairwise pipeline uses, so the two pool cleanly: each set has a
    ``reference`` (the per-scene group id) and a list of ``[winner, loser]`` image
    ids, where the scene's pseudo-reference uses the ``__ref__<reference>``
    pseudo-item (matching the contributed-comparison convention).
    """
    out: list[tuple[str, str, str]] = []
    for cset in manifest.get("human_comparisons", []):
        if cset.get("task") != task:
            continue
        ref = cset.get("reference")
        for winner, loser in cset.get("comparisons", []):
            out.append((ref, winner, loser))
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
