"""Helpers for exploring image-level metric results.

This module is GUI-independent: it works from a dataset manifest and stored
ResultRecord objects, and returns plain dictionaries that GUI/CLI code can
render as needed.
"""

from __future__ import annotations

import re

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from ipasc_iqa.core.result import ResultRecord

_NATURAL_PART_RE = re.compile(r"(\d+)")


def _natural_sort_key(value: object) -> tuple:
    """Sort strings naturally, e.g. alg2 before alg10 and sparse_2x before sparse_16x."""
    text = str(value)
    parts = _NATURAL_PART_RE.split(text)
    key = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def image_groups_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Group related image variants from a dataset manifest.

    Preferred grouping:
    - metadata.stem groups variants of the same underlying image.
    - metadata.algorithm labels the reconstruction/version/configuration.

    Fallback:
    - if no stem is present, each image becomes its own group.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for entry in manifest.get("images", []):
        image_id = entry.get("id")
        if not image_id:
            continue

        metadata = entry.get("metadata", {}) or {}

        stem = metadata.get("stem")
        if stem is None:
            stem = image_id

        algorithm = metadata.get("algorithm")
        if algorithm is None:
            algorithm = metadata.get("configuration")
        if algorithm is None:
            algorithm = metadata.get("reconstruction")
        if algorithm is None:
            algorithm = image_id

        grouped[str(stem)].append(
            {
                "image_id": image_id,
                "label": str(algorithm),
                "metadata": metadata,
                "path": entry.get("path"),
                "references": entry.get("references") or {},
                "signal_roi": entry.get("signal_roi"),
                "noise_roi": entry.get("noise_roi"),
            }
        )

    out: list[dict[str, Any]] = []
    for stem, items in sorted(grouped.items(), key=lambda pair: pair[0]):
        def _item_sort_key(item: dict[str, Any]) -> tuple:
            metadata = item.get("metadata", {}) or {}
            algorithm_index = _optional_int(metadata.get("algorithm_index"))

            if algorithm_index is not None:
                return (0, algorithm_index, _natural_sort_key(item["label"]), _natural_sort_key(item["image_id"]))

            return (1, _natural_sort_key(item["label"]), _natural_sort_key(item["image_id"]))


        items_sorted = sorted(items, key=_item_sort_key)
        out.append(
            {
                "stem": stem,
                "label": stem,
                "items": items_sorted,
                "n_items": len(items_sorted),
            }
        )

    return out


def image_metric_scores(
    records: Iterable[ResultRecord],
    image_id: str,
    dataset_name: str,
    *,
    higher_is_better_by_measure: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Return all metric scores available for one image."""
    higher_is_better_by_measure = higher_is_better_by_measure or {}

    scores: list[dict[str, Any]] = []

    for rec in records:
        if rec.dataset != dataset_name or rec.image_id != image_id:
            continue

        higher_is_better = higher_is_better_by_measure.get(rec.measure, True)

        if rec.value is not None:
            scores.append(
                {
                    "measure": rec.measure,
                    "value": float(rec.value),
                    "higher_is_better": bool(higher_is_better),
                    "kind": "main",
                }
            )

        if rec.sub_scores:
            for sub_name, sub_value in sorted(rec.sub_scores.items()):
                if sub_value is None:
                    continue

                scores.append(
                    {
                        "measure": f"{rec.measure}.{sub_name}",
                        "value": float(sub_value),
                        "higher_is_better": bool(higher_is_better),
                        "kind": "sub_score",
                    }
                )

    return sorted(scores, key=lambda row: row["measure"])

def score_comparison_table(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one metric-by-image table for a group of image variants.

    Cell colors are computed within the selected group:
    - green means best among the displayed variants for that metric;
    - red means worst among the displayed variants for that metric;
    - direction is respected, so lower-is-better metrics are inverted.
    """
    measure_names = sorted(
        {
            score["measure"]
            for item in items
            for score in item.get("scores", [])
        },
        key=_natural_sort_key,
    )

    table: list[dict[str, Any]] = []

    for measure in measure_names:
        scores_by_image = {
            item["image_id"]: next(
                (
                    score
                    for score in item.get("scores", [])
                    if score["measure"] == measure
                ),
                None,
            )
            for item in items
        }

        available = [
            score
            for score in scores_by_image.values()
            if score is not None and score.get("value") is not None
        ]

        higher_is_better = True
        if available:
            higher_is_better = bool(available[0].get("higher_is_better", True))

        values = [float(score["value"]) for score in available]
        lo = min(values) if values else None
        hi = max(values) if values else None

        cells = []
        for item in items:
            score = scores_by_image[item["image_id"]]

            if score is None or score.get("value") is None:
                cells.append(
                    {
                        "image_id": item["image_id"],
                        "label": item["label"],
                        "value": None,
                        "quality_position": None,
                        "quality_percent": None,
                        "color_hue": None,
                    }
                )
                continue

            value = float(score["value"])

            if lo is None or hi is None or hi == lo:
                quality_position = 0.5
            else:
                numeric_position = (value - lo) / (hi - lo)
                quality_position = numeric_position if higher_is_better else 1.0 - numeric_position

            quality_position = max(0.0, min(1.0, quality_position))

            cells.append(
                {
                    "image_id": item["image_id"],
                    "label": item["label"],
                    "value": value,
                    "quality_position": quality_position,
                    "quality_percent": round(quality_position * 100),
                    "color_hue": round(quality_position * 120),
                }
            )

        table.append(
            {
                "measure": measure,
                "higher_is_better": higher_is_better,
                "values": cells,
            }
        )

    return table

def image_group_detail(
    manifest: dict[str, Any],
    records: Iterable[ResultRecord],
    dataset_name: str,
    *,
    stem: str | None = None,
    higher_is_better_by_measure: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Return one grouped image set with metric scores for each variant."""
    groups = image_groups_from_manifest(manifest)

    if not groups:
        return {
            "error": "no images found in manifest",
            "groups": [],
            "active_stem": None,
            "items": [],
        }

    if stem is None:
        active_group = groups[0]
    else:
        active_group = next((group for group in groups if group["stem"] == stem), groups[0])

    items = []
    for item in active_group["items"]:
        image_id = item["image_id"]
        items.append(
            {
                **item,
                "scores": image_metric_scores(
                    records,
                    image_id,
                    dataset_name,
                    higher_is_better_by_measure=higher_is_better_by_measure,
                ),
            }
        )

    return {
        "error": None,
        "groups": groups,
        "active_stem": active_group["stem"],
        "items": items,
        "score_table": score_comparison_table(items),
    }