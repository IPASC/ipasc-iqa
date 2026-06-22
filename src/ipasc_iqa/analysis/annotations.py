"""Loading expert annotation CSV files.

An annotation CSV (one per *quality task*, e.g. ``overall_quality``) follows the
PhotIQA format::

    filename;rater1;...;rater5;mos;z-score rater1;...;z-score rater5;z-score mos

We only need the per-image aggregate scores (``mos`` and ``z-score mos``) for
correlation; individual rater columns are preserved but optional.

The loader is format-agnostic about delimiter (',' or ';') and tolerant of a
trailing ``.png``/``.npy`` extension on the filename, which it strips to obtain a
stable join key (the *annotation key*).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


def _strip_ext(name: str) -> str:
    """Normalise an annotation filename to a join key (drop a known extension)."""
    for ext in (".png", ".npy", ".jpg", ".jpeg", ".tif", ".tiff"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


@dataclass
class AnnotationTable:
    """Per-image scores for a single quality task."""

    task: str
    #: join key (filename stem) -> mean opinion score
    mos: dict[str, float] = field(default_factory=dict)
    #: join key (filename stem) -> z-scored mean opinion score
    zscore_mos: dict[str, float] = field(default_factory=dict)
    #: join key (filename stem) -> {rater_id: raw score} (individual annotators)
    raters: dict[str, dict[str, float]] = field(default_factory=dict)

    def image_keys(self) -> set[str]:
        """Return the set of join keys (filename stems) that have any score."""
        return set(self.mos) | set(self.zscore_mos) | set(self.raters)

    def __len__(self) -> int:
        return len(self.image_keys())


def _sniff_delimiter(header_line: str) -> str:
    return ";" if header_line.count(";") >= header_line.count(",") else ","


def load_annotation_csv(path: Path, task: str | None = None) -> AnnotationTable:
    """Load a single annotation CSV into an :class:`AnnotationTable`.

    Parameters
    ----------
    path:
        Path to the CSV file.
    task:
        Task name; defaults to the filename with a leading ``annotations_`` removed.
    """
    path = Path(path)
    if task is None:
        task = path.stem
        if task.startswith("annotations_"):
            task = task[len("annotations_") :]

    text = path.read_text(encoding="utf-8").splitlines()
    if not text:
        return AnnotationTable(task=task)

    delimiter = _sniff_delimiter(text[0])
    reader = csv.DictReader(text, delimiter=delimiter)

    # Raw per-rater columns (e.g. "rater1".."rater5"); the "z-score rater*"
    # columns are deliberately excluded (they start with "z-score").
    rater_cols = [c for c in (reader.fieldnames or []) if c and c.startswith("rater")]

    table = AnnotationTable(task=task)
    for row in reader:
        raw_name = (row.get("filename") or "").strip()
        if not raw_name:
            continue
        key = _strip_ext(raw_name)

        mos_raw = row.get("mos")
        if mos_raw not in (None, ""):
            table.mos[key] = float(mos_raw)

        z_raw = row.get("z-score mos")
        if z_raw not in (None, ""):
            table.zscore_mos[key] = float(z_raw)

        ratings = {c: float(row[c]) for c in rater_cols if row.get(c) not in (None, "")}
        if ratings:
            table.raters[key] = ratings

    return table


def load_annotation_dir(directory: Path) -> dict[str, AnnotationTable]:
    """Load every ``annotations_*.csv`` in *directory*, keyed by task name."""
    directory = Path(directory)
    tables: dict[str, AnnotationTable] = {}
    for csv_path in sorted(directory.glob("annotations_*.csv")):
        table = load_annotation_csv(csv_path)
        tables[table.task] = table
    return tables
