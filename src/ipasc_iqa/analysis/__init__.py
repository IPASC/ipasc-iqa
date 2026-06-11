"""Analysis layer: annotation loading, measure↔MOS correlation, standings.

This is pure logic (no I/O framework, no web). The GUI and CLI both call it.
"""

from ipasc_iqa.analysis.annotations import (
    AnnotationTable,
    load_annotation_csv,
    load_annotation_dir,
)
from ipasc_iqa.analysis.correlation import (
    CorrelationRow,
    correlate_measures_with_annotations,
)
from ipasc_iqa.analysis.standings import StandingRow, compute_standings

__all__ = [
    "AnnotationTable",
    "load_annotation_csv",
    "load_annotation_dir",
    "CorrelationRow",
    "correlate_measures_with_annotations",
    "StandingRow",
    "compute_standings",
]
