"""Unit tests for the Alladien 2AFC dataset adapters (no network)."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ipasc_iqa.datasets.alladien import (
    REFERENCE_TYPE,
    AlladienDataset,
    DavoudiDataset,
    PHIREDataset,
    _variant_labels,
)

CSV_HEADER = [
    "user_id",
    "sceneId",
    "imageA",
    "imageB",
    "winner",
    "duration_ms",
    "device_type",
    "expertise_level",
    "description",
    "source",
]


def _webp_bytes(value: int, size: tuple[int, int] = (8, 6)) -> bytes:
    """A solid-colour webp encoded to bytes (RGB so colour datasets are covered)."""
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", size, (value, value, value)).save(buf, "WEBP", lossless=True)
    return buf.getvalue()


def _csv_bytes(rows: list[list[str]]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_HEADER)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _row(scene: str, a: str, b: str, winner: str, source: str) -> list[str]:
    return ["1", scene, a, b, winner, "1000", "Desktop", "Expert", source, source]


def _write_zip(
    root: Path,
    *,
    raw_zip: str,
    subfolder: str,
    files: dict[str, bytes],
    rows: list[list[str]],
) -> None:
    """Create <root>/<raw_zip> with results_clean.csv + images/<subfolder>/... files.

    ``files`` maps ``<scene>/<filename>.webp`` to image bytes.
    """
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / raw_zip, "w") as z:
        z.writestr("results_clean.csv", _csv_bytes(rows))
        for rel, data in files.items():
            z.writestr(f"images/{subfolder}/{rel}", data)


def _manifest(root: Path) -> dict:
    return json.loads((root / "converted" / "manifest.json").read_text(encoding="utf-8"))


def _davoudi_fixture(root: Path) -> None:
    """One Davoudi scene where 'full' is preferred most → becomes the reference.

    CSV tokens are 1-based/un-padded (``image_2_full``); files are 0-based and
    zero-padded to 4 (``image_0001_full.webp``), exercising the token mapping.
    """
    scene = "scene_2201"
    files = {
        f"{scene}/image_0001_full.webp": _webp_bytes(240),
        f"{scene}/image_0001_32.webp": _webp_bytes(120),
        f"{scene}/image_0001_16.webp": _webp_bytes(40),
    }
    full, v32, v16 = "image_2_full", "image_2_32", "image_2_16"
    rows: list[list[str]] = []
    rows += [_row(scene, full, v16, full, "Sparse Sampling") for _ in range(5)]
    rows += [_row(scene, full, v32, full, "Sparse Sampling") for _ in range(5)]
    rows += [_row(scene, v32, v16, v32, "Sparse Sampling") for _ in range(5)]
    _write_zip(root, raw_zip="alladien_davoudi.zip", subfolder="Davoudi", files=files, rows=rows)


def test_reference_is_most_preferred_variant(tmp_path: Path) -> None:
    root = tmp_path / "davoudi"
    _davoudi_fixture(root)

    DavoudiDataset(root, download=False)
    manifest = _manifest(root)
    entries = manifest["images"]

    assert manifest["reference_type"] == REFERENCE_TYPE
    # 'full' won the most comparisons → reference, so it is NOT a scored entry;
    # the other two variants are.
    assert len(entries) == 2
    assert {e["metadata"]["variant_token"] for e in entries} == {"image_2_32", "image_2_16"}
    for e in entries:
        assert e["metadata"]["reference_token"] == "image_2_full"
        assert REFERENCE_TYPE in e["references"]
        assert e["shape"] == [6, 8, 1]  # (H, W, 1)
        assert e["dtype"] == "float32"

    # The pseudo-reference array is the 'full' image (value 240/255).
    ref_rel = entries[0]["references"][REFERENCE_TYPE]["path"]
    ref = np.load(root / "converted" / ref_rel)
    assert np.allclose(ref, 240.0 / 255.0, atol=1e-3)


def test_quality_tiers_are_numeric_and_ordered(tmp_path: Path) -> None:
    root = tmp_path / "davoudi"
    _davoudi_fixture(root)
    DavoudiDataset(root, download=False)
    entries = _manifest(root)["images"]
    by_token = {e["metadata"]["variant_token"]: e for e in entries}

    # 32 beat 16, so 32 is the better (tier 1) scored variant, 16 tier 2;
    # the reference ('full') is excluded. Groups are numeric, not raw labels.
    assert by_token["image_2_32"]["metadata"]["algorithm"] == "1"
    assert by_token["image_2_16"]["metadata"]["algorithm"] == "2"
    assert by_token["image_2_32"]["metadata"]["quality_tier"] == 1
    assert {e["metadata"]["algorithm"] for e in entries} == {"1", "2"}

    # No fake MOS annotations remain; raw win/loss tallies are kept.
    assert "annotations" not in by_token["image_2_16"]["metadata"]
    assert by_token["image_2_16"]["metadata"]["pairwise"] == {
        "wins": 0,
        "losses": 10,
        "n_comparisons": 10,
        "n_annotators": 1,  # the fixture's single annotator
    }


def test_human_comparisons_feed_bradley_terry(tmp_path: Path) -> None:
    from ipasc_iqa.analysis.manifest_query import (
        comparison_tasks_in_manifest,
        comparisons_from_manifest,
    )
    from ipasc_iqa.contrib.bradley_terry import fit_bradley_terry

    root = tmp_path / "davoudi"
    _davoudi_fixture(root)
    DavoudiDataset(root, download=False)
    manifest = _manifest(root)

    assert comparison_tasks_in_manifest(manifest) == ["overall_quality"]
    triples = comparisons_from_manifest(manifest, "overall_quality")
    assert len(triples) == 15  # 5 + 5 + 5 rows

    # The reference participates as the __ref__ anchor pseudo-item, not an entry id.
    ref_item = "__ref__scene_2201"
    entry_ids = {e["id"] for e in manifest["images"]}
    assert ref_item not in entry_ids
    refs = {r for r, _, _ in triples}
    assert refs == {"scene_2201"}

    # Re-fitting anchored at the reference (as the GUI does) reproduces the order:
    # reference at 0, both variants below it, the 32-variant above the 16-variant.
    comps = [(w, lo) for _, w, lo in triples]
    items = sorted({i for pair in comps for i in pair})
    res = fit_bradley_terry(items, comps, anchor=ref_item)
    assert res.beta[ref_item] == 0.0
    assert res.beta["scene_2201_image_2_32"] > res.beta["scene_2201_image_2_16"]
    assert res.beta["scene_2201_image_2_32"] < 0  # below the reference


def test_raw_comparisons_csv_is_retained(tmp_path: Path) -> None:
    root = tmp_path / "davoudi"
    _davoudi_fixture(root)
    DavoudiDataset(root, download=False)

    manifest = _manifest(root)
    assert manifest["human_comparisons_csv"] == "results_clean.csv"
    assert (root / "converted" / "results_clean.csv").exists()


def test_ranking_count_excludes_reference_and_counts_sets_once(tmp_path: Path) -> None:
    from ipasc_iqa.analysis.manifest_query import ranking_observations_from_manifest

    scene = "scene_2201"
    files = {
        f"{scene}/image_0001_full.webp": _webp_bytes(240),
        f"{scene}/image_0001_32.webp": _webp_bytes(120),
        f"{scene}/image_0001_16.webp": _webp_bytes(40),
    }
    full, v32, v16 = "image_2_full", "image_2_32", "image_2_16"

    def row(user: str, a: str, b: str, winner: str) -> list[str]:
        r = _row(scene, a, b, winner, "Sparse Sampling")
        r[0] = user
        return r

    # Two annotators; 'full' is preferred by both, so it is the reference.
    rows = [row("1", full, v16, full), row("1", full, v32, full), row("1", v32, v16, v32)]
    rows += [row("2", full, v16, full), row("2", v32, v16, v32)]
    root = tmp_path / "davoudi"
    _write_zip(root, raw_zip="alladien_davoudi.zip", subfolder="Davoudi", files=files, rows=rows)

    DavoudiDataset(root, download=False)
    manifest = _manifest(root)

    # One comparison set per annotator (not one per pairwise comparison).
    sets = manifest["human_comparisons"]
    assert {c["user"] for c in sets} == {"1", "2"}
    assert all(c["reference"] == scene for c in sets)

    # Each annotator ranked exactly the two non-reference variants {32, 16}, so
    # placements = 2 annotators x 2 variants = 4 — NOT the 5 raw comparisons, and
    # the reference ('full') is excluded even though it appears in comparisons.
    assert ranking_observations_from_manifest(manifest, "overall_quality") == 4
    items = {i for c in sets for pair in c["comparisons"] for i in pair}
    assert "__ref__scene_2201" in items  # reference participates but isn't counted

    by_token = {e["metadata"]["variant_token"]: e for e in manifest["images"]}
    assert by_token[v16]["metadata"]["pairwise"]["n_annotators"] == 2
    assert by_token[v32]["metadata"]["pairwise"]["n_annotators"] == 2


def test_download_false_without_data_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        DavoudiDataset(tmp_path / "davoudi", download=False)


def test_variant_labels_split_on_segments() -> None:
    # Whole-segment stripping keeps '128' intact (no truncation to '8') and drops
    # the shared anatomy/slice segments.
    tokens = ["image_5_16", "image_5_32", "image_5_128", "image_5_full"]
    assert _variant_labels(tokens) == {
        "image_5_16": "16",
        "image_5_32": "32",
        "image_5_128": "128",
        "image_5_full": "full",
    }
    # OADAT-style: comma tag survives as an underscore-joined label; the plain
    # backprojection (nothing distinguishing) falls back to 'full'.
    oadat = [
        "SWFD_semicircle_sc_BP_slice0031",
        "SWFD_semicircle_sc,ss128_BP_slice0031",
        "SWFD_semicircle_sc,lv128_BP_slice0031",
    ]
    labels = _variant_labels(oadat)
    assert labels["SWFD_semicircle_sc_BP_slice0031"] == "full"
    assert labels["SWFD_semicircle_sc,ss128_BP_slice0031"] == "ss128"
    assert labels["SWFD_semicircle_sc,lv128_BP_slice0031"] == "lv128"


def test_phire_downscales_to_max_long_edge(tmp_path: Path) -> None:
    # PHIRE caps the longest edge at 1024; a scene's variants share one source
    # size, so reference and variants stay shape-matched after resizing.
    assert PHIREDataset.MAX_LONG_EDGE == 1024
    scene = "scene_6000"
    big = (2200, 1100)  # (w, h) > 1024 on the long edge
    files = {
        f"{scene}/Liver_1 51-80.webp": _webp_bytes(40, big),
        f"{scene}/Liver_1 54-80.webp": _webp_bytes(200, big),
        f"{scene}/Liver_1 56-80.webp": _webp_bytes(120, big),
    }
    a, b, c = "Liver_1 51-80", "Liver_1 54-80", "Liver_1 56-80"
    rows = [_row(scene, b, a, b, "PHIRE") for _ in range(4)]  # 'b' preferred -> reference
    rows += [_row(scene, c, a, c, "PHIRE") for _ in range(4)]
    root = tmp_path / "phire"
    _write_zip(root, raw_zip="alladien_phire.zip", subfolder="PHIRE", files=files, rows=rows)

    PHIREDataset(root, download=False)
    entries = _manifest(root)["images"]
    for e in entries:
        h, w, ch = e["shape"]
        assert max(h, w) == 1024 and ch == 1
        assert h == 512 and w == 1024  # 2200x1100 -> 1024x512, aspect preserved
    ref_rel = entries[0]["references"][REFERENCE_TYPE]["path"]
    assert list(np.load(root / "converted" / ref_rel).shape) == entries[0]["shape"]


def test_base_adapter_requires_name() -> None:
    # The abstract base must not be registrable / usable without a DATASET_NAME.
    assert AlladienDataset.DATASET_NAME == "local"
