"""Tests for composite MOS from individual annotator ratings."""

from __future__ import annotations

import numpy as np

from ipasc_iqa.analysis.mos import composite_mos


def test_matches_average_of_per_annotator_zscores() -> None:
    # Three annotators, four images; composite = mean of per-annotator z-scores.
    raw = {
        "img0": {"a": 1.0, "b": 2.0, "c": 4.0},
        "img1": {"a": 2.0, "b": 2.0, "c": 3.0},
        "img2": {"a": 3.0, "b": 3.0, "c": 2.0},
        "img3": {"a": 4.0, "b": 3.0, "c": 1.0},
    }
    comp = composite_mos(raw)

    # Independent reference computation (population std).
    annotators: dict[str, dict[str, float]] = {}
    for img, scores in raw.items():
        for ann, v in scores.items():
            annotators.setdefault(ann, {})[img] = v
    z = {}
    for ann, vals in annotators.items():
        arr = np.array(list(vals.values()))
        mu, sd = arr.mean(), arr.std(ddof=0)
        z[ann] = {img: (v - mu) / sd for img, v in vals.items()}
    expected = {img: float(np.mean([z[a][img] for a in scores])) for img, scores in raw.items()}
    for img in raw:
        assert abs(comp[img] - expected[img]) < 1e-12


def test_zero_variance_annotator_contributes_zero() -> None:
    # An annotator who gives every image the same score adds 0 (after z-norm).
    raw = {
        "img0": {"flat": 3.0, "varied": 1.0},
        "img1": {"flat": 3.0, "varied": 4.0},
    }
    comp = composite_mos(raw)
    # 'flat' contributes 0; composite is half the 'varied' z-score.
    varied = np.array([1.0, 4.0])
    vz = (varied - varied.mean()) / varied.std(ddof=0)
    assert abs(comp["img0"] - vz[0] / 2) < 1e-12
    assert abs(comp["img1"] - vz[1] / 2) < 1e-12


def test_empty_input() -> None:
    assert composite_mos({}) == {}


def test_annotator_ratings_from_manifest_namespaces() -> None:
    from ipasc_iqa.analysis.manifest_query import annotator_ratings_from_manifest

    manifest = {
        "images": [
            {
                "id": "img0_alg0",
                "metadata": {
                    "annotations": {
                        "overall_quality": {
                            "mos": 2.8,
                            "zscore_mos": 0.1,
                            "ratings": {"rater1": 3, "rater2": 2},
                        }
                    }
                },
            },
            {"id": "img1_alg0", "metadata": {}},  # no annotations → skipped
        ]
    }
    out = annotator_ratings_from_manifest(manifest, "overall_quality")
    assert out == {"img0_alg0": {"manifest:rater1": 3.0, "manifest:rater2": 2.0}}
