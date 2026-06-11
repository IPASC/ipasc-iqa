"""Manifest generation and loading."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from ipasc_iqa.core.roi import hash_file


def generate_manifest(
    folder: Path,
    *,
    dataset_name: str = "",
    dataset_version: str = "1.0.0",
    description: str = "",
    output: Path | None = None,
) -> dict:
    """Scan *folder/images/* and produce a manifest dict (and optionally write it)."""
    images_dir = folder / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"images/ directory not found in {folder}")

    # Collect all .npy files, identify roles by suffix pattern.
    # Reconstruction:  <stem>.npy  (no extra dots before .npy other than .var.X)
    # Signal ROI:      <stem>.signal_roi.npy
    # Noise ROI:       <stem>.noise_roi.npy
    # Reference:       <stem>.ref.<type>.npy

    signal_roi_re = re.compile(r"^(.+)\.signal_roi$")
    noise_roi_re = re.compile(r"^(.+)\.noise_roi$")
    reference_re = re.compile(r"^(.+)\.ref\.([^.]+)$")

    all_npy = sorted(images_dir.glob("*.npy"))

    signal_rois: dict[str, Path] = {}
    noise_rois: dict[str, Path] = {}
    references: dict[str, dict[str, Path]] = {}
    reconstructions: dict[str, Path] = {}

    for p in all_npy:
        stem = p.stem  # filename without last .npy
        if m := signal_roi_re.match(stem):
            signal_rois[m.group(1)] = p
        elif m := noise_roi_re.match(stem):
            noise_rois[m.group(1)] = p
        elif m := reference_re.match(stem):
            base, ref_type = m.group(1), m.group(2)
            references.setdefault(base, {})[ref_type] = p
        else:
            # Reconstruction — strip .var. from id
            image_id = re.sub(r"\.var\.", "_", stem)
            reconstructions[image_id] = p

    import numpy as np

    entries = []
    for image_id, img_path in sorted(reconstructions.items()):
        # Resolve the base stem for sibling lookup (undo our _var_ → .var. rewrite)
        raw_stem = img_path.stem
        # base stem for sibling lookup: strip .var.<X> suffix if present
        base_stem_m = re.match(r"^(.+)\.var\.[^.]+$", raw_stem)
        base_stem = base_stem_m.group(1) if base_stem_m else raw_stem

        arr = np.load(img_path)
        entry: dict = {
            "id": image_id,
            "path": str(img_path.relative_to(folder)),
            "sha256": hash_file(img_path),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

        if base_stem in signal_rois:
            rp = signal_rois[base_stem]
            entry["signal_roi"] = {
                "path": str(rp.relative_to(folder)),
                "sha256": hash_file(rp),
            }

        if base_stem in noise_rois:
            rp = noise_rois[base_stem]
            entry["noise_roi"] = {
                "path": str(rp.relative_to(folder)),
                "sha256": hash_file(rp),
            }

        if base_stem in references:
            entry["references"] = {
                ref_type: {
                    "path": str(rp.relative_to(folder)),
                    "sha256": hash_file(rp),
                }
                for ref_type, rp in references[base_stem].items()
            }

        entries.append(entry)

    manifest = {
        "schema_version": "1.0.0",
        "dataset_name": dataset_name or folder.name,
        "dataset_version": dataset_version,
        "description": description,
        "created": datetime.now(UTC).isoformat(),
        "images": entries,
    }

    out_path = output or (folder / "manifest.json")
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
