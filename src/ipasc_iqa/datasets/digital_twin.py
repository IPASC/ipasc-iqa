"""Digital Twin dataset adapter.

Source:
    "Digital twins enable full-reference quality assessment of photoacoustic
    image reconstructions", https://zenodo.org/records/15388429 (data.zip).

This dataset ships *no* human quality annotations — it is a full-reference set
where the simulated initial-pressure distribution (``p0``) is the true ground
truth and several reconstruction algorithms are compared against it.

Source layout (inside data.zip), per split ``calibration/`` and ``testing/``
(``full_view/`` is ignored — different array geometry):

    <split>/
        p0/                         # ground-truth initial pressure, (300, 300)
        raw/{exp,sim,sim_raw}/       # time series — not used here
        recons/<algo>/sim/           # reconstruction from the calibrated sim data
            <name>.npy               # (300, 300); same <name> as the p0 file

Reconstruction algorithms: bp, bph, fft, ittr, mb, tr. For each scene the p0
file is the reference and the six reconstructions are variants of it, paired by
identical filename. ``calibration`` and ``testing`` are pooled (split-prefixed
ids avoid collisions).

Converted layout written to ``<root>/converted/`` follows the data contract:

    <root>/converted/
        manifest.json
        images/
            <split>_<stem>.ref.ground_truth.npy
            <split>_<stem>.var.<algo>.npy

Arrays are stored as float32 of shape ``(H, W, 1)`` (the framework's channel
convention; measures consume ``(H, W, C)``).
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ipasc_iqa.core.roi import hash_file
from ipasc_iqa.datasets.local import LocalDataset

ZENODO_BASE = "https://zenodo.org/records/15388429/files"
DATA_ZIP = "data.zip"

REFERENCE_TYPE = "ground_truth"
RECON_ALGORITHMS = ("bp", "bph", "fft", "ittr", "mb", "tr")
SPLITS = ("calibration", "testing")
RECON_SOURCE = "sim"  # only the calibrated-simulation reconstructions are used


class DigitalTwinDataset(LocalDataset):
    """Digital Twin photoacoustic reconstruction dataset (no human ratings).

    Parameters
    ----------
    root:
        Directory where the dataset is stored (downloaded + unzipped + converted).
    download:
        If True, download ``data.zip`` from Zenodo when neither it nor the
        unzipped ``data/`` folder is already present.
    force_convert:
        Re-run the conversion even if ``converted/`` already exists.
    """

    DATASET_NAME = "digital_twin"
    DATASET_VERSION = "1.0.0"
    DATASET_DESCRIPTION = (
        "Digital Twin Dataset: photoacoustic reconstructions (bp, bph, fft, ittr, mb, tr) "
        "of calibrated simulations, with the simulated initial pressure (p0) as full-reference "
        "ground truth. No human quality ratings (Zenodo 15388429)."
    )

    @classmethod
    def prepared_manifest_path(cls, root: Path | str) -> Path:
        return Path(root) / "converted" / "manifest.json"

    def __init__(
        self,
        root: Path | str,
        *,
        download: bool = True,
        force_convert: bool = False,
    ) -> None:
        self._force_convert = force_convert
        super().__init__(root, download=download)

    # ------------------------------------------------------------------
    # Download + convert before the manifest is read
    # ------------------------------------------------------------------

    def _prepare(self) -> None:
        converted_dir = self.root / "converted"
        manifest_path = converted_dir / "manifest.json"

        if manifest_path.exists() and not self._force_convert:
            self.root = converted_dir  # already prepared
            return

        data_root = self._ensure_unzipped()
        self._convert(data_root, converted_dir)
        self.root = converted_dir

    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    # ------------------------------------------------------------------
    # Download / unzip (each step skipped if its output already exists)
    # ------------------------------------------------------------------

    def _ensure_unzipped(self) -> Path:
        """Return the directory holding the unzipped source, fetching as needed."""
        self.root.mkdir(parents=True, exist_ok=True)
        zip_path = self.root / DATA_ZIP
        data_root = self.root / "data"

        if not data_root.exists():
            if not zip_path.exists():
                if not self._download:
                    raise FileNotFoundError(
                        f"{zip_path} not found and download=False. "
                        f"Place data.zip there or allow downloading."
                    )
                url = f"{ZENODO_BASE}/{DATA_ZIP}?download=1"
                print(f"Downloading {DATA_ZIP} (~2.9 GB) from Zenodo …")
                urllib.request.urlretrieve(url, zip_path)  # noqa: S310 — known Zenodo URL
                print(f"  → saved to {zip_path}")
            print(f"Extracting {DATA_ZIP} → {data_root} …")
            data_root.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(zip_path), str(data_root))

        return data_root

    # ------------------------------------------------------------------
    # Conversion: source .npy → framework .npy + manifest
    # ------------------------------------------------------------------

    @staticmethod
    def _find_split_dir(data_root: Path, split: str) -> Path | None:
        """Locate a split directory (e.g. 'calibration') that contains a p0/ folder.

        Robust to an extra wrapping folder inside the archive.
        """
        for candidate in data_root.rglob(split):
            if candidate.is_dir() and (candidate / "p0").is_dir():
                return candidate
        return None

    @staticmethod
    def _to_npy_array(path: Path) -> np.ndarray:
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr

    def _convert(self, data_root: Path, converted_dir: Path) -> None:
        images_out = converted_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        manifest_entries: list[dict] = []
        n_scenes = 0

        for split in SPLITS:
            split_dir = self._find_split_dir(data_root, split)
            if split_dir is None:
                print(f"  · split '{split}' not found under {data_root}; skipping")
                continue

            p0_dir = split_dir / "p0"
            for p0_path in sorted(p0_dir.glob("*.npy")):
                name = p0_path.name  # identical filename used in p0/ and recons/*/sim/
                stem = p0_path.stem
                scene = f"{split}_{stem}"

                # Reference (ground truth) — written once per scene.
                ref_rel = f"images/{scene}.ref.{REFERENCE_TYPE}.npy"
                np.save(converted_dir / ref_rel, self._to_npy_array(p0_path))
                ref_sha = hash_file(converted_dir / ref_rel)

                wrote_any_variant = False
                for algo in RECON_ALGORITHMS:
                    recon_path = split_dir / "recons" / algo / RECON_SOURCE / name
                    if not recon_path.exists():
                        continue
                    var_rel = f"images/{scene}.var.{algo}.npy"
                    arr = self._to_npy_array(recon_path)
                    np.save(converted_dir / var_rel, arr)
                    manifest_entries.append(
                        {
                            "id": f"{scene}_{algo}",
                            "path": var_rel,
                            "sha256": hash_file(converted_dir / var_rel),
                            "shape": list(arr.shape),
                            "dtype": str(arr.dtype),
                            "references": {
                                REFERENCE_TYPE: {"path": ref_rel, "sha256": ref_sha}
                            },
                            "metadata": {
                                "split": split,
                                "stem": scene,  # groups variants of one reference
                                "algorithm": algo,
                                "source_file": name,
                            },
                        }
                    )
                    wrote_any_variant = True

                if wrote_any_variant:
                    n_scenes += 1
                else:
                    # No reconstructions for this scene — drop the orphan reference.
                    (converted_dir / ref_rel).unlink(missing_ok=True)

        manifest = {
            "schema_version": "1.0.0",
            "dataset_name": self.DATASET_NAME,
            "dataset_version": self.DATASET_VERSION,
            "description": self.DATASET_DESCRIPTION,
            "citation": (
                "Digital twins enable full-reference quality assessment of photoacoustic "
                "image reconstructions. https://zenodo.org/records/15388429"
            ),
            "license": "CC-BY-4.0",
            "created": datetime.now(UTC).isoformat(),
            "images": manifest_entries,
        }
        manifest_path = converted_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"Manifest written to {manifest_path} "
            f"({len(manifest_entries)} reconstructions across {n_scenes} scenes)"
        )
