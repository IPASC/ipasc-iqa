"""PhotIQA dataset adapter.

Source:
    Breger et al., "PhotIQA: A photoacoustic image data set with image quality ratings"
    https://zenodo.org/records/13325196

On-disk source layout (after download):
    <root>/
        normalised_images/
            reference/   imageX.png          (378 reference images)
            algorithms/  imageX_Y.png        (1134 algorithm images, Y in {0,1,2})
        annotations/
            photoacoustic/
                annotations_overall_quality.csv
                ... (other quality CSV files)

Converted layout written to <root>/converted/ follows the framework data contract:
    <root>/converted/
        manifest.json
        images/
            imageX.var.alg0.npy
            imageX.var.alg1.npy
            imageX.var.alg2.npy
            imageX.ref.pseudo_reference.npy   (shared per stem X)

Images are stored as float32 arrays of shape (H, W, 1), normalised to [0, 1].
The source PNGs are already linearly normalised relative to the reference image,
so we divide by 255 to obtain [0, 1] floats.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from ipasc_iqa.core.dataset import Dataset, ImageHashes, ImageRecord
from ipasc_iqa.core.roi import hash_array, hash_file
from ipasc_iqa.datasets.local import LocalDataset

ZENODO_BASE = "https://zenodo.org/records/13325196/files"
NORMALISED_IMAGES_ZIP = "normalised_images.zip"
ANNOTATIONS_ZIP = "annotations.zip"

ALGORITHM_NAMES = {0: "alg0", 1: "alg1", 2: "alg2"}
REFERENCE_TYPE = "pseudo_reference"

_ALG_FILE_RE = re.compile(r"^image(\d+)_(\d+)\.png$")
_REF_FILE_RE = re.compile(r"^image(\d+)\.png$")


class PhotIQADataset(LocalDataset):
    """PhotIQA photoacoustic image quality dataset.

    Parameters
    ----------
    root:
        Directory where the dataset will be stored (downloaded + converted).
        The directory is created if it does not exist.
    download:
        If True, download source files from Zenodo when missing.
    force_convert:
        If True, re-run the PNG→NPY conversion even if ``converted/`` already exists.
    """

    DATASET_NAME = "phot_iqa"
    DATASET_VERSION = "1.0.0"

    def __init__(
        self,
        root: Path | str,
        *,
        download: bool = True,
        force_convert: bool = False,
    ) -> None:
        self._download = download
        self._force_convert = force_convert
        # LocalDataset.__init__ calls self._prepare() then reads the manifest.
        super().__init__(root)

    # ------------------------------------------------------------------
    # Overridden hook: download + convert before manifest is read
    # ------------------------------------------------------------------

    def _prepare(self) -> None:
        converted_dir = self.root / "converted"
        manifest_path = converted_dir / "manifest.json"

        if manifest_path.exists() and not self._force_convert:
            # Already prepared; re-point root so LocalDataset reads from converted/
            self.root = converted_dir
            return

        if self._download:
            self._download_sources()

        self._convert(converted_dir)
        # Re-point root to converted layout
        self.root = converted_dir

    # ------------------------------------------------------------------
    # Manifest path — point at converted/
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_sources(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for fname in (NORMALISED_IMAGES_ZIP, ANNOTATIONS_ZIP):
            dest = self.root / fname
            if dest.exists():
                continue
            url = f"{ZENODO_BASE}/{fname}?download=1"
            print(f"Downloading {fname} from Zenodo …")
            urllib.request.urlretrieve(url, dest)  # noqa: S310 — known Zenodo URL
            print(f"  → saved to {dest}")

    # ------------------------------------------------------------------
    # Conversion: PNG → NPY + manifest
    # ------------------------------------------------------------------

    def _convert(self, converted_dir: Path) -> None:
        try:
            from PIL import Image as PILImage
        except ImportError as exc:
            raise ImportError("Pillow is required to convert PhotIQA images.") from exc

        images_out = converted_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        src_root = self.root
        # Unzip if needed
        ref_dir = src_root / "normalised_images" / "reference"
        alg_dir = src_root / "normalised_images" / "algorithms"

        if not ref_dir.exists() or not alg_dir.exists():
            self._unzip_sources(src_root)
            ref_dir = src_root / "normalised_images" / "reference"
            alg_dir = src_root / "normalised_images" / "algorithms"

        if not ref_dir.exists():
            raise FileNotFoundError(
                f"Expected reference images at {ref_dir}. "
                "Ensure the normalised_images.zip was downloaded."
            )

        # Map stem → Path for references
        ref_map: dict[str, Path] = {}
        for p in ref_dir.glob("*.png"):
            if m := _REF_FILE_RE.match(p.name):
                ref_map[m.group(1)] = p

        # Collect algorithm files grouped by stem
        alg_map: dict[str, dict[int, Path]] = {}
        for p in alg_dir.glob("*.png"):
            if m := _ALG_FILE_RE.match(p.name):
                stem, alg_idx = m.group(1), int(m.group(2))
                alg_map.setdefault(stem, {})[alg_idx] = p

        print(f"Converting {sum(len(v) for v in alg_map.values())} algorithm images …")

        manifest_entries = []
        written_refs: set[str] = set()  # track which reference npy files we've written

        for stem in sorted(alg_map.keys(), key=lambda s: int(s)):
            ref_npy_relpath: str | None = None

            # Write shared reference once per stem
            if stem in ref_map and stem not in written_refs:
                ref_arr = self._png_to_npy(ref_map[stem])
                ref_npy_name = f"image{stem}.ref.{REFERENCE_TYPE}.npy"
                ref_npy_path = images_out / ref_npy_name
                np.save(ref_npy_path, ref_arr)
                written_refs.add(stem)
                ref_npy_relpath = f"images/{ref_npy_name}"
            elif stem in written_refs:
                ref_npy_relpath = f"images/image{stem}.ref.{REFERENCE_TYPE}.npy"

            for alg_idx, alg_path in sorted(alg_map[stem].items()):
                alg_name = ALGORITHM_NAMES.get(alg_idx, f"alg{alg_idx}")
                npy_name = f"image{stem}.var.{alg_name}.npy"
                npy_path = images_out / npy_name
                arr = self._png_to_npy(alg_path)
                np.save(npy_path, arr)

                entry: dict = {
                    "id": f"image{stem}_{alg_name}",
                    "path": f"images/{npy_name}",
                    "sha256": hash_file(npy_path),
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "metadata": {
                        "stem": stem,
                        "algorithm": alg_name,
                        "algorithm_index": alg_idx,
                        "source_file": alg_path.name,
                    },
                }

                if ref_npy_relpath:
                    ref_abs = converted_dir / ref_npy_relpath
                    entry["references"] = {
                        REFERENCE_TYPE: {
                            "path": ref_npy_relpath,
                            "sha256": hash_file(ref_abs),
                        }
                    }

                manifest_entries.append(entry)

        manifest = {
            "schema_version": "1.0.0",
            "dataset_name": self.DATASET_NAME,
            "dataset_version": self.DATASET_VERSION,
            "description": (
                "PhotIQA photoacoustic image quality dataset. "
                "1134 reconstructed images from 3 algorithms paired with 378 reference images. "
                "Source: https://zenodo.org/records/13325196"
            ),
            "citation": (
                "A. Breger et al., PhotIQA: A photoacoustic image data set with "
                "image quality ratings. https://zenodo.org/records/13325196"
            ),
            "license": "CC-BY-4.0",
            "created": datetime.now(timezone.utc).isoformat(),
            "images": manifest_entries,
        }

        manifest_path = converted_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest written to {manifest_path} ({len(manifest_entries)} images)")

    @staticmethod
    def _unzip_sources(root: Path) -> None:
        import shutil

        for fname in (NORMALISED_IMAGES_ZIP, ANNOTATIONS_ZIP):
            zip_path = root / fname
            if not zip_path.exists():
                continue
            print(f"Extracting {fname} …")
            dest = root / "normalised_images" if "normalised" in fname else root
            shutil.unpack_archive(str(zip_path), str(dest if "normalised" in fname else root))

    @staticmethod
    def _png_to_npy(png_path: Path) -> np.ndarray:
        """Load a PNG, convert to float32 [0, 1], shape (H, W, 1)."""
        from PIL import Image as PILImage

        img = PILImage.open(png_path).convert("L")  # grayscale
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr[:, :, np.newaxis]
