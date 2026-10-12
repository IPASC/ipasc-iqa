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
            imageX.ref.ground_truth.npy   (shared per stem X)

Images are stored as float32 arrays of shape (H, W, 1), normalised to [0, 1].
The source PNGs are already linearly normalised relative to the reference image,
so we divide by 255 to obtain [0, 1] floats.
"""

from __future__ import annotations

import json
import re
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ipasc_iqa.core.roi import hash_file
from ipasc_iqa.datasets.local import LocalDataset
from ipasc_iqa.datasets.progress import report_preparation

ZENODO_BASE = "https://zenodo.org/records/19920440/files"
NORMALISED_IMAGES_ZIP = "normalised_images.zip"
ANNOTATIONS_ZIP = "annotations.zip"
EPS = 1e-10

ALGORITHM_NAMES = {0: "alg0", 1: "alg1", 2: "alg2"}
REFERENCE_TYPE = "ground_truth"

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
    DATASET_DISPLAY_NAME = "PhotIQA"
    DATASET_VERSION = "1.0.0"
    DATASET_DESCRIPTION = (
        "PhotIQA: 1134 reconstructed photoacoustic images from 3 algorithms, paired "
        "with 378 references and expert quality ratings (Breger et al., Zenodo 13325196)."
    )
    DATASET_BIBTEX = (
        "@dataset{breger2024photiqa,\n"
        "  author    = {Breger, Anna and others},\n"
        "  title     = {{PhotIQA}: A Photoacoustic Image Data Set with Image Quality Ratings},\n"
        "  year      = {2024},\n"
        "  publisher = {Zenodo},\n"
        "  doi       = {10.5281/zenodo.13325196},\n"
        "  url       = {https://zenodo.org/records/13325196}\n"
        "}"
    )

    @classmethod
    def prepared_manifest_path(cls, root: Path | str) -> Path:
        # PhotIQA converts source data into a <root>/converted/ sub-folder.
        return Path(root) / "converted" / "manifest.json"

    def __init__(
        self,
        root: Path | str,
        *,
        download: bool = True,
        force_convert: bool = False,
    ) -> None:
        self._force_convert = force_convert
        # LocalDataset.__init__ stores self._download, calls _prepare(), reads manifest.
        super().__init__(root, download=download)

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

        report_preparation("converting", "Converting images")
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
            report_preparation("downloading", "Downloading source archive")
            urllib.request.urlretrieve(url, dest)  # noqa: S310 — known Zenodo URL
            print(f"  → saved to {dest}")

    # ------------------------------------------------------------------
    # Conversion: PNG → NPY + manifest
    # ------------------------------------------------------------------

    def _convert(self, converted_dir: Path) -> None:
        import importlib.util

        if importlib.util.find_spec("PIL") is None:
            raise ImportError("Pillow is required to convert PhotIQA images.")

        images_out = converted_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        src_root = self.root
        # Unzip if needed
        ref_dir = src_root / "normalised_images" / "reference"
        alg_dir = src_root / "normalised_images" / "algorithms"

        if not ref_dir.exists() or not alg_dir.exists():
            self._unzip_sources(src_root)
            report_preparation("converting", "Converting images")
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

        # Load expert annotations (keyed by source stem, e.g. "image0_0")
        annotations = self._load_annotations(src_root)

        print(f"Converting {sum(len(v) for v in alg_map.values())} algorithm images …")

        manifest_entries = []
        written_refs: set[str] = set()  # track which reference npy files we've written

        for stem in sorted(alg_map.keys(), key=lambda s: int(s)):
            ref_npy_relpath: str | None = None
            signal_roi_relpath: str | None = None
            noise_roi_relpath: str | None = None

            # The reference and its derived ROI masks are shared by all variants of
            # this stem. Their filenames are deterministic, so resolve the relpaths
            # for every variant and write the files once.
            if stem in ref_map:
                ref_npy_relpath = f"images/image{stem}.ref.{REFERENCE_TYPE}.npy"
                signal_roi_relpath = f"images/image{stem}.signal_roi.npy"
                noise_roi_relpath = f"images/image{stem}.noise_roi.npy"

                if stem not in written_refs:
                    ref_arr = self._png_to_npy(ref_map[stem])
                    np.save(converted_dir / ref_npy_relpath, ref_arr)
                    written_refs.add(stem)

                    # Background = the darkest (near-minimum) pixels of the reference;
                    # signal = everything else. Stored as boolean masks.
                    noise_roi = ref_arr <= (np.min(ref_arr) + EPS)
                    signal_roi = ~noise_roi
                    np.save(converted_dir / noise_roi_relpath, noise_roi)
                    np.save(converted_dir / signal_roi_relpath, signal_roi)

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

                # Attach expert annotations keyed by the source stem (e.g. "image0_0")
                ann_for_image = annotations.get(alg_path.stem)
                if ann_for_image:
                    entry["metadata"]["annotations"] = ann_for_image

                # Register the shared reference + ROI masks on every variant entry,
                # using the same {path, sha256} structure the framework discovers.
                if ref_npy_relpath:
                    entry["references"] = {
                        REFERENCE_TYPE: {
                            "path": ref_npy_relpath,
                            "sha256": hash_file(converted_dir / ref_npy_relpath),
                        }
                    }
                if signal_roi_relpath:
                    entry["signal_roi"] = {
                        "path": signal_roi_relpath,
                        "sha256": hash_file(converted_dir / signal_roi_relpath),
                    }
                if noise_roi_relpath:
                    entry["noise_roi"] = {
                        "path": noise_roi_relpath,
                        "sha256": hash_file(converted_dir / noise_roi_relpath),
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
            "created": datetime.now(UTC).isoformat(),
            "images": manifest_entries,
        }

        manifest_path = converted_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Manifest written to {manifest_path} ({len(manifest_entries)} images)")

    @staticmethod
    def _load_annotations(src_root: Path) -> dict[str, dict[str, dict]]:
        """Load expert annotation CSVs into ``{source_stem: {task: {...}}}``.

        Each task slot holds the aggregate ``mos`` / ``zscore_mos`` *and* the
        individual per-rater scores under ``ratings`` ({rater_id: raw}), so a
        composite MOS can be recomputed (and pooled with contributed ratings).
        Returns an empty mapping if no annotation directory is found.
        """
        from ipasc_iqa.analysis.annotations import load_annotation_dir

        # Locate a directory containing annotations_*.csv (handles annotations/<modality>/).
        candidates = list(src_root.rglob("annotations_overall_quality.csv"))
        if not candidates:
            return {}
        ann_dir = candidates[0].parent

        tables = load_annotation_dir(ann_dir)

        # The project starts with a single generic "overall quality" task. The
        # other PhotIQA sub-tasks are disabled for now — re-enable one by
        # uncommenting its name below.
        enabled_tasks = {
            "overall_quality",
            # "edge_visibility",
            # "object_homogeneity",
            # "object_intensity_(inclusions)",
            # "object_intensity_(background)",
        }

        per_image: dict[str, dict[str, dict]] = {}
        for task, table in tables.items():
            if task not in enabled_tasks:
                continue
            for key in table.image_keys():
                slot = per_image.setdefault(key, {})
                slot[task] = {}
                if key in table.mos:
                    slot[task]["mos"] = table.mos[key]
                if key in table.zscore_mos:
                    slot[task]["zscore_mos"] = table.zscore_mos[key]
                if key in table.raters:
                    slot[task]["ratings"] = table.raters[key]
        return per_image

    @staticmethod
    def _unzip_sources(root: Path) -> None:
        import shutil

        for fname in (NORMALISED_IMAGES_ZIP, ANNOTATIONS_ZIP):
            zip_path = root / fname
            if not zip_path.exists():
                continue
            print(f"Extracting {fname} …")
            dest = root / "normalised_images" if "normalised" in fname else root
            report_preparation("extracting", "Extracting source archive")
            shutil.unpack_archive(str(zip_path), str(dest if "normalised" in fname else root))

    @staticmethod
    def _png_to_npy(png_path: Path) -> np.ndarray:
        """Load a PNG, convert to float32 [0, 1], shape (H, W, 1)."""
        from PIL import Image as PILImage

        img = PILImage.open(png_path).convert("L")  # grayscale
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr[:, :, np.newaxis]
