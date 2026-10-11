"""Duke PAM dataset adapter.

Source:
    Photoacoustic microscopy (PAM) images of mouse brain vasculature,
    https://zenodo.org/records/4042171 (``raw_train.zip``). Used by DiSpirito
    et al. for deep-learning reconstruction of *under-sampled* PAM.

This adapter turns the raw PAM PNGs into a *full-reference* IQA dataset. Each
source image is min-max normalised, then tiled into non-overlapping PATCHxPATCH
patches; each patch is a **reference** (ground truth). From every reference we
synthesise sparse-sampled **variants** that emulate the paper's under-sampling
strategy: keep every Nth pixel on a 2D grid (N in 2, 4, 8, 16) and restore the
size with nearest-neighbour interpolation. Higher N = sparser = lower quality,
giving a natural quality ordering with no human ratings (annotatable later via
contributed rankings/ratings, like the digital_twin set).

Converted layout written to ``<root>/converted/`` follows the data contract:

    <root>/converted/
        manifest.json
        images/
            <patch>.ref.ground_truth.npy     # the PATCHxPATCH reference patch
            <patch>.signal_roi.npy           # foreground mask (intensity >= mean)
            <patch>.noise_roi.npy            # background mask (intensity < mean)
            <patch>.var.sparse_<N>x.npy      # NN-upscaled sparse variant

Arrays are stored as float32 of shape ``(PATCH, PATCH, 1)`` in [0, 1].
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

ZENODO_BASE = "https://zenodo.org/records/4042171/files"
RAW_ZIP = "raw_train.zip"

REFERENCE_TYPE = "ground_truth"
PATCH = 256
SUBSAMPLE_FACTORS = (2, 4, 8, 16)  # each must divide PATCH
MAX_PATCHES_PER_IMAGE = 2  # cap, evenly spread across the source image


class DukePAMDataset(LocalDataset):
    """Duke under-sampled-PAM full-reference IQA dataset (no human ratings).

    Parameters
    ----------
    root:
        Directory holding ``raw_train.zip`` (or its unzipped ``raw_train/``);
        downloaded/converted in place.
    download:
        If True, fetch ``raw_train.zip`` from Zenodo when it is missing.
    force_convert:
        Re-run the conversion even if ``converted/`` already exists.
    """

    DATASET_NAME = "duke_pam"
    DATASET_DISPLAY_NAME = "Duke PAM"
    DATASET_VERSION = "1.0.1"
    DATASET_DESCRIPTION = (
        "Duke PAM: We use 256x256 reference patches from photoacoustic-microscopy images, "
        "each paired with sparse-sampled (2x, 4x, 8x, 16x) nearest-neighbour-upscaled variants "
        "for full-reference IQA (Zenodo 4042171)."
    )
    DATASET_BIBTEX = (
        "@article{dispirito2021reconstructing,\n"
        "  title     = {Reconstructing undersampled photoacoustic microscopy images "
        "using deep learning},\n"
        "  author    = {DiSpirito, Anthony and Li, Daiwei and Vu, Tri and Chen, Maomao and "
        "Zhang, Dong and Luo, Jianwen and Horstmeyer, Roarke and Yao, Junjie},\n"
        "  journal   = {IEEE Transactions on Medical Imaging},\n"
        "  volume    = {40},\n"
        "  number    = {2},\n"
        "  pages     = {562--570},\n"
        "  year      = {2021},\n"
        "  publisher = {IEEE},\n"
        "  doi       = {10.1109/TMI.2020.3031541}\n"
        "}"
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

        png_dir = self._ensure_unzipped()
        report_preparation("converting", "Converting images")
        self._convert(png_dir, converted_dir)
        self.root = converted_dir

    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _ensure_unzipped(self) -> Path:
        """Return the directory holding the unzipped PNGs, fetching as needed."""
        import shutil

        self.root.mkdir(parents=True, exist_ok=True)
        zip_path = self.root / RAW_ZIP
        png_root = self.root / "raw_train"

        if not png_root.exists():
            if not zip_path.exists():
                if not self._download:
                    raise FileNotFoundError(
                        f"{zip_path} not found and download=False. "
                        f"Place {RAW_ZIP} there or allow downloading."
                    )
                url = f"{ZENODO_BASE}/{RAW_ZIP}?download=1"
                print(f"Downloading {RAW_ZIP} from Zenodo …")
                report_preparation("downloading", "Downloading source archive")
                urllib.request.urlretrieve(url, zip_path)  # noqa: S310 — known Zenodo URL
                print(f"  → saved to {zip_path}")
            print(f"Extracting {RAW_ZIP} → {png_root} …")
            png_root.mkdir(parents=True, exist_ok=True)
            report_preparation("extracting", "Extracting source archive")
            shutil.unpack_archive(str(zip_path), str(png_root))

        return png_root

    # ------------------------------------------------------------------
    # Conversion: PNG → patches + sparse variants + manifest
    # ------------------------------------------------------------------

    @staticmethod
    def _read_gray(path: Path) -> np.ndarray:
        """Load a (16-bit) PNG as a 2D float64 array, min-max normalised to [0, 1]."""
        from PIL import Image as PILImage

        arr = np.asarray(PILImage.open(path)).astype(np.float64)
        if arr.ndim == 3:  # defensive: collapse any stray channel axis
            arr = arr.mean(axis=2)
        lo, hi = float(arr.min()), float(arr.max())
        return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    @staticmethod
    def _patch_positions(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
        """Row-major (row, col) tile indices, capped + evenly spread across the image."""
        positions = [(r, c) for r in range(n_rows) for c in range(n_cols)]
        if len(positions) <= MAX_PATCHES_PER_IMAGE:
            return positions
        idx = np.linspace(0, len(positions) - 1, MAX_PATCHES_PER_IMAGE + 2).round().astype(int)
        idx = idx[1:-1]  # exlude the startpoint and endpoint
        return [positions[i] for i in sorted(set(int(i) for i in idx))]

    @staticmethod
    def _subsample_nn(patch: np.ndarray, n: int) -> np.ndarray:
        """Keep every Nth pixel on a 2D grid, then nearest-neighbour upscale to PATCH.

        Exact NN because ``n`` divides ``PATCH``: ``patch[::n, ::n]`` is
        ``(PATCH/n, PATCH/n)`` and each kept pixel is tiled back into an n×n block.
        """
        sub = patch[::n, ::n, :]
        return np.repeat(np.repeat(sub, n, axis=0), n, axis=1)

    @staticmethod
    def _rois(ref: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Foreground/background masks split at the patch's mean intensity.

        Returns ``(None, None)`` for a degenerate (flat) patch where one side
        would be empty — such a patch simply gets no ROI (CNR/SNR skip it).
        """
        flat = ref[:, :, 0]
        threshold = float(flat.mean())
        signal = flat >= threshold
        noise = ~signal
        if signal.any() and noise.any():
            return signal[:, :, np.newaxis], noise[:, :, np.newaxis]
        return None, None

    def _convert(self, png_dir: Path, converted_dir: Path) -> None:
        images_out = converted_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        manifest_entries: list[dict] = []
        n_patches = 0
        seen_source_hashes: dict[str, Path] = {}
        n_duplicate_sources = 0

        png_paths = sorted(png_dir.rglob("*.png"))
        print(f"Converting {len(png_paths)} PAM images → {PATCH}x{PATCH} patches …")

        for png_path in png_paths:
            # raw_train.zip contains files with different names but identical PNG
            # content. Keep only the first occurrence (deterministic because
            # png_paths is sorted) so the same source image is not weighted twice.
            source_sha = hash_file(png_path)
            duplicate_of = seen_source_hashes.get(source_sha)
            if duplicate_of is not None:
                n_duplicate_sources += 1
                print(
                    f"  Skipping duplicate source {png_path.name!r} "
                    f"(identical to {duplicate_of.name!r})"
                )
                continue
            seen_source_hashes[source_sha] = png_path

            image = self._read_gray(png_path)
            h, w = image.shape
            n_rows, n_cols = h // PATCH, w // PATCH
            if n_rows == 0 or n_cols == 0:
                continue  # smaller than a single patch in some dimension

            base = re.sub(r"[^0-9A-Za-z]+", "_", png_path.stem).strip("_")

            for r, c in self._patch_positions(n_rows, n_cols):
                ref = image[r * PATCH : (r + 1) * PATCH, c * PATCH : (c + 1) * PATCH]
                ref = ref[:, :, np.newaxis].astype(np.float32)
                patch_id = f"{base}_r{r}_c{c}"

                ref_rel = f"images/{patch_id}.ref.{REFERENCE_TYPE}.npy"
                np.save(converted_dir / ref_rel, ref)
                ref_sha = hash_file(converted_dir / ref_rel)

                # ROI masks (shared by all variants of this patch), if non-degenerate.
                signal_roi, noise_roi = self._rois(ref)
                roi_fields: dict[str, dict] = {}
                if signal_roi is not None and noise_roi is not None:
                    sig_rel = f"images/{patch_id}.signal_roi.npy"
                    noi_rel = f"images/{patch_id}.noise_roi.npy"
                    np.save(converted_dir / sig_rel, signal_roi)
                    np.save(converted_dir / noi_rel, noise_roi)
                    roi_fields = {
                        "signal_roi": {
                            "path": sig_rel,
                            "sha256": hash_file(converted_dir / sig_rel),
                        },
                        "noise_roi": {
                            "path": noi_rel,
                            "sha256": hash_file(converted_dir / noi_rel),
                        },
                    }

                for n in SUBSAMPLE_FACTORS:
                    variant = self._subsample_nn(ref, n).astype(np.float32)
                    algo = f"sparse_{n}x"
                    var_rel = f"images/{patch_id}.var.{algo}.npy"
                    np.save(converted_dir / var_rel, variant)
                    manifest_entries.append(
                        {
                            "id": f"{patch_id}_{algo}",
                            "path": var_rel,
                            "sha256": hash_file(converted_dir / var_rel),
                            "shape": list(variant.shape),
                            "dtype": str(variant.dtype),
                            "references": {REFERENCE_TYPE: {"path": ref_rel, "sha256": ref_sha}},
                            **roi_fields,
                            "metadata": {
                                "stem": patch_id,  # groups variants of one reference
                                "algorithm": algo,
                                "subsample_factor": n,
                                "source_file": png_path.name,
                            },
                        }
                    )
                n_patches += 1

        manifest = {
            "schema_version": "1.0.0",
            "dataset_name": self.DATASET_NAME,
            "dataset_version": self.DATASET_VERSION,
            "description": self.DATASET_DESCRIPTION,
            "citation": (
                "A. DiSpirito et al. Reconstructing undersampled photoacoustic microscopy "
                "images using deep learning. IEEE TMI 40.2 (2021): 562-570. "
                "Data: https://zenodo.org/records/4042171"
            ),
            "license": "CC-BY-4.0",
            "created": datetime.now(UTC).isoformat(),
            "images": manifest_entries,
        }
        manifest_path = converted_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if n_duplicate_sources:
            print(
                f"Skipped {n_duplicate_sources} duplicate source PNG(s); "
                f"kept {len(seen_source_hashes)} unique source image(s)."
            )
        print(
            f"Manifest written to {manifest_path} "
            f"({len(manifest_entries)} variants across {n_patches} reference patches)"
        )