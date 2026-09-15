"""Ear-vessel PAM dataset adapter (Gaussian-noise sensitivity).

Source:
    Photoacoustic microscopy (PAM) images of ear vasculature,
    https://data.mendeley.com/datasets/dp5jgrkd6k/2 (``ear_vessel_dataset.zip``).
    The unzipped archive holds source PNGs below ``train/image`` and ``test/image``;
    in the published archive they are grouped in patch subfolders such as
    ``test/image/patch_6/1.png``.

This adapter turns the raw PAM PNGs into a *full-reference* IQA dataset that
probes each measure's sensitivity to additive Gaussian noise in a PAM context.
Each source image is min-max normalised to [0, 1] and kept as a **reference**
(ground truth). From every reference we synthesise noisy **variants** by adding
zero-mean Gaussian noise whose standard deviation is a fixed percentage
(1, 2, 5, 10, 20%) of the image's maximum signal; results are clipped to [0, 1].
More noise = lower quality, giving a natural quality ordering with no human
ratings (annotatable later via contributed rankings/ratings).

Signal/noise ROIs come from the dataset's **manually annotated** vessel
segmentations, stored as PNG masks alongside each image in the sibling
``<split>/groundtruth/`` folder (same filename). The vessel (non-zero) pixels
form the signal ROI; the remaining pixels form the noise ROI.

Converted layout written to ``<root>/converted/`` follows the data contract:

    <root>/converted/
        manifest.json
        images/
            <img>.ref.ground_truth.npy   # the 256x256 reference
            <img>.signal_roi.npy         # vessel mask (annotated segmentation)
            <img>.noise_roi.npy          # background mask (complement)
            <img>.var.gauss_<pct>pct.npy # noisy variant

Arrays are stored as float32 of shape ``(H, W, 1)`` in [0, 1].
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

DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/dp5jgrkd6k/files/"
    "2bd72dfc-b693-445a-9230-187e863143a4/file_downloaded"
)
RAW_ZIP = "ear_vessel_dataset.zip"

REFERENCE_TYPE = "ground_truth"
NOISE_PERCENTAGES = (1, 2, 5, 10, 20)  # std as % of each image's max signal
NOISE_SEED = 0  # deterministic noise for reproducible variants


class EarVesselsDataset(LocalDataset):
    """Ear-vessel PAM Gaussian-noise sensitivity IQA dataset (no human ratings).

    Parameters
    ----------
    root:
        Directory holding ``ear_vessel_dataset.zip`` (or its unzipped contents);
        downloaded/converted in place.
    download:
        If True, fetch the archive from Mendeley Data when it is missing.
    force_convert:
        Re-run the conversion even if ``converted/`` already exists.
    """

    DATASET_NAME = "ear_vessels"
    DATASET_DISPLAY_NAME = "Meng et al."
    DATASET_VERSION = "1.0.0"
    DATASET_DESCRIPTION = (
        "Ear vessel dataset: 256x256 reference photoacoustic-microscopy images, each paired with "
        "additive-Gaussian-noise variants (std = 1, 2, 5, 10, 20% of the image max) for "
        "full-reference IQA noise-sensitivity testing (Mendeley Data dp5jgrkd6k)."
    )
    DATASET_BIBTEX = (
        "@misc{earvessel_dataset,\n"
        "  title     = {Ear vessel dataset},\n"
        "  howpublished = {Mendeley Data},\n"
        "  doi       = {10.17632/dp5jgrkd6k.2},\n"
        "  url       = {https://data.mendeley.com/datasets/dp5jgrkd6k/2}\n"
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
        self._convert(png_dir, converted_dir)
        self.root = converted_dir

    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _ensure_unzipped(self) -> Path:
        """Return the directory holding the unzipped PNGs, fetching as needed."""
        import shutil

        self.root.mkdir(parents=True, exist_ok=True)
        zip_path = self.root / RAW_ZIP
        png_root = self.root / "ear_vessel_dataset"

        if not png_root.exists():
            if not zip_path.exists():
                if not self._download:
                    raise FileNotFoundError(
                        f"{zip_path} not found and download=False. "
                        f"Place {RAW_ZIP} there or allow downloading."
                    )
                print(f"Downloading {RAW_ZIP} from Mendeley Data …")
                self._download_file(DOWNLOAD_URL, zip_path)
                print(f"  → saved to {zip_path}")
            print(f"Extracting {RAW_ZIP} → {png_root} …")
            png_root.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(zip_path), str(png_root))

        return png_root

    @staticmethod
    def _download_file(url: str, dest: Path) -> None:
        """Stream ``url`` to ``dest``, sending a browser User-Agent.

        Mendeley's file CDN rejects the default ``Python-urllib`` agent with
        HTTP 403, so we present a browser-like one (the link itself is public).
        """
        import shutil

        req = urllib.request.Request(  # noqa: S310 — known https Mendeley URL
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as resp, dest.open("wb") as fh:  # noqa: S310
            shutil.copyfileobj(resp, fh)

    # ------------------------------------------------------------------
    # Conversion: PNG → reference + noisy variants + manifest
    # ------------------------------------------------------------------

    @staticmethod
    def _read_gray(path: Path) -> np.ndarray:
        """Load a PNG as a 2D float64 array, min-max normalised to [0, 1]."""
        from PIL import Image as PILImage

        arr = np.asarray(PILImage.open(path)).astype(np.float64)
        if arr.ndim == 3:  # defensive: collapse any stray channel axis
            arr = arr.mean(axis=2)
        lo, hi = float(arr.min()), float(arr.max())
        return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    @staticmethod
    def _add_noise(ref: np.ndarray, pct: int, rng: np.random.Generator) -> np.ndarray:
        """Add zero-mean Gaussian noise (std = pct% of the image max), clipped to [0, 1]."""
        std = (pct / 100.0) * float(ref.max())
        noisy = ref + rng.normal(0.0, std, size=ref.shape)
        return np.clip(noisy, 0.0, 1.0)

    #: Folder names the archive has used for the segmentation masks.
    _MASK_DIRS = ("groundtruth", "ground_truth")

    @staticmethod
    def _image_dir(png_path: Path) -> Path | None:
        """Return the nearest ancestor folder named ``image``.

        The archive may store source PNGs directly in ``<split>/image`` or in
        nested patch folders such as ``<split>/image/patch_6/1.png``.
        """
        for parent in png_path.parents:
            if parent.name == "image":
                return parent
        return None

    @classmethod
    def _mask_path(cls, png_path: Path) -> Path | None:
        """Return the matching segmentation mask, preserving subfolders.

        Examples
        --------
        ``test/image/1.png`` -> ``test/groundtruth/1.png``
        ``test/image/patch_6/1.png`` -> ``test/groundtruth/patch_6/1.png``

        """
        image_dir = cls._image_dir(png_path)
        if image_dir is None:
            return None

        relative_path = png_path.relative_to(image_dir)
        split_dir = image_dir.parent

        for folder in cls._MASK_DIRS:
            candidate = split_dir / folder / relative_path
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def _rois(cls, png_path: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Load the manually annotated vessel segmentation as signal/noise ROIs.

        The mask lives in the ``<split>/groundtruth/`` folder next to the image's
        own ``<split>/image/`` folder, under the same filename. Non-zero pixels
        are the vessel (signal) ROI; the complement is the noise ROI. Returns
        ``(None, None)`` if the mask is missing or degenerate (one side empty) so
        CNR/SNR simply skip it.
        """
        from PIL import Image as PILImage

        mask_path = cls._mask_path(png_path)
        if mask_path is None:
            return None, None
        mask = np.asarray(PILImage.open(mask_path))
        if mask.ndim == 3:  # defensive: collapse any stray channel axis
            mask = mask.mean(axis=2)
        signal = mask > 0
        noise = ~signal
        if signal.any() and noise.any():
            return signal[:, :, np.newaxis], noise[:, :, np.newaxis]
        return None, None

    def _convert(self, png_dir: Path, converted_dir: Path) -> None:
        images_out = converted_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        manifest_entries: list[dict] = []
        rng = np.random.default_rng(NOISE_SEED)

        # Source images may be directly below ``<split>/image/`` or in nested
        # patch folders such as ``<split>/image/patch_6/1.png``.  Select every
        # PNG whose ancestor chain contains an ``image`` directory; masks under
        # ``groundtruth`` / ``ground_truth`` are therefore excluded.
        png_paths = sorted(
            p for p in png_dir.rglob("*.png") if self._image_dir(p) is not None
        )
        print(f"Converting {len(png_paths)} ear-vessel PAM images …")

        if not png_paths:
            raise RuntimeError(
                f"No source PNGs found below {png_dir}. Expected paths such as "
                "<split>/image/<file>.png or <split>/image/patch_x/<file>.png."
            )

        seen: set[str] = set()
        for png_path in png_paths:
            image_dir = self._image_dir(png_path)
            if image_dir is None:  # defensive; png_paths already enforces this
                continue

            ref = self._read_gray(png_path)[:, :, np.newaxis].astype(np.float32)

            # ``image_dir.parent`` is the split (train/test), independently of
            # how many patch subfolders exist below ``image``.
            split = image_dir.parent.name

            # Preserve the path relative to ``image`` in the id so files such as
            # patch_1/1.png and patch_6/1.png cannot collide.
            relative_source = png_path.relative_to(image_dir).with_suffix("")
            relative_id = "_".join(relative_source.parts)
            base = re.sub(
                r"[^0-9A-Za-z]+", "_", f"{split}_{relative_id}"
            ).strip("_")
            while base in seen:
                base += "_x"
            seen.add(base)

            ref_rel = f"images/{base}.ref.{REFERENCE_TYPE}.npy"
            np.save(converted_dir / ref_rel, ref)
            ref_sha = hash_file(converted_dir / ref_rel)

            # ROI masks (shared by all variants of this image), if available.
            signal_roi, noise_roi = self._rois(png_path)
            roi_fields: dict[str, dict] = {}
            if signal_roi is not None and noise_roi is not None:
                sig_rel = f"images/{base}.signal_roi.npy"
                noi_rel = f"images/{base}.noise_roi.npy"
                np.save(converted_dir / sig_rel, signal_roi)
                np.save(converted_dir / noi_rel, noise_roi)
                roi_fields = {
                    "signal_roi": {"path": sig_rel, "sha256": hash_file(converted_dir / sig_rel)},
                    "noise_roi": {"path": noi_rel, "sha256": hash_file(converted_dir / noi_rel)},
                }

            for pct in NOISE_PERCENTAGES:
                variant = self._add_noise(ref, pct, rng).astype(np.float32)
                algo = f"gauss_{pct}pct"
                var_rel = f"images/{base}.var.{algo}.npy"
                np.save(converted_dir / var_rel, variant)
                manifest_entries.append(
                    {
                        "id": f"{base}_{algo}",
                        "path": var_rel,
                        "sha256": hash_file(converted_dir / var_rel),
                        "shape": list(variant.shape),
                        "dtype": str(variant.dtype),
                        "references": {REFERENCE_TYPE: {"path": ref_rel, "sha256": ref_sha}},
                        **roi_fields,
                        "metadata": {
                            "stem": base,  # groups variants of one reference
                            "algorithm": algo,
                            "noise_pct": pct,
                            "split": split,
                            "source_file": png_path.name,
                            "source_relative_path": png_path.relative_to(image_dir).as_posix(),
                        },
                    }
                )

        manifest = {
            "schema_version": "1.0.0",
            "dataset_name": self.DATASET_NAME,
            "dataset_version": self.DATASET_VERSION,
            "description": self.DATASET_DESCRIPTION,
            "citation": (
                "Ear vessel dataset. Mendeley Data, V2 (2024). "
                "doi: 10.17632/dp5jgrkd6k.2. "
                "https://data.mendeley.com/datasets/dp5jgrkd6k/2"
            ),
            "license": "CC-BY-4.0",
            "created": datetime.now(UTC).isoformat(),
            "images": manifest_entries,
        }
        manifest_path = converted_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"Manifest written to {manifest_path} "
            f"({len(manifest_entries)} variants across {len(seen)} references)"
        )
