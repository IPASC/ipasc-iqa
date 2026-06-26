"""PAMMOTH speed-of-sound reconstruction dataset adapter.

Source:
    Maximum-intensity projections (MIPs) exported from PAMMOTH breast
    photoacoustic tomography reconstructions. For every volunteer/scan, the same
    anatomy is reconstructed with four speed-of-sound (SOS) strategies — ``1SOS``
    (single global SOS), ``2SOS`` (two-compartment SOS), ``TOF`` (time-of-flight
    SOS) and ``FWI`` (full-waveform-inversion SOS) — across multiple wavelengths
    and three projection planes (``axial``, ``sagittal``, ``frontal``).

    The archive ``pammoth_soundspeed_iqa_subset.zip`` mirrors the acquisition
    layout::

        pammoth_soundspeed_iqa_subset/<volunteer>/<scan>/PA/<method>/rec_<wl>_<plane>.npy

    where each ``.npy`` is a single 2D maximum-intensity projection (float32).

This adapter turns the MIPs into a *full-reference* IQA dataset. The **FWI**
reconstruction is treated as the ground-truth reference; the other three SOS
methods (``1SOS``, ``2SOS``, ``TOF``) become the **variants** that measures score
against that reference. Each MIP passes through :meth:`PAMMOTHDataset._process_image`
before being stored (currently a pass-through; an intensity transform will be
added there).

There are no signal/noise segmentations for this data, so no ROIs are written
(CNR/SNR simply skip these images).

Converted layout written to ``<root>/converted/`` follows the data contract:

    <root>/converted/
        manifest.json
        images/
            <stem>.ref.ground_truth.npy   # the FWI MIP
            <stem>.var.<method>.npy       # 1SOS / 2SOS / TOF MIP

Arrays are stored as float32 of shape ``(H, W, 1)``.
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

# Mocked Zenodo location — the archive is not yet published. Until the design is
# confirmed with collaborators, place the zip in the dataset root manually.
ZENODO_BASE = "https://zenodo.org/records/0000000/files"
RAW_ZIP = "pammoth_soundspeed_iqa_subset.zip"
UNPACKED_DIR = "pammoth_soundspeed_iqa_subset"

REFERENCE_TYPE = "ground_truth"
REFERENCE_METHOD = "FWI"  # full-waveform-inversion SOS = ground truth
VARIANT_METHODS = ("1SOS", "2SOS", "TOF")
LOG_FLOOR = 5  # clamp floor for the (currently disabled) log transform

# MIP filenames look like ``rec_<wl>_<plane>.npy``. Older slice exports carried a
# trailing offset (``rec_<wl>_<plane>_+00.npy``); this matches those so they can
# be filtered out — only the offset-free maximum-intensity projections are used.
_SLICE_OFFSET_RE = re.compile(r"_[+-]\d+\.npy$")


class PAMMOTHDataset(LocalDataset):
    """PAMMOTH speed-of-sound full-reference IQA dataset (no human ratings).

    Each cropped 2D slice is reconstructed with four SOS strategies. FWI is the
    reference; ``1SOS``/``2SOS``/``TOF`` are the variants scored against it.

    Parameters
    ----------
    root:
        Directory holding ``pammoth_soundspeed_iqa_subset.zip`` (or its unzipped
        contents); converted in place.
    download:
        If True, fetch the archive from Zenodo when it is missing (URL mocked).
    force_convert:
        Re-run the conversion even if ``converted/`` already exists.
    """

    DATASET_NAME = "pammoth"
    DATASET_VERSION = "1.0.0"
    DATASET_DESCRIPTION = (
        "PAMMOTH SOS: maximum-intensity projections from breast photoacoustic-tomography "
        "reconstructions, where the full-waveform-inversion (FWI) speed-of-sound "
        "reconstruction is the ground-truth reference and the 1SOS/2SOS/TOF "
        "reconstructions are the variants for full-reference IQA."
    )
    DATASET_BIBTEX = (
        "@misc{pammoth_soundspeed_iqa,\n"
        "  title        = {PAMMOTH speed-of-sound reconstruction IQA subset},\n"
        "  howpublished = {Zenodo (to appear)},\n"
        "  note         = {Download URL mocked pending publication}\n"
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

        slice_dir = self._ensure_unzipped()
        self._convert(slice_dir, converted_dir)
        self.root = converted_dir

    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _ensure_unzipped(self) -> Path:
        """Return the directory holding the unzipped slices, fetching as needed."""
        import shutil

        self.root.mkdir(parents=True, exist_ok=True)
        zip_path = self.root / RAW_ZIP
        slice_root = self.root / UNPACKED_DIR

        if not slice_root.exists():
            if not zip_path.exists():
                if not self._download:
                    raise FileNotFoundError(
                        f"{zip_path} not found and download=False. "
                        f"Place {RAW_ZIP} there or allow downloading."
                    )
                url = f"{ZENODO_BASE}/{RAW_ZIP}?download=1"
                print(f"Downloading {RAW_ZIP} from Zenodo (mocked URL) …")
                urllib.request.urlretrieve(url, zip_path)  # noqa: S310 — known Zenodo URL
                print(f"  → saved to {zip_path}")
            print(f"Extracting {RAW_ZIP} → {slice_root} …")
            shutil.unpack_archive(str(zip_path), str(self.root))

        return slice_root

    # ------------------------------------------------------------------
    # Conversion: slices → FWI reference + SOS variants + manifest
    # ------------------------------------------------------------------

    @staticmethod
    def _process_image(path: Path) -> np.ndarray:
        """Load a 2D MIP and return it as float32 ``(H, W, 1)``.

        Intensity processing is applied here. It is currently disabled (the MIP
        is passed through unchanged) pending a replacement transform; the former
        log compression is kept below, commented out, for reference::

            img[img < LOG_FLOOR] = LOG_FLOOR
            img = np.log10(img)
        """
        img = np.load(path).astype(np.float64)
        if img.ndim == 3:  # defensive: collapse any stray channel axis
            img = img[..., 0]
        return img[:, :, np.newaxis].astype(np.float32)

    def _convert(self, slice_dir: Path, converted_dir: Path) -> None:
        images_out = converted_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)

        # Group the four method reconstructions of one MIP together. A MIP is
        # identified by everything except the method folder:
        #   <volunteer>/<scan>/PA/<method>/<filename>  →  key (volunteer, scan, filename)
        # Only offset-free MIP files are kept; any leftover slice exports
        # (``rec_<wl>_<plane>_+00.npy``) are skipped.
        groups: dict[tuple[str, str, str], dict[str, Path]] = {}
        for npy_path in sorted(slice_dir.rglob("*.npy")):
            parts = npy_path.relative_to(slice_dir).parts
            # expect: volunteer, scan, "PA", method, filename
            if len(parts) != 5 or parts[2] != "PA":
                continue
            volunteer, scan, _pa, method, filename = parts
            if _SLICE_OFFSET_RE.search(filename):
                continue  # legacy 2D slice, not a MIP
            groups.setdefault((volunteer, scan, filename), {})[method] = npy_path

        print(f"Converting {len(groups)} PAMMOTH MIPs (FWI reference + SOS variants) …")

        manifest_entries: list[dict] = []
        n_refs = 0
        for (volunteer, scan, filename), method_paths in sorted(groups.items()):
            ref_path = method_paths.get(REFERENCE_METHOD)
            if ref_path is None:
                continue  # no FWI ground truth for this MIP — skip

            # Stem uniquely identifies the MIP (volunteer/scan/wavelength/plane).
            slice_id = Path(filename).stem  # e.g. rec_800nm_sagittal
            base = re.sub(r"[^0-9A-Za-z]+", "_", f"{volunteer}_{scan}_{slice_id}").strip("_")

            ref = self._process_image(ref_path)
            ref_rel = f"images/{base}.ref.{REFERENCE_TYPE}.npy"
            np.save(converted_dir / ref_rel, ref)
            ref_sha = hash_file(converted_dir / ref_rel)
            n_refs += 1

            for method in VARIANT_METHODS:
                var_path = method_paths.get(method)
                if var_path is None:
                    continue
                variant = self._process_image(var_path)
                var_rel = f"images/{base}.var.{method}.npy"
                np.save(converted_dir / var_rel, variant)
                manifest_entries.append(
                    {
                        "id": f"{base}_{method}",
                        "path": var_rel,
                        "sha256": hash_file(converted_dir / var_rel),
                        "shape": list(variant.shape),
                        "dtype": str(variant.dtype),
                        "references": {REFERENCE_TYPE: {"path": ref_rel, "sha256": ref_sha}},
                        "metadata": {
                            "stem": base,  # groups variants of one reference
                            "algorithm": method,
                            "volunteer": volunteer,
                            "scan": scan,
                            "source_file": filename,
                        },
                    }
                )

        manifest = {
            "schema_version": "1.0.0",
            "dataset_name": self.DATASET_NAME,
            "dataset_version": self.DATASET_VERSION,
            "description": self.DATASET_DESCRIPTION,
            "citation": (
                "PAMMOTH speed-of-sound reconstruction IQA subset. "
                "Zenodo (to appear). Download URL mocked pending publication."
            ),
            "license": "CC-BY-4.0",
            "created": datetime.now(UTC).isoformat(),
            "images": manifest_entries,
        }
        manifest_path = converted_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"Manifest written to {manifest_path} "
            f"({len(manifest_entries)} variants across {n_refs} FWI references)"
        )
