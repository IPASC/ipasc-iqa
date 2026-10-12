"""PAMMOTH speed-of-sound reconstruction dataset adapter.

Source:
    Maximum-intensity projections (MIPs) exported from PAMMOTH breast
    photoacoustic tomography reconstructions. For every volunteer/scan, the same
    anatomy is reconstructed with four speed-of-sound (SOS) strategies — ``1SOS``
    (single global SOS), ``2SOS`` (two-compartment SOS), ``TOF`` (time-of-flight
    SOS) and ``FWI`` (full-waveform-inversion SOS) — across multiple wavelengths
    and three projection planes (``axial``, ``sagittal``, ``frontal``).

    The full public PAM3 archive is downloaded from Zenodo. Complete sets of
    3D PA reconstructions are converted to cropped 2D MIPs in::

        pammoth_soundspeed_iqa_subset/<volunteer>/<scan>/PA/<method>/rec_<wl>_<plane>.npy

    where each ``.npy`` is a cropped 2D maximum-intensity projection (float32).
    The original archive and extracted source are kept for reuse.

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

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy.io as sio

from ipasc_iqa.core.roi import hash_file
from ipasc_iqa.datasets.local import LocalDataset
from ipasc_iqa.datasets.progress import report_preparation

# Pinned Zenodo version, ensuring reproducible data source.
ZENODO_BASE = "https://zenodo.org/records/20612980/files"
RAW_ZIP = "PAM3SystemIntroductionImagesVisualization.zip"
RAW_ZIP_MD5 = "ca9566c4538e30f74e397f8a48f77e96"
UNPACKED_DIR = "pammoth_soundspeed_iqa_subset"
RECONSTRUCTIONS_DIR = "reconstructions"
REQUIRED_SUBFOLDERS = ("1SOS", "2SOS", "FWI", "TOF")
PLANES = ("sagittal", "frontal", "axial")
VOLUME_KEY = "p0_rec"

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

    Each cropped 2D MIP is reconstructed with four SOS strategies. FWI is the
    reference; ``1SOS``/``2SOS``/``TOF`` are the variants scored against it.

    Parameters
    ----------
    root:
        Directory holding the official PAM3 ZIP, extracted reconstructions,
        and generated MIPs; converted in place.
    download:
        If True, fetch the official archive from Zenodo when missing.
    force_convert:
        Re-run the conversion even if ``converted/`` already exists.
    """

    DATASET_NAME = "pammoth"
    DATASET_DISPLAY_NAME = "PAM3"
    DATASET_VERSION = "1.0.0"
    DATASET_DESCRIPTION = (
        "PAM3 data: maximum-intensity projections from breast photoacoustic-tomography "
        "reconstructions, where the full-waveform-inversion (FWI) speed-of-sound "
        "reconstruction is the ground-truth reference and the 1SOS/2SOS/TOF "
        "reconstructions are the variants for full-reference IQA."
    )
    DATASET_BIBTEX = (
        "@misc{pammoth_soundspeed_iqa,\n"
        "  title        = {Photoacoustic and ultrasound tomography images of phantom and "
        "in-vivo data acquired with the PAM3 scanner},\n"
        "  author       = {Dantuma, Maura and Lucka, Felix and De Santi, Bruno and "
        "Bosman, Esther and Manohar, Srirang},\n"
        "  year         = {2026},\n"
        "  doi          = {10.5281/zenodo.20612980}\n"
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

        slice_dir = self._ensure_mips()
        report_preparation("converting", "Converting MIPs to benchmark images")
        self._convert(slice_dir, converted_dir)
        self.root = converted_dir

    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @staticmethod
    def _md5(path: Path) -> str:
        digest = hashlib.md5()  # noqa: S324 - checksum published by Zenodo
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _download_archive(cls, url: str, target: Path, *, retries: int = 5) -> None:
        """Download with retries and HTTP Range resume, retaining partial data.

        If the server ignores Range, restart from byte zero rather than append
        a full response to an incomplete ZIP. Never delete a partial download
        after a transient network failure.
        """
        partial = target.with_name(target.name + ".part")
        for attempt in range(retries):
            report_preparation("downloading", f"Downloading PAM3 archive (attempt {attempt + 1}/{retries})")
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "ipasc-iqa/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                    status = response.status
                    content_range = response.headers.get("Content-Range", "")
                    if offset and status == 206:
                        match = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", content_range)
                        if not match or int(match.group(1)) != offset:
                            raise OSError(f"Unexpected Content-Range: {content_range!r}")
                        mode = "ab"
                        total = int(match.group(3)) if match.group(3) != "*" else None
                    elif status == 200:
                        # Zenodo (or an intermediary) may ignore Range requests.
                        # Replace the incomplete file only when a full response starts.
                        mode = "wb"
                        length = response.headers.get("Content-Length")
                        total = int(length) if length is not None else None
                    else:
                        raise OSError(f"Unexpected HTTP status {status} for download")
                    with partial.open(mode) as output:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                    size = partial.stat().st_size
                    if total is not None and size != total:
                        raise OSError(f"Incomplete download: {size} of {total} bytes")
                print("  -> download complete, checking MD5 ...")
                if cls._md5(partial) != RAW_ZIP_MD5:
                    raise ValueError(
                        "PAM3 ZIP failed MD5 verification; partial file retained "
                        f"at {partial}. Remove it before retrying if corrupted."
                    )
                partial.replace(target)
                return
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                if attempt == retries - 1:
                    raise RuntimeError(
                        f"PAM3 download interrupted after {retries} attempts. "
                        f"Saved {partial.stat().st_size if partial.exists() else 0} "
                        f"bytes in {partial}; retry the run to resume."
                    ) from exc
                print(f"  -> download interrupted ({exc}); retrying ({attempt + 1}/{retries}) ...")
                time.sleep(min(2 ** attempt, 12))

    def _ensure_reconstructions(self) -> Path:
        """Download and unpack the official archive unless already extracted."""
        self.root.mkdir(parents=True, exist_ok=True)
        existing = self._find_reconstructions()
        if existing is not None:
            return existing

        zip_path = self.root / RAW_ZIP
        if not zip_path.exists():
            if not self._download:
                raise FileNotFoundError(
                    f"No extracted reconstructions or {RAW_ZIP} found in {self.root}, "
                    "and download=False."
                )
            url = f"{ZENODO_BASE}/{RAW_ZIP}?download=1"
            print(f"Downloading {RAW_ZIP} (~11.8 GB) from Zenodo ...")
            self._download_archive(url, zip_path)
            print(f"  -> saved to {zip_path}")
        elif self._md5(zip_path) != RAW_ZIP_MD5:
            raise ValueError(
                f"Existing archive {zip_path} does not match Zenodo's checksum. "
                "Remove it and retry."
            )

        report_preparation("extracting", "Extracting PAM3 source archive")
        print(f"Extracting {RAW_ZIP} into {self.root} ...")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(self.root)
        found = self._find_reconstructions()
        if found is None:
            raise FileNotFoundError(
                f"No '{RECONSTRUCTIONS_DIR}' directory found inside {RAW_ZIP}"
            )
        return found

    def _find_reconstructions(self) -> Path | None:
        for candidate in sorted(self.root.rglob(RECONSTRUCTIONS_DIR)):
            if candidate.is_dir() and any(candidate.rglob("PA")):
                return candidate
        return None

    @staticmethod
    def _complete_sets(reconstructions: Path) -> list[tuple[Path, list[str]]]:
        matches: list[tuple[Path, list[str]]] = []
        for pa_dir in sorted(reconstructions.rglob("PA")):
            if not pa_dir.is_dir():
                continue
            common: set[str] | None = None
            for method in REQUIRED_SUBFOLDERS:
                method_dir = pa_dir / method
                if not method_dir.is_dir():
                    common = set()
                    break
                names = {f.name for f in method_dir.glob("*.mat")}
                common = names if common is None else common & names
                if not common:
                    break
            if common:
                # Matches the original script: only common filenames are used.
                matches.append((pa_dir, sorted(common)))
        return matches

    @staticmethod
    def _mip(volume: np.ndarray, plane: str) -> np.ndarray:
        axis = {"sagittal": 0, "frontal": 1, "axial": 2}[plane]
        projection = np.max(volume, axis=axis)
        return projection[150:-150, 150:].T

    def _ensure_mips(self) -> Path:
        """Generate precisely the MIPs exported by the source scripts."""
        slice_root = self.root / UNPACKED_DIR
        # A completed MIP export can be reused without reading the large MATs.
        done = slice_root / ".complete"
        if done.exists():
            return slice_root

        reconstructions = self._ensure_reconstructions()
        matches = self._complete_sets(reconstructions)
        if not matches:
            raise ValueError(f"No complete PA reconstruction sets in {reconstructions}")

        report_preparation("converting", "Generating MIPs from 3D reconstructions")
        count = 0
        print(f"Generating PAM3 MIPs from {len(matches)} complete PA folder(s) ...")
        for pa_dir, shared in matches:
            relative = pa_dir.relative_to(reconstructions)
            # Layout required by the existing converter: <volunteer>/<scan>/PA.
            if len(relative.parts) != 3 or relative.parts[-1] != "PA":
                print(f"Skipping unsupported PA folder layout: {relative}")
                continue
            for filename in shared:
                stem = Path(filename).stem
                for method in REQUIRED_SUBFOLDERS:
                    source = pa_dir / method / filename
                    volume = np.asarray(sio.loadmat(source)[VOLUME_KEY])
                    if volume.ndim != 3:
                        raise ValueError(f"Expected 3D p0_rec in {source}, got {volume.shape}")
                    out_dir = slice_root / relative / method
                    out_dir.mkdir(parents=True, exist_ok=True)
                    for plane in PLANES:
                        mip = self._mip(volume, plane)
                        if not mip.size:
                            raise ValueError(f"Empty {plane} crop from {source}")
                        np.save(out_dir / f"{stem}_{plane}.npy", mip)
                        count += 1
        if not count:
            raise ValueError("No MIPs generated from the discovered PA folders")
        done.write_text(f"{count} MIPs\n", encoding="utf-8")
        print(f"Generated {count} MIPs in {slice_root}")
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
                "Dantuma et al., Photoacoustic and ultrasound tomography images "
                "of phantom and in-vivo data acquired with the PAM3 scanner, "
                "Zenodo, 2026. doi:10.5281/zenodo.20612980."
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
