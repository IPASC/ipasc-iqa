"""Alladien et al. perceptual photoacoustic IQA datasets (2AFC pairwise study).

Source:
    A large crowd-sourced two-alternative-forced-choice (2AFC) study in which
    observers repeatedly picked the better of two photoacoustic reconstructions.
    The published archive bundles four sub-studies, each shipped here as a
    standalone dataset that shares this common adapter:

        - ``davoudi`` — sparse-sampling reconstructions (Davoudi et al. 2019)
        - ``oadat``   — forearm sparse-/limited-view reconstructions (Ozdemir et al. 2022)
        - ``efa``     — phantom + in-vivo handheld-probe reconstructions
        - ``phire``   — in-vivo mouse cross-sections (Horizon Europe PHIRE)

On-disk source layout (inside ``alladien_<name>.zip``)::

    results_clean.csv                         # pairwise comparisons for this sub-study
    images/<Subfolder>/<scene>/<token>.webp   # the compared reconstructions
    images/<Subfolder>/<scene>/<scene>.txt    # human-readable scene description

Each *scene* is a set of reconstruction *variants* of the same anatomy. Observers
compared variants pairwise; we fit a regularised Bradley-Terry model per scene
(:func:`ipasc_iqa.contrib.bradley_terry.fit_bradley_terry`) to turn those
preferences into a continuous quality score per variant. The variant the crowd
preferred most (highest strength) is taken as the scene's **pseudo-reference** —
mirroring the authors' choice of the "best" image as the reference — and the
remaining variants are scored against it by full-reference measures.

These are *pairwise* (Bradley-Terry), not absolute (MOS), data, so — like the
PhotIQA adapter keeps its published ratings in the manifest — the raw comparisons
are embedded in the manifest under ``human_comparisons``, one set per annotator
per scene (``{user, task, reference, comparisons: [[winner_id, loser_id], ...]}``,
with the reference image as the ``__ref__<scene>`` anchor pseudo-item). The GUI's
Bradley-Terry source re-fits these (anchored at the reference), pooling them with
any contributed comparisons, so the scatter shows correctly-labelled BT scores;
the per-annotator grouping also lets ranking coverage be counted as one set per
annotator (reference excluded), not once per pairwise comparison. Per-variant win/loss tallies
also live in ``metadata.pairwise`` and the full ``results_clean.csv`` is copied
into ``converted/``. Each scored variant carries a numeric ``algorithm`` quality
tier (1 = best non-reference variant) for consistent scatter grouping.

Converted layout written to ``<root>/converted/`` follows the data contract::

    <root>/converted/
        manifest.json
        results_clean.csv                      # the raw pairwise comparisons (retained)
        images/
            <scene>.ref.pseudo_reference.npy   # the most-preferred variant
            <scene>.var.<token>.npy            # the other variants

Arrays are stored as float32 of shape ``(H, W, 1)`` in [0, 1] — the luminance of
the rendered webp divided by 255. The Davoudi/OADAT renders are already greyscale
(R==G==B); the EFA/PHIRE renders are colour, so luminance is a lossy but
contract-compliant reduction. There are no signal/noise ROIs for this study.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ipasc_iqa.contrib.bradley_terry import fit_bradley_terry
from ipasc_iqa.core.roi import hash_file
from ipasc_iqa.datasets.local import LocalDataset

# Mocked Zenodo location — the perceptual-study archive is not yet published.
# Until then, place ``alladien_<name>.zip`` in the dataset root manually (this is
# what the bundled ``data/<name>/`` directories already do).
ZENODO_BASE = "https://zenodo.org/records/0000000/files"

REFERENCE_TYPE = "pseudo_reference"
ANNOTATION_TASK = "overall_quality"
BT_PRIOR_SD = 3.0  # Gaussian-prior std for the Bradley-Terry fit (see contrib)
# Anchor pseudo-item id for a scene's reference in the comparison records — must
# match gui.service._REF_PREFIX so manifest comparisons pool with contributed ones.
REF_ITEM_PREFIX = "__ref__"

_SANITISE_RE = re.compile(r"[^0-9A-Za-z]+")


def _safe(name: str) -> str:
    """Filesystem/id-safe token (collapse runs of non-alphanumerics to '_')."""
    return _SANITISE_RE.sub("_", name).strip("_")


def _variant_labels(tokens: list[str]) -> dict[str, str]:
    """Map each scene token to a grouping label = the part that varies in the scene.

    Tokens are split into alphanumeric *segments*; the segments shared by every
    token at the start (anatomy / acquisition prefix) and end (slice index / file
    suffix) are dropped, leaving the reconstruction setting (sampling level,
    method, frame count, ...). Working in whole segments avoids mangling values
    like ``128`` into ``8``. Falls back to ``"full"`` when nothing remains — that
    is typically the un-degraded reconstruction.
    """
    seglists = [[s for s in _SANITISE_RE.split(t) if s] for t in tokens]
    if not seglists:
        return {}

    lead = 0
    while all(len(s) > lead for s in seglists) and len({s[lead] for s in seglists}) == 1:
        lead += 1
    trail = 0
    while (
        all(len(s) > lead + trail for s in seglists)
        and len({s[len(s) - 1 - trail] for s in seglists}) == 1
    ):
        trail += 1

    labels: dict[str, str] = {}
    for token, segs in zip(tokens, seglists, strict=True):
        middle = segs[lead : len(segs) - trail] if len(segs) - trail > lead else []
        labels[token] = "_".join(middle) or "full"
    return labels


class AlladienDataset(LocalDataset):
    """Base adapter for the four Alladien-study 2AFC photoacoustic IQA datasets.

    Concrete subclasses set :attr:`DATASET_NAME`, :attr:`SOURCE` (the value of the
    CSV ``source`` column), :attr:`SUBFOLDER` (the ``images/<Subfolder>`` the
    sub-study lives in) and :attr:`RAW_ZIP` (the archive name in the dataset
    root). Only :meth:`_token_to_filename` differs between sub-studies; all
    Bradley-Terry scoring, reference selection and manifest writing are shared.

    Parameters
    ----------
    root:
        Directory holding ``alladien_<name>.zip`` (or its unzipped contents);
        downloaded/converted in place.
    download:
        If True, fetch the archive from Zenodo when it is missing (URL mocked).
    force_convert:
        Re-run the conversion even if ``converted/`` already exists.
    """

    #: Value of the CSV ``source`` column selecting this sub-study's rows.
    SOURCE: str = ""
    #: Image sub-folder inside the archive (``images/<SUBFOLDER>/<scene>/``).
    SUBFOLDER: str = ""
    #: Archive name expected in the dataset root.
    RAW_ZIP: str = ""
    #: If set, downscale each image so its longest edge is at most this many
    #: pixels (aspect ratio preserved). ``None`` keeps the native resolution.
    MAX_LONG_EDGE: int | None = None

    DATASET_VERSION = "1.0.0"

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
        if (converted_dir / "manifest.json").exists() and not self._force_convert:
            self.root = converted_dir  # already prepared
            return

        src_root = self._ensure_unzipped()
        self._convert(src_root, converted_dir)
        self.root = converted_dir

    def _manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def _ensure_unzipped(self) -> Path:
        """Return the root holding the unzipped sub-study, fetching as needed."""
        self.root.mkdir(parents=True, exist_ok=True)
        zip_path = self.root / self.RAW_ZIP
        images_root = self.root / "images" / self.SUBFOLDER

        if not images_root.exists():
            if not zip_path.exists():
                if not self._download:
                    raise FileNotFoundError(
                        f"{zip_path} not found and download=False. "
                        f"Place {self.RAW_ZIP} there or allow downloading."
                    )
                url = f"{ZENODO_BASE}/{self.RAW_ZIP}?download=1"
                print(f"Downloading {self.RAW_ZIP} from Zenodo (mocked URL)")
                urllib.request.urlretrieve(url, zip_path)  # noqa: S310 — known Zenodo URL
                print(f"  -> saved to {zip_path}")
            print(f"Extracting {self.RAW_ZIP} -> {self.root}")
            shutil.unpack_archive(str(zip_path), str(self.root))

        return self.root

    # ------------------------------------------------------------------
    # Per-sub-study hook
    # ------------------------------------------------------------------

    def _token_to_filename(self, token: str) -> str:
        """Map a CSV image token to its webp filename. Default: verbatim + .webp."""
        return f"{token}.webp"

    # ------------------------------------------------------------------
    # Conversion: webp + pairwise CSV -> Bradley-Terry scores + manifest
    # ------------------------------------------------------------------

    def _read_image(self, path: Path) -> np.ndarray:
        """Load a webp as float32 ``(H, W, 1)`` luminance in [0, 1].

        If :attr:`MAX_LONG_EDGE` is set and the image's longest edge exceeds it,
        the image is downscaled (Lanczos, aspect ratio preserved) first. A scene's
        variants share one source size, so they stay shape-matched after resizing.
        """
        from PIL import Image as PILImage

        img = PILImage.open(path).convert("L")
        if self.MAX_LONG_EDGE and max(img.size) > self.MAX_LONG_EDGE:
            scale = self.MAX_LONG_EDGE / max(img.size)
            new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            img = img.resize(new_size, PILImage.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr[:, :, np.newaxis]

    def _load_rows(self, src_root: Path) -> list[dict[str, str]]:
        """Read the pairwise CSV, keeping only this sub-study's rows."""
        csv_path = src_root / "results_clean.csv"
        with csv_path.open(encoding="utf-8", newline="") as fh:
            return [row for row in csv.DictReader(fh) if row["source"] == self.SOURCE]

    def _convert(self, src_root: Path, converted_dir: Path) -> None:
        if importlib.util.find_spec("PIL") is None:
            raise ImportError("Pillow is required to convert Alladien webp images.")

        images_out = converted_dir / "images"
        images_out.mkdir(parents=True, exist_ok=True)
        img_base = src_root / "images" / self.SUBFOLDER

        # Group pairwise rows by scene (each scene = variants of one anatomy).
        scenes: dict[str, list[dict[str, str]]] = {}
        for row in self._load_rows(src_root):
            scenes.setdefault(row["sceneId"], []).append(row)

        print(f"Converting {self.DATASET_NAME}: {len(scenes)} scenes")

        manifest_entries: list[dict] = []
        comparison_sets: list[dict] = []
        n_refs = 0
        for scene in sorted(scenes):
            srows = scenes[scene]
            tokens = sorted({t for r in srows for t in (r["imageA"], r["imageB"])})

            # (winner, loser) comparisons -> per-variant Bradley-Terry strengths.
            # The BT fit here is only used to pick the reference and order the
            # quality tiers; the GUI re-fits (anchored at the reference) for the
            # scores it correlates against, pooling these with any contributed
            # comparisons.
            valid = [r for r in srows if r["winner"] in (r["imageA"], r["imageB"])]
            comparisons = [
                (r["winner"], r["imageA"] if r["winner"] == r["imageB"] else r["imageB"])
                for r in valid
            ]
            bt = fit_bradley_terry(tokens, comparisons, prior_sd=BT_PRIOR_SD)

            # The most-preferred variant is the scene's pseudo-reference; in the
            # comparison records it becomes the __ref__ anchor pseudo-item.
            ref_token = max(tokens, key=lambda t: bt.beta[t])
            ref_item = f"{REF_ITEM_PREFIX}{scene}"
            item_of = {
                t: (ref_item if t == ref_token else f"{scene}_{_safe(t)}") for t in tokens
            }

            # Numeric quality tier: rank the scored (non-reference) variants by
            # descending strength (1 = best variant), a consistent ordering across
            # scenes for grouping the scatter points.
            scored = sorted((t for t in tokens if t != ref_token), key=lambda t: -bt.beta[t])
            tier_of = {t: i + 1 for i, t in enumerate(scored)}

            wins: Counter[str] = Counter(w for w, _ in comparisons)
            losses: Counter[str] = Counter(lo for _, lo in comparisons)
            labels = _variant_labels(tokens)

            # Per-annotator ranking sessions for this scene: each user's own
            # comparisons (mapped to item ids). These pool for the BT fit and let
            # the GUI count ranking coverage as one set per annotator (reference
            # excluded), rather than once per pairwise comparison.
            user_comps: dict[str, list[list[str]]] = defaultdict(list)
            annotators_of: dict[str, set[str]] = defaultdict(set)
            for r in valid:
                loser = r["imageA"] if r["winner"] == r["imageB"] else r["imageB"]
                user_comps[r["user_id"]].append([item_of[r["winner"]], item_of[loser]])
                annotators_of[r["winner"]].add(r["user_id"])
                annotators_of[loser].add(r["user_id"])
            for uid, comps in user_comps.items():
                comparison_sets.append(
                    {
                        "user": uid,
                        "task": ANNOTATION_TASK,
                        "reference": scene,
                        "comparisons": comps,
                    }
                )

            scene_dir = img_base / scene

            # Write the pseudo-reference array once for the scene.
            ref_arr = self._read_image(scene_dir / self._token_to_filename(ref_token))
            ref_rel = f"images/{scene}.ref.{REFERENCE_TYPE}.npy"
            np.save(converted_dir / ref_rel, ref_arr)
            ref_sha = hash_file(converted_dir / ref_rel)
            n_refs += 1

            for token in scored:
                arr = self._read_image(scene_dir / self._token_to_filename(token))
                var_rel = f"images/{scene}.var.{_safe(token)}.npy"
                np.save(converted_dir / var_rel, arr)

                manifest_entries.append(
                    {
                        "id": item_of[token],
                        "path": var_rel,
                        "sha256": hash_file(converted_dir / var_rel),
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                        "references": {REFERENCE_TYPE: {"path": ref_rel, "sha256": ref_sha}},
                        "metadata": {
                            "stem": scene,  # groups variants of one scene
                            "scene": scene,
                            "source_subset": self.SOURCE,
                            "algorithm": str(tier_of[token]),  # numeric quality tier (1 = best)
                            "quality_tier": tier_of[token],
                            "variant_token": token,
                            "variant_label": labels[token],
                            "reference_token": ref_token,
                            "source_file": self._token_to_filename(token),
                            "pairwise": {
                                "wins": int(wins[token]),
                                "losses": int(losses[token]),
                                "n_comparisons": int(wins[token] + losses[token]),
                                "n_annotators": len(annotators_of[token]),
                            },
                        },
                    }
                )

        # Retain the raw pairwise comparisons (as a CSV) alongside the converted data.
        shutil.copyfile(src_root / "results_clean.csv", converted_dir / "results_clean.csv")

        manifest = {
            "schema_version": "1.0.0",
            "dataset_name": self.DATASET_NAME,
            "dataset_version": self.DATASET_VERSION,
            "description": self.DATASET_DESCRIPTION,
            "citation": self._citation(),
            "license": "CC-BY-4.0",
            "created": datetime.now(UTC).isoformat(),
            "reference_type": REFERENCE_TYPE,
            "human_comparisons_csv": "results_clean.csv",
            "human_comparisons": comparison_sets,
            "images": manifest_entries,
        }
        manifest_path = converted_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"Manifest written to {manifest_path} "
            f"({len(manifest_entries)} variants across {n_refs} scene pseudo-references)"
        )

    def _citation(self) -> str:
        return (
            "Alladien et al., crowd-sourced 2AFC perceptual quality study of "
            f"photoacoustic reconstructions ({self.SOURCE} sub-study). "
            "Pairwise comparisons in results_clean.csv; per-scene quality from a "
            "regularised Bradley-Terry fit."
        )


# ----------------------------------------------------------------------
# Concrete sub-study adapters
# ----------------------------------------------------------------------


class DavoudiDataset(AlladienDataset):
    """Davoudi sparse-sampling sub-study (CSV ``source`` = "Sparse Sampling")."""

    DATASET_NAME = "davoudi"
    SOURCE = "Sparse Sampling"
    SUBFOLDER = "Davoudi"
    RAW_ZIP = "alladien_davoudi.zip"
    DATASET_DESCRIPTION = (
        "Alladien 2AFC sub-study (Davoudi sparse sampling): cross-sectional mouse "
        "optoacoustic reconstructions at varying sampling densities (16/32/64/128/256 "
        "and full), ranked by crowd pairwise comparisons. The most-preferred (full) "
        "reconstruction is the per-scene pseudo-reference."
    )
    DATASET_BIBTEX = (
        "@article{davoudi2019deep,\n"
        "  title   = {Deep learning optoacoustic tomography with sparse data},\n"
        "  author  = {Davoudi, Neda and Dean-Ben, Xose Luis and Razansky, Daniel},\n"
        "  journal = {Nature Machine Intelligence},\n"
        "  volume  = {1}, number = {10}, pages = {453--460}, year = {2019}\n"
        "}"
    )

    _TOKEN_RE = re.compile(r"^image_(\d+)_(.+)$")

    def _token_to_filename(self, token: str) -> str:
        # CSV uses a 1-based, un-padded index; files are 0-based, zero-padded to 4:
        #   image_113_128  ->  image_0112_128.webp
        m = self._TOKEN_RE.match(token)
        if not m:
            return f"{token}.webp"
        index, level = int(m.group(1)), m.group(2)
        return f"image_{index - 1:04d}_{level}.webp"


class OADATDataset(AlladienDataset):
    """OADAT forearm sub-study (CSV ``source`` = "OADAT")."""

    DATASET_NAME = "oadat"
    SOURCE = "OADAT"
    SUBFOLDER = "OADAT"
    RAW_ZIP = "alladien_oadat.zip"
    DATASET_DESCRIPTION = (
        "Alladien 2AFC sub-study (OADAT): forearm optoacoustic backprojection "
        "reconstructions degraded by sparse sampling (ss32/ss64/ss128) and limited "
        "view (lv128), ranked by crowd pairwise comparisons. The most-preferred "
        "(full backprojection) reconstruction is the per-scene pseudo-reference. "
        "Source data: Ozdemir et al., OADAT (2022)."
    )
    DATASET_BIBTEX = (
        "@article{lafci2022oadat,\n"
        "  title   = {OADAT: Experimental and synthetic clinical optoacoustic data "
        "for standardized image processing},\n"
        "  author  = {Lafci, Berkan and Ozdemir, Firat and others},\n"
        "  year    = {2022}\n"
        "}"
    )

    def _token_to_filename(self, token: str) -> str:
        # CSV separates the degradation tag with a comma; files use an underscore:
        #   SWFD_semicircle_sc,ss128_BP_slice0031 -> SWFD_semicircle_sc_ss128_BP_slice0031.webp
        return f"{token.replace(',', '_')}.webp"


class EFADataset(AlladienDataset):
    """EFA handheld-probe sub-study (CSV ``source`` = "EFA")."""

    DATASET_NAME = "efa"
    SOURCE = "EFA"
    SUBFOLDER = "EFA"
    RAW_ZIP = "alladien_efa.zip"
    DATASET_DESCRIPTION = (
        "Alladien 2AFC sub-study (EFA): handheld-probe photoacoustic reconstructions "
        "of phantoms (with a simulated ground truth) and in-vivo wrist/radial/finger "
        "scans, ranked by crowd pairwise comparisons. The most-preferred variant per "
        "scene is the pseudo-reference."
    )
    DATASET_BIBTEX = (
        "@article{xia2018handheld,\n"
        "  title   = {Handheld real-time LED-based photoacoustic and ultrasound "
        "imaging system ...},\n"
        "  author  = {Xia, Wenfeng and Kuniyil Ajith Singh, Mithun and others},\n"
        "  journal = {Sensors}, volume = {18}, number = {5}, pages = {1394}, year = {2018}\n"
        "}"
    )


class PHIREDataset(AlladienDataset):
    """PHIRE in-vivo mouse sub-study (CSV ``source`` = "PHIRE")."""

    DATASET_NAME = "phire"
    SOURCE = "PHIRE"
    SUBFOLDER = "PHIRE"
    RAW_ZIP = "alladien_phire.zip"
    # Native PHIRE renders are huge (up to ~6300x4100); full-res arrays make
    # measure computation glacial, so cap the longest edge at 1024 px.
    MAX_LONG_EDGE = 1024
    DATASET_DESCRIPTION = (
        "Alladien 2AFC sub-study (PHIRE): in-vivo mouse cross-sections (liver, "
        "bladder, brain, hindlimb, kidneys) reconstructed from differing frame "
        "counts, ranked by crowd pairwise comparisons. There is no physical ground "
        "truth; the most-preferred variant per scene is the pseudo-reference. "
        "Data source: Horizon Europe innovation programme — PHIRE."
    )
    DATASET_BIBTEX = (
        "@misc{phire_dataset,\n"
        "  title        = {PHIRE in-vivo photoacoustic reconstructions},\n"
        "  howpublished = {Horizon Europe innovation programme — PHIRE}\n"
        "}"
    )
