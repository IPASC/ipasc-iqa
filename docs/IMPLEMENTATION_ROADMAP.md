# `ipasc_iqa` — Implementation Roadmap

**A benchmarking framework for Image Quality Assessment (IQA) of photoacoustic image reconstructions.**

---

## 1. Project summary

`ipasc_iqa` is a Python framework that evaluates IQA measures on standardized photoacoustic reconstruction datasets. 
It supports both classical and deep-learning measures, ships reference datasets with integrity verification, 
produces machine-comparable results, and offers a command line interface (CLI) plus a browser-based GUI.

**Design pillars**

1. **Single plugin axis** — only IQA measures are pluggable; datasets follow a fixed on-disk contract.
2. **Reproducibility-first** — every result JSON carries dataset version, file hashes, measure version, ROI hash, and reference-type used.
3. **Ecosystem-friendly** — third-party measures live in their own packages and register via Python entry points, preserving authorship.
4. **Researcher-friendly** — CLI is the primary interface; the GUI is a thin layer over the same Python API.

---

## 2. Decisions of record

| Area | Decision |
|---|---|
| Domain focus | Photoacoustic IQA; data layer is generic enough to extend later. |
| Pluggable axis | IQA measures only. Datasets ship reconstructions; no reconstruction plugin layer. |
| DL backend | PyTorch as reference; base class is backend-agnostic. |
| ROI handling | Per-image masks as sibling files. |
| References | Multiple typed references per image (e.g. `ground_truth`, `pseudo_reference`, `expert`). |
| Versioning | Manifest + SHA256 checksums |
| Plugin model | Python entry points (`[project.entry-points."ipasc_iqa.measures"]`). |
| GUI | Streamlit, importing the core Python API directly (web-hostable later). |
| Python floor | 3.12. |
| Distribution | PyPI + conda-forge. |
| License | MIT (IPASC default). |
| Scale | ~thousands of images per dataset; result caching on disk. |
| GPU | Framework auto-detects device; measures own their batching. |

---

## 3. Repository layout

```
ipasc_iqa/
├── pyproject.toml
├── README.md
├── LICENSE                       # MIT
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── ARCHITECTURE.md               # this document
│
├── src/
│   └── ipasc_iqa/
│       ├── __init__.py           # public API surface
│       ├── _version.py           # set by hatch-vcs
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── measure.py        # BaseMeasure, MeasureMetadata
│       │   ├── dataset.py        # Dataset, DatasetManifest, ImageRecord
│       │   ├── reference.py      # ReferenceType, ReferenceSpec
│       │   ├── roi.py            # ROI loading + hashing
│       │   ├── result.py         # ResultRecord, RunResults (pydantic models)
│       │   ├── registry.py       # entry-point discovery, MeasureRegistry
│       │   ├── runner.py         # Evaluation orchestration
│       │   ├── cache.py          # diskcache-backed result cache
│       │   └── device.py         # GPU/CPU detection helpers
│       │
│       ├── measures/
│       │   ├── __init__.py
│       │   ├── classical/
│       │   │   ├── psnr.py
│       │   │   ├── ssim.py
│       │   │   ├── cnr.py
│       │   │   ├── snr.py
│       │   │   └── ...
│       │   └── deep/
│       │       ├── lpips.py
│       │       └── ...
│       │
│       ├── datasets/
│       │   ├── __init__.py
│       │   └── local.py          # Downloader from online ressource.
│       │
│       ├── io/
│       │   ├── __init__.py
│       │   ├── npy.py
│       │   ├── manifest.py
│       │   └── results.py        # JSON schema validation + I/O
│       │
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py           # typer app
│       │   ├── run.py            # `ipasc_iqa run`
│       │   ├── list.py           # `ipasc_iqa list measures/datasets`
│       │   ├── verify.py         # `ipasc_iqa verify <dataset>`
│       │   └── compare.py        # `ipasc_iqa compare results/*.json`
│       │
│       ├── gui/
│       │   ├── __init__.py
│       │   ├── app.py            # Streamlit entrypoint
│       │   └── pages/
│       │       ├── 1_Run.py
│       │       ├── 2_Compare.py
│       │       └── 3_Datasets.py
│       │
│       └── schemas/
│           ├── results.schema.json
│           └── manifest.schema.json
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/                 # tiny synthetic datasets
│
├── docs/                         # mkdocs-material
│   ├── index.md
│   ├── tutorials/
│   ├── how-to/
│   ├── reference/                # auto-generated from docstrings
│   └── explanation/
│
└── .github/
    ├── workflows/
    │   ├── ci.yml                # pytest, ruff, mypy, build
    │   ├── docs.yml              # deploy mkdocs to gh-pages
    │   └── release.yml           # PyPI on tag; conda-forge via separate feedstock
    ├── ISSUE_TEMPLATE/
    └── PULL_REQUEST_TEMPLATE.md
```

The layout follows the **`src/` layout** (PEP 621-compatible, prevents accidental imports of in-repo code during 
testing) and the [Diátaxis](https://diataxis.fr/) documentation structure (tutorials / how-to / reference / 
explanation).

---

## 4. Data contract

### 4.1 Dataset on-disk layout

```
my_dataset/
├── manifest.json
├── images/
│   ├── img_0001.npy
│   ├── img_0001.signal_roi.npy
│   ├── img_0001.noise_roi.npy
│   ├── img_0001.ref.ground_truth.npy
│   ├── img_0001.ref.pseudo_reference.npy
│   ├── img_0002.npy
│   ├── img_0002.signal_roi.npy
│   └── ...
└── README.md
```

**Naming convention** (case-sensitive):

- Reconstruction: `<stem>.npy`
- Signal ROI mask (boolean or {0,1} array): `<stem>.signal_roi.npy`
- Noise ROI mask: `<stem>.noise_roi.npy`
- Typed reference: `<stem>.ref.<reference_type>.npy`

Any of the sibling files (ROIs, references) may be absent for a given image. The framework filters measures 
per image based on what is available and what each measure requires.

There is the possibility for several reconstructions to share the same reference.
In these cases, the images will be named as multiple variants with the qualifying unique filename after a ".var." separator:

```
my_dataset/
├── manifest.json
├── images/
│   ├── img_0001.var.algorithm1.npy
│   ├── img_0001.var.algorithm2.npy
│   ├── ...
│   ├── img_0001.signal_roi.npy
│   ├── img_0001.noise_roi.npy
│   ├── img_0001.ref.ground_truth.npy
│   ├── img_0001.ref.pseudo_reference.npy
│   ├── img_0002.npy
│   ├── img_0002.signal_roi.npy
│   └── ...
└── README.md
```

For the later benchmarking and GUI visualisations, each variant will be treated as an independent image.

### 4.2 `manifest.json`

```jsonc
{
  "schema_version": "1.0.0",
  "dataset_name": "phantom_study_A",
  "dataset_version": "2.1.0",
  "description": "Photoacoustic phantom study, agar 1%, 5 wavelengths.",
  "citation": "Doe et al. 2025, DOI:10.xxxx/yyyy",
  "license": "CC-BY-4.0",
  "created": "2026-05-26T10:00:00Z",
  "images": [
    {
      "id": "img_0001",
      "path": "images/img_0001.npy",
      "sha256": "abc123...",
      "shape": [512, 512],
      "dtype": "float32",
      "signal_roi": { "path": "images/img_0001.signal_roi.npy", "sha256": "..." },
      "noise_roi":  { "path": "images/img_0001.noise_roi.npy",  "sha256": "..." },
      "references": {
        "ground_truth":     { "path": "images/img_0001.ref.ground_truth.npy",     "sha256": "..." },
        "pseudo_reference": { "path": "images/img_0001.ref.pseudo_reference.npy", "sha256": "..." }
      },
      "metadata": { "wavelength_nm": 800, "subject_id": "phantom_03" }
    }
  ]
}
```

A `ipasc_iqa manifest generate <folder>` CLI command produces the manifest by scanning the folder and hashing 
files; users do not hand-write it. If there are multiple variants for the images, then a separate entry is created for each
and the `.var.` separator is ommitted from the `id`. 

### 4.3 Reference types

`reference_type` is a free-form string but the framework reserves three canonical values: `ground_truth`, 
`pseudo_reference`, `expert`. Measures declare `accepted_reference_types: list[str] | None` — `None` means 
"any reference type acceptable."

---

## 5. Core API

### 5.1 `BaseMeasure`

```python
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import torch  # optional dependency at use site only

@dataclass(frozen=True)
class MeasureMetadata:
    name: str                                # canonical, lowercase, hyphen-free
    version: str                             # semver of THIS measure impl
    measure_type: MeasureType                     # needs a reference image?
    requires_signal_roi: bool
    requires_noise_roi: bool
    accepted_reference_types: tuple[str, ...] | None = None  # None = any
    higher_is_better: bool = True
    value_range: tuple[float, float] | None = None
    citation: str | None = None
    description: str = ""

class BaseMeasure:
    metadata: MeasureMetadata  # subclass sets as class attribute

    def __init__(self, device: str = "auto", **kwargs): ...

    def evaluate(
        self,
        image: np.ndarray,
        *,
        reference: np.ndarray | None = None,
        reference_type: str | None = None,
        signal_roi: np.ndarray | None = None,
        noise_roi: np.ndarray | None = None,
    ) -> float | dict[str, float]:
        """Return a scalar score, or a dict of named sub-scores."""
        raise NotImplementedError

    def batch_evaluate(self, items: list[EvaluationInput]) -> list[float | dict]:
        """Optional override for batched/GPU evaluation. Default loops over evaluate()."""
        return [self.evaluate(**it.as_kwargs()) for it in items]
```

**Conventions**

- All arrays are NumPy on the public API. Measures that use PyTorch convert internally.
- `device="auto"` resolves to CUDA → MPS → CPU via `core.device.resolve()`.
- A measure returning a dict produces multiple result rows in the output JSON (e.g., SSIM might return `{"ssim": ..., "ms_ssim": ...}`).
- Subclasses **must** set a class-level `metadata` attribute. The registry validates this at discovery time.

### 5.2 Dataset API

```python
class Dataset:
    name: str
    version: str
    manifest_sha256: str  # hash of the canonicalized manifest

    def __iter__(self) -> Iterator[ImageRecord]: ...
    def __len__(self) -> int: ...
    def verify(self) -> VerificationReport: ...  # re-hash all files

@dataclass
class ImageRecord:
    id: str
    image: np.ndarray
    signal_roi: np.ndarray | None
    noise_roi: np.ndarray | None
    references: dict[str, np.ndarray]   # type → array
    metadata: dict[str, Any]
    hashes: ImageHashes                  # for cache keying
```

The dataset class is responsible for:
- downloading the data from a url to the local harddrive (if not done so before)
- converting the data layout from the format in the online ressource to the one dictated in the data contract.
- normalisation of the images

### 5.3 Runner

```python
def run_evaluation(
    datasets: list[Dataset],
    measures: list[BaseMeasure],
    *,
    reference_preference: list[str] | None = None,
    cache: ResultCache | None = None,
    output_path: Path,
    progress: bool = True,
) -> RunResults:
    ...
```

- `reference_preference` lets the user say "use `ground_truth` if available, else `pseudo_reference`" for full-reference measures.
- The runner skips (measure, image) pairs whose requirements aren't met and records the skip with a reason.
- The runner is the **single entry point used by both CLI and GUI**.

---

## 6. Results JSON schema

One run produces one JSON file. Schema is versioned and validated with `jsonschema` against `schemas/results.schema.json`.

```jsonc
{
  "schema_version": "1.0.0",
  "run_id": "2026-05-26T14-32-01_a1b2c3",
  "tool": { "name": "ipasc_iqa", "version": "0.3.1" },
  "environment": {
    "python": "3.12.4",
    "platform": "Linux-6.8-x86_64",
    "torch": "2.3.0",
    "cuda": "12.1",
    "git_commit": "abc123..."           // if invoked from a git checkout
  },
  "started": "2026-05-26T14:32:01Z",
  "finished": "2026-05-26T14:48:17Z",
  "config": {
    "reference_preference": ["ground_truth", "pseudo_reference"],
    "cache_enabled": true
  },
  "datasets": [
    {
      "name": "phantom_study_A",
      "version": "2.1.0",
      "manifest_sha256": "...",
      "n_images": 240
    }
  ],
  "measures": [
    {
      "name": "psnr",
      "version": "1.0.0",
      "plugin_distribution": "ipasc_iqa",   // or third-party package name
      "metadata": { "measure_type": "FULL_REFERENCE", "higher_is_better": true }
    }
  ],
  "results": [
    {
      "dataset": "phantom_study_A",
      "image_id": "img_0001",
      "image_sha256": "...",
      "measure": "psnr",
      "measure_version": "1.0.0",
      "reference_type": "ground_truth",
      "signal_roi_sha256": null,
      "noise_roi_sha256": null,
      "value": 28.41,
      "sub_scores": null,
      "wall_time_s": 0.012,
      "device": "cuda:0",
      "from_cache": false
    }
  ],
  "skipped": [
    {
      "dataset": "phantom_study_A",
      "image_id": "img_0015",
      "measure": "lpips",
      "reason": "no reference of accepted type available"
    }
  ]
}
```

Design notes:
- Flat `results` list (one row per `(dataset, image, measure, sub_score)` tuple) is trivial to load into pandas / Polars and pivot.
- All hashes recorded inline so a `compare` command can warn if two runs evaluated *nominally* the same image but the underlying file changed.
- `from_cache` lets you audit how much was actually recomputed.

---

## 7. Caching

`core.cache.ResultCache` is backed by [`diskcache`](https://grantjenks.com/docs/diskcache/) 
(pure Python, no server, transactional, well-tested).

**Cache key** (the precise composition is critical):

```
sha256(
    measure_name | measure_version |
    image_sha256 |
    reference_type or "" | reference_sha256 or "" |
    signal_roi_sha256 or "" | noise_roi_sha256 or "" |
    measure_config_canonical_json
)
```

`measure_config_canonical_json` is the JSON-serialized `__init__` kwargs of the measure instance 
(sorted keys, no whitespace). This means changing a measure's hyperparameter naturally invalidates its cache without 
the user having to think about it.

The cache lives at `$XDG_CACHE_HOME/ipasc_iqa/` (or platform equivalent via `platformdirs`), with a CLI command to 
inspect and clear it.

---

## 8. Plugin discovery

Third-party measures register via entry points:

```toml
# In a third-party package's pyproject.toml
[project.entry-points."ipasc_iqa.measures"]
my_custom_metric = "my_pkg.metrics:MyCustomMeasure"
```

At first use, `core.registry.MeasureRegistry` calls `importlib.metadata.entry_points(group="ipasc_iqa.measures")`, 
instantiates each class lazily, validates that it subclasses `BaseMeasure` and has well-formed `metadata`, 
and records the **distribution name + version** that provided it. That distribution info is written into the 
results JSON, so a result row is always attributable to a specific published package version.

A `[project.entry-points."ipasc_iqa.datasets"]` group is reserved but unused in v1 (datasets are pure data, not code).

---

## 9. CLI

Built with [`typer`](https://typer.tiangolo.com/) (gives us `--help`, completions, type-checked args for free). 
All commands map directly to functions in the public API; the CLI is glue, not logic.

```
ipasc_iqa list measures
ipasc_iqa list datasets [--root PATH]
ipasc_iqa verify <dataset_path>
ipasc_iqa manifest generate <folder> [--output manifest.json]
ipasc_iqa run \
    --dataset path/to/dataset_a \
    --dataset path/to/dataset_b \
    --measure psnr --measure ssim --measure lpips \
    --reference-preference ground_truth,pseudo_reference \
    --output results/run_2026-05-26.json \
    [--no-cache] [--device cuda]
ipasc_iqa compare results/run_a.json results/run_b.json [--output report.html]
ipasc_iqa cache info | clear
```

Subcommands return non-zero exit codes on failure for shell scripting; `--json` flag on listing commands emits 
machine-readable output.

---

## 10. GUI

Streamlit multi-page app, **importing `ipasc_iqa` as a library**, never shelling out to the CLI. Pages mirror the 
main CLI verbs:

- **Run** — select datasets (file picker / known-datasets registry), select measures (checkboxes from `registry.list_measures()`), set reference preference, launch and stream progress.
- **Compare** — upload two or more result JSONs, render per-measure tables and box plots, flag dataset/image hash mismatches.
- **Datasets** — verify integrity, view manifest, generate a manifest for a new folder.

Because the GUI calls the same Python functions as the CLI, deploying it as a web service later is a matter of running 
`streamlit run` behind a reverse proxy plus authentication — no rearchitecture.

---

## 11. Tooling and quality gates

| Concern | Tool |
|---|---|
| Build backend | `hatchling` + `hatch-vcs` (version from git tags) |
| Lint + format | `ruff` (replaces black, isort, flake8) |
| Type-check | `mypy --strict` on `src/`, relaxed on `tests/` |
| Tests | `pytest`, `pytest-cov` (target ≥85% on `core/`), `hypothesis` for property tests on hashing/ROI logic |
| Pre-commit | `pre-commit` config with ruff, mypy-light, end-of-file-fixer, check-toml |
| Docs | `mkdocs-material` + `mkdocstrings` for API reference |
| Schema validation | `jsonschema` for results + manifest |
| Versioning | semver; release notes in `CHANGELOG.md` (Keep a Changelog format) |
| CI | GitHub Actions: matrix over (3.12, 3.13) × (Linux, macOS, Windows) |

`pyproject.toml` defines optional dependency groups:

```toml
[project.optional-dependencies]
deep    = ["torch>=2.2", "torchvision>=0.17"]
gui     = ["streamlit>=1.32", "plotly>=5.20"]
dev     = ["pytest", "pytest-cov", "ruff", "mypy", "hypothesis", "pre-commit"]
docs    = ["mkdocs-material", "mkdocstrings[python]"]
all     = ["ipasc_iqa[deep,gui]"]
```

Base install is lightweight: `numpy`, `scipy`, `scikit-image`, `pydantic`, `typer`, `jsonschema`, `diskcache`, `platformdirs`, `tqdm`. PyTorch is opt-in via `pip install ipasc_iqa[deep]`.

---

## 12. Versioning and release

- **`ipasc_iqa` itself** uses semver. Breaking changes to the results JSON schema or the `BaseMeasure` interface bump the major version.
- **Results schema** has its own `schema_version` independent of tool version, so old results remain loadable.
- **Each measure** has its own `metadata.version` semver (changed when its algorithm changes, not when the wrapping package is rereleased for unrelated reasons). Cache invalidation hinges on this — emphasize it in the contributor guide.
- Releases: tag `vX.Y.Z` → GitHub Actions builds sdist + wheel → publishes to PyPI via [trusted publishing (OIDC)](https://docs.pypi.org/trusted-publishers/) (no API tokens in CI). conda-forge feedstock auto-PRs from PyPI via the `regro-cf-autotick-bot`.

---

## 13. Documentation plan

Four document categories (Diátaxis):

1. **Tutorials** — "Run your first benchmark in 5 minutes," "Write your first custom measure."
2. **How-to guides** — "Add a deep-learning measure with GPU batching," "Publish a third-party measure to PyPI," "Compare two runs."
3. **Reference** — auto-generated API docs via `mkdocstrings`, results JSON schema, manifest schema.
4. **Explanation** — architecture (this doc, trimmed), the caching key design, why the plugin model is entry-point-based.

Hosted on GitHub Pages via the docs workflow.

---

## 14. Roadmap

**v0.1 — MVP (core working end-to-end on one dataset)**
- `BaseMeasure`, `LocalDataset`, `manifest generate`, `run`, results JSON write + validate.
- Built-in measures: PSNR, SSIM, CNR, SNR.
- CLI only. No caching, no GUI, no DL measures.
- Tests + CI for Linux/Python 3.12.

**v0.2 — Reproducibility layer**
- Manifest verification, hashes in results, `compare` command (CLI text output).
- Result caching with `diskcache`.
- Property-based tests for cache keys.

**v0.3 — Plugins + DL** ✓
- Entry-point discovery, reference LPIPS measure as `ipasc_iqa[deep]`. ✓
- `device.resolve()` with CUDA/MPS/CPU. ✓
- Contributor guide for third-party measures. ✓

**v0.4 — GUI**
- Streamlit webapp with Run / Compare / Datasets pages.
- Cross-platform CI (macOS, Windows).
- conda-forge feedstock submitted.

**v1.0 — Stabilization**
- API freeze on `BaseMeasure`, `Dataset`, and results schema (v1.0.0).
- Full docs site live.
- At least one IPASC reference dataset published with a known manifest hash.

---

## 15. Agreed upon conventions:


1. **Multi-channel / multi-wavelength images.** We assume 3D arrays data arrays. Photoacoustic data is often multi-wavelength or images at multiple positions, each wavelength and position is to be treated as an individual image. Is is the task of the data converter to convert 2D images into a (N, M, 1) 3D shape.
2. **Numerical precision in cached hashes.** Float arrays should be hashed on their raw bytes (`array.tobytes()`) — but different dtypes of "the same" data will produce different hashes. Document this and recommend canonical `float32`.
3. **Result aggregation conventions.** When reporting "PSNR on dataset X," do we report mean ± std, median, or per-image? Defer to the `compare` command's design, but pick one default.
