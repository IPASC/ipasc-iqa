# Adding a dataset

Datasets are **adapter-driven**: only datasets that have a registered adapter class are offered in 
and GUI, and selecting one downloads and converts it on demand. An adapter is a small Python 
class that knows how to fetch a specific dataset and lay it out in the framework's on-disk **data
contract**. This keeps raw published data (any format) separate from the normalized form the
framework consumes.

This guide covers adding an adapter **inside this repository** and opening a pull request. A
third-party package can register an adapter the same way via an entry point (last section).

---

## 1. The data contract

A prepared dataset is a directory containing a `manifest.json` and an `images/` folder. Filenames
are case-sensitive:

```
my_dataset/
├── manifest.json
└── images/
    ├── img_0001.npy                       # reconstruction, float32, shape (H, W, 1)
    ├── img_0001.ref.ground_truth.npy      # a typed reference (optional)
    ├── img_0001.ref.pseudo_reference.npy  # another reference type (optional)
    ├── img_0001.signal_roi.npy            # boolean signal ROI mask (optional)
    ├── img_0001.noise_roi.npy             # boolean noise ROI mask (optional)
    └── ...
```

Rules:

- **Images are 3-D `float32` arrays of shape `(H, W, C)`.** Convert 2-D images to `(H, W, 1)`. Treat
  each wavelength / position as an independent image.
- **References** are siblings named `<stem>.ref.<reference_type>.npy`. The three canonical reference
  types are `ground_truth`, `pseudo_reference`, and `expert`, but the field is free-form.
- **ROI masks** are siblings named `<stem>.signal_roi.npy` / `<stem>.noise_roi.npy` (boolean).
- Any sibling may be absent — the framework filters measures per image based on what is available.
- **Variants** that share a reference/ROI use `<stem>.var.<variant>.npy` (e.g.
  `img_0001.var.algorithm1.npy`); each variant becomes an independent image with id `img_0001_algorithm1`.

`manifest.json` records, per image, the relative path, a SHA-256 hash, shape, dtype, and the paths +
hashes of its references/ROIs. You normally **generate it programmatically** (see below) rather than
writing it by hand.

### Expert annotations (optional, enables correlation & standings)

If your dataset has expert quality ratings, embed them into each image entry's `metadata` so the
**Detailed analysis** and **Standings** pages can correlate measures against them:

```jsonc
"metadata": {
  "algorithm": "alg0",
  "annotations": {
    "overall_quality": { "mos": 2.8, "zscore_mos": 0.149 },
    "edge_visibility": { "mos": 3.0, "zscore_mos": 0.144 }
  }
}
```

When individual annotator ratings are available, they are z-normalised per annotator and combined into 
a composite MOS. Otherwise, correlation uses the pre-computed zscore_mos. Different datasets may expose 
different task names — all of them are shown automatically.

---

## 2. Write the adapter

Subclass [`LocalDataset`](../../src/ipasc_iqa/datasets/local.py), declare the registry metadata, and
override `_prepare()` to download + convert the source data into the contract under `self.root`
(the framework passes a managed per-dataset directory). Honor `self._download`: only fetch from the
network when it is `True`.

```python
# src/ipasc_iqa/datasets/my_dataset.py
"""Adapter for the Example PA dataset."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np

from ipasc_iqa.datasets.local import LocalDataset
from ipasc_iqa.io.manifest import generate_manifest

_SOURCE_URL = "https://example.org/example_pa.zip"


class ExampleDataset(LocalDataset):
    DATASET_NAME = "example_pa"
    DATASET_DISPLAY_NAME = "Example PA"
    DATASET_VERSION = "1.0.0"
    DATASET_DESCRIPTION = "Example paired reconstruction/reference photoacoustic dataset."
    DATASET_BIBTEX = "..."

    def _prepare(self) -> None:
        # self.root is the managed directory for this dataset; self._download is set
        # by LocalDataset.__init__. Skip work if already prepared.
        if (self.root / "manifest.json").exists():
            return

        if self._download:
            self._download_sources()

        self._convert_to_contract()

    # -- helpers -----------------------------------------------------------

    def _download_sources(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        archive = self.root / "source.zip"
        if not archive.exists():
            urllib.request.urlretrieve(_SOURCE_URL, archive)  # noqa: S310
        import shutil
        shutil.unpack_archive(str(archive), str(self.root / "raw"))

    def _convert_to_contract(self) -> None:
        images_out = self.root / "images"
        images_out.mkdir(parents=True, exist_ok=True)
        raw = self.root / "raw"

        # Example: each sample is a recon PNG + a reference PNG. Convert to the
        # contract: float32 (H, W, 1) arrays, reference as a typed sibling.
        from PIL import Image

        for recon_png in sorted(raw.glob("*_recon.png")):
            stem = recon_png.name.replace("_recon.png", "")
            recon = np.asarray(Image.open(recon_png).convert("L"), np.float32)[:, :, None] / 255.0
            ref_png = raw / f"{stem}_reference.png"
            ref = np.asarray(Image.open(ref_png).convert("L"), np.float32)[:, :, None] / 255.0

            np.save(images_out / f"{stem}.npy", recon)
            np.save(images_out / f"{stem}.ref.ground_truth.npy", ref)

        # Scan images/ and hash everything into a manifest.json at self.root.
        generate_manifest(
            self.root,
            dataset_name=self.DATASET_NAME,
            dataset_version=self.DATASET_VERSION,
            description=self.DATASET_DESCRIPTION,
        )
```

`generate_manifest()` discovers the reconstruction/reference/ROI siblings by naming convention and
writes a hashed `manifest.json`. If you need to embed expert annotations (section 1), build the
manifest dict yourself instead and add the `metadata.annotations` block — see
[`PhotIQADataset._convert`](../../src/ipasc_iqa/datasets/phot_iqa.py) for a complete worked example
that also handles `.var.` variants and per-image annotations.

### Keeping raw and converted data separate

If you prefer to keep downloaded data and the converted contract in different folders (as PhotIQA
does with a `converted/` sub-directory), write the manifest into that sub-folder, re-point
`self.root` to it at the end of `_prepare()`, and override the classmethod so the registry knows
where the prepared manifest lives:

```python
@classmethod
def prepared_manifest_path(cls, root):
    from pathlib import Path
    return Path(root) / "converted" / "manifest.json"
```

---

## 3. Register the adapter

Add it to the built-in adapters in
[`src/ipasc_iqa/datasets/registry.py`](../../src/ipasc_iqa/datasets/registry.py), inside
`DatasetRegistry._load`:

```python
def _load(self) -> None:
    if self._loaded:
        return
    from ipasc_iqa.datasets.phot_iqa import PhotIQADataset
    from ipasc_iqa.datasets.my_dataset import ExampleDataset   # <- import

    self.register(PhotIQADataset, distribution="ipasc_iqa")
    self.register(ExampleDataset, distribution="ipasc_iqa")    # <- register
    ...
```

Verify discovery:

```bash
pip install -e ".[dev,gui]"
ipasc_iqa-gui --workspace . --data-dir .data    # 'example_pa' should appear on the Run page
```

---

## 4. Add a test

You generally don't want a test to hit the network. Test the adapter against a tiny **pre-staged**
copy of the contract (no download), the way the GUI tests do — see
[`tests/integration/test_gui.py`](../../tests/integration/test_gui.py), which writes a minimal
manifest + `.npy` files into a temp dir and registers a test adapter. A good adapter test:

- writes a handful of fake source files into a temp dir,
- runs `_convert_to_contract()` (or instantiates with `download=False` over pre-staged data), and
- asserts the resulting `manifest.json` and arrays satisfy the contract (shape `(H, W, 1)`,
  float32, references present).

---

## 5. Open the pull request

```bash
git checkout -b add-example-dataset
ruff check .
pytest
git add -A
git commit -m "Add Example PA dataset adapter"
git push -u origin add-example-dataset
gh pr create --fill
```

In the PR, include the dataset's source URL, license, citation, and a note on which reference types,
ROIs, and annotation tasks it provides.

---

## Publishing from your own package (no PR needed)

Register an adapter from any installed distribution via the reserved entry-point group:

```toml
# in your_package/pyproject.toml
[project.entry-points."ipasc_iqa.datasets"]
example_pa = "your_package.datasets:ExampleDataset"
```

The class must subclass `LocalDataset` and declare `DATASET_NAME` / `DATASET_VERSION` /
`DATASET_DESCRIPTION`. Once installed alongside `ipasc_iqa`, it is offered automatically.
