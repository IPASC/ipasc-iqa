# Writing a custom IQA measure

IQA measures are the one pluggable axis of `ipasc_iqa`. A measure is a Python class that subclasses
`BaseMeasure`, declares its requirements via a `MeasureMetadata`, and implements `evaluate()`. It is
discovered at runtime through a Python **entry point**, so it can live in this repository *or* in
your own separate package — the framework records which distribution provided each measure, in every
results file.

This guide covers adding a measure **inside this repository** and opening a pull request. Publishing
a measure from a *third-party* package is identical except you put the entry point in your own
`pyproject.toml` (see the last section).

---

## 1. The interface

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class MeasureMetadata:
    name: str                                # canonical, lowercase, no spaces
    version: str                             # semver of THIS measure's algorithm
    full_reference: bool                     # does it need a reference image?
    requires_signal_roi: bool                # does it need a signal ROI mask?
    requires_noise_roi: bool                 # does it need a noise ROI mask?
    accepted_reference_types: tuple[str, ...] | None = None  # None = any
    higher_is_better: bool = True
    value_range: tuple[float, float] | None = None
    citation: str | None = None
    description: str = ""
```

`evaluate()` receives **NumPy arrays** of shape `(H, W, C)` (float32 by convention) and returns a
single `float`, or a `dict[str, float]` of named sub-scores:

```python
def evaluate(
    self,
    image: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    reference_type: str | None = None,
    signal_roi: np.ndarray | None = None,
    noise_roi: np.ndarray | None = None,
) -> float | dict[str, float]:
    ...
```

The runner only calls your measure when its declared requirements are met (it skips images that lack
a required reference or ROI and records the reason), so inside `evaluate()` you can assume the inputs
your metadata asked for are present.

`metadata.version` is important: it is part of the result-cache key, so **bump it whenever the
algorithm changes** — that automatically invalidates stale cached values.

---

## 2. Write the measure

Create a file under `src/ipasc_iqa/measures/classical/` (or `measures/deep/` for PyTorch-based
ones). Here is a complete, minimal example — a mean-absolute-error measure:

```python
# src/ipasc_iqa/measures/classical/mae.py
"""Mean Absolute Error."""

from __future__ import annotations

import numpy as np

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata


class MAEMeasure(BaseMeasure):
    metadata = MeasureMetadata(
        name="mae",
        version="1.0.0",
        full_reference=True,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,   # accepts any reference type
        higher_is_better=False,          # lower error is better
        value_range=(0.0, float("inf")),
        description="Mean absolute error between image and reference.",
    )

    def evaluate(
        self,
        image: np.ndarray,
        *,
        reference: np.ndarray | None = None,
        reference_type: str | None = None,
        signal_roi: np.ndarray | None = None,
        noise_roi: np.ndarray | None = None,
    ) -> float:
        if reference is None:
            raise ValueError("MAE requires a reference image.")
        return float(np.mean(np.abs(image.astype(np.float64) - reference.astype(np.float64))))
```

For a real-world template, read the bundled
[`PSNRMeasure`](../../src/ipasc_iqa/measures/classical/psnr.py) and
[`CNRMeasure`](../../src/ipasc_iqa/measures/classical/cnr.py) (the latter shows ROI usage).

### Notes for GPU / deep-learning measures

- Put the file under `measures/deep/` and import `torch` *inside* `__init__`/`evaluate`, not at
  module top level, so the base install stays lightweight.
- `BaseMeasure.__init__` resolves `self.device` (CUDA → MPS → CPU). Convert NumPy → tensor inside
  `evaluate()` and move to `self.device`.
- Override `batch_evaluate()` if you want true batched GPU evaluation.
- Add the dependency to the `deep` optional-dependency group in `pyproject.toml`.

---

## 3. Register the entry point

Add one line to the `[project.entry-points."ipasc_iqa.measures"]` table in `pyproject.toml`. The key
is the entry-point name; the value is `module.path:ClassName`:

```toml
[project.entry-points."ipasc_iqa.measures"]
psnr = "ipasc_iqa.measures.classical.psnr:PSNRMeasure"
ssim = "ipasc_iqa.measures.classical.ssim:SSIMMeasure"
cnr  = "ipasc_iqa.measures.classical.cnr:CNRMeasure"
snr  = "ipasc_iqa.measures.classical.snr:SNRMeasure"
mae  = "ipasc_iqa.measures.classical.mae:MAEMeasure"   # <- new
```

Re-install so the entry point is picked up, and confirm it is discovered:

```bash
pip install -e ".[dev,gui]"
ipasc_iqa list measures        # 'mae' should appear in the table
```

---

## 4. Add a test

Add a unit test under `tests/unit/` mirroring `tests/unit/test_measures.py`:

```python
import numpy as np
from ipasc_iqa.measures.classical.mae import MAEMeasure

def test_mae_zero_for_identical_images():
    img = np.random.default_rng(0).random((32, 32, 1), dtype=np.float32)
    assert MAEMeasure().evaluate(img, reference=img) == 0.0
```

---

## 5. Open the pull request

```bash
git checkout -b add-mae-measure
ruff check .          # lint must pass
pytest                # tests must pass
git add -A
git commit -m "Add MAE measure"
git push -u origin add-mae-measure
gh pr create --fill   # or open the PR on GitHub
```

In the PR description, include: what the measure computes, a citation if applicable, and its
`value_range` / `higher_is_better` semantics.

---

## Publishing from your own package (no PR needed)

You don't have to contribute the measure here. Any installed distribution can register measures by
declaring the same entry-point group in *its* `pyproject.toml`:

```toml
# in your_package/pyproject.toml
[project.entry-points."ipasc_iqa.measures"]
my_metric = "your_package.metrics:MyMeasure"
```

Once your package is installed alongside `ipasc_iqa`, `ipasc_iqa list measures` shows it, and result
files record your distribution's name and version next to every value it produced.
