

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
from ipasc_iqa.core.measure import MeasureType

@dataclass(frozen=True)
class MeasureMetadata:
    name: str                                # canonical, lowercase, no spaces
    version: str                             # semver of THIS measure's algorithm
    measure_type: MeasureType                     # does it need a reference image?
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

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


class MAEMeasure(BaseMeasure):
  metadata = MeasureMetadata(
    name="mae",
    version="1.0.0",
    measure_type=MeasureType.FULL_REFERENCE,
    requires_signal_roi=False,
    requires_noise_roi=False,
    accepted_reference_types=None,  # accepts any reference type
    higher_is_better=False,  # lower error is better
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

### Deep-learning (PyTorch) measures

Put the file under `measures/deep/`. The bundled
[LPIPS](../../src/ipasc_iqa/measures/deep/lpips.py) and
[DISTS](../../src/ipasc_iqa/measures/deep/dists.py) adapters are the canonical
references; their design decisions are worth following exactly:

#### Defer all heavy imports to first use

Do **not** import `torch` or a deep-learning library at module top level. Import inside the method
that needs them. This lets `from ipasc_iqa.measures.deep.lpips import LPIPSMeasure` succeed even
when `ipasc_iqa[deep]` is not installed, and means registry discovery never triggers a slow or
failing import.

```python
# wrong — breaks base install
import torch
import lpips

class LPIPSMeasure(BaseMeasure):
    ...

# correct
class LPIPSMeasure(BaseMeasure):
    def _get_model(self):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("Requires ipasc_iqa[deep]") from exc
        import torch
        ...
```

#### Lazy model initialisation

Pretrained deep models download weights (tens to hundreds of MB) on first construction.
Loading in `__init__` would trigger a download for every `ipasc_iqa list measures` call.
Use a `None` sentinel and load on the first `evaluate()` call instead:

```python
def __init__(self, device: str = "auto", net: str = "alex") -> None:
    super().__init__(device=device, net=net)  # pass hyperparams as kwargs
    self._net = net
    self._model: Any = None  # loaded lazily

def _get_model(self) -> Any:
    if self._model is None:
        import lpips, torch
        self._model = lpips.LPIPS(net=self._net, verbose=False)
        self._model = self._model.to(torch.device(self.device)).eval()
    return self._model
```

#### Pass all hyperparameters as kwargs to `super().__init__()`

`BaseMeasure.__init__` captures `**kwargs` into `self._config`, which is serialised into
`config_canonical_json()` — the component of the cache key that changes when a measure is
reconfigured.  If you forget this, changing the backbone (e.g. `net="vgg"`) reuses stale
cached results silently.

```python
super().__init__(device=device, net=net)  # net ends up in the cache key
```

#### Device handling

`BaseMeasure.__init__` resolves `self.device` via `core.device.resolve()` (CUDA → MPS → CPU,
never assumes CUDA is available). Move the model to this device at lazy-init time, and move
tensors to the same device when building them:

```python
self._model = model.to(torch.device(self.device)).eval()
# ...
tensor = tensor.to(torch.device(self.device))
```

#### Single-channel to 3-channel expansion

Photoacoustic images are single-channel `(H, W, 1)`. Networks pretrained on RGB images expect
3 channels. Use `.expand()` rather than `.repeat()` — it creates a zero-copy view:

```python
if t.shape[0] == 1:
    t = t.expand(3, -1, -1)   # view, no data copy
    # not: t = t.repeat(3, 1, 1)  # allocates 3× the memory
```

#### Joint normalisation

Normalise image and reference together using the **reference's** data range. This preserves
intensity differences (a systematically brighter reconstruction is penalised), unlike
per-image normalisation which would wash out those differences:

```python
data_min   = float(reference.min())
data_range = float(reference.max()) - data_min or 1.0
# apply the same transform to both
img_t = (img_t - data_min) / data_range   # result in [0, 1] for ref
ref_t = (ref_t - data_min) / data_range   # values outside range are clamped
```

Use the library's built-in rescaling flag when available (`normalize=True` in the lpips
package maps `[0, 1] → [−1, 1]`) rather than doing it manually.

#### Always use `torch.no_grad()`

```python
with torch.no_grad():
    score = model(img_t, ref_t, normalize=True)
```

#### Functional measures need no lazy model — but still defer imports

Some `piq` measures (MS-SSIM, IW-SSIM, BRISQUE, …) are pure functions with no
pretrained weights to download.  You still **must** defer the `piq` / `torch`
imports to first use so that the base install (which omits `torch`) can import
your class without error, but you do **not** need the `_model = None` sentinel or
a `_get_model()` method.  Keep the import inside `evaluate()`:

```python
class MSSSIMMeasure(BaseMeasure):
    # No __init__ override needed unless you expose hyperparameters.
    # If you do, pass them to super() for the cache key:
    def __init__(self, kernel_size: int = 11, kernel_sigma: float = 1.5) -> None:
        super().__init__(kernel_size=kernel_size, kernel_sigma=kernel_sigma)
        self._kernel_size = kernel_size

    def evaluate(self, image, *, reference=None, **_):
        try:
            from piq import multi_scale_ssim
        except ImportError as exc:
            raise ImportError("Requires ipasc_iqa[deep]") from exc
        import torch
        # ... build tensors, call function, return float
```

#### Adaptive spatial parameters for multi-scale measures

MS-SSIM processes the image at multiple resolutions, halving spatial dimensions
at each scale.  The Gaussian kernel (`kernel_size × kernel_size`) must still fit
at the coarsest scale.  For small images (e.g. test fixtures at 64×64) the
default 5-scale pyramid may not fit — detect this and clip the number of scales:

```python
min_dim = min(image.shape[0], image.shape[1])
# At scale s (1-indexed from finest) the dim is min_dim / 2^(s-1)
n_scales = min(5, max(1, int(np.floor(np.log2(min_dim / kernel_size))) + 1))
weights = default_weights[:n_scales]
weights = weights / weights.sum()   # renormalise
```

Without this guard, evaluating on small test images raises a `RuntimeError`
inside piq.

---

## 3. Register the entry point

Add one line to the `[project.entry-points."ipasc_iqa.measures"]` table in `pyproject.toml`. The key
is the entry-point name; the value is `module.path:ClassName`:

```toml
[project.entry-points."ipasc_iqa.measures"]
psnr  = "ipasc_iqa.measures.classical.psnr:PSNRMeasure"
ssim  = "ipasc_iqa.measures.classical.ssim:SSIMMeasure"
cnr   = "ipasc_iqa.measures.classical.cnr:CNRMeasure"
snr   = "ipasc_iqa.measures.classical.snr:SNRMeasure"
mae   = "ipasc_iqa.measures.classical.mae:MAEMeasure"
lpips = "ipasc_iqa.measures.deep.lpips:LPIPSMeasure"   # <- new deep measure
dists = "ipasc_iqa.measures.deep.dists:DISTSMeasure"   # <- new deep measure
```

For deep measures, also add the extra dependency to the `deep` group:

```toml
[project.optional-dependencies]
deep = ["torch>=2.2", "torchvision>=0.17", "lpips>=0.1.4", "DISTS-pytorch>=0.1"]
```

Re-install so the entry point is picked up, and confirm it is discovered:

```bash
pip install -e ".[dev,gui,deep]"
ipasc_iqa list measures        # 'lpips' should appear in the table
```

---

## 4. Add a test

Add a unit test under `tests/unit/` mirroring `tests/unit/test_measures.py`.  For deep measures,
guard the entire test class with `pytest.importorskip` so the suite stays green without
`ipasc_iqa[deep]`:

```python
import numpy as np
import pytest

pytest.importorskip("torch",  reason="requires ipasc_iqa[deep]")
pytest.importorskip("lpips",  reason="requires ipasc_iqa[deep]")

from ipasc_iqa.measures.deep.lpips import LPIPSMeasure

rng = np.random.default_rng(42)
IMG = rng.random((32, 32, 1), dtype=np.float32).astype(np.float32)
REF = rng.random((32, 32, 1), dtype=np.float32).astype(np.float32)

class TestLPIPS:
    def test_returns_non_negative_float(self):
        v = LPIPSMeasure(device="cpu").evaluate(IMG, reference=REF)
        assert isinstance(v, float) and v >= 0.0

    def test_identical_images_near_zero(self):
        assert LPIPSMeasure(device="cpu").evaluate(IMG, reference=IMG) < 0.01

    def test_requires_reference(self):
        with pytest.raises(ValueError):
            LPIPSMeasure(device="cpu").evaluate(IMG)
```

`pytest.importorskip` at module level causes pytest to **skip** (not fail) the whole module
when the packages are absent, keeping CI green on the base install.

---

## 5. Open the pull request

```bash
git checkout -b add-lpips-measure
ruff check .          # lint must pass
pytest                # tests must pass
git add -A
git commit -m "Add LPIPS deep measure"
git push -u origin add-lpips-measure
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
