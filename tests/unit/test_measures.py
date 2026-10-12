"""Unit tests for built-in measures."""

from __future__ import annotations

import numpy as np
import pytest

from ipasc_iqa.measures.classical.cnr import CNRMeasure
from ipasc_iqa.measures.classical.mae import MAEMeasure
from ipasc_iqa.measures.classical.psnr import PSNRMeasure
from ipasc_iqa.measures.classical.snr import SNRMeasure
from ipasc_iqa.measures.classical.ssim import SSIMMeasure
from ipasc_iqa.measures.classical.tv import TVMeasure

rng = np.random.default_rng(0)
IMAGE = rng.random((256, 256, 1), dtype=np.float32).astype(np.float32)
REFERENCE = rng.random((256, 256, 1), dtype=np.float32).astype(np.float32)
SIGNAL_ROI = (IMAGE[:, :, 0] > 0.7)[:, :, np.newaxis].astype(bool)
NOISE_ROI = (IMAGE[:, :, 0] < 0.2)[:, :, np.newaxis].astype(bool)


class TestMAE:
    def test_uses_mean_absolute_error(self) -> None:
        image = np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float32)
        reference = np.zeros_like(image)

        # Mean absolute error is 1.0; the previous median implementation
        # returned 0.0 for this deliberately asymmetric example.
        assert MAEMeasure().evaluate(image, reference=reference) == 1.0

    def test_identical_images_is_zero(self) -> None:
        assert MAEMeasure().evaluate(IMAGE, reference=IMAGE) == 0.0

    def test_requires_reference(self) -> None:
        with pytest.raises(ValueError, match="requires a reference"):
            MAEMeasure().evaluate(IMAGE)


class TestPSNR:
    def test_returns_positive_float(self) -> None:
        m = PSNRMeasure()
        v = m.evaluate(IMAGE, reference=REFERENCE)
        assert isinstance(v, float)
        assert v > 0

    def test_identical_images_is_inf(self) -> None:
        m = PSNRMeasure()
        v = m.evaluate(IMAGE, reference=IMAGE)
        assert v == float("inf")

    def test_requires_reference(self) -> None:
        m = PSNRMeasure()
        with pytest.raises(ValueError):
            m.evaluate(IMAGE)


class TestSSIM:
    def test_returns_float_in_range(self) -> None:
        m = SSIMMeasure()
        v = m.evaluate(IMAGE, reference=REFERENCE)
        assert isinstance(v, float)
        assert -1.0 <= v <= 1.0

    def test_identical_images_is_one(self) -> None:
        m = SSIMMeasure()
        v = m.evaluate(IMAGE, reference=IMAGE)
        assert abs(v - 1.0) < 1e-5


class TestCNR:
    def test_returns_non_negative(self) -> None:
        m = CNRMeasure()
        v = m.evaluate(IMAGE, signal_roi=SIGNAL_ROI, noise_roi=NOISE_ROI)
        assert isinstance(v, float)
        assert v >= 0

    def test_requires_rois(self) -> None:
        m = CNRMeasure()
        with pytest.raises(ValueError):
            m.evaluate(IMAGE)


class TestSNR:
    def test_returns_float(self) -> None:
        m = SNRMeasure()
        v = m.evaluate(IMAGE, signal_roi=SIGNAL_ROI, noise_roi=NOISE_ROI)
        assert isinstance(v, float)

    def test_requires_rois(self) -> None:
        m = SNRMeasure()
        with pytest.raises(ValueError):
            m.evaluate(IMAGE)


class TestTV:
    def test_returns_non_negative_float(self) -> None:
        v = TVMeasure().evaluate(IMAGE)
        assert isinstance(v, float)
        assert v >= 0.0

    def test_uniform_image_is_zero(self) -> None:
        uniform = np.ones((32, 32, 1), dtype=np.float32) * 0.5
        assert TVMeasure().evaluate(uniform) == 0.0

    def test_noisy_image_higher_than_smooth(self) -> None:
        rng_tv = np.random.default_rng(99)
        smooth = np.ones((64, 64, 1), dtype=np.float32) * 0.5
        noisy = rng_tv.random((64, 64, 1)).astype(np.float32)
        assert TVMeasure().evaluate(noisy) > TVMeasure().evaluate(smooth)

    def test_ignores_reference(self) -> None:
        # TV is no-reference — passing a reference should not affect the score.
        m = TVMeasure()
        assert m.evaluate(IMAGE) == m.evaluate(IMAGE, reference=REFERENCE)


# ---------------------------------------------------------------------------
# LPIPS — skipped when PyTorch-based metric dependencies are unavailable
# ---------------------------------------------------------------------------

pytest.importorskip("torch", reason="requires PyTorch-based metric dependencies")
pytest.importorskip("lpips", reason="requires PyTorch-based metric dependencies")

from ipasc_iqa.measures.deep.lpips import LPIPSMeasure  # noqa: E402

_rng = np.random.default_rng(42)
_IMG_3C = _rng.random((32, 32, 3)).astype(np.float32)
_REF_3C = _rng.random((32, 32, 3)).astype(np.float32)


class TestLPIPS:
    def test_returns_non_negative_float(self) -> None:
        v = LPIPSMeasure(device="cpu").evaluate(IMAGE, reference=REFERENCE)
        assert isinstance(v, float)
        assert v >= 0.0

    def test_identical_images_near_zero(self) -> None:
        v = LPIPSMeasure(device="cpu").evaluate(IMAGE, reference=IMAGE)
        assert v < 0.01

    def test_requires_reference(self) -> None:
        with pytest.raises(ValueError):
            LPIPSMeasure(device="cpu").evaluate(IMAGE)

    def test_three_channel_input(self) -> None:
        v = LPIPSMeasure(device="cpu").evaluate(_IMG_3C, reference=_REF_3C)
        assert isinstance(v, float)
        assert v >= 0.0

    def test_model_reused_across_calls(self) -> None:
        m = LPIPSMeasure(device="cpu")
        assert m._model is None
        m.evaluate(IMAGE, reference=REFERENCE)
        model_ref = m._model
        m.evaluate(IMAGE, reference=REFERENCE)
        assert m._model is model_ref  # same object, not reloaded


# ---------------------------------------------------------------------------
# DISTS — skipped when PyTorch-based metric dependencies are unavailable
# ---------------------------------------------------------------------------

pytest.importorskip("torch", reason="requires PyTorch-based metric dependencies")
pytest.importorskip("DISTS_pytorch", reason="requires PyTorch-based metric dependencies")

from ipasc_iqa.measures.deep.dists import DISTSMeasure  # noqa: E402


class TestDISTS:
    def test_returns_non_negative_float(self) -> None:
        v = DISTSMeasure(device="cpu").evaluate(IMAGE, reference=REFERENCE)
        assert isinstance(v, float)
        assert v >= 0.0

    def test_identical_images_near_zero(self) -> None:
        v = DISTSMeasure(device="cpu").evaluate(IMAGE, reference=IMAGE)
        assert v < 0.01

    def test_requires_reference(self) -> None:
        with pytest.raises(ValueError):
            DISTSMeasure(device="cpu").evaluate(IMAGE)

    def test_three_channel_input(self) -> None:
        v = DISTSMeasure(device="cpu").evaluate(_IMG_3C, reference=_REF_3C)
        assert isinstance(v, float)
        assert v >= 0.0

    def test_model_reused_across_calls(self) -> None:
        m = DISTSMeasure(device="cpu")
        assert m._model is None
        m.evaluate(IMAGE, reference=REFERENCE)
        model_ref = m._model
        m.evaluate(IMAGE, reference=REFERENCE)
        assert m._model is model_ref  # same object, not reloaded


# ---------------------------------------------------------------------------
# MS-SSIM — skipped when PyTorch-based metric dependencies are unavailable
# ---------------------------------------------------------------------------

pytest.importorskip("piq", reason="requires PyTorch-based metric dependencies")

from ipasc_iqa.measures.deep.msssim import MSSSIMMeasure  # noqa: E402

# Use a larger image so the adaptive scale logic gets to exercise ≥2 scales.
_rng2 = np.random.default_rng(7)
_IMG_128 = _rng2.random((256, 256, 1)).astype(np.float32)
_REF_128 = _rng2.random((256, 256, 1)).astype(np.float32)


class TestMSSSIM:
    def test_returns_float_in_range(self) -> None:
        v = MSSSIMMeasure().evaluate(IMAGE, reference=REFERENCE)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

    def test_identical_images_near_one(self) -> None:
        v = MSSSIMMeasure().evaluate(IMAGE, reference=IMAGE)
        assert abs(v - 1.0) < 1e-4

    def test_requires_reference(self) -> None:
        with pytest.raises(ValueError):
            MSSSIMMeasure().evaluate(IMAGE)

    def test_three_channel_input(self) -> None:
        v = MSSSIMMeasure().evaluate(_IMG_3C, reference=_REF_3C)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

    def test_adaptive_scales_small_image(self) -> None:
        # 64×64 can support at most 3 of the 5 default scales with kernel=11.
        m = MSSSIMMeasure()
        weights = m._adaptive_scale_weights(64)
        assert len(weights) == 3
        assert abs(float(weights.sum().item()) - 1.0) < 1e-5

    def test_adaptive_scales_large_image(self) -> None:
        # 256×256 supports all 5 scales.
        m = MSSSIMMeasure()
        weights = m._adaptive_scale_weights(256)
        assert len(weights) == 5

    def test_larger_image_runs(self) -> None:
        v = MSSSIMMeasure().evaluate(_IMG_128, reference=_REF_128)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# IW-SSIM — skipped when PyTorch-based metric dependencies are unavailable
# ---------------------------------------------------------------------------

pytest.importorskip("piq", reason="requires PyTorch-based metric dependencies")

from ipasc_iqa.measures.deep.iwssim import IWSSIMMeasure  # noqa: E402


class TestIWSSIM:
    def test_returns_float_in_range(self) -> None:
        v = IWSSIMMeasure().evaluate(_IMG_128, reference=_REF_128)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

    def test_identical_images_near_one(self) -> None:
        v = IWSSIMMeasure().evaluate(_IMG_128, reference=_IMG_128)
        assert abs(v - 1.0) < 1e-4

    def test_requires_reference(self) -> None:
        with pytest.raises(ValueError):
            IWSSIMMeasure().evaluate(_IMG_128)

    def test_three_channel_input(self) -> None:
        img_3c = _rng2.random((256, 256, 3)).astype(np.float32)
        ref_3c = _rng2.random((256, 256, 3)).astype(np.float32)
        v = IWSSIMMeasure().evaluate(img_3c, reference=ref_3c)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# VIF — skipped when PyTorch-based metric dependencies are unavailable
# ---------------------------------------------------------------------------

pytest.importorskip("piq", reason="requires PyTorch-based metric dependencies")

from ipasc_iqa.measures.deep.vif import VIFMeasure  # noqa: E402

# VIF uses a 4-scale Gaussian pyramid; recommended minimum is ~41×41.
_rng3 = np.random.default_rng(13)
_IMG_64 = _rng3.random((64, 64, 1)).astype(np.float32)
_REF_64 = _rng3.random((64, 64, 1)).astype(np.float32)


class TestVIF:
    def test_returns_non_negative_float(self) -> None:
        v = VIFMeasure().evaluate(_IMG_64, reference=_REF_64)
        assert isinstance(v, float)
        assert v >= 0.0

    def test_identical_images_near_one(self) -> None:
        v = VIFMeasure().evaluate(_IMG_128, reference=_IMG_128)
        assert abs(v - 1.0) < 0.01

    def test_requires_reference(self) -> None:
        with pytest.raises(ValueError):
            VIFMeasure().evaluate(_IMG_64)


# ---------------------------------------------------------------------------
# FSIM — skipped when PyTorch-based metric dependencies are unavailable
# ---------------------------------------------------------------------------

pytest.importorskip("piq", reason="requires PyTorch-based metric dependencies")

from ipasc_iqa.measures.deep.fsim import FSIMMeasure  # noqa: E402


class TestFSIM:
    def test_returns_float_in_range(self) -> None:
        v = FSIMMeasure().evaluate(IMAGE, reference=REFERENCE)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

    def test_identical_images_near_one(self) -> None:
        v = FSIMMeasure().evaluate(IMAGE, reference=IMAGE)
        assert abs(v - 1.0) < 1e-4

    def test_requires_reference(self) -> None:
        with pytest.raises(ValueError):
            FSIMMeasure().evaluate(IMAGE)

    def test_chromatic_kwarg_in_cache_key(self) -> None:
        # Different chromatic values → different config → different cache keys
        assert FSIMMeasure(chromatic=False)._config != FSIMMeasure(chromatic=True)._config


# ---------------------------------------------------------------------------
# GMSD — skipped when PyTorch-based metric dependencies are unavailable
# ---------------------------------------------------------------------------

pytest.importorskip("piq", reason="requires PyTorch-based metric dependencies")

from ipasc_iqa.measures.deep.gmsd import GMSDMeasure  # noqa: E402


class TestGMSD:
    def test_returns_non_negative_float(self) -> None:
        v = GMSDMeasure().evaluate(IMAGE, reference=REFERENCE)
        assert isinstance(v, float)
        assert v >= 0.0

    def test_identical_images_near_zero(self) -> None:
        v = GMSDMeasure().evaluate(IMAGE, reference=IMAGE)
        assert v < 1e-4

    def test_requires_reference(self) -> None:
        with pytest.raises(ValueError):
            GMSDMeasure().evaluate(IMAGE)


# ---------------------------------------------------------------------------
# MS-GMSD — skipped when PyTorch-based metric dependencies are unavailable
# ---------------------------------------------------------------------------

pytest.importorskip("piq", reason="requires PyTorch-based metric dependencies")

from ipasc_iqa.measures.deep.msgmsd import MSGMSDMeasure  # noqa: E402


class TestMSGMSD:
    def test_returns_non_negative_float(self) -> None:
        v = MSGMSDMeasure().evaluate(IMAGE, reference=REFERENCE)
        assert isinstance(v, float)
        assert v >= 0.0

    def test_identical_images_near_zero(self) -> None:
        v = MSGMSDMeasure().evaluate(IMAGE, reference=IMAGE)
        assert v < 1e-4

    def test_requires_reference(self) -> None:
        with pytest.raises(ValueError):
            MSGMSDMeasure().evaluate(IMAGE)

    def test_adaptive_scales_small_image(self) -> None:
        # 8×8 image → only 2 of the default 4 scales fit (min_dim=8, threshold=4)
        m = MSGMSDMeasure()
        weights = m._adaptive_scale_weights(8)
        assert len(weights) == 2
        assert abs(float(weights.sum().item()) - 1.0) < 1e-5

    def test_adaptive_scales_normal_image(self) -> None:
        # 64×64 → all 4 scales fit (64/4 = 16 ≥ 4 after 3 halvings)

        weights = MSGMSDMeasure()._adaptive_scale_weights(64)
        assert len(weights) == 4


# ---------------------------------------------------------------------------
# HaarPSI — skipped when PyTorch-based metric dependencies are unavailable
# ---------------------------------------------------------------------------

pytest.importorskip("piq", reason="requires PyTorch-based metric dependencies")

from ipasc_iqa.measures.deep.haarpsi import HaarPSIMeasure  # noqa: E402


class TestHaarPSI:
    def test_returns_float_in_range(self) -> None:
        v = HaarPSIMeasure().evaluate(IMAGE, reference=REFERENCE)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

    def test_identical_images_near_one(self) -> None:
        v = HaarPSIMeasure().evaluate(IMAGE, reference=IMAGE)
        assert abs(v - 1.0) < 1e-4

    def test_requires_reference(self) -> None:
        with pytest.raises(ValueError):
            HaarPSIMeasure().evaluate(IMAGE)


# ---------------------------------------------------------------------------
# CLIP-IQA — skipped when PyTorch-based metric dependencies are unavailable
# ---------------------------------------------------------------------------

pytest.importorskip("piq", reason="requires PyTorch-based metric dependencies")

from ipasc_iqa.measures.deep.clipiqa import CLIPIQAMeasure  # noqa: E402


class TestCLIPIQA:
    def test_returns_float_in_range(self) -> None:
        v = CLIPIQAMeasure(device="cpu").evaluate(_IMG_128)
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

    def test_single_channel_broadcast(self) -> None:
        # Grayscale (C=1) should not raise; expands internally to C=3.
        v = CLIPIQAMeasure(device="cpu").evaluate(IMAGE)
        assert isinstance(v, float)

    def test_no_reference_required(self) -> None:
        # Should not raise without a reference.
        CLIPIQAMeasure(device="cpu").evaluate(IMAGE)

    def test_model_reused_across_calls(self) -> None:
        m = CLIPIQAMeasure(device="cpu")
        assert m._model is None
        m.evaluate(IMAGE)
        model_ref = m._model
        m.evaluate(IMAGE)
        assert m._model is model_ref


# ---------------------------------------------------------------------------
# Device propagation for functional PIQ wrappers
# ---------------------------------------------------------------------------

from ipasc_iqa.measures.deep.brisque import BRISQUEMeasure  # noqa: E402
from ipasc_iqa.measures.deep.haarpsimed import HaarPSIMedMeasure  # noqa: E402


@pytest.mark.parametrize(
    ("measure_cls", "piq_function", "needs_reference"),
    [
        (BRISQUEMeasure, "brisque", False),
        (FSIMMeasure, "fsim", True),
        (GMSDMeasure, "gmsd", True),
        (HaarPSIMeasure, "haarpsi", True),
        (HaarPSIMedMeasure, "haarpsi", True),
        (IWSSIMMeasure, "information_weighted_ssim", True),
        (MSGMSDMeasure, "multi_scale_gmsd", True),
        (MSSSIMMeasure, "multi_scale_ssim", True),
        (VIFMeasure, "vif_p", True),
    ],
)
def test_functional_piq_wrapper_uses_requested_device(
    monkeypatch: pytest.MonkeyPatch,
    measure_cls,
    piq_function: str,
    needs_reference: bool,
) -> None:
    """Verify device propagation without requiring CUDA or MPS hardware."""
    import piq
    import torch

    # The meta device lets the test observe tensor placement without requiring
    # an accelerator. Returning a CPU scalar avoids calling item() on meta.
    monkeypatch.setattr(
        "ipasc_iqa.core.device.resolve",
        lambda preference="auto": preference,
    )

    def fake_piq_function(*args, **kwargs):
        tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        assert tensor_args
        assert all(tensor.device.type == "meta" for tensor in tensor_args)

        scale_weights = kwargs.get("scale_weights")
        if scale_weights is not None:
            assert scale_weights.device.type == "meta"

        return torch.tensor(0.5)

    monkeypatch.setattr(piq, piq_function, fake_piq_function)

    measure = measure_cls(device="meta")
    kwargs = {"reference": REFERENCE} if needs_reference else {}
    assert measure.evaluate(IMAGE, **kwargs) == 0.5
