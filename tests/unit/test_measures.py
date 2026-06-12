"""Unit tests for built-in measures."""

from __future__ import annotations

import numpy as np
import pytest

from ipasc_iqa.measures.classical.cnr import CNRMeasure
from ipasc_iqa.measures.classical.psnr import PSNRMeasure
from ipasc_iqa.measures.classical.snr import SNRMeasure
from ipasc_iqa.measures.classical.ssim import SSIMMeasure

rng = np.random.default_rng(0)
IMAGE = rng.random((64, 64, 1), dtype=np.float32).astype(np.float32)
REFERENCE = rng.random((64, 64, 1), dtype=np.float32).astype(np.float32)
SIGNAL_ROI = (IMAGE[:, :, 0] > 0.7)[:, :, np.newaxis].astype(bool)
NOISE_ROI = (IMAGE[:, :, 0] < 0.2)[:, :, np.newaxis].astype(bool)


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


# ---------------------------------------------------------------------------
# LPIPS — skipped when ipasc_iqa[deep] is not installed
# ---------------------------------------------------------------------------

pytest.importorskip("torch", reason="requires ipasc_iqa[deep]")
pytest.importorskip("lpips", reason="requires ipasc_iqa[deep]")

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
