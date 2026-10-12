"""Unit tests for the experimental S3IM measure."""

from __future__ import annotations

import numpy as np
import pytest

from ipasc_iqa.measures.classical.s3im import S3IMMeasure
from ipasc_iqa.measures.classical.ssim import SSIMMeasure


def _sparse_pair() -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros((64, 64, 1), dtype=np.float32)
    reference[12:52, 30:34, 0] = 1.0
    reference[30:34, 12:52, 0] = 0.7
    image = reference.copy()
    image[30:34, 32:52, 0] = 0.0
    return image, reference


class TestS3IM:
    def test_identical_sparse_images_are_one(self) -> None:
        _, reference = _sparse_pair()
        score = S3IMMeasure().evaluate(reference, reference=reference)
        assert score == pytest.approx(1.0)

    def test_degraded_sparse_image_returns_finite_score(self) -> None:
        image, reference = _sparse_pair()
        score = S3IMMeasure().evaluate(image, reference=reference)
        assert np.isfinite(score)
        assert -1.0 <= score < 1.0

    def test_sparse_degradation_is_more_visible_than_with_ssim(self) -> None:
        image, reference = _sparse_pair()
        s3im = S3IMMeasure().evaluate(image, reference=reference)
        ssim = SSIMMeasure().evaluate(image, reference=reference)
        assert s3im < ssim

    def test_is_invariant_to_common_positive_scaling(self) -> None:
        image, reference = _sparse_pair()
        measure = S3IMMeasure()
        score_unit = measure.evaluate(image, reference=reference)
        score_scaled = measure.evaluate(image * 250.0, reference=reference * 250.0)
        assert score_scaled == pytest.approx(score_unit)

    def test_requires_reference(self) -> None:
        image, _ = _sparse_pair()
        with pytest.raises(ValueError, match="requires a reference"):
            S3IMMeasure().evaluate(image)

    def test_rejects_multichannel_images(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="single-channel"):
            S3IMMeasure().evaluate(image, reference=image)

    def test_parameters_are_part_of_cache_configuration(self) -> None:
        default = S3IMMeasure()
        changed = S3IMMeasure(sensitivity=0.7)
        assert default.config_canonical_json() != changed.config_canonical_json()
