"""Sparse Structural Similarity Index (S3IM).

Repository adapter for the implementation supplied by the S3IM authors.
The numerical formulation is intentionally preserved; only the
``BaseMeasure`` interface and repository metadata are added.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from skimage.filters import threshold_local
from skimage.metrics import structural_similarity

from ipasc_iqa.core.measure import BaseMeasure, MeasureMetadata, MeasureType


def _adaptive_mask(
    image: np.ndarray,
    sensitivity: float = 0.5,
) -> np.ndarray:
    """Return pixels that stand out from their local neighbourhood."""
    patch_size = 2 * (np.array(image.shape) // 16) + 1
    block_size = int(np.mean(patch_size))

    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    local_threshold = threshold_local(
        image,
        block_size,
        method="gaussian",
        offset=-(0.5 - sensitivity),
    )
    return image > local_threshold


def calculate_single_mask(
    image: np.ndarray,
    sensitivity: float = 0.5,
) -> np.ndarray:
    """Return the adaptive-threshold mask for one image."""
    return _adaptive_mask(image, sensitivity)


def calculate_combined_mask(
    image_1: np.ndarray,
    image_2: np.ndarray,
    sensitivity: float = 0.5,
    combine: Literal["or", "and"] = "or",
) -> np.ndarray:
    """Combine the two adaptive-threshold masks (OR by default)."""
    mask_1 = _adaptive_mask(image_1, sensitivity)
    mask_2 = _adaptive_mask(image_2, sensitivity)

    if combine == "or":
        return np.logical_or(mask_1, mask_2)
    if combine == "and":
        return np.logical_and(mask_1, mask_2)

    raise ValueError(f"Unknown combine mode: {combine!r}")


def cw_ssim_map(
    image_1: np.ndarray,
    image_2: np.ndarray,
    width: int = 30,
    k: float = 0.01,
) -> np.ndarray:
    """Compute the Complex Wavelet SSIM map from the supplied implementation."""
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "The S3IM 'cw' backend requires PyWavelets. Install it with "
            "`python -m pip install PyWavelets`."
        ) from exc

    if image_1.shape != image_2.shape:
        raise ValueError("image_1 and image_2 must have the same shape")

    height, image_width = image_1.shape
    widths = np.arange(1, width + 1)

    cwt_1, _ = pywt.cwt(image_1.ravel(), widths, "mexh")
    cwt_2, _ = pywt.cwt(image_2.ravel(), widths, "mexh")

    c1c2 = np.abs(cwt_1) * np.abs(cwt_2)
    c1_squared = np.abs(cwt_1) ** 2
    c2_squared = np.abs(cwt_2) ** 2

    numerator_1 = 2 * np.sum(c1c2, axis=0) + k
    denominator_1 = (
        np.sum(c1_squared, axis=0)
        + np.sum(c2_squared, axis=0)
        + k
    )

    c1c2_conjugate = cwt_1 * np.conjugate(cwt_2)
    numerator_2 = 2 * np.abs(np.sum(c1c2_conjugate, axis=0)) + k
    denominator_2 = 2 * np.sum(np.abs(c1c2_conjugate), axis=0) + k

    ssim_1d = (
        (numerator_1 / denominator_1)
        * (numerator_2 / denominator_2)
    )
    return ssim_1d.reshape((height, image_width))


def s3im(
    image_1: np.ndarray,
    image_2: np.ndarray,
    sensitivity: float = 0.5,
    window_size: int = 7,
    backend: Literal["ssim", "cw"] = "ssim",
    data_range: float = 1.0,
    increase_sensitivity: bool = True,
    max_sensitivity: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute S3IM exactly as in the supplied standalone implementation."""
    if image_1.shape != image_2.shape:
        raise ValueError("image_1 and image_2 must have the same shape")

    current_sensitivity = sensitivity

    while True:
        mask = calculate_combined_mask(
            image_1,
            image_2,
            current_sensitivity,
        )

        if np.any(mask):
            if backend == "ssim":
                _, ssim_map = structural_similarity(
                    image_1,
                    image_2,
                    full=True,
                    win_size=11,
                    gaussian_weights=True,
                    sigma=1.5,
                    use_sample_covariance=False,
                    data_range=data_range,
                )
                # _, ssim_map = structural_similarity(
                #     image_1,
                #     image_2,
                #     full=True,
                #     win_size=window_size,
                #     data_range=data_range,
                # )
            elif backend == "cw":
                ssim_map = cw_ssim_map(image_1, image_2)
            else:
                raise ValueError(f"Unknown backend: {backend!r}")

            masked_map = ssim_map * mask

            if np.any(masked_map):
                score = np.sum(masked_map) / np.sum(mask)
                # score = float(
                #     np.sum(masked_map)
                #     / np.sum(masked_map != 0)
                # )
                return score, mask, masked_map

        if (
            not increase_sensitivity
            or current_sensitivity >= max_sensitivity
        ):
            empty_map = np.zeros_like(image_1, dtype=float)
            return float("nan"), mask, empty_map

        current_sensitivity = min(
            current_sensitivity + 0.1,
            max_sensitivity,
        )


class S3IMMeasure(BaseMeasure):
    """Adapter exposing the supplied S3IM implementation to ``ipasc_iqa``."""

    metadata = MeasureMetadata(
        name="S3IM",
        version="1.0.0",
        measure_type=MeasureType.FULL_REFERENCE,
        requires_signal_roi=False,
        requires_noise_roi=False,
        accepted_reference_types=None,
        higher_is_better=True,
        value_range=(-1.0, 1.0),
        citation=(
            "Kalloor Joseph, 'Sparse Structural Similarity Index (S3IM): "
            "a new full reference image quality metric for photoacoustic "
            "imaging', Proc. SPIE 13319, 1331917 (2025). "
            "https://doi.org/10.1117/12.3040764"
        ),
        bibtex=(
            "@inproceedings{kalloorjoseph2025s3im,\n"
            "  author    = {Kalloor Joseph, Francis},\n"
            "  title     = {Sparse Structural Similarity Index ({S3IM}): "
            "A New Full Reference Image Quality Metric for "
            "Photoacoustic Imaging},\n"
            "  booktitle = {Photons Plus Ultrasound: Imaging and "
            "Sensing 2025},\n"
            "  series    = {Proceedings of SPIE},\n"
            "  volume    = {13319},\n"
            "  pages     = {1331917},\n"
            "  year      = {2025},\n"
            "  doi       = {10.1117/12.3040764}\n"
            "}"
        ),
        description=(
            "Sparse Structural Similarity Index (S3IM): SSIM averaged "
            "over an adaptively selected sparse-region mask. The data "
            "range is inferred from the reference unless explicitly "
            "provided."
        ),
    )

    def __init__(
        self,
        sensitivity: float = 0.5,
        window_size: int = 7,
        backend: Literal["ssim", "cw"] = "ssim",
        data_range: float | None = None,
        increase_sensitivity: bool = True,
        max_sensitivity: float = 1.0,
        device: str = "auto",
    ) -> None:
        super().__init__(
            device=device,
            sensitivity=sensitivity,
            window_size=window_size,
            backend=backend,
            data_range=data_range,
            increase_sensitivity=increase_sensitivity,
            max_sensitivity=max_sensitivity,
        )

        self._sensitivity = sensitivity
        self._window_size = window_size
        self._backend = backend
        self._data_range = data_range
        self._increase_sensitivity = increase_sensitivity
        self._max_sensitivity = max_sensitivity

    @staticmethod
    def _to_grayscale(
        array: np.ndarray,
        name: str,
    ) -> np.ndarray:
        image = np.asarray(array, dtype=float)

        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        if image.ndim != 2:
            raise ValueError(
                f"S3IM requires a single-channel 2D {name}."
            )

        return image

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
            raise ValueError("S3IM requires a reference image.")

        image_gray = self._to_grayscale(image, "image")
        reference_gray = self._to_grayscale(
            reference,
            "reference",
        )

        data_range = self._data_range

        if data_range is None:
            data_range = float(
                reference_gray.max() - reference_gray.min()
            )
            if data_range == 0.0:
                data_range = 1.0
        
        score, _, _ = s3im(
            reference_gray,
            image_gray,
            sensitivity=self._sensitivity,
            window_size=self._window_size,
            backend=self._backend,
            data_range=data_range,
            increase_sensitivity=self._increase_sensitivity,
            max_sensitivity=self._max_sensitivity,
        )

        return score