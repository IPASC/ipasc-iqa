"""NumPy array I/O helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load a .npy image and ensure float32 dtype and 3-D (H, W, C) shape."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return arr


def save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))
