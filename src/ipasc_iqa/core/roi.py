"""ROI loading and hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def load_roi(path: Path) -> np.ndarray:
    """Load a boolean ROI mask from a .npy file."""
    arr = np.load(path)
    return arr.astype(bool)


def hash_array(arr: np.ndarray) -> str:
    """SHA-256 of raw array bytes (dtype-sensitive by design)."""
    return hashlib.sha256(arr.tobytes()).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
