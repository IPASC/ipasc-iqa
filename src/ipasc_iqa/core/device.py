"""GPU/CPU device resolution."""

from __future__ import annotations


def resolve(preference: str = "auto") -> str:
    """Return a device string: 'cuda:0', 'mps', or 'cpu'.

    preference: 'auto' | 'cpu' | 'cuda' | 'mps'
    """
    if preference == "cpu":
        return "cpu"

    try:
        import torch

        if preference in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda:0"
        if preference in ("auto", "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"
