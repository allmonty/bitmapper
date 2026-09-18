"""Post-quantization visual effects applied to the full-resolution output."""
from __future__ import annotations

import numpy as np


def apply_scanlines(image: np.ndarray, strength: float) -> np.ndarray:
    """Darken every other row of ``image`` by ``strength`` (0 = no effect,
    1 = alternate rows go fully black), simulating a CRT scanline effect.
    """
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"scanlines strength must be between 0 and 1, got {strength}")
    if strength == 0.0:
        return image

    out = image.astype(np.float64)
    out[1::2] *= 1.0 - strength
    return np.clip(out, 0, 255).astype(np.uint8)
