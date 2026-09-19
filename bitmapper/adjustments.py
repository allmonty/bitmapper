"""Tone/color pre-adjustments applied to the canvas before quantization."""
from __future__ import annotations

import numpy as np

from .color import luminance


def adjust_contrast(image: np.ndarray, amount: float) -> np.ndarray:
    """Scale pixel values around mid-gray. ``amount`` 1.0 is a no-op, >1
    increases contrast, <1 decreases it, 0 collapses the image to flat gray.
    """
    if amount < 0:
        raise ValueError(f"contrast amount must be >= 0, got {amount}")
    img = image.astype(np.float64)
    out = (img - 127.5) * amount + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_saturation(image: np.ndarray, amount: float) -> np.ndarray:
    """Blend each pixel toward its luminance-weighted grayscale. ``amount``
    1.0 is a no-op, 0 is grayscale, >1 boosts saturation.
    """
    if amount < 0:
        raise ValueError(f"saturation amount must be >= 0, got {amount}")
    img = image.astype(np.float64)
    luma = luminance(img[..., :3])
    gray = np.stack([luma, luma, luma], axis=-1)
    out = gray + (img - gray) * amount
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma-correct ``image``: ``out = 255 * (in / 255) ** (1 / gamma)``.
    ``gamma`` 1.0 is a no-op, >1 brightens midtones, <1 darkens them.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    img = image.astype(np.float64) / 255.0
    out = np.power(img, 1.0 / gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply(image: np.ndarray, contrast: float = 1.0, saturation: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Apply contrast, then saturation, then gamma, skipping any that are a no-op."""
    out = image
    if contrast != 1.0:
        out = adjust_contrast(out, contrast)
    if saturation != 1.0:
        out = adjust_saturation(out, saturation)
    if gamma != 1.0:
        out = adjust_gamma(out, gamma)
    return out
