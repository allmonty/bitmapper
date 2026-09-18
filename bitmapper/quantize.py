"""Nearest-color mapping of pixels onto a fixed palette."""
from __future__ import annotations

import numpy as np


def nearest_color(image: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each pixel in ``image`` (..., C) to the closest color (Euclidean,
    in RGB space) in ``palette`` (K, C).

    Returns ``(quantized_image, indices)`` where ``indices`` has the same
    leading shape as ``image`` and gives the chosen palette row per pixel.
    """
    shape = image.shape
    flat = image.reshape(-1, shape[-1]).astype(np.float64)
    pal = palette.astype(np.float64)

    dists = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    idx = dists.argmin(axis=1)

    quantized = pal[idx].reshape(shape).astype(np.uint8)
    return quantized, idx.reshape(shape[:-1])
