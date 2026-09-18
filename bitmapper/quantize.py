"""Nearest-color mapping of pixels onto a fixed palette."""
from __future__ import annotations

import numpy as np

# The naive distance computation allocates a (pixels, palette, channels)
# intermediate, which is hundreds of MB for a large grid against a big
# palette (a 300x300 grid against vga256 needs ~550MB). Pixels are
# independent, so they are processed in chunks sized to keep that
# intermediate bounded instead.
_MAX_DISTANCE_ELEMENTS = 8_000_000  # ~64MB as float64


def nearest_index(pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Index of the closest ``palette`` (K, C) entry for each of ``pixels``
    (N, C), by squared Euclidean distance in RGB space. Ties go to the
    lowest palette index.
    """
    flat = pixels.astype(np.float64, copy=False)
    pal = palette.astype(np.float64, copy=False)

    n_pixels, n_channels = flat.shape
    chunk = max(1, _MAX_DISTANCE_ELEMENTS // max(len(pal) * n_channels, 1))

    idx = np.empty(n_pixels, dtype=np.intp)
    for start in range(0, n_pixels, chunk):
        block = flat[start:start + chunk]
        dists = ((block[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
        idx[start:start + chunk] = dists.argmin(axis=1)
    return idx


def nearest_color(image: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each pixel in ``image`` (..., C) to the closest color (Euclidean,
    in RGB space) in ``palette`` (K, C).

    Returns ``(quantized_image, indices)`` where ``indices`` has the same
    leading shape as ``image`` and gives the chosen palette row per pixel.
    """
    shape = image.shape
    flat = image.reshape(-1, shape[-1])
    idx = nearest_index(flat, palette)

    quantized = palette.astype(np.float64, copy=False)[idx].reshape(shape).astype(np.uint8)
    return quantized, idx.reshape(shape[:-1])
