"""Dithering: Floyd-Steinberg error diffusion and ordered (Bayer) dithering.

Both run at grid resolution (not full canvas resolution), so the dither
pattern reads as part of the chunky pixel art rather than as full-resolution
noise underneath the blocks.
"""
from __future__ import annotations

import numpy as np

from .quantize import nearest_color

_BAYER_4X4 = np.array(
    [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ],
    dtype=np.float64,
) / 16.0 - 0.5  # centered in [-0.5, 0.5)


def floyd_steinberg(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Classic Floyd-Steinberg error-diffusion dithering onto ``palette``."""
    img = image.astype(np.float64).copy()
    pal = palette.astype(np.float64)
    h, w = img.shape[:2]

    for y in range(h):
        for x in range(w):
            old = img[y, x].copy()
            dists = ((pal - old) ** 2).sum(axis=1)
            new = pal[int(dists.argmin())]
            img[y, x] = new
            error = old - new

            if x + 1 < w:
                img[y, x + 1] += error * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    img[y + 1, x - 1] += error * 3 / 16
                img[y + 1, x] += error * 5 / 16
                if x + 1 < w:
                    img[y + 1, x + 1] += error * 1 / 16

    return np.clip(img, 0, 255).astype(np.uint8)


def ordered(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Ordered (Bayer 4x4) dithering: perturb pixels by a tiled threshold
    map, scaled to roughly the palette's color spacing, before nearest-color
    quantization.
    """
    h, w = image.shape[:2]
    tiled = np.tile(_BAYER_4X4, (h // 4 + 1, w // 4 + 1))[:h, :w]

    n_colors = max(len(palette), 2)
    step = 255.0 / (n_colors ** (1 / 3))

    perturbed = image.astype(np.float64) + tiled[..., None] * step
    quantized, _ = nearest_color(np.clip(perturbed, 0, 255), palette)
    return quantized


def apply(image: np.ndarray, palette: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        quantized, _ = nearest_color(image, palette)
        return quantized
    if method == "floyd_steinberg":
        return floyd_steinberg(image, palette)
    if method == "ordered":
        return ordered(image, palette)
    raise ValueError(f"unknown dither method: {method!r}")
