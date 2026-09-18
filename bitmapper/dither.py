"""Dithering: error-diffusion and ordered (Bayer) dithering methods.

All methods run at grid resolution (not full canvas resolution), so the
dither pattern reads as part of the chunky pixel art rather than as
full-resolution noise underneath the blocks.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from .quantize import nearest_color

# Error-diffusion kernels as (dx, dy, weight) offsets from the current pixel,
# plus the divisor each weight is normalized by.
_DIFFUSION_KERNELS: dict[str, tuple[list[tuple[int, int, float]], float]] = {
    "floyd_steinberg": (
        [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)],
        16,
    ),
    # Apple/HyperCard-era: only diffuses 6/8 of the error, for a lighter,
    # higher-contrast look.
    "atkinson": (
        [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)],
        8,
    ),
    "jarvis_judice_ninke": (
        [
            (1, 0, 7), (2, 0, 5),
            (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
            (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1),
        ],
        48,
    ),
    "stucki": (
        [
            (1, 0, 8), (2, 0, 4),
            (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2),
            (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1),
        ],
        42,
    ),
    "sierra": (
        [
            (1, 0, 5), (2, 0, 3),
            (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2),
            (-1, 2, 2), (0, 2, 3), (1, 2, 2),
        ],
        32,
    ),
    "sierra_lite": (
        [(1, 0, 2), (-1, 1, 1), (0, 1, 1)],
        4,
    ),
    "burkes": (
        [
            (1, 0, 8), (2, 0, 4),
            (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2),
        ],
        32,
    ),
}


def _error_diffusion(
    image: np.ndarray,
    palette: np.ndarray,
    kernel: list[tuple[int, int, float]],
    divisor: float,
    strength: float = 1.0,
) -> np.ndarray:
    img = image.astype(np.float64).copy()
    pal = palette.astype(np.float64)
    h, w = img.shape[:2]
    # Fold divisor/strength into each tap's weight once, instead of doing
    # the same division and multiplication on every pixel in the hot loop.
    scaled_kernel = [(dx, dy, weight * strength / divisor) for dx, dy, weight in kernel]

    for y in range(h):
        for x in range(w):
            old = img[y, x].copy()
            dists = ((pal - old) ** 2).sum(axis=1)
            new = pal[int(dists.argmin())]
            img[y, x] = new
            error = old - new

            for dx, dy, scaled_weight in scaled_kernel:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    img[ny, nx] += error * scaled_weight

    return np.clip(img, 0, 255).astype(np.uint8)


def error_diffusion(image: np.ndarray, palette: np.ndarray, method: str, strength: float = 1.0) -> np.ndarray:
    """Error-diffusion dithering onto ``palette`` using the named kernel
    from ``_DIFFUSION_KERNELS`` (floyd_steinberg, atkinson, stucki, ...).
    """
    try:
        kernel, divisor = _DIFFUSION_KERNELS[method]
    except KeyError:
        raise ValueError(f"unknown error-diffusion kernel: {method!r}") from None
    return _error_diffusion(image, palette, kernel, divisor, strength)


def _bayer_matrix(size: int) -> np.ndarray:
    """Recursively build an ``size``x``size`` Bayer threshold matrix
    (``size`` must be a power of two), centered in [-0.5, 0.5).
    """
    if size == 1:
        return np.array([[0.0]])
    smaller = _bayer_matrix(size // 2)
    return np.block(
        [
            [4 * smaller + 0, 4 * smaller + 2],
            [4 * smaller + 3, 4 * smaller + 1],
        ]
    )


def _ordered_with_matrix(image: np.ndarray, palette: np.ndarray, matrix: np.ndarray, strength: float = 1.0) -> np.ndarray:
    size = matrix.shape[0]
    threshold = matrix / (size * size) - 0.5
    h, w = image.shape[:2]
    tiled = np.tile(threshold, (h // size + 1, w // size + 1))[:h, :w]

    n_colors = max(len(palette), 2)
    step = 255.0 / (n_colors ** (1 / 3))

    perturbed = image.astype(np.float64) + tiled[..., None] * step * strength
    quantized, _ = nearest_color(np.clip(perturbed, 0, 255), palette)
    return quantized


def ordered(image: np.ndarray, palette: np.ndarray, matrix_size: int = 4, strength: float = 1.0) -> np.ndarray:
    """Ordered (Bayer) dithering: perturb pixels by a tiled threshold map,
    scaled to roughly the palette's color spacing, before nearest-color
    quantization. ``matrix_size`` must be a power of two (2, 4, 8, ...).
    """
    return _ordered_with_matrix(image, palette, _bayer_matrix(matrix_size), strength)


def ordered_2x2(image: np.ndarray, palette: np.ndarray, strength: float = 1.0) -> np.ndarray:
    return ordered(image, palette, matrix_size=2, strength=strength)


def ordered_8x8(image: np.ndarray, palette: np.ndarray, strength: float = 1.0) -> np.ndarray:
    return ordered(image, palette, matrix_size=8, strength=strength)


def random_dither(
    image: np.ndarray, palette: np.ndarray, strength: float = 1.0, seed: int | None = None
) -> np.ndarray:
    """White-noise dithering: perturb pixels with uniform random noise,
    scaled to roughly the palette's color spacing, before nearest-color
    quantization. Grungier and less structured than ordered dithering.
    """
    rng = np.random.default_rng(seed)
    n_colors = max(len(palette), 2)
    step = 255.0 / (n_colors ** (1 / 3))

    noise = rng.uniform(-0.5, 0.5, size=image.shape[:2]) * step * strength
    perturbed = image.astype(np.float64) + noise[..., None]
    quantized, _ = nearest_color(np.clip(perturbed, 0, 255), palette)
    return quantized


# Error-diffusion methods are registered straight from the kernel table, so
# adding a kernel to _DIFFUSION_KERNELS is all it takes to expose a new
# method (and to pick up the parametrized tests that read list_methods()).
_METHODS = {
    **{name: partial(error_diffusion, method=name) for name in _DIFFUSION_KERNELS},
    "ordered": ordered,
    "ordered_2x2": ordered_2x2,
    "ordered_8x8": ordered_8x8,
    "random": random_dither,
}


def list_methods() -> list[str]:
    return ["none"] + sorted(_METHODS.keys())


def apply(image: np.ndarray, palette: np.ndarray, method: str, strength: float = 1.0) -> np.ndarray:
    """Apply dither ``method`` onto ``palette``. ``strength`` (0-1) scales
    how much dithering pattern is applied: 0 behaves like "none" (plain
    nearest-color quantization), 1 is full-strength dithering.
    """
    if strength < 0:
        raise ValueError(f"dither strength must be >= 0, got {strength}")
    if method == "none":
        quantized, _ = nearest_color(image, palette)
        return quantized
    try:
        return _METHODS[method](image, palette, strength=strength)
    except KeyError:
        raise ValueError(f"unknown dither method: {method!r}") from None
