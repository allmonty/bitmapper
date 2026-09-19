"""Sprite-style ink outlines, drawn on the quantized grid.

A cell becomes "ink" when one of its 4 neighbours is brighter than it by
more than a threshold: the line lands on the dark side of every strong edge,
one cell thick, like the outlines around hand-drawn sprites. Ink is the
palette's darkest color, so the output stays within the palette.
"""
from __future__ import annotations

import numpy as np


def _luminance(colors: np.ndarray) -> np.ndarray:
    """Rec. 601 luma, as ``(r*0.299 + g*0.587) + b*0.114`` in float64 (the
    Dart port computes it in the same order, so results match exactly)."""
    c = colors.astype(np.float64)
    return c[..., 0] * 0.299 + c[..., 1] * 0.587 + c[..., 2] * 0.114


def outline_threshold(strength: float) -> float:
    """Brightness jump that counts as an edge: 128 at strength 0+, down to
    16 at strength 1 (stronger = more edges outlined)."""
    return 128.0 - 112.0 * strength


def darkest_color(palette: np.ndarray) -> np.ndarray:
    """The palette entry with the lowest luminance (first one on ties)."""
    return palette[int(np.argmin(_luminance(palette)))]


def apply_outline(grid: np.ndarray, palette: np.ndarray, strength: float) -> np.ndarray:
    """Ink the dark side of strong edges in ``grid`` (rows, cols, 3) with the
    darkest color of ``palette``. ``strength`` 0 is off; 1 outlines the
    faintest edges."""
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"outline strength must be between 0 and 1, got {strength}")
    if strength == 0.0 or grid.size == 0:
        return grid

    lum = _luminance(grid)
    threshold = outline_threshold(strength)
    mask = np.zeros(lum.shape, dtype=bool)
    # Each neighbour direction: is the neighbour brighter by more than threshold?
    mask[:, :-1] |= lum[:, 1:] - lum[:, :-1] > threshold   # right
    mask[:, 1:] |= lum[:, :-1] - lum[:, 1:] > threshold    # left
    mask[:-1, :] |= lum[1:, :] - lum[:-1, :] > threshold   # below
    mask[1:, :] |= lum[:-1, :] - lum[1:, :] > threshold    # above

    out = grid.copy()
    out[mask] = darkest_color(palette)
    return out
