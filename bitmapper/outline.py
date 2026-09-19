"""Sprite-style ink outlines, drawn on the quantized grid.

Like dithering, outlining offers several methods (``list_methods()``):

- ``brightness`` (the original): a cell is inked when one of its 4
  neighbours is brighter than it by more than the threshold. The line
  lands on the dark side of every strong brightness edge, one cell thick.
- ``color``: the same neighbour comparison, but by RGB distance instead of
  brightness, so it also catches edges between hues of similar brightness
  (e.g. red next to green). The darker of each pair is inked.
- ``sobel``: a 3x3 Sobel gradient of the brightness, which also finds
  diagonal and gradual edges; only cells darker than their 3x3
  neighbourhood average are inked, keeping the line on the dark side.

Ink is either the palette's darkest color (``darkest``, the original) or,
for a softer sprite look, the palette color closest to a half-brightness
copy of the outlined color (``shaded``). Either way the output stays
within the palette.
"""
from __future__ import annotations

import numpy as np

from .color import luminance
from .quantize import nearest_index

METHODS = ("brightness", "color", "sobel")
INKS = ("darkest", "shaded")


def list_methods() -> list[str]:
    return list(METHODS)


def list_inks() -> list[str]:
    return list(INKS)


def outline_threshold(strength: float) -> float:
    """Edge threshold: 128 at strength 0+, down to 16 at strength 1
    (stronger = more edges outlined). A brightness step for ``brightness``
    and ``sobel``; a color distance for ``color``."""
    return 128.0 - 112.0 * strength


def darkest_color(palette: np.ndarray) -> np.ndarray:
    """The palette entry with the lowest luminance (first one on ties)."""
    return palette[int(np.argmin(luminance(palette)))]


def _squared_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared RGB distance, as ``(dr² + dg²) + db²`` in float64."""
    d = a.astype(np.float64) - b.astype(np.float64)
    return d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]


def _brightness_mask(grid: np.ndarray, threshold: float) -> np.ndarray:
    lum = luminance(grid)
    mask = np.zeros(lum.shape, dtype=bool)
    # Each neighbour direction: is the neighbour brighter by more than threshold?
    mask[:, :-1] |= lum[:, 1:] - lum[:, :-1] > threshold   # right
    mask[:, 1:] |= lum[:, :-1] - lum[:, 1:] > threshold    # left
    mask[:-1, :] |= lum[1:, :] - lum[:-1, :] > threshold   # below
    mask[1:, :] |= lum[:-1, :] - lum[1:, :] > threshold    # above
    return mask


def _color_mask(grid: np.ndarray, threshold: float) -> np.ndarray:
    lum = luminance(grid)
    limit = 3.0 * threshold * threshold
    mask = np.zeros(lum.shape, dtype=bool)
    # Horizontal pairs (a = left, b = right), then vertical (a = above, b = below).
    for a_idx, b_idx in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        edge = _squared_distance(grid[a_idx], grid[b_idx]) > limit
        # Ink the darker cell of the pair; on equal brightness, the first one.
        a_darker = lum[a_idx] <= lum[b_idx]
        mask[a_idx] |= edge & a_darker
        mask[b_idx] |= edge & ~a_darker
    return mask


def _sobel_mask(grid: np.ndarray, threshold: float) -> np.ndarray:
    lum = np.pad(luminance(grid), 1, mode="edge")
    tl, t, tr = lum[:-2, :-2], lum[:-2, 1:-1], lum[:-2, 2:]
    ml, mc, mr = lum[1:-1, :-2], lum[1:-1, 1:-1], lum[1:-1, 2:]
    bl, b, br = lum[2:, :-2], lum[2:, 1:-1], lum[2:, 2:]
    gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl)
    gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr)
    # A brightness step of `threshold` between neighbours gives a Sobel
    # magnitude of about 4 * threshold.
    strong = gx * gx + gy * gy > 16.0 * threshold * threshold
    total = ((tl + t + tr) + (ml + mc + mr)) + (bl + b + br)
    darker = mc * 9.0 < total
    return strong & darker


def _shaded_ink(sources: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """For each source color, the palette color closest to it at half
    brightness, falling back to the darkest color when that would be the
    source itself (so the line still shows)."""
    idx = nearest_index(sources.astype(np.int64) >> 1, palette)
    ink = palette[idx]
    same = (ink == sources).all(axis=1)
    ink[same] = darkest_color(palette)
    return ink


def apply_outline(
    grid: np.ndarray,
    palette: np.ndarray,
    strength: float,
    method: str = "brightness",
    ink: str = "darkest",
    edge_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Ink the edges of ``grid`` (rows, cols, 3) found by ``method`` with
    colors from ``palette``, using ``ink`` to pick the color. ``strength``
    0 is off; 1 outlines the faintest edges.

    Edges are detected on ``edge_grid`` (defaulting to ``grid`` itself) but
    ink is always painted onto ``grid``. The pipeline passes the grid
    quantized before dithering as ``edge_grid``, so a dither pattern's
    color noise in flat regions isn't mistaken for real edges, while the
    ink color/placement still reflects the actually rendered pixels.
    """
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"outline strength must be between 0 and 1, got {strength}")
    if method not in METHODS:
        raise ValueError(f"invalid outline method: {method!r}")
    if ink not in INKS:
        raise ValueError(f"invalid outline ink: {ink!r}")
    if strength == 0.0 or grid.size == 0:
        return grid
    if edge_grid is None:
        edge_grid = grid
    elif edge_grid.shape != grid.shape:
        raise ValueError(
            f"edge_grid shape {edge_grid.shape} doesn't match grid shape {grid.shape}"
        )

    threshold = outline_threshold(strength)
    if method == "brightness":
        mask = _brightness_mask(edge_grid, threshold)
    elif method == "color":
        mask = _color_mask(edge_grid, threshold)
    else:
        mask = _sobel_mask(edge_grid, threshold)

    out = grid.copy()
    out[mask] = darkest_color(palette) if ink == "darkest" else _shaded_ink(grid[mask], palette)
    return out
