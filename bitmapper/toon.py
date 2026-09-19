"""Toon (cel) shading stages that run on the grid.

- ``apply_shade_bands`` flattens brightness into a few bands while keeping
  each cell's hue, before palette mapping.
- ``despeckle`` removes isolated cells after quantization, so flat regions
  stay flat.

Together with ``outline.apply_outline`` (ink lines), they give a cel-shaded
look.
"""
from __future__ import annotations

import numpy as np

from .color import luminance

MIN_SHADE_BANDS = 2
MAX_SHADE_BANDS = 8

# Neighbour order for despeckle: row by row, top-left first. Ties between
# equally common neighbour colors go to the first one in this order.
_NEIGHBOURS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def apply_shade_bands(grid: np.ndarray, bands: int) -> np.ndarray:
    """Snap each cell's luminance to the centre of one of ``bands`` equal
    bands, rescaling its RGB so the hue is kept. ``bands`` 0 is off.

    Per cell, in this operation order (the Dart port matches it exactly):
    ``band = min(bands - 1, floor(luma * bands / 256))``,
    ``target = (band + 0.5) * 255 / bands``, then ``factor = target / luma``
    capped so no channel would exceed 255 (``min(factor, 255 / max_channel)``,
    which only ever reduces a brightening factor, never a darkening one),
    then each channel is ``channel * factor``, clipped and truncated; black
    cells become gray at ``target``.
    """
    if bands == 0:
        return grid
    if not (MIN_SHADE_BANDS <= bands <= MAX_SHADE_BANDS):
        raise ValueError(
            f"shade bands must be 0 (off) or {MIN_SHADE_BANDS}..{MAX_SHADE_BANDS}, got {bands}"
        )
    luma = luminance(grid)
    band = np.minimum(bands - 1, np.floor(luma * bands / 256.0))
    target = (band + 0.5) * 255.0 / bands
    safe_luma = np.where(luma > 0, luma, 1.0)
    max_channel = np.where(luma > 0, grid.astype(np.float64).max(axis=-1), 1.0)
    factor = np.minimum(target / safe_luma, 255.0 / max_channel)
    scaled = grid.astype(np.float64) * factor[..., None]
    gray = np.broadcast_to(target[..., None], scaled.shape)
    out = np.where((luma > 0)[..., None], scaled, gray)
    return np.clip(out, 0, 255).astype(np.uint8)


def despeckle(grid: np.ndarray) -> np.ndarray:
    """Replace every isolated cell (one whose color matches none of its 8
    neighbours) with its most common neighbour color. Reads the original
    grid, so the result doesn't depend on scan order."""
    h, w = grid.shape[:2]
    packed = (grid[..., 0].astype(np.int64) << 16) | (grid[..., 1].astype(np.int64) << 8) | grid[..., 2]
    out = grid.copy()
    for y in range(h):
        for x in range(w):
            me = packed[y, x]
            counts: dict[int, int] = {}
            order: list[int] = []
            isolated = True
            for dx, dy in _NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    c = int(packed[ny, nx])
                    if c == me:
                        isolated = False
                        break
                    if c not in counts:
                        counts[c] = 0
                        order.append(c)
                    counts[c] += 1
            if isolated and order:
                best = order[0]
                for c in order[1:]:
                    if counts[c] > counts[best]:
                        best = c
                out[y, x] = ((best >> 16) & 255, (best >> 8) & 255, best & 255)
    return out
