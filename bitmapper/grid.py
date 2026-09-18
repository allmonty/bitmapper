"""Pixel-grid downsampling and blocky upscaling.

These two functions are the core of the "chunky pixel" look: an image is
first collapsed onto a small logical grid (e.g. 200x200), then that grid is
replicated back up to the full canvas size using nearest-neighbor repetition
so each grid cell becomes a hard-edged NxM block of flat color.
"""
from __future__ import annotations

import numpy as np


def _splits(length: int, n: int) -> list[np.ndarray]:
    return np.array_split(np.arange(length), n)


def downsample(image: np.ndarray, grid_size: tuple[int, int], mode: str = "average") -> np.ndarray:
    """Collapse ``image`` (H, W, C) onto a (grid_h, grid_w, C) grid.

    ``grid_size`` is (cols, rows). ``mode`` is "average" (box-filter mean of
    each block) or "nearest" (sample the center pixel of each block).
    """
    if mode not in ("average", "nearest"):
        raise ValueError(f"unknown mode: {mode!r}")

    h, w = image.shape[:2]
    grid_cols, grid_rows = grid_size
    row_splits = _splits(h, grid_rows)
    col_splits = _splits(w, grid_cols)

    if mode == "nearest":
        row_idx = np.array([rows[len(rows) // 2] for rows in row_splits])
        col_idx = np.array([cols[len(cols) // 2] for cols in col_splits])
        return image[np.ix_(row_idx, col_idx)].copy()

    # average: fully vectorized box filter via reduceat, works for uneven splits too
    row_starts = np.array([rows[0] for rows in row_splits])
    col_starts = np.array([cols[0] for cols in col_splits])
    row_counts = np.array([len(rows) for rows in row_splits]).reshape(-1, 1, 1)
    col_counts = np.array([len(cols) for cols in col_splits]).reshape(1, -1, 1)

    summed = np.add.reduceat(image.astype(np.float64), row_starts, axis=0)
    summed = np.add.reduceat(summed, col_starts, axis=1)
    out = summed / row_counts / col_counts
    return np.clip(out, 0, 255).astype(np.uint8)


def upscale(
    grid_image: np.ndarray,
    output_size: tuple[int, int],
    gap_px: int = 0,
    gap_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Replicate each cell of ``grid_image`` (grid_h, grid_w, C) up to
    ``output_size`` = (width, height) pixels, nearest-neighbor style so
    blocks stay flat and hard-edged.

    If ``gap_px`` > 0, a gutter of that width, filled with ``gap_color``, is
    drawn at every block boundary (between cells, not around the canvas
    edge), giving the blocks a separated-tile look.
    """
    out_w, out_h = output_size
    grid_h, grid_w = grid_image.shape[:2]

    row_repeats = [len(rows) for rows in _splits(out_h, grid_h)]
    col_repeats = [len(cols) for cols in _splits(out_w, grid_w)]

    out = np.repeat(grid_image, row_repeats, axis=0)
    out = np.repeat(out, col_repeats, axis=1)

    if gap_px > 0:
        out = out.copy()
        fill = np.array(gap_color, dtype=out.dtype)
        half = gap_px // 2

        for edge in np.cumsum(row_repeats)[:-1]:
            lo, hi = max(0, edge - half), min(out_h, edge + (gap_px - half))
            out[lo:hi, :] = fill
        for edge in np.cumsum(col_repeats)[:-1]:
            lo, hi = max(0, edge - half), min(out_w, edge + (gap_px - half))
            out[:, lo:hi] = fill

    return out
