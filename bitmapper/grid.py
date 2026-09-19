"""Pixel-grid downsampling and blocky upscaling.

These two functions are the core of the "chunky pixel" look: an image is
first collapsed onto a small logical grid (e.g. 200x200), then that grid is
replicated back up to the full canvas size using nearest-neighbor repetition
so each grid cell becomes a hard-edged NxM block of flat color.
"""
from __future__ import annotations

import numpy as np


def _splits(length: int, n: int) -> list[np.ndarray]:
    """Split ``range(length)`` into ``n`` consecutive parts whose sizes differ
    by at most 1, with the larger parts spread evenly: part ``i`` covers
    ``[i * length // n, (i + 1) * length // n)``.

    Deliberately not ``np.array_split``, which gives every leftover element to
    the first parts: that squeezes the start of an image into its cells and
    stretches the rest (e.g. 1024 px in 120 columns: 64 cells of 9 px, then 56
    of 8), a visible distortion that shifts with the grid size. The Dart port
    (``splitSizes``) uses the same formula so the two stay comparable.
    """
    bounds = [i * length // n for i in range(n + 1)]
    return [np.arange(bounds[i], bounds[i + 1]) for i in range(n)]


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


def _min_positive_run(repeats: list[int]) -> int:
    """The smallest non-empty block size in ``repeats`` (0 if every block is
    empty, which only happens when the grid has more cells than output
    pixels along that axis)."""
    positive = [r for r in repeats if r > 0]
    return min(positive) if positive else 0


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
    edge), giving the blocks a separated-tile look. The gap shrinks (and,
    below 2px, disappears) when a block is too small to show any of its own
    color around a full-width gutter — otherwise, with enough grid cells,
    adjacent gutters would tile the whole canvas and the image would render
    as solid ``gap_color`` instead of a fine chunky grid.
    """
    out_w, out_h = output_size
    grid_h, grid_w = grid_image.shape[:2]

    row_repeats = [len(rows) for rows in _splits(out_h, grid_h)]
    col_repeats = [len(cols) for cols in _splits(out_w, grid_w)]

    out = np.repeat(grid_image, row_repeats, axis=0)
    out = np.repeat(out, col_repeats, axis=1)

    if gap_px > 0:
        min_run = min(_min_positive_run(row_repeats), _min_positive_run(col_repeats))
        effective_gap = min(gap_px, min_run - 1) if min_run > 0 else 0
        if effective_gap > 0:
            out = out.copy()
            fill = np.array(gap_color, dtype=out.dtype)
            half = effective_gap // 2

            for edge in np.cumsum(row_repeats)[:-1]:
                lo, hi = max(0, edge - half), min(out_h, edge + (effective_gap - half))
                out[lo:hi, :] = fill
            for edge in np.cumsum(col_repeats)[:-1]:
                lo, hi = max(0, edge - half), min(out_w, edge + (effective_gap - half))
                out[:, lo:hi] = fill

    return out
