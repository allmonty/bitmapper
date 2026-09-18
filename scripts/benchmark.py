"""Rough timing of each pipeline stage at a realistic grid size.

Not a correctness test (see tests/ for that) — a manual tool for spotting
performance regressions and seeing where time actually goes before
reimplementing a stage in Dart. Run with:

    python scripts/benchmark.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from bitmapper import BitmapFilterConfig, apply_bitmap_filter
from bitmapper.dither import list_methods

OUTPUT_SIZE = (1200, 1200)
GRID_SIZE = (150, 150)


def _random_image(size=1200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _time(config: BitmapFilterConfig, image: np.ndarray) -> float:
    t0 = time.perf_counter()
    apply_bitmap_filter(image, config)
    return time.perf_counter() - t0


def main() -> None:
    image = _random_image(OUTPUT_SIZE[0])
    print(f"output_size={OUTPUT_SIZE}, grid_size={GRID_SIZE}\n")

    print("-- dither methods (bit_depth=4, palette_mode=auto) --")
    for method in list_methods():
        config = BitmapFilterConfig(output_size=OUTPUT_SIZE, grid_size=GRID_SIZE, bit_depth=4, dither=method)
        elapsed = _time(config, image)
        print(f"  {method:<22} {elapsed:.3f}s")

    print("\n-- grid size scaling (dither=floyd_steinberg) --")
    for grid in [(50, 50), (100, 100), (150, 150), (300, 300)]:
        config = BitmapFilterConfig(output_size=OUTPUT_SIZE, grid_size=grid, bit_depth=4, dither="floyd_steinberg")
        elapsed = _time(config, image)
        print(f"  grid={grid!s:<12} {elapsed:.3f}s")


if __name__ == "__main__":
    main()
