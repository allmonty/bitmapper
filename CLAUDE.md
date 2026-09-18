# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A retro bitmap image filter that simulates 80s/90s/early-2000s pixel-art aesthetics: chunky pixel grids, low bit-depth color quantization, curated retro palettes, and classic dithering. It's a reference implementation for a future Flutter/Dart port — the algorithms (box-average downsampling, nearest-neighbor upscaling, median-cut/k-means quantization, Bayer ordered dithering, error-diffusion dithering) are standard and documented enough to reimplement in Dart, so `bitmapper/` is kept as pure NumPy with no notebook/plotting dependencies.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Commands

```bash
pytest -v                       # run the full test suite
pytest tests/test_dither.py -v  # run a single test file
pytest tests/test_dither.py::test_apply_rejects_unknown_method -v  # run a single test
```

There is no lint/format/build step configured beyond `pytest`.

## Architecture

The filter is a pipeline of pure-function stages, each in its own module under `bitmapper/`, wired together by `pipeline.py`:

1. **`grid.py`** — downsamples the source image onto a logical grid (average or nearest-neighbor sampling), and upscales a grid back to a full canvas as flat blocks.
2. **`palette_gen.py`** — automatic palette generation from the grid's colors (median cut, k-means), used in `palette_mode="auto"`.
3. **`palettes.py`** — curated, era-authentic fixed palettes (CGA, EGA, VGA-ish 256-color, Game Boy, C64, ZX Spectrum, PICO-8), used in `palette_mode="fixed"`. `list_palettes()` / `get_palette(name)` are the public API; add new palettes to the `_PALETTES` dict.
4. **`quantize.py`** — nearest-color mapping of grid pixels onto a palette.
5. **`dither.py`** — dithering applied before/during quantization. Error-diffusion methods (Floyd-Steinberg, Atkinson, Jarvis-Judice-Ninke, Stucki, Sierra, Sierra Lite, Burkes) share a generic kernel-based `_error_diffusion` helper — add a new one by adding its kernel/divisor to `_DIFFUSION_KERNELS` rather than writing a new pixel loop. Ordered (Bayer) dithering supports any power-of-two matrix size via a recursively-built Bayer matrix. `list_methods()` is the source of truth for valid `dither` config values — `pipeline.py` validates against it, so new methods registered in `_METHODS` are automatically valid config and automatically covered by the parametrized pipeline test.
6. **`pipeline.py`** — `BitmapFilterConfig` (a dataclass that validates itself in `__post_init__`) and `apply_bitmap_filter`, which runs stages 1→4/5 in order and returns a `FilterResult` (`output`: full-res blocky image, `grid`: low-res quantized grid, `palette`: colors actually used).

At `bit_depth` 16+ (`_TRUE_COLOR_THRESHOLD`), palette generation and dithering are skipped entirely — the grid is treated as true-color and block-sampled colors pass straight through, since a small grid rarely has anywhere near that many unique colors. Fixed palettes remain available at any bit depth.

The package's public API (`bitmapper/__init__.py`) is intentionally small: `BitmapFilterConfig`, `FilterResult`, `apply_bitmap_filter`, `list_palettes`.

## Testing conventions

- Shared image fixtures (`solid_image`, `checkerboard_image`, `gradient_image`) live in `tests/conftest.py`.
- Enumerable things (palettes, dither methods) are tested via parametrization over the module's own listing function (`list_palettes()`, `list_dither_methods()`) rather than a hardcoded list in the test file, so new entries get test coverage automatically once registered.
- When adding a new palette or dither method, extend the corresponding parametrized test (`tests/test_palettes.py`, `tests/test_dither.py`) in the same change — don't rely solely on the pipeline-level smoke test.

## Iterating visually

Open `notebooks/01_bitmapper_playground.ipynb` and run all cells — it loads a sample image (or generates a synthetic one) and lets you compare different `BitmapFilterConfig` combinations side by side. This notebook (plus `matplotlib`/`jupyter`) is the only place plotting dependencies are used; keep them out of `bitmapper/`.
