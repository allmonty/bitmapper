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
python scripts/benchmark.py     # rough per-stage timing, not a correctness check
```

There is no lint/format/build step configured beyond `pytest`.

## Architecture

The filter is a pipeline of pure-function stages, each in its own module under `bitmapper/`, wired together by `pipeline.py`:

1. **`grid.py`** — downsamples the source image onto a logical grid (average or nearest-neighbor sampling), and upscales a grid back to a full canvas as flat blocks. `upscale()` also takes `gap_px`/`gap_color` to draw a gutter at block boundaries, reusing the same boundary computation as block replication so gutters land exactly on block edges.
2. **`palette_gen.py`** — automatic palette generation from the grid's colors (median cut, k-means), used in `palette_mode="auto"`.
3. **`palettes.py`** — curated, era-authentic fixed palettes (14: CGA, EGA, VGA-ish 256-color, Game Boy, C64, ZX Spectrum, PICO-8, NES, Apple II, MSX, Teletext, green/amber monochrome, sepia), used in `palette_mode="fixed"`. `list_palettes()` / `get_palette(name)` are the public API; add new palettes to the `_PALETTES` dict. `subsample(palette, n_colors)` picks evenly-spaced entries to cap a palette to a smaller color budget — used by the pipeline so `bit_depth` limits fixed/custom palettes too, not just auto-generated ones.
4. **`quantize.py`** — nearest-color mapping of grid pixels onto a palette. `nearest_index(pixels, palette)` is the single implementation of "closest palette entry" (ties go to the lowest index) and is used by both `nearest_color` and `palette_gen.kmeans`. It processes pixels in chunks bounded by `_MAX_DISTANCE_ELEMENTS` because the distance computation's natural `(pixels, palette, channels)` intermediate is otherwise hundreds of MB on a large grid against a big palette. Chunk size is purely a memory knob — results are identical at any granularity, which `tests/test_quantize.py` pins.
5. **`dither.py`** — dithering applied before/during quantization. Error-diffusion methods (Floyd-Steinberg, Atkinson, Jarvis-Judice-Ninke, Stucki, Sierra, Sierra Lite, Burkes) all run through one `error_diffusion(image, palette, method, strength)` entry point over the `_DIFFUSION_KERNELS` table — **adding a kernel to that table is all it takes** to register a new method, since `_METHODS` is derived from it (and `list_methods()` from that, so the parametrized tests and config validation pick it up automatically). Ordered (Bayer) dithering supports any power-of-two matrix size via a recursively-built Bayer matrix (`ordered`, `ordered_2x2`, `ordered_8x8` are exposed). `random` does white-noise dithering. Every method takes a `strength` (0-1+) that scales the diffused-error/threshold-perturbation magnitude — 0 behaves exactly like `"none"`. `apply()` dispatches to `_METHODS[method]` passing `strength` **as a keyword** — passing it positionally is a trap, since `ordered()`'s third positional parameter is `matrix_size`, not `strength` (this bug shipped once; caught by the strength tests).
6. **`adjustments.py`** — contrast/saturation/gamma pre-adjustments (`apply()`) run on the canvas before downsampling, so they affect what the palette/dither stages see.
7. **`effects.py`** — post-quantization effects applied to the full-resolution output, currently `apply_scanlines`.
8. **`presets.py`** — named bundles of style-related config (palette, dither, effects, tone) for common looks. `get_preset(name, **overrides)` merges caller overrides (typically `output_size`/`grid_size`, which are per-image and never baked into a preset) on top of the preset dict, mirroring `palettes.py`'s list/get pattern.
9. **`pipeline.py`** — `BitmapFilterConfig` (a dataclass that validates itself in `__post_init__`) and `apply_bitmap_filter`, which runs canvas resize → adjustments → downsample → palette/dither → upscale (+ grid gap) → scanlines, and returns a `FilterResult` (`output`: full-res blocky image, `grid`: low-res quantized grid, `palette`: colors actually used). `_resolve_palette` picks the palette for the configured mode and returns `None` for the true-color path, keeping the dither call in one place rather than once per palette mode.

At `bit_depth` 16+ (`_TRUE_COLOR_THRESHOLD`), palette generation and dithering are skipped entirely — the grid is treated as true-color and block-sampled colors pass straight through, since a small grid rarely has anywhere near that many unique colors. Fixed and custom palettes remain available at any bit depth (subsampled to the `bit_depth` budget rather than skipped).

The package's public API (`bitmapper/__init__.py`) is intentionally small: `BitmapFilterConfig`, `FilterResult`, `apply_bitmap_filter`, `list_palettes`, `list_presets`, `get_preset`.

## Testing conventions

- Shared image fixtures (`solid_image`, `checkerboard_image`, `gradient_image`) live in `tests/conftest.py`.
- Enumerable things (palettes, dither methods, presets) are tested via parametrization over the module's own listing function (`list_palettes()`, `list_dither_methods()`, `list_presets()`) rather than a hardcoded list in the test file, so new entries get test coverage automatically once registered.
- When adding a new palette, dither method, or preset, extend the corresponding parametrized test (`tests/test_palettes.py`, `tests/test_dither.py`, `tests/test_presets.py`) in the same change — don't rely solely on the pipeline-level smoke test.

## Iterating visually

Open `notebooks/01_bitmapper_playground.ipynb` and run all cells — it loads a sample image (or generates a synthetic one) and lets you compare different `BitmapFilterConfig` combinations side by side. Every comparison cell calls a shared `plot_grid(items, config_fn, title_fn)` helper (defined near the top of the notebook) that renders one output per item in a 3-column grid — add a new comparison by calling it, not by hand-rolling `plt.subplots` again. This notebook (plus `matplotlib`/`jupyter`) is the only place plotting dependencies are used; keep them out of `bitmapper/`.

After editing the notebook, re-execute it end-to-end and check for errors rather than trusting unexecuted cells:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_bitmapper_playground.ipynb
```

## Performance and the Dart port

`scripts/benchmark.py` times each stage; run it after touching a hot path. As of writing (1200x1200 output, 150x150 grid): `grid.downsample`/`upscale`, `quantize.nearest_color`, `adjustments`, `effects`, and the ordered/random dither methods are all vectorized NumPy and take well under 100ms. The error-diffusion methods (Floyd-Steinberg and its 6 relatives) are the one slow stage — 200-450ms at this size, scaling roughly with `grid pixels × kernel taps`, because each pixel's quantized color depends on the already-diffused error of its neighbors, so it can't be vectorized across the whole grid the way the other stages are; it runs as a genuine nested Python loop. `_error_diffusion` folds each kernel tap's `weight * strength / divisor` once per call (not per pixel) to keep that loop as cheap as reasonably possible without changing behavior.

`docs/FLUTTER_MIGRATION.md` holds the port plan and, more importantly, the list of places where Python and Dart semantics diverge (resampling filters, `np.array_split` block boundaries, uint8 truncation, NumPy's banker's rounding, RNG reproducibility). Read it before changing anything in those areas — several look like harmless cleanups but are part of a numeric contract.

**Don't reach for numba/Cython to fix this.** The slowness is a Python-interpreter artifact, not an algorithmic one — the same nested loop in Dart (AOT-compiled) runs at native speed, so this isn't a problem to engineer around in the reference implementation, just to be aware of when comparing notebook timings to what the Flutter app will feel like. Optimizing further here would also pull the reference implementation away from the plain, directly-portable form that's the whole point of keeping `bitmapper/` pure NumPy. If the notebook feels sluggish, prefer a smaller `grid_size` for iteration (see the "Grid resolution" cell) over adding a Python-specific speedup.

When porting a stage to Dart: the vectorized NumPy stages (grid, quantize, ordered/random dither, adjustments, effects) need to be re-expressed as explicit loops over typed arrays (`Uint8List`/`Float64List`) — Dart has no NumPy — so budget for that even though they're "fast" here. The error-diffusion stages, by contrast, translate close to line-for-line since they're already a nested loop.
