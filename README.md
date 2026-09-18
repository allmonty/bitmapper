# Bitmapper

A retro bitmap image filter that simulates 80s/90s/early-2000s pixel-art
aesthetics: chunky pixel grids, low bit-depth color quantization, curated
retro palettes, classic dithering, and CRT-style effects.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest -v
```

## Iterate visually

Open `notebooks/01_bitmapper_playground.ipynb` and run all cells. It loads a
sample image (or generates a synthetic one if you don't have one handy) and
lets you try different `BitmapFilterConfig` combinations side by side.

## Usage

```python
from bitmapper import BitmapFilterConfig, apply_bitmap_filter

config = BitmapFilterConfig(
    output_size=(2000, 2000),   # final canvas size in pixels
    grid_size=(200, 200),       # logical bitmap grid -> 10x10px "pixels"
    bit_depth=4,                # 1-24; also caps fixed/custom palette size
    block_sampling="average",   # "average" or "nearest"
    palette_mode="auto",        # "auto" | "fixed" | "custom"
    palette_algorithm="median_cut",  # "median_cut" or "kmeans" (auto mode)
    fixed_palette="ega",        # any name from list_palettes() (fixed mode)
    custom_palette=None,        # list of (r, g, b) tuples (custom mode)
    dither="floyd_steinberg",   # any name from bitmapper.dither.list_methods()
    dither_strength=1.0,        # 0 = no dither pattern, 1 = full strength
    scanlines=0.0,               # 0-1, darkens alternate output rows
    grid_gap_px=0,                # gutter width between blocks, in output px
    grid_gap_color=(0, 0, 0),
    contrast=1.0,                # around mid-gray; 0 = flat gray
    saturation=1.0,               # 0 = grayscale
    gamma=1.0,
)

result = apply_bitmap_filter(source_image_array, config)
result.output   # full-resolution blocky image (H, W, 3) uint8
result.grid     # low-res quantized grid, useful as an export source
result.palette  # colors actually used
```

Note: at `bit_depth` 16+, palette generation and dithering are skipped
(treated as true-color) since a small grid rarely has anywhere near that many
unique colors — the block-sampled colors are passed straight through. Fixed
and custom palettes remain available (and get capped to `2 ** bit_depth`
colors) at any depth.

### Palettes

`list_palettes()` returns the curated fixed palettes: `cga`, `ega`,
`gameboy`, `vga256`, `c64`, `zxspectrum`, `pico8`, `nes`, `appleii`, `msx`,
`teletext`, `monochrome_green`, `monochrome_amber`, `sepia`. Pass
`custom_palette=[(r, g, b), ...]` with `palette_mode="custom"` for your own.

### Dithering

`bitmapper.dither.list_methods()` returns `none` plus 7 error-diffusion
methods (`floyd_steinberg`, `atkinson`, `jarvis_judice_ninke`, `stucki`,
`sierra`, `sierra_lite`, `burkes`), 3 ordered/Bayer variants (`ordered`,
`ordered_2x2`, `ordered_8x8`), and `random` (white-noise). `dither_strength`
blends toward `"none"` without introducing off-palette colors.

### Presets

`bitmapper.presets.list_presets()` bundles palette/dither/effect/tone combos
into named looks (`gameboy_camera`, `arcade_cabinet`, `crt_terminal`, `vhs`,
`sepia_photo`, `newspaper`, `vaporwave`, `pico8_game`):

```python
from bitmapper import get_preset, apply_bitmap_filter

config = get_preset("vhs", output_size=(1200, 1200), grid_size=(150, 150))
result = apply_bitmap_filter(source_image_array, config)
```

## Architecture

Core filter logic lives in `bitmapper/` as pure NumPy functions with no
notebook/plotting dependencies, so it's fully unit tested (`tests/`) and easy
to port to Dart/Flutter later:

- `grid.py` — downsample an image onto a logical grid (average or nearest
  sampling), and upscale a grid back to a full canvas as flat blocks (with
  an optional gutter drawn at block boundaries).
- `palette_gen.py` — automatic palette generation (median cut, k-means).
- `palettes.py` — curated fixed retro palettes, plus `subsample()` to cap a
  palette to a smaller color budget.
- `quantize.py` — nearest-color mapping onto a palette.
- `dither.py` — error-diffusion (Floyd-Steinberg and 6 relatives sharing a
  kernel-based helper) and ordered/Bayer/random dithering, all with a
  `strength` knob.
- `adjustments.py` — contrast/saturation/gamma pre-adjustments applied to
  the canvas before quantization.
- `effects.py` — post-quantization effects (currently CRT scanlines).
- `presets.py` — named bundles of the above, layered under caller overrides.
- `pipeline.py` — `BitmapFilterConfig` + `apply_bitmap_filter`, wiring the
  above stages together.

## Roadmap

This notebook/Python project is the reference implementation for a future
Flutter app. The algorithms here (box-average downsampling, nearest-neighbor
upscaling, median-cut/k-means quantization, Bayer-matrix ordered dithering,
error-diffusion dithering) are standard and documented enough to reimplement
in Dart.
