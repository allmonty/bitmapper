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
    outline=0.0,                  # 0-1, ink on edges found by outline_method (0 = off)
    outline_method="brightness",  # any name from bitmapper.outline.list_methods()
    outline_ink="darkest",        # any name from bitmapper.outline.list_inks()
    shade_bands=0,                # 0 = off, or 2-8 flat brightness bands (toon)
    despeckle=False,              # replace isolated cells with their neighbours' color
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

`list_palettes()` returns the curated fixed palettes:

- **Systems:** `cga`, `cga_palette0`, `ega`, `vga256`, `windows16`, `mac16`,
  `gameboy`, `gameboy_pocket`, `virtualboy`, `nes`, `master_system`,
  `c64`, `zxspectrum`, `amstrad_cpc`, `appleii`, `msx`, `teletext`,
  `pico8`.
- **Pixel-art palettes:** `db16` (DawnBringer), `sweetie16` (TIC-80),
  `endesga32`.
- **Tones:** `monochrome_green`, `monochrome_amber`, `one_bit`,
  `grayscale16`, `sepia`, `thermal`.

Pass
`custom_palette=[(r, g, b), ...]` with `palette_mode="custom"` for your own.

### Dithering

`bitmapper.dither.list_methods()` returns `none` plus:

- **Error diffusion (10):** `floyd_steinberg`, `atkinson`,
  `jarvis_judice_ninke`, `stucki`, `sierra`, `sierra_two_row`,
  `sierra_lite`, `burkes`, `false_floyd_steinberg` and `simple`.
- **Serpentine error diffusion:** `floyd_steinberg_serpentine`, which
  alternates the scan direction each row to break up diagonal artifacts.
- **Ordered:** `ordered`, `ordered_2x2`, `ordered_8x8` and `ordered_16x16`
  (Bayer), plus `clustered_dot` (halftone).
- **Noise:** `interleaved_gradient_noise` (a deterministic, blue-noise-like
  grain) and `random` (white noise). `dither_strength`
blends toward `"none"` without introducing off-palette colors.

### Outlining

`bitmapper.outline.list_methods()` picks how edges are found:

- `brightness`: a cell is inked when a 4-neighbour is brighter by more
  than the threshold. Simple and fast, but on photos it can pick up noisy
  texture edges as small dots rather than clean lines.
- `color`: the same neighbour comparison, by RGB distance instead of
  brightness, so it also finds edges between hues of similar brightness.
- `sobel`: a 3x3 gradient of the brightness. It finds diagonal and gradual
  edges that `brightness` misses, closing gaps in the line; generally the
  best default for photos.

`bitmapper.outline.list_inks()` picks the ink color: `darkest` (the
palette's darkest color, bold sprite-style lines) or `shaded` (the palette
color closest to a half-brightness version of the outlined pixel, a
softer line that varies with what it outlines).

### Presets

`bitmapper.presets.list_presets()` bundles palette/dither/effect/tone combos
into named looks:

- `gameboy_camera`, `gameboy_pocket`, `virtual_boy`
- `arcade_cabinet`, `crt_terminal`, `vhs`
- `sepia_photo`, `newspaper`, `comic_halftone`, `vaporwave`
- `pico8_game`, `tic80`, `dawnbringer`, `endesga_art`
- `windows98`, `classic_mac`, `macpaint`
- `amstrad_cpc`, `master_system`
- `pixel_art`, `pixel_art_soft`, `pixel_art_rich`, `pixel_art_earthy`,
  `pixel_art_mono`: flat colors with no dithering, meant for a chunky grid.
- `pixel_art_sprite`: the same, with ink outlines (see `outline`).
- `toon`, `toon_pastel`: cel shading, combining `shade_bands`, `despeckle`
  and `outline`.

`preset_columns(name)` gives the grid width a preset is designed for (the
pixel-art ones suggest 48–80), or `None`.

For example:

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
- `toon.py` — cel-shading extras applied to the grid: `apply_shade_bands`
  (flatten luma into 2-8 bands) and `despeckle` (replace isolated cells
  with their most common neighbour).
- `outline.py` — sprite-style ink outlines on the quantized grid, with a
  choice of edge-finding method (`brightness`, `color`, `sobel`) and ink
  style (`darkest`, `shaded`); see [Outlining](#outlining) above.
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

See [docs/FLUTTER_MIGRATION.md](docs/FLUTTER_MIGRATION.md) for the port plan:
where the pixel work should run, a stage-by-stage breakdown, the golden-file
strategy for verifying the Dart port against this implementation, and the
numeric gotchas that stop it being bit-exact.
