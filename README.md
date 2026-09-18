# Bitmapper

A retro bitmap image filter that simulates 80s/90s/early-2000s pixel-art
aesthetics: chunky pixel grids, low bit-depth color quantization, curated
retro palettes (CGA/EGA/VGA/Game Boy), and classic dithering.

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
    bit_depth=4,                # 2, 4, 8, 16, or 32
    block_sampling="average",   # "average" or "nearest"
    palette_mode="auto",        # "auto" or "fixed"
    palette_algorithm="median_cut",  # "median_cut" or "kmeans" (auto mode)
    fixed_palette="ega",        # "cga" | "ega" | "vga256" | "gameboy" (fixed mode)
    dither="floyd_steinberg",   # "none" | "floyd_steinberg" | "ordered"
)

result = apply_bitmap_filter(source_image_array, config)
result.output   # full-resolution blocky image (H, W, 3) uint8
result.grid     # low-res quantized grid, useful as an export source
result.palette  # colors actually used
```

Note: at `bit_depth` 16 or 32, palette generation and dithering are skipped
(treated as true-color) since a small grid rarely has anywhere near that many
unique colors — the block-sampled colors are passed straight through.

## Architecture

Core filter logic lives in `bitmapper/` as pure NumPy functions with no
notebook/plotting dependencies, so it's fully unit tested (`tests/`) and easy
to port to Dart/Flutter later:

- `grid.py` — downsample an image onto a logical grid (average or nearest
  sampling), and upscale a grid back to a full canvas as flat blocks.
- `palette_gen.py` — automatic palette generation (median cut, k-means).
- `palettes.py` — curated fixed retro palettes (CGA, EGA, VGA-ish 256, Game Boy).
- `quantize.py` — nearest-color mapping onto a palette.
- `dither.py` — Floyd-Steinberg error diffusion and ordered (Bayer) dithering.
- `pipeline.py` — `BitmapFilterConfig` + `apply_bitmap_filter`, wiring the
  above stages together.

## Roadmap

This notebook/Python project is the reference implementation for a future
Flutter app. The algorithms here (box-average downsampling, nearest-neighbor
upscaling, median-cut/k-means quantization, Bayer-matrix ordered dithering,
Floyd-Steinberg diffusion) are standard and documented enough to reimplement
in Dart.
