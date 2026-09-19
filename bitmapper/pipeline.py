"""High-level bitmap filter: config + the function that applies it."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import adjustments
from . import grid as gridmod
from . import palette_gen
from . import palettes
from .dither import apply as apply_dither
from .dither import list_methods as list_dither_methods
from .effects import apply_scanlines
from .outline import apply_outline
from .outline import list_inks as list_outline_inks
from .outline import list_methods as list_outline_methods
from .quantize import nearest_color
from .toon import MAX_SHADE_BANDS, MIN_SHADE_BANDS, apply_shade_bands, despeckle

MIN_BIT_DEPTH = 1
MAX_BIT_DEPTH = 24  # 2**24 = 16.7M colors: full precision for 8-bit-per-channel RGB
# 16+ bit color is effectively "true color" for a photo — generating and
# dithering onto a palette of 65536+ colors from a small grid is both
# pointless (the grid rarely has that many unique colors) and expensive, so
# those depths skip palette generation/dithering and pass the block-sampled
# colors straight through. Fixed palettes remain available at any depth.
_TRUE_COLOR_THRESHOLD = 16


@dataclass
class BitmapFilterConfig:
    output_size: tuple[int, int] = (2000, 2000)  # (width, height)
    grid_size: tuple[int, int] = (200, 200)  # (cols, rows)
    bit_depth: int = 8
    block_sampling: str = "average"  # "average" | "nearest"
    palette_mode: str = "auto"  # "auto" | "fixed" | "custom"
    palette_algorithm: str = "median_cut"  # "median_cut" | "kmeans"
    fixed_palette: str | None = None
    custom_palette: np.ndarray | list[tuple[int, int, int]] | None = None
    dither: str = "none"  # see bitmapper.dither.list_methods()
    dither_strength: float = 1.0  # 0 = no dithering pattern, 1 = full strength
    scanlines: float = 0.0  # 0 = off, 1 = alternate rows fully black
    grid_gap_px: int = 0  # gutter width between blocks, in output pixels
    outline: float = 0.0  # 0 = off; ink on strong edges (found by outline_method), 1 = most edges
    outline_method: str = "brightness"  # see bitmapper.outline.list_methods()
    outline_ink: str = "darkest"  # see bitmapper.outline.list_inks()
    shade_bands: int = 0  # 0 = off; else 2-8 flat brightness bands (toon shading)
    despeckle: bool = False  # replace isolated cells with their neighbours' color
    grid_gap_color: tuple[int, int, int] = (0, 0, 0)
    contrast: float = 1.0  # 1.0 = no-op, >1 boosts, <1 flattens, 0 = flat gray
    saturation: float = 1.0  # 1.0 = no-op, >1 boosts, 0 = grayscale
    gamma: float = 1.0  # 1.0 = no-op, >1 brightens midtones, <1 darkens them

    def __post_init__(self) -> None:
        if not (MIN_BIT_DEPTH <= self.bit_depth <= MAX_BIT_DEPTH):
            raise ValueError(
                f"bit_depth must be between {MIN_BIT_DEPTH} and {MAX_BIT_DEPTH}, got {self.bit_depth}"
            )
        if self.block_sampling not in ("average", "nearest"):
            raise ValueError(f"invalid block_sampling: {self.block_sampling!r}")
        if self.palette_mode not in ("auto", "fixed", "custom"):
            raise ValueError(f"invalid palette_mode: {self.palette_mode!r}")
        if self.dither not in list_dither_methods():
            raise ValueError(f"invalid dither: {self.dither!r}")
        if self.dither_strength < 0:
            raise ValueError(f"dither_strength must be >= 0, got {self.dither_strength}")
        if not (0.0 <= self.scanlines <= 1.0):
            raise ValueError(f"scanlines must be between 0 and 1, got {self.scanlines}")
        if not (0.0 <= self.outline <= 1.0):
            raise ValueError(f"outline must be between 0 and 1, got {self.outline}")
        if self.outline_method not in list_outline_methods():
            raise ValueError(f"invalid outline_method: {self.outline_method!r}")
        if self.outline_ink not in list_outline_inks():
            raise ValueError(f"invalid outline_ink: {self.outline_ink!r}")
        if self.shade_bands != 0 and not (MIN_SHADE_BANDS <= self.shade_bands <= MAX_SHADE_BANDS):
            raise ValueError(
                f"shade_bands must be 0 or {MIN_SHADE_BANDS}..{MAX_SHADE_BANDS}, got {self.shade_bands}"
            )
        if self.grid_gap_px < 0:
            raise ValueError(f"grid_gap_px must be >= 0, got {self.grid_gap_px}")
        if self.contrast < 0:
            raise ValueError(f"contrast must be >= 0, got {self.contrast}")
        if self.saturation < 0:
            raise ValueError(f"saturation must be >= 0, got {self.saturation}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}")
        if self.palette_mode == "fixed" and not self.fixed_palette:
            raise ValueError("fixed_palette must be set when palette_mode='fixed'")
        if self.palette_mode == "custom":
            if self.custom_palette is None or len(self.custom_palette) == 0:
                raise ValueError("custom_palette must be set when palette_mode='custom'")
            arr = np.asarray(self.custom_palette)
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"custom_palette must be an (N, 3) array of RGB colors, got shape {arr.shape}")

    @property
    def n_colors(self) -> int:
        return 2 ** self.bit_depth


@dataclass
class FilterResult:
    output: np.ndarray  # full canvas, output_size
    grid: np.ndarray  # low-res quantized grid, grid_size
    palette: np.ndarray  # colors actually used


def _resize_to_canvas(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    if image.shape[1::-1] == tuple(output_size):
        return image
    pil_img = Image.fromarray(image).resize(output_size, Image.Resampling.LANCZOS)
    return np.array(pil_img)


def _resolve_palette(grid_colors: np.ndarray, config: BitmapFilterConfig) -> np.ndarray | None:
    """The palette to quantize ``grid_colors`` onto, or ``None`` for the
    true-color path (no palette, no dithering — colors pass through).
    """
    if config.palette_mode == "fixed":
        return palettes.subsample(palettes.get_palette(config.fixed_palette), config.n_colors)
    if config.palette_mode == "custom":
        custom = np.asarray(config.custom_palette, dtype=np.uint8)
        return palettes.subsample(custom, config.n_colors)
    if config.bit_depth >= _TRUE_COLOR_THRESHOLD:
        return None
    return palette_gen.generate_palette(grid_colors, config.n_colors, config.palette_algorithm)


def apply_bitmap_filter(image: np.ndarray, config: BitmapFilterConfig) -> FilterResult:
    """Apply the full retro-bitmap pipeline to an RGB ``image`` array."""
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image must be an (H, W, 3) or (H, W, 4) array")
    if image.shape[2] == 4:
        image = image[:, :, :3]

    canvas = _resize_to_canvas(image, config.output_size)
    canvas = adjustments.apply(canvas, config.contrast, config.saturation, config.gamma)
    grid_colors = gridmod.downsample(canvas, config.grid_size, mode=config.block_sampling)
    grid_colors = apply_shade_bands(grid_colors, config.shade_bands)

    palette = _resolve_palette(grid_colors, config)
    pre_dither_grid = None
    if palette is None:
        palette = np.unique(grid_colors.reshape(-1, 3), axis=0)
        quantized_grid = grid_colors
    else:
        if config.outline > 0.0:
            pre_dither_grid, _ = nearest_color(grid_colors, palette)
        quantized_grid = apply_dither(grid_colors, palette, config.dither, config.dither_strength)

    if config.despeckle:
        quantized_grid = despeckle(quantized_grid)
    if config.outline > 0.0:
        quantized_grid = apply_outline(
            quantized_grid,
            palette,
            config.outline,
            config.outline_method,
            config.outline_ink,
            edge_grid=pre_dither_grid,
        )

    output = gridmod.upscale(
        quantized_grid, config.output_size, gap_px=config.grid_gap_px, gap_color=config.grid_gap_color
    )
    if config.scanlines > 0.0:
        output = apply_scanlines(output, config.scanlines)
    return FilterResult(output=output, grid=quantized_grid, palette=palette)
