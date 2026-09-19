"""Named presets: curated bundles of BitmapFilterConfig overrides for
common retro looks, so callers don't have to hand-tune every knob.

Presets only set style-related fields (palette, dither, effects, tone).
``output_size``/``grid_size`` are left to the caller via ``**overrides``,
since those depend on the source image rather than the look.
"""
from __future__ import annotations

from .pipeline import BitmapFilterConfig

_PRESETS: dict[str, dict] = {
    "gameboy_camera": dict(
        palette_mode="fixed", fixed_palette="gameboy", bit_depth=2,
        dither="ordered",
    ),
    "arcade_cabinet": dict(
        palette_mode="fixed", fixed_palette="cga", bit_depth=2,
        dither="floyd_steinberg", scanlines=0.35, grid_gap_px=1,
    ),
    "crt_terminal": dict(
        palette_mode="fixed", fixed_palette="monochrome_green", bit_depth=2,
        dither="ordered", scanlines=0.4,
    ),
    "vhs": dict(
        palette_mode="auto", bit_depth=5, dither="floyd_steinberg",
        dither_strength=0.6, saturation=1.3, contrast=0.9, scanlines=0.25,
    ),
    "sepia_photo": dict(
        palette_mode="fixed", fixed_palette="sepia", bit_depth=5,
        dither="floyd_steinberg", contrast=1.1,
    ),
    "newspaper": dict(
        palette_mode="custom", custom_palette=[(0, 0, 0), (255, 255, 255)],
        dither="ordered", saturation=0.0, contrast=1.3,
    ),
    "vaporwave": dict(
        palette_mode="custom",
        custom_palette=[(9, 3, 46), (255, 105, 180), (0, 255, 255), (255, 253, 208)],
        dither="floyd_steinberg", saturation=1.4, gamma=1.2,
    ),
    "pico8_game": dict(
        palette_mode="fixed", fixed_palette="pico8", bit_depth=4,
        dither="none", grid_gap_px=1,
    ),
    "windows98": dict(
        palette_mode="fixed", fixed_palette="windows16", bit_depth=4,
        dither="ordered",
    ),
    # Bill Atkinson's dither was the Mac's own.
    "classic_mac": dict(
        palette_mode="fixed", fixed_palette="mac16", bit_depth=4,
        dither="atkinson",
    ),
    "macpaint": dict(
        palette_mode="fixed", fixed_palette="one_bit", bit_depth=1,
        dither="atkinson", contrast=1.1,
    ),
    "comic_halftone": dict(
        palette_mode="custom",
        custom_palette=[(255, 255, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)],
        dither="clustered_dot", dither_strength=0.5, saturation=1.3, gamma=1.2,
    ),
    "virtual_boy": dict(
        palette_mode="fixed", fixed_palette="virtualboy", bit_depth=2,
        dither="ordered", saturation=0.0,
    ),
    "gameboy_pocket": dict(
        palette_mode="fixed", fixed_palette="gameboy_pocket", bit_depth=2,
        dither="ordered",
    ),
    "amstrad_cpc": dict(
        palette_mode="fixed", fixed_palette="amstrad_cpc", bit_depth=5,
        dither="floyd_steinberg", scanlines=0.2,
    ),
    "master_system": dict(
        palette_mode="fixed", fixed_palette="master_system", bit_depth=6,
        dither="sierra_lite",
    ),
    "tic80": dict(
        palette_mode="fixed", fixed_palette="sweetie16", bit_depth=4,
        dither="interleaved_gradient_noise",
    ),
    "dawnbringer": dict(
        palette_mode="fixed", fixed_palette="db16", bit_depth=4,
        dither="floyd_steinberg_serpentine",
    ),
    "endesga_art": dict(
        palette_mode="fixed", fixed_palette="endesga32", bit_depth=5,
        dither="interleaved_gradient_noise", saturation=1.1,
    ),
    # Pixel-art looks: flat colors (no dithering), curated pixel-art
    # palettes, a little extra punch, and a suggested chunky grid (see
    # _PRESET_COLUMNS).
    "pixel_art": dict(
        palette_mode="fixed", fixed_palette="pico8", bit_depth=4,
        dither="none", contrast=1.15, saturation=1.25,
    ),
    "pixel_art_soft": dict(
        palette_mode="fixed", fixed_palette="sweetie16", bit_depth=4,
        dither="none", saturation=1.1,
    ),
    "pixel_art_rich": dict(
        palette_mode="fixed", fixed_palette="endesga32", bit_depth=5,
        dither="none", contrast=1.1, saturation=1.15,
    ),
    "pixel_art_earthy": dict(
        palette_mode="fixed", fixed_palette="db16", bit_depth=4,
        dither="none", contrast=1.1,
    ),
    "pixel_art_mono": dict(
        palette_mode="fixed", fixed_palette="gameboy", bit_depth=2,
        dither="none", contrast=1.3,
    ),
    # Pixel art with sprite-style ink outlines on strong edges.
    "pixel_art_sprite": dict(
        palette_mode="fixed", fixed_palette="pico8", bit_depth=4,
        dither="none", contrast=1.15, saturation=1.25,
        outline=0.4, outline_method="sobel",
    ),
    # Cel shading: flat brightness bands, no stray cells, ink outlines.
    "toon": dict(
        palette_mode="auto", bit_depth=4, dither="none", saturation=1.3,
        shade_bands=3, despeckle=True, outline=0.5, outline_method="sobel",
    ),
    "toon_pastel": dict(
        palette_mode="fixed", fixed_palette="sweetie16", bit_depth=4,
        dither="none", saturation=1.3, shade_bands=4, despeckle=True,
        outline=0.5, outline_method="sobel",
    ),
}

# Grid width (columns) a preset is designed for. The row count depends on
# the image's aspect ratio, so presets suggest columns instead of setting
# ``grid_size``; callers apply it (or not) when building the config.
_PRESET_COLUMNS: dict[str, int] = {
    "pixel_art": 64,
    "pixel_art_soft": 64,
    "pixel_art_rich": 80,
    "pixel_art_earthy": 64,
    "pixel_art_mono": 48,
    "pixel_art_sprite": 64,
    "toon": 96,
    "toon_pastel": 96,
}


def list_presets() -> list[str]:
    return sorted(_PRESETS.keys())


def preset_columns(name: str) -> int | None:
    """The grid width preset ``name`` is designed for, or ``None`` if it
    works at any grid size."""
    if name not in _PRESETS:
        raise ValueError(f"unknown preset {name!r}, available: {list_presets()}")
    return _PRESET_COLUMNS.get(name)


def get_preset(name: str, **overrides) -> BitmapFilterConfig:
    """Build a ``BitmapFilterConfig`` from preset ``name``, layering
    ``overrides`` on top (e.g. ``output_size``, ``grid_size``, or a
    deliberate tweak to one of the preset's own fields).
    """
    try:
        base = _PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}, available: {list_presets()}") from exc
    return BitmapFilterConfig(**{**base, **overrides})
