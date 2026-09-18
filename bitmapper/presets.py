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
}


def list_presets() -> list[str]:
    return sorted(_PRESETS.keys())


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
