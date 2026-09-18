"""Curated, era-authentic fixed retro palettes."""
from __future__ import annotations

import numpy as np

# CGA mode 4, palette 1, high intensity: black, cyan, magenta, white.
_CGA = [
    (0, 0, 0),
    (85, 255, 255),
    (255, 85, 255),
    (255, 255, 255),
]

# Classic 16-color EGA/VGA text-mode palette.
_EGA = [
    (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
    (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
    (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
    (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255),
]

# Original Game Boy's 4-shade green LCD palette.
_GAMEBOY = [
    (15, 56, 15),
    (48, 98, 48),
    (139, 172, 15),
    (155, 188, 15),
]

# Commodore 64's fixed 16-color palette (Pepto/VICE reference values).
_C64 = [
    (0, 0, 0),
    (255, 255, 255),
    (136, 0, 0),
    (170, 255, 238),
    (204, 68, 204),
    (0, 204, 85),
    (0, 0, 170),
    (238, 238, 119),
    (221, 136, 85),
    (102, 68, 0),
    (255, 119, 119),
    (51, 51, 51),
    (119, 119, 119),
    (170, 255, 102),
    (0, 136, 255),
    (187, 187, 187),
]

# ZX Spectrum's 8 base + 8 "bright" attribute colors.
_ZX_SPECTRUM = [
    (0, 0, 0), (0, 0, 215), (215, 0, 0), (215, 0, 215),
    (0, 215, 0), (0, 215, 215), (215, 215, 0), (215, 215, 215),
    (0, 0, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255),
    (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 255, 255),
]

# PICO-8 fantasy console's official 16-color palette.
_PICO8 = [
    (0, 0, 0),
    (29, 43, 83),
    (126, 37, 83),
    (0, 135, 81),
    (171, 82, 54),
    (95, 87, 79),
    (194, 195, 199),
    (255, 241, 232),
    (255, 0, 77),
    (255, 163, 0),
    (255, 236, 39),
    (0, 228, 54),
    (41, 173, 255),
    (131, 118, 156),
    (255, 119, 168),
    (255, 204, 170),
]


def _vga256() -> np.ndarray:
    """Approximate the 256-color VGA palette with a 6x6x6 color cube (216
    colors) plus a 40-step grayscale ramp, the same layout the classic "web
    safe" / VGA 8-bit palettes are built from.
    """
    levels = [0, 51, 102, 153, 204, 255]
    cube = np.array([(r, g, b) for r in levels for g in levels for b in levels], dtype=np.uint8)
    grays = np.linspace(0, 255, 40).astype(np.uint8)
    grayscale = np.stack([grays, grays, grays], axis=1)
    return np.vstack([cube, grayscale])


_PALETTES = {
    "cga": np.array(_CGA, dtype=np.uint8),
    "ega": np.array(_EGA, dtype=np.uint8),
    "gameboy": np.array(_GAMEBOY, dtype=np.uint8),
    "vga256": _vga256(),
    "c64": np.array(_C64, dtype=np.uint8),
    "zxspectrum": np.array(_ZX_SPECTRUM, dtype=np.uint8),
    "pico8": np.array(_PICO8, dtype=np.uint8),
}


def list_palettes() -> list[str]:
    return sorted(_PALETTES.keys())


def get_palette(name: str) -> np.ndarray:
    try:
        return _PALETTES[name].copy()
    except KeyError as exc:
        raise ValueError(
            f"unknown fixed palette {name!r}, available: {list_palettes()}"
        ) from exc
