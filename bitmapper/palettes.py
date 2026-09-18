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
