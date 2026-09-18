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


def _from_hex(colors: list[str]) -> list[tuple[int, int, int]]:
    return [(int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)) for c in colors]


# NES/Famicom 2C02 PPU's 64-entry master palette (including its duplicate
# "unused black" slots, kept for hardware authenticity).
_NES = _from_hex(
    [
        "7C7C7C", "0000FC", "0000BC", "4428BC", "940084", "A80020", "A81000", "881400",
        "503000", "007800", "006800", "005800", "004058", "000000", "000000", "000000",
        "BCBCBC", "0078F8", "0058F8", "6844FC", "D800CC", "E40058", "F83800", "E45C10",
        "AC7C00", "00B800", "00A800", "00A844", "008888", "000000", "000000", "000000",
        "F8F8F8", "3CBCFC", "6888FC", "9878F8", "F878F8", "F85898", "F87858", "FCA044",
        "F8B800", "B8F818", "58D854", "58F898", "00E8D8", "787878", "000000", "000000",
        "FCFCFC", "A4E4FC", "B8B8F8", "D8B8F8", "F8B8F8", "F8A4C0", "F0D0B0", "FCE0A8",
        "F8D878", "D8F878", "B8F8B8", "B8F8D8", "00FCFC", "F8D8F8", "000000", "000000",
    ]
)

# Apple II lo-res 16-color palette: approximate NTSC composite artifact
# colors (dark/light gray share one RGB value on real hardware).
_APPLEII = _from_hex(
    [
        "000000", "A72B4F", "3B33BF", "FF44FD",
        "007D21", "7E7E7E", "2296F1", "BBB9FF",
        "855300", "FF6A32", "7E7E7E", "FF9AD1",
        "29DC2E", "D9D956", "4DFFC8", "FFFFFF",
    ]
)

# MSX1 (TMS9918 VDP) 16-color fixed palette.
_MSX = _from_hex(
    [
        "000000", "000000", "21C842", "5EDC78",
        "5455ED", "7D76FC", "D4524D", "42EBF5",
        "FC5554", "FF7978", "D4C154", "E6CE80",
        "21B03B", "C95BBA", "CCCCCC", "FFFFFF",
    ]
)

# Teletext's 8 pure-combination foreground colors.
_TELETEXT = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]

# Green and amber phosphor monochrome terminal palettes, 4 intensity levels.
_MONOCHROME_GREEN = [
    (0, 0, 0),
    (0, 68, 0),
    (0, 153, 0),
    (51, 255, 51),
]
_MONOCHROME_AMBER = [
    (0, 0, 0),
    (68, 34, 0),
    (153, 85, 0),
    (255, 176, 0),
]


def _sepia(steps: int = 32) -> np.ndarray:
    """A sepia-toned ramp: apply the classic sepia color matrix to a
    grayscale ramp, the same way ``_vga256``'s grayscale ramp is built.
    """
    tones = np.linspace(0, 255, steps)
    r = np.clip(tones * 1.351, 0, 255)
    g = np.clip(tones * 1.203, 0, 255)
    b = np.clip(tones * 0.937, 0, 255)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


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
    "nes": np.array(_NES, dtype=np.uint8),
    "appleii": np.array(_APPLEII, dtype=np.uint8),
    "msx": np.array(_MSX, dtype=np.uint8),
    "teletext": np.array(_TELETEXT, dtype=np.uint8),
    "monochrome_green": np.array(_MONOCHROME_GREEN, dtype=np.uint8),
    "monochrome_amber": np.array(_MONOCHROME_AMBER, dtype=np.uint8),
    "sepia": _sepia(),
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
