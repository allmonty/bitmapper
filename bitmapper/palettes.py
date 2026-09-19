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


# CGA mode 4, palette 0, high intensity: black, green, red, yellow.
_CGA_PALETTE0 = [
    (0, 0, 0),
    (85, 255, 85),
    (255, 85, 85),
    (255, 255, 85),
]

# Windows 3.x / 95 / 98 default 16-color (VGA) system palette.
_WINDOWS16 = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
    (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
]

# Classic Macintosh (Mac II) 16-color system palette.
_MAC16 = _from_hex(
    [
        "FFFFFF", "FCF400", "FF6400", "DD0202", "F00285", "4600A5", "0000D5", "00AEE9",
        "1AB90C", "006407", "572800", "917035", "C1C1C1", "818181", "3E3E3E", "000000",
    ]
)

# Game Boy Pocket's 4-shade gray LCD.
_GAMEBOY_POCKET = _from_hex(["1F1F1F", "4D533C", "8B956D", "C4CFA1"])

# Virtual Boy's red-on-black display.
_VIRTUALBOY = _from_hex(["000000", "550000", "AA0000", "FF0000"])

# DawnBringer's 16-color palette (DB16).
_DB16 = _from_hex(
    [
        "140C1C", "442434", "30346D", "4E4A4E", "854C30", "346524", "D04648", "757161",
        "597DCE", "D27D2C", "8595A1", "6DAA2C", "D2AA99", "6DC2CA", "DAD45E", "DEEED6",
    ]
)

# Sweetie 16, the TIC-80 fantasy console's default palette (by GrafxKid).
_SWEETIE16 = _from_hex(
    [
        "1A1C2C", "5D275D", "B13E53", "EF7D57", "FFCD75", "A7F070", "38B764", "257179",
        "29366F", "3B5DC9", "41A6F6", "73EFF7", "F4F4F4", "94B0C2", "566C86", "333C57",
    ]
)

# Endesga 32 (by ENDESGA).
_ENDESGA32 = _from_hex(
    [
        "BE4A2F", "D77643", "EAD4AA", "E4A672", "B86F50", "733E39", "3E2731", "A22633",
        "E43B44", "F77622", "FEAE34", "FEE761", "63C74D", "3E8948", "265C42", "193C3E",
        "124E89", "0099DB", "2CE8F5", "FFFFFF", "C0CBDC", "8B9BB4", "5A6988", "3A4466",
        "262B44", "181425", "FF0044", "68386C", "B55088", "F6757A", "E8B796", "C28569",
    ]
)

# Pure 1-bit: black and white.
_ONE_BIT = [(0, 0, 0), (255, 255, 255)]


def _rgb_levels(levels: list[int]) -> np.ndarray:
    """Every RGB combination of ``levels`` (r-major, then g, then b)."""
    return np.array([(r, g, b) for r in levels for g in levels for b in levels], dtype=np.uint8)


def _grayscale(steps: int = 16) -> np.ndarray:
    """An even black-to-white ramp (truncated like every uint8 cast here)."""
    tones = np.linspace(0, 255, steps).astype(np.uint8)
    return np.stack([tones, tones, tones], axis=1)


def _thermal(steps: int = 16) -> np.ndarray:
    """A thermal-camera ramp: black, purple, red, orange, yellow, white,
    linearly interpolated between those stops."""
    stops = np.array(
        [(0, 0, 0), (80, 0, 140), (200, 0, 60), (255, 110, 0), (255, 220, 0), (255, 255, 255)],
        dtype=np.float64,
    )
    positions = np.linspace(0, len(stops) - 1, steps)
    lower = np.floor(positions).astype(int).clip(0, len(stops) - 2)
    frac = (positions - lower)[:, None]
    ramp = stops[lower] + (stops[lower + 1] - stops[lower]) * frac
    return np.clip(ramp, 0, 255).astype(np.uint8)


# Register a palette by adding its RGB tuples here; the uint8 conversion is
# applied once below, so entries can be plain lists or computed arrays.
_PALETTES = {
    name: np.asarray(colors, dtype=np.uint8)
    for name, colors in {
        "cga": _CGA,
        "ega": _EGA,
        "gameboy": _GAMEBOY,
        "vga256": _vga256(),
        "c64": _C64,
        "zxspectrum": _ZX_SPECTRUM,
        "pico8": _PICO8,
        "nes": _NES,
        "appleii": _APPLEII,
        "msx": _MSX,
        "teletext": _TELETEXT,
        "monochrome_green": _MONOCHROME_GREEN,
        "monochrome_amber": _MONOCHROME_AMBER,
        "sepia": _sepia(),
        "cga_palette0": _CGA_PALETTE0,
        "windows16": _WINDOWS16,
        "mac16": _MAC16,
        "gameboy_pocket": _GAMEBOY_POCKET,
        "virtualboy": _VIRTUALBOY,
        # Amstrad CPC hardware palette: 3 levels per channel.
        "amstrad_cpc": _rgb_levels([0, 128, 255]),
        # Sega Master System: 2 bits per channel.
        "master_system": _rgb_levels([0, 85, 170, 255]),
        "db16": _DB16,
        "sweetie16": _SWEETIE16,
        "endesga32": _ENDESGA32,
        "one_bit": _ONE_BIT,
        "grayscale16": _grayscale(),
        "thermal": _thermal(),
    }.items()
}


def subsample(palette: np.ndarray, n_colors: int) -> np.ndarray:
    """Pick ``n_colors`` evenly-spaced entries from ``palette`` (keeping the
    first and last), for capping a fixed palette to a smaller color budget.
    Returns ``palette`` unchanged (copied) if it already has <= ``n_colors``.
    """
    if n_colors < 1:
        raise ValueError(f"n_colors must be >= 1, got {n_colors}")
    if n_colors >= len(palette):
        return palette.copy()
    idx = np.linspace(0, len(palette) - 1, n_colors).round().astype(int)
    return palette[idx].copy()


def list_palettes() -> list[str]:
    return sorted(_PALETTES.keys())


def get_palette(name: str) -> np.ndarray:
    try:
        return _PALETTES[name].copy()
    except KeyError as exc:
        raise ValueError(
            f"unknown fixed palette {name!r}, available: {list_palettes()}"
        ) from exc
