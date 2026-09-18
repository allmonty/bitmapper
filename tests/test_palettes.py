import numpy as np
import pytest

from bitmapper.palettes import get_palette, list_palettes, subsample

EXPECTED_SIZES = {
    "cga": 4,
    "ega": 16,
    "gameboy": 4,
    "vga256": 256,
    "c64": 16,
    "zxspectrum": 16,
    "pico8": 16,
    "nes": 64,
    "appleii": 16,
    "msx": 16,
    "teletext": 8,
    "monochrome_green": 4,
    "monochrome_amber": 4,
    "sepia": 32,
}


def test_list_palettes_matches_expected_names():
    assert set(list_palettes()) == set(EXPECTED_SIZES)


@pytest.mark.parametrize("name,size", EXPECTED_SIZES.items())
def test_palette_has_expected_color_count_and_valid_rgb(name, size):
    palette = get_palette(name)
    assert palette.shape == (size, 3)
    assert palette.dtype == np.uint8
    assert palette.min() >= 0 and palette.max() <= 255


def test_get_palette_returns_a_copy():
    a = get_palette("cga")
    a[0] = (1, 2, 3)
    b = get_palette("cga")
    assert not np.array_equal(a[0], b[0])


def test_unknown_palette_raises():
    with pytest.raises(ValueError):
        get_palette("does_not_exist")


def test_subsample_smaller_than_palette_picks_first_and_last():
    palette = get_palette("ega")
    out = subsample(palette, 4)
    assert out.shape == (4, 3)
    np.testing.assert_array_equal(out[0], palette[0])
    np.testing.assert_array_equal(out[-1], palette[-1])
    assert {tuple(c) for c in out}.issubset({tuple(c) for c in palette})


def test_subsample_larger_than_palette_returns_unchanged_copy():
    palette = get_palette("cga")
    out = subsample(palette, 100)
    np.testing.assert_array_equal(out, palette)
    out[0] = (1, 2, 3)
    assert not np.array_equal(out[0], get_palette("cga")[0])


def test_subsample_rejects_non_positive_n_colors():
    palette = get_palette("cga")
    with pytest.raises(ValueError):
        subsample(palette, 0)
