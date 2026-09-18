import numpy as np
import pytest

from bitmapper.palettes import get_palette, list_palettes

EXPECTED_SIZES = {
    "cga": 4,
    "ega": 16,
    "gameboy": 4,
    "vga256": 256,
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
