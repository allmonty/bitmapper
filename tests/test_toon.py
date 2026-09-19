import numpy as np
import pytest

from bitmapper.pipeline import BitmapFilterConfig, apply_bitmap_filter
from bitmapper.presets import get_preset, preset_columns
from bitmapper.toon import apply_shade_bands, despeckle
from conftest import random_image


# --- Shade bands -----------------------------------------------------------

def _gray_ramp():
    v = np.arange(256, dtype=np.uint8)
    return np.stack([v, v, v], axis=1).reshape(16, 16, 3)


def test_zero_bands_is_a_no_op():
    grid = random_image(8)
    assert apply_shade_bands(grid, 0) is grid


@pytest.mark.parametrize("bands", [2, 3, 4, 8])
def test_a_gray_ramp_collapses_to_exactly_n_band_centres(bands):
    out = apply_shade_bands(_gray_ramp(), bands)
    levels = sorted(set(out[..., 0].reshape(-1).tolist()))
    assert len(levels) == bands
    expected = [int((b + 0.5) * 255.0 / bands) for b in range(bands)]
    assert levels == expected


def test_hue_is_kept():
    grid = np.array([[[200, 100, 50]]], dtype=np.uint8)
    r, g, b = apply_shade_bands(grid, 3)[0, 0].astype(float)
    assert r / g == pytest.approx(2.0, rel=0.02)
    assert g / b == pytest.approx(2.0, rel=0.05)


def test_black_becomes_the_darkest_band_gray():
    out = apply_shade_bands(np.zeros((1, 1, 3), dtype=np.uint8), 4)
    assert out[0, 0].tolist() == [31, 31, 31]  # (0 + 0.5) * 255 / 4


@pytest.mark.parametrize("bands", [1, 9, -1])
def test_rejects_invalid_band_counts(bands):
    with pytest.raises(ValueError):
        apply_shade_bands(_gray_ramp(), bands)
    with pytest.raises(ValueError):
        BitmapFilterConfig(shade_bands=bands)


# --- Despeckle -------------------------------------------------------------

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


def test_isolated_cells_take_the_most_common_neighbour_color():
    grid = np.full((3, 3, 3), GREEN, dtype=np.uint8)
    grid[1, 1] = RED
    out = despeckle(grid)
    assert out[1, 1].tolist() == list(GREEN)
    assert (out == GREEN).all()


def test_cells_with_a_matching_neighbour_are_kept():
    grid = np.full((3, 4, 3), GREEN, dtype=np.uint8)
    grid[1, 1] = grid[1, 2] = RED  # a pair, not a speck
    np.testing.assert_array_equal(despeckle(grid), grid)


def test_ties_go_to_the_first_neighbour_in_reading_order():
    # Centre is isolated; its neighbours are 4 blue (top row + left) and 4 green.
    grid = np.array(
        [[BLUE, BLUE, BLUE], [BLUE, RED, GREEN], [GREEN, GREEN, GREEN]], dtype=np.uint8
    )
    assert despeckle(grid)[1, 1].tolist() == list(BLUE)


def test_despeckle_never_adds_colors():
    grid = random_image(12)
    before = {tuple(c) for c in grid.reshape(-1, 3).tolist()}
    after = {tuple(c) for c in despeckle(grid).reshape(-1, 3).tolist()}
    assert after <= before


def test_a_single_cell_grid_is_unchanged():
    grid = np.array([[RED]], dtype=np.uint8)
    np.testing.assert_array_equal(despeckle(grid), grid)


# --- Pipeline and presets ----------------------------------------------------

@pytest.mark.parametrize("name", ["toon", "toon_pastel"])
def test_toon_presets(name):
    config = get_preset(name, output_size=(48, 48), grid_size=(12, 12))
    assert config.shade_bands > 0 and config.despeckle and config.outline > 0
    assert preset_columns(name) == 96
    result = apply_bitmap_filter(random_image(48), config)
    palette = {tuple(c) for c in result.palette.tolist()}
    assert {tuple(c) for c in result.grid.reshape(-1, 3).tolist()} <= palette


def test_shade_bands_run_before_the_palette_is_built():
    # A smooth gray ramp with 2 bands and a generous budget: the auto
    # palette can only contain the 2 band grays.
    img = np.repeat(_gray_ramp(), 2, axis=0).repeat(2, axis=1)
    config = BitmapFilterConfig(output_size=(32, 32), grid_size=(16, 16), bit_depth=4, shade_bands=2)
    result = apply_bitmap_filter(img, config)
    assert len({tuple(c) for c in result.grid.reshape(-1, 3).tolist()}) <= 2


# --- Shared with the Dart port (bitmapper_core/test/toon_test.dart) ---------

def _shared_grid():
    g = (np.arange(6 * 5 * 3) * 53 % 256).astype(np.uint8).reshape(5, 6, 3)
    g[0, 0] = (0, 0, 0)
    return g


def _shared_speckles():
    d = np.zeros((5, 6, 3), dtype=np.uint8)
    for y in range(5):
        for x in range(6):
            d[y, x] = [RED, GREEN, BLUE][((x * 7 + y * 3) // 5) % 3]
    d[2, 2] = (9, 9, 9)
    return d


def test_shade_bands_match_the_dart_port():
    assert apply_shade_bands(_shared_grid(), 3).reshape(-1).tolist() == [42, 42, 42, 195, 255, 11, 75, 139, 203, 110, 9, 35, 94, 134, 175, 16, 48, 80, 197, 253, 38, 85, 136, 188, 255, 51, 111, 116, 157, 0, 70, 140, 209, 117, 5, 34, 92, 135, 177, 12, 49, 85, 196, 255, 30, 83, 137, 191, 255, 45, 110, 163, 223, 255, 65, 141, 217, 126, 1, 34, 91, 135, 180, 8, 50, 91, 196, 255, 23, 80, 138, 196, 255, 38, 109, 160, 224, 255, 19, 47, 75, 198, 251, 47, 89, 136, 183, 2, 51, 100]
    assert apply_shade_bands(_shared_grid(), 5).reshape(-1).tolist() == [25, 25, 25, 164, 218, 9, 75, 139, 203, 199, 16, 64, 132, 188, 245, 29, 87, 144, 165, 213, 32, 85, 136, 188, 255, 51, 111, 163, 220, 1, 42, 84, 125, 212, 9, 62, 130, 189, 248, 23, 88, 153, 165, 214, 26, 83, 137, 191, 255, 45, 110, 137, 187, 238, 39, 84, 130, 228, 2, 61, 91, 135, 180, 15, 90, 165, 164, 216, 19, 80, 138, 196, 185, 23, 65, 135, 188, 241, 35, 85, 135, 214, 255, 51, 89, 136, 183, 1, 30, 60]


def test_despeckle_matches_the_dart_port():
    assert despeckle(_shared_speckles()).reshape(-1).tolist() == [255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 255, 0, 0, 255, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255]
