import numpy as np
import pytest

from bitmapper.outline import apply_outline, darkest_color, outline_threshold
from bitmapper.pipeline import BitmapFilterConfig, apply_bitmap_filter
from conftest import random_image

GRAY, WHITE, BLACK = (128, 128, 128), (255, 255, 255), (0, 0, 0)
PALETTE = np.array([WHITE, GRAY, BLACK], dtype=np.uint8)


def _square():
    """7x7 gray grid with a white 3x3 square in the middle."""
    grid = np.full((7, 7, 3), GRAY, dtype=np.uint8)
    grid[2:5, 2:5] = WHITE
    return grid


def test_strength_zero_is_a_no_op():
    grid = _square()
    assert apply_outline(grid, PALETTE, 0.0) is grid


def test_inks_the_dark_side_of_strong_edges_one_cell_thick():
    out = apply_outline(_square(), PALETTE, 0.5)
    inked = (out == BLACK).all(axis=2)
    expected = np.zeros((7, 7), dtype=bool)
    # Gray cells 4-adjacent to the white square (not the corners).
    expected[1, 2:5] = expected[5, 2:5] = True
    expected[2:5, 1] = expected[2:5, 5] = True
    np.testing.assert_array_equal(inked, expected)
    # The bright square itself is untouched.
    assert (out[2:5, 2:5] == WHITE).all()


def test_weak_edges_need_more_strength():
    grid = np.full((3, 3, 3), GRAY, dtype=np.uint8)
    grid[1, 1] = (178, 178, 178)  # 50 brighter than its neighbours
    assert (apply_outline(grid, PALETTE, 0.5) == grid).all()  # threshold 72
    out = apply_outline(grid, PALETTE, 1.0)  # threshold 16
    assert (out[0, 1] == BLACK).all() and (out[1, 1] == (178, 178, 178)).all()


def test_threshold_range():
    assert outline_threshold(0.0) == 128.0
    assert outline_threshold(1.0) == 16.0


def test_darkest_color_takes_the_first_on_ties():
    palette = np.array([WHITE, (10, 10, 10), (10, 10, 10), GRAY], dtype=np.uint8)
    assert darkest_color(palette).tolist() == [10, 10, 10]
    assert darkest_color(palette[:1]).tolist() == list(WHITE)


def test_rejects_out_of_range_strength():
    with pytest.raises(ValueError):
        apply_outline(_square(), PALETTE, 1.5)
    with pytest.raises(ValueError):
        BitmapFilterConfig(outline=-0.1)


@pytest.mark.parametrize("mode", ["auto", "fixed", "true_color"])
def test_pipeline_output_stays_within_the_palette(mode):
    kwargs = dict(output_size=(40, 40), grid_size=(10, 10), outline=0.6)
    if mode == "fixed":
        kwargs.update(palette_mode="fixed", fixed_palette="pico8", bit_depth=4)
    elif mode == "true_color":
        kwargs.update(bit_depth=24)
    else:
        kwargs.update(bit_depth=3)
    result = apply_bitmap_filter(random_image(), BitmapFilterConfig(**kwargs))
    palette = {tuple(c) for c in result.palette.tolist()}
    assert {tuple(c) for c in result.grid.reshape(-1, 3).tolist()} <= palette


def test_matches_the_dart_port():
    # Shared expected values with bitmapper-app/packages/bitmapper_core/test/outline_test.dart.
    grid = (np.arange(6 * 5 * 3) * 37 % 256).astype(np.uint8).reshape(5, 6, 3)
    palette = np.array([(200, 30, 30), (5, 60, 90), (240, 240, 240), (12, 40, 20)], dtype=np.uint8)
    out = apply_outline(grid, palette, 0.35)
    assert out.reshape(-1).tolist() == EXPECTED_SHARED


EXPECTED_SHARED = [12, 40, 20, 111, 148, 185, 222, 3, 40, 77, 114, 151, 188, 225, 6, 12, 40, 20, 154, 191, 228, 12, 40, 20, 120, 157, 194, 231, 12, 49, 86, 123, 160, 197, 234, 15, 12, 40, 20, 163, 200, 237, 12, 40, 20, 129, 166, 203, 240, 21, 58, 95, 132, 169, 206, 243, 24, 12, 40, 20, 172, 209, 246, 12, 40, 20, 138, 175, 212, 249, 30, 67, 104, 141, 178, 215, 252, 33, 12, 40, 20, 181, 218, 255, 12, 40, 20, 147, 184, 221]
