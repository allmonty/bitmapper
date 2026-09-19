import numpy as np
import pytest

from bitmapper.dither import apply as apply_dither
from bitmapper.outline import apply_outline, darkest_color, list_inks, list_methods, outline_threshold
from bitmapper.pipeline import BitmapFilterConfig, apply_bitmap_filter
from bitmapper.quantize import nearest_color
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


def test_rejects_invalid_method_or_ink():
    with pytest.raises(ValueError):
        apply_outline(_square(), PALETTE, 0.5, method="nonsense")
    with pytest.raises(ValueError):
        apply_outline(_square(), PALETTE, 0.5, ink="nonsense")
    with pytest.raises(ValueError):
        BitmapFilterConfig(outline_method="nonsense")
    with pytest.raises(ValueError):
        BitmapFilterConfig(outline_ink="nonsense")


@pytest.mark.parametrize("thickness", [0, 4])
def test_rejects_an_out_of_range_thickness(thickness):
    with pytest.raises(ValueError):
        apply_outline(_square(), PALETTE, 0.5, thickness=thickness)
    with pytest.raises(ValueError):
        BitmapFilterConfig(outline_thickness=thickness)


def test_list_methods_and_inks():
    assert list_methods() == ["brightness", "color", "sobel"]
    assert list_inks() == ["darkest", "shaded"]


def test_color_method_also_catches_same_brightness_hue_edges():
    # Red and green cells of identical luminance: brightness sees no edge,
    # color does.
    red, green = (255, 0, 0), (0, 130, 0)
    grid = np.array([[red, green]], dtype=np.uint8)
    palette = np.array([red, green, BLACK], dtype=np.uint8)
    assert (apply_outline(grid, palette, 0.9, method="brightness") == grid).all()
    out = apply_outline(grid, palette, 0.9, method="color")
    assert not (out == grid).all()


def test_sobel_method_also_catches_diagonal_edges():
    grid = np.full((3, 3, 3), GRAY, dtype=np.uint8)
    grid[0, 0] = WHITE  # only a diagonal neighbour of the center
    assert (apply_outline(grid, PALETTE, 0.9, method="brightness")[1, 1] == GRAY).all()
    assert (apply_outline(grid, PALETTE, 0.9, method="sobel")[1, 1] == BLACK).all()


def test_shaded_ink_uses_a_half_brightness_palette_match():
    grid = _square()
    palette = np.array([WHITE, GRAY, (64, 64, 64), BLACK], dtype=np.uint8)
    out = apply_outline(grid, palette, 0.5, ink="shaded")
    # Half of GRAY (128) is 64, which is in the palette, so shaded ink picks
    # it instead of falling back to the darkest color.
    assert (out[1, 2] == (64, 64, 64)).all()


def test_omitting_edge_grid_equals_passing_the_same_grid():
    grid = _square()
    np.testing.assert_array_equal(
        apply_outline(grid, PALETTE, 0.5, method="sobel", edge_grid=grid),
        apply_outline(grid, PALETTE, 0.5, method="sobel"),
    )


def test_a_dither_boundary_invisible_in_edge_grid_is_not_inked():
    # The middle cell's final color is darker than its neighbours (as
    # dither noise would leave it in a flat white region), but its
    # pre-dither mapping agrees with them: it should only be inked
    # (darkened further, to the palette's darkest color) without edge_grid.
    final_grid = np.array([[WHITE, GRAY, WHITE]], dtype=np.uint8)
    pre_dither = np.array([[WHITE, WHITE, WHITE]], dtype=np.uint8)
    without_edge_grid = apply_outline(final_grid, PALETTE, 0.9, method="brightness")
    with_edge_grid = apply_outline(
        final_grid, PALETTE, 0.9, method="brightness", edge_grid=pre_dither
    )
    assert not (without_edge_grid[0, 1] == GRAY).all(), "sees a real jump and inks it"
    assert (with_edge_grid[0, 1] == GRAY).all(), "pre-dither, all three cells agree"


def test_rejects_an_edge_grid_of_a_different_shape():
    with pytest.raises(ValueError):
        apply_outline(_square(), PALETTE, 0.5, edge_grid=np.zeros((1, 1, 3), dtype=np.uint8))


def test_detects_fewer_edges_from_dither_noise_than_from_the_final_grid(gradient_image):
    # A smooth gradient dithered onto a small palette: Floyd-Steinberg
    # scatters color noise across the whole gradient, which the final grid
    # alone can't tell apart from a real edge. The pre-dither grid has far
    # fewer real edges (only at the palette's band boundaries).
    gray4 = np.array([(0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255)], dtype=np.uint8)
    dithered = apply_dither(gradient_image, gray4, "floyd_steinberg")
    pre_dither, _ = nearest_color(gradient_image, gray4)

    def inked_count(out, original):
        return int((out != original).any(axis=-1).sum())

    without_edge_grid = apply_outline(dithered, gray4, 1.0, method="sobel")
    with_edge_grid = apply_outline(dithered, gray4, 1.0, method="sobel", edge_grid=pre_dither)
    noisy_count = inked_count(without_edge_grid, dithered)
    clean_count = inked_count(with_edge_grid, dithered)
    assert clean_count < noisy_count
    # Empirically ~35-40% fewer across several dither methods; leave
    # headroom so the test isn't brittle to small algorithm tweaks.
    assert clean_count < round(noisy_count * 0.75)


@pytest.mark.parametrize("mode", ["auto", "fixed", "true_color", "dithered"])
def test_pipeline_output_stays_within_the_palette(mode):
    kwargs = dict(output_size=(40, 40), grid_size=(10, 10), outline=0.6)
    if mode == "fixed":
        kwargs.update(palette_mode="fixed", fixed_palette="pico8", bit_depth=4)
    elif mode == "true_color":
        kwargs.update(bit_depth=24)
    elif mode == "dithered":
        kwargs.update(bit_depth=3, dither="floyd_steinberg", outline_method="sobel")
    else:
        kwargs.update(bit_depth=3)
    result = apply_bitmap_filter(random_image(), BitmapFilterConfig(**kwargs))
    palette = {tuple(c) for c in result.palette.tolist()}
    assert {tuple(c) for c in result.grid.reshape(-1, 3).tolist()} <= palette


def test_thickness_1_reproduces_the_one_cell_thick_line():
    np.testing.assert_array_equal(
        apply_outline(_square(), PALETTE, 0.5, thickness=1), apply_outline(_square(), PALETTE, 0.5)
    )


def test_thickness_grows_the_line_by_one_cell_per_step_tapering_diagonally():
    # A single inked cell in the middle of an otherwise-uninked field:
    # thickness 2 should grow it to a plus shape (4-neighbours), and
    # thickness 3 should reach the diagonals too (dilated twice).
    grid = np.full((5, 5, 3), WHITE, dtype=np.uint8)
    grid[2, 2] = BLACK
    palette = np.array([WHITE, BLACK], dtype=np.uint8)

    def is_black(out, y, x):
        return out[y, x, 0] == 0

    t1 = apply_outline(grid, palette, 0.5, thickness=1)
    assert not is_black(t1, 2, 1)
    assert is_black(t1, 2, 2)

    t2 = apply_outline(grid, palette, 0.5, thickness=2)
    assert is_black(t2, 2, 1), "plus shape: left neighbour"
    assert is_black(t2, 1, 2), "plus shape: top neighbour"
    assert not is_black(t2, 1, 1), "not yet reached diagonally"

    t3 = apply_outline(grid, palette, 0.5, thickness=3)
    assert is_black(t3, 1, 1), "diagonal, reached after 2 dilation steps"


def test_thickness_matches_the_dart_port():
    # Shared expected values with bitmapper-app/packages/bitmapper_core/test/outline_test.dart.
    grid = (np.arange(6 * 5 * 3) * 37 % 256).astype(np.uint8).reshape(5, 6, 3)
    palette = np.array([(200, 30, 30), (5, 60, 90), (240, 240, 240), (12, 40, 20)], dtype=np.uint8)
    out2 = apply_outline(grid, palette, 0.35, method="sobel", thickness=2)
    assert out2.reshape(-1).tolist() == EXPECTED_SHARED_THICKNESS_2
    out3 = apply_outline(grid, palette, 0.35, method="sobel", thickness=3)
    assert out3.reshape(-1).tolist() == EXPECTED_SHARED_THICKNESS_3


def test_close_gaps_off_reproduces_every_prior_golden():
    np.testing.assert_array_equal(
        apply_outline(_square(), PALETTE, 0.5, close_gaps=False),
        apply_outline(_square(), PALETTE, 0.5),
    )


def _ring_grid():
    """7x7, bright everywhere except a dark 3x3 ring (rows/cols 2-4)
    around a bright center cell (3,3): a single-cell gap in an otherwise
    fully-enclosed dark ring."""
    n = 7
    g = np.full((n, n, 3), WHITE, dtype=np.uint8)
    for y in range(2, 5):
        for x in range(2, 5):
            if not (x == 3 and y == 3):
                g[y, x] = BLACK
    return g


def test_close_gaps_bridges_a_gap_fully_enclosed_by_inked_cells():
    palette = np.array([WHITE, BLACK], dtype=np.uint8)
    without_close = apply_outline(_ring_grid(), palette, 0.5, method="brightness")
    with_close = apply_outline(_ring_grid(), palette, 0.5, method="brightness", close_gaps=True)
    assert tuple(without_close[3, 3]) == WHITE, "the ring has a 1-cell gap at its centre"
    assert tuple(with_close[3, 3]) == BLACK, "closing bridges a fully-enclosed gap"


def test_close_gaps_matches_the_dart_port():
    # Shared expected values with bitmapper-app/packages/bitmapper_core/test/outline_test.dart.
    grid = (np.arange(6 * 5 * 3) * 37 % 256).astype(np.uint8).reshape(5, 6, 3)
    palette = np.array([(200, 30, 30), (5, 60, 90), (240, 240, 240), (12, 40, 20)], dtype=np.uint8)
    out = apply_outline(grid, palette, 0.35, method="brightness", close_gaps=True)
    assert out.reshape(-1).tolist() == EXPECTED_SHARED_CLOSE_GAPS


def test_matches_the_dart_port():
    # Shared expected values with bitmapper-app/packages/bitmapper_core/test/outline_test.dart.
    grid = (np.arange(6 * 5 * 3) * 37 % 256).astype(np.uint8).reshape(5, 6, 3)
    palette = np.array([(200, 30, 30), (5, 60, 90), (240, 240, 240), (12, 40, 20)], dtype=np.uint8)
    out = apply_outline(grid, palette, 0.35)
    assert out.reshape(-1).tolist() == EXPECTED_SHARED


@pytest.mark.parametrize(
    "method,ink,expected_name",
    [
        ("color", "darkest", "EXPECTED_SHARED_COLOR_METHOD"),
        ("sobel", "darkest", "EXPECTED_SHARED_SOBEL_METHOD"),
        ("brightness", "shaded", "EXPECTED_SHARED_SHADED_INK"),
    ],
)
def test_matches_the_dart_port_for_every_method_and_ink(method, ink, expected_name):
    # Shared expected values with bitmapper-app/packages/bitmapper_core/test/outline_test.dart.
    grid = (np.arange(6 * 5 * 3) * 37 % 256).astype(np.uint8).reshape(5, 6, 3)
    palette = np.array([(200, 30, 30), (5, 60, 90), (240, 240, 240), (12, 40, 20)], dtype=np.uint8)
    out = apply_outline(grid, palette, 0.35, method=method, ink=ink)
    assert out.reshape(-1).tolist() == globals()[expected_name]


EXPECTED_SHARED = [12, 40, 20, 111, 148, 185, 222, 3, 40, 77, 114, 151, 188, 225, 6, 12, 40, 20, 154, 191, 228, 12, 40, 20, 120, 157, 194, 231, 12, 49, 86, 123, 160, 197, 234, 15, 12, 40, 20, 163, 200, 237, 12, 40, 20, 129, 166, 203, 240, 21, 58, 95, 132, 169, 206, 243, 24, 12, 40, 20, 172, 209, 246, 12, 40, 20, 138, 175, 212, 249, 30, 67, 104, 141, 178, 215, 252, 33, 12, 40, 20, 181, 218, 255, 12, 40, 20, 147, 184, 221]
EXPECTED_SHARED_COLOR_METHOD = [12, 40, 20, 111, 148, 185, 12, 40, 20, 12, 40, 20, 188, 225, 6, 12, 40, 20, 154, 191, 228, 12, 40, 20, 120, 157, 194, 12, 40, 20, 12, 40, 20, 197, 234, 15, 12, 40, 20, 163, 200, 237, 12, 40, 20, 129, 166, 203, 12, 40, 20, 12, 40, 20, 206, 243, 24, 12, 40, 20, 172, 209, 246, 12, 40, 20, 138, 175, 212, 12, 40, 20, 12, 40, 20, 215, 252, 33, 12, 40, 20, 181, 218, 255, 12, 40, 20, 147, 184, 221]
EXPECTED_SHARED_SOBEL_METHOD = [12, 40, 20, 111, 148, 185, 222, 3, 40, 77, 114, 151, 188, 225, 6, 12, 40, 20, 154, 191, 228, 9, 46, 83, 120, 157, 194, 231, 12, 49, 86, 123, 160, 197, 234, 15, 52, 89, 126, 163, 200, 237, 18, 55, 92, 129, 166, 203, 240, 21, 58, 95, 132, 169, 206, 243, 24, 61, 98, 135, 172, 209, 246, 27, 64, 101, 138, 175, 212, 249, 30, 67, 104, 141, 178, 215, 252, 33, 70, 107, 144, 181, 218, 255, 36, 73, 110, 147, 184, 221]
EXPECTED_SHARED_SHADED_INK = [12, 40, 20, 111, 148, 185, 222, 3, 40, 77, 114, 151, 188, 225, 6, 12, 40, 20, 154, 191, 228, 12, 40, 20, 120, 157, 194, 231, 12, 49, 86, 123, 160, 197, 234, 15, 5, 60, 90, 163, 200, 237, 12, 40, 20, 129, 166, 203, 240, 21, 58, 95, 132, 169, 206, 243, 24, 5, 60, 90, 172, 209, 246, 12, 40, 20, 138, 175, 212, 249, 30, 67, 104, 141, 178, 215, 252, 33, 5, 60, 90, 181, 218, 255, 12, 40, 20, 147, 184, 221]
EXPECTED_SHARED_THICKNESS_2 = [12, 40, 20, 12, 40, 20, 222, 3, 40, 77, 114, 151, 12, 40, 20, 12, 40, 20, 12, 40, 20, 9, 46, 83, 120, 157, 194, 231, 12, 49, 86, 123, 160, 12, 40, 20, 52, 89, 126, 163, 200, 237, 18, 55, 92, 129, 166, 203, 240, 21, 58, 95, 132, 169, 206, 243, 24, 61, 98, 135, 172, 209, 246, 27, 64, 101, 138, 175, 212, 249, 30, 67, 104, 141, 178, 215, 252, 33, 70, 107, 144, 181, 218, 255, 36, 73, 110, 147, 184, 221]
EXPECTED_SHARED_THICKNESS_3 = [12, 40, 20, 12, 40, 20, 12, 40, 20, 12, 40, 20, 12, 40, 20, 12, 40, 20, 12, 40, 20, 12, 40, 20, 120, 157, 194, 231, 12, 49, 12, 40, 20, 12, 40, 20, 12, 40, 20, 163, 200, 237, 18, 55, 92, 129, 166, 203, 240, 21, 58, 12, 40, 20, 206, 243, 24, 61, 98, 135, 172, 209, 246, 27, 64, 101, 138, 175, 212, 249, 30, 67, 104, 141, 178, 215, 252, 33, 70, 107, 144, 181, 218, 255, 36, 73, 110, 147, 184, 221]
EXPECTED_SHARED_CLOSE_GAPS = [12, 40, 20, 12, 40, 20, 222, 3, 40, 77, 114, 151, 188, 225, 6, 12, 40, 20, 12, 40, 20, 12, 40, 20, 120, 157, 194, 231, 12, 49, 86, 123, 160, 197, 234, 15, 12, 40, 20, 12, 40, 20, 12, 40, 20, 12, 40, 20, 240, 21, 58, 95, 132, 169, 12, 40, 20, 12, 40, 20, 12, 40, 20, 12, 40, 20, 138, 175, 212, 249, 30, 67, 12, 40, 20, 12, 40, 20, 12, 40, 20, 12, 40, 20, 12, 40, 20, 12, 40, 20]
