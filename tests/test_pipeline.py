import numpy as np
import pytest

from bitmapper.dither import list_methods as list_dither_methods
from bitmapper.pipeline import BitmapFilterConfig, apply_bitmap_filter


def _random_image(size=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _assert_blocky(output, grid_size, output_size):
    grid_cols, grid_rows = grid_size
    out_w, out_h = output_size
    block_h = out_h // grid_rows
    block_w = out_w // grid_cols
    for i in range(grid_rows):
        for j in range(grid_cols):
            block = output[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
            assert (block == block[0, 0]).all(), f"block ({i},{j}) is not flat"


@pytest.mark.parametrize("bit_depth", [2, 4, 8])
@pytest.mark.parametrize("dither", list_dither_methods())
def test_auto_palette_pipeline_produces_blocky_output_within_color_budget(bit_depth, dither):
    config = BitmapFilterConfig(
        output_size=(40, 40),
        grid_size=(8, 8),
        bit_depth=bit_depth,
        palette_mode="auto",
        dither=dither,
    )
    result = apply_bitmap_filter(_random_image(), config)

    assert result.output.shape == (40, 40, 3)
    assert result.grid.shape == (8, 8, 3)
    _assert_blocky(result.output, config.grid_size, config.output_size)

    used_colors = {tuple(c) for c in result.output.reshape(-1, 3)}
    assert len(used_colors) <= 2 ** bit_depth


def test_fixed_palette_pipeline_uses_only_fixed_colors():
    config = BitmapFilterConfig(
        output_size=(40, 40),
        grid_size=(8, 8),
        bit_depth=4,
        palette_mode="fixed",
        fixed_palette="ega",
        dither="floyd_steinberg",
    )
    result = apply_bitmap_filter(_random_image(), config)

    ega_set = {tuple(c) for c in result.palette}
    used_colors = {tuple(c) for c in result.output.reshape(-1, 3)}
    assert used_colors.issubset(ega_set)


def test_true_color_bit_depth_skips_dithering_and_passes_through():
    config = BitmapFilterConfig(
        output_size=(40, 40),
        grid_size=(8, 8),
        bit_depth=24,
        palette_mode="auto",
        dither="floyd_steinberg",  # should be ignored at true-color depth
    )
    result = apply_bitmap_filter(_random_image(), config)

    direct_grid = apply_bitmap_filter(
        _random_image(),
        BitmapFilterConfig(output_size=(40, 40), grid_size=(8, 8), bit_depth=24),
    ).grid
    np.testing.assert_array_equal(result.grid, direct_grid)


def test_block_sampling_nearest_vs_average_differ():
    img = _random_image(size=40, seed=2)
    cfg_avg = BitmapFilterConfig(output_size=(40, 40), grid_size=(4, 4), bit_depth=8, block_sampling="average")
    cfg_near = BitmapFilterConfig(output_size=(40, 40), grid_size=(4, 4), bit_depth=8, block_sampling="nearest")

    out_avg = apply_bitmap_filter(img, cfg_avg)
    out_near = apply_bitmap_filter(img, cfg_near)

    assert not np.array_equal(out_avg.grid, out_near.grid)


def test_rgba_input_is_handled():
    rgba = np.dstack([_random_image(size=16), np.full((16, 16), 255, dtype=np.uint8)])
    config = BitmapFilterConfig(output_size=(16, 16), grid_size=(4, 4), bit_depth=4)
    result = apply_bitmap_filter(rgba, config)
    assert result.output.shape == (16, 16, 3)


def test_invalid_bit_depth_rejected():
    with pytest.raises(ValueError):
        BitmapFilterConfig(bit_depth=0)
    with pytest.raises(ValueError):
        BitmapFilterConfig(bit_depth=25)


def test_fixed_palette_mode_requires_a_name():
    with pytest.raises(ValueError):
        BitmapFilterConfig(palette_mode="fixed", fixed_palette=None)


def test_scanlines_darken_alternate_output_rows():
    config = BitmapFilterConfig(output_size=(40, 40), grid_size=(8, 8), bit_depth=8, scanlines=0.0)
    plain = apply_bitmap_filter(_random_image(), config)

    scanlined_config = BitmapFilterConfig(output_size=(40, 40), grid_size=(8, 8), bit_depth=8, scanlines=1.0)
    scanlined = apply_bitmap_filter(_random_image(), scanlined_config)

    np.testing.assert_array_equal(scanlined.output[1::2], 0)
    np.testing.assert_array_equal(scanlined.output[0::2], plain.output[0::2])


def test_invalid_scanlines_rejected():
    with pytest.raises(ValueError):
        BitmapFilterConfig(scanlines=-0.1)
    with pytest.raises(ValueError):
        BitmapFilterConfig(scanlines=1.1)
