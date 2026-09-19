import numpy as np
import pytest

from bitmapper.pipeline import BitmapFilterConfig, apply_bitmap_filter
from bitmapper.presets import get_preset, list_presets, preset_columns
from conftest import random_image



@pytest.mark.parametrize("name", list_presets())
def test_preset_builds_a_valid_config(name):
    config = get_preset(name, output_size=(40, 40), grid_size=(8, 8))
    assert isinstance(config, BitmapFilterConfig)
    assert config.output_size == (40, 40)
    assert config.grid_size == (8, 8)


@pytest.mark.parametrize("name", list_presets())
def test_preset_runs_through_the_pipeline(name):
    config = get_preset(name, output_size=(40, 40), grid_size=(8, 8))
    result = apply_bitmap_filter(random_image(), config)
    assert result.output.shape == (40, 40, 3)


def test_get_preset_overrides_take_precedence():
    config = get_preset("gameboy_camera", output_size=(40, 40), grid_size=(8, 8), bit_depth=4)
    assert config.bit_depth == 4  # overridden from the preset's own bit_depth=2


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        get_preset("does_not_exist")


def test_list_presets_is_sorted_and_nonempty():
    presets = list_presets()
    assert presets == sorted(presets)
    assert len(presets) > 0


PIXEL_ART = ["pixel_art", "pixel_art_soft", "pixel_art_rich", "pixel_art_earthy", "pixel_art_mono"]


@pytest.mark.parametrize("name", PIXEL_ART)
def test_pixel_art_presets_are_flat_with_a_pixel_art_palette_and_chunky_grid(name):
    config = get_preset(name)
    assert config.dither == "none"
    assert config.palette_mode == "fixed"
    assert 32 <= preset_columns(name) <= 96


@pytest.mark.parametrize("name", PIXEL_ART)
def test_pixel_art_output_uses_only_palette_colors(name):
    cols = preset_columns(name)
    config = get_preset(name, output_size=(cols * 2, cols * 2), grid_size=(cols, cols))
    result = apply_bitmap_filter(random_image(size=cols * 3), config)
    palette = {tuple(c) for c in result.palette.tolist()}
    assert {tuple(c) for c in result.grid.reshape(-1, 3).tolist()} <= palette


def test_preset_columns_is_none_for_presets_without_a_suggestion():
    assert preset_columns("vhs") is None
    assert preset_columns("pixel_art") == 64


def test_preset_columns_rejects_unknown_presets():
    with pytest.raises(ValueError):
        preset_columns("does_not_exist")
