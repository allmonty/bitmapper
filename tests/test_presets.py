import numpy as np
import pytest

from bitmapper.pipeline import BitmapFilterConfig, apply_bitmap_filter
from bitmapper.presets import get_preset, list_presets
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
