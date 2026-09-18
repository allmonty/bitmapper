import numpy as np
import pytest

import bitmapper.quantize as quantize
from bitmapper.quantize import nearest_color, nearest_index


def test_nearest_color_picks_closest():
    palette = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0]], dtype=np.uint8)
    image = np.array([[[10, 10, 10], [240, 10, 10]]], dtype=np.uint8)

    quantized, idx = nearest_color(image, palette)

    np.testing.assert_array_equal(quantized[0, 0], (0, 0, 0))
    np.testing.assert_array_equal(quantized[0, 1], (255, 0, 0))
    np.testing.assert_array_equal(idx, [[0, 2]])


def test_nearest_color_output_uses_only_palette_colors():
    rng = np.random.default_rng(1)
    palette = rng.integers(0, 256, size=(5, 3)).astype(np.uint8)
    image = rng.integers(0, 256, size=(10, 10, 3)).astype(np.uint8)

    quantized, _ = nearest_color(image, palette)

    flat = quantized.reshape(-1, 3)
    palette_set = {tuple(c) for c in palette}
    assert all(tuple(c) in palette_set for c in flat)


def test_nearest_index_breaks_ties_toward_the_lower_index():
    # (10, 0, 0) is exactly equidistant from both entries.
    palette = np.array([[0, 0, 0], [20, 0, 0]], dtype=np.uint8)
    idx = nearest_index(np.array([[10, 0, 0]], dtype=np.uint8), palette)
    assert idx[0] == 0


def test_nearest_index_handles_float_input():
    palette = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    pixels = np.array([[10.4, 10.6, 10.5], [200.2, 200.9, 200.1]])
    np.testing.assert_array_equal(nearest_index(pixels, palette), [0, 1])


@pytest.mark.parametrize("max_elements", [1, 7, 64, 10_000_000])
def test_chunking_does_not_change_results(monkeypatch, max_elements):
    """Pixels are independent, so the chunk size is purely a memory knob —
    every chunking granularity must agree, including one pixel at a time.
    """
    rng = np.random.default_rng(3)
    palette = rng.integers(0, 256, size=(17, 3)).astype(np.uint8)
    pixels = rng.integers(0, 256, size=(97, 3)).astype(np.uint8)

    monkeypatch.setattr(quantize, "_MAX_DISTANCE_ELEMENTS", 10_000_000)
    expected = nearest_index(pixels, palette)

    monkeypatch.setattr(quantize, "_MAX_DISTANCE_ELEMENTS", max_elements)
    np.testing.assert_array_equal(nearest_index(pixels, palette), expected)
