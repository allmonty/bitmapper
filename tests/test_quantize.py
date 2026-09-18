import numpy as np

from bitmapper.quantize import nearest_color


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
