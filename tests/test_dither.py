import numpy as np
import pytest

from bitmapper.dither import apply, floyd_steinberg, ordered

PALETTE = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)


def _assert_only_palette_colors(image, palette):
    flat = image.reshape(-1, 3)
    palette_set = {tuple(c) for c in palette}
    assert all(tuple(c) in palette_set for c in flat)


def test_floyd_steinberg_only_uses_palette_colors(gradient_image):
    out = floyd_steinberg(gradient_image, PALETTE)
    assert out.shape == gradient_image.shape
    _assert_only_palette_colors(out, PALETTE)


def test_floyd_steinberg_preserves_overall_brightness_roughly(gradient_image):
    out = floyd_steinberg(gradient_image, PALETTE)
    original_mean = gradient_image.astype(np.float64).mean()
    out_mean = out.astype(np.float64).mean()
    assert abs(original_mean - out_mean) < 40  # loose bound, error diffusion is approximate


def test_ordered_only_uses_palette_colors(gradient_image):
    out = ordered(gradient_image, PALETTE)
    assert out.shape == gradient_image.shape
    _assert_only_palette_colors(out, PALETTE)


def test_ordered_and_floyd_steinberg_differ_on_gradient(gradient_image):
    fs = floyd_steinberg(gradient_image, PALETTE)
    od = ordered(gradient_image, PALETTE)
    assert not np.array_equal(fs, od)


def test_apply_none_is_plain_nearest_color(gradient_image):
    out = apply(gradient_image, PALETTE, "none")
    _assert_only_palette_colors(out, PALETTE)


def test_apply_rejects_unknown_method(gradient_image):
    with pytest.raises(ValueError):
        apply(gradient_image, PALETTE, "bogus")
