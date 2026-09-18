import numpy as np
import pytest

from bitmapper.dither import (
    _DIFFUSION_KERNELS,
    apply,
    error_diffusion,
    list_methods,
    ordered,
    ordered_2x2,
    ordered_8x8,
    random_dither,
)

PALETTE = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)

# Read from the kernel table rather than a hardcoded list, so a newly
# registered kernel is covered by these tests automatically.
DIFFUSION_METHODS = sorted(_DIFFUSION_KERNELS)
ORDERED_METHODS = ["ordered", "ordered_2x2", "ordered_8x8"]
ALL_METHODS = DIFFUSION_METHODS + ORDERED_METHODS + ["random"]


def _assert_only_palette_colors(image, palette):
    flat = image.reshape(-1, 3)
    palette_set = {tuple(c) for c in palette}
    assert all(tuple(c) in palette_set for c in flat)


def test_list_methods_includes_none_and_is_sorted_otherwise():
    methods = list_methods()
    assert methods[0] == "none"
    assert set(methods) == {"none", *ALL_METHODS}
    assert methods[1:] == sorted(methods[1:])


@pytest.mark.parametrize("method", ALL_METHODS)
def test_apply_only_uses_palette_colors(gradient_image, method):
    out = apply(gradient_image, PALETTE, method)
    assert out.shape == gradient_image.shape
    assert out.dtype == np.uint8
    _assert_only_palette_colors(out, PALETTE)


@pytest.mark.parametrize("method", DIFFUSION_METHODS)
def test_diffusion_preserves_overall_brightness_roughly(gradient_image, method):
    out = apply(gradient_image, PALETTE, method)
    original_mean = gradient_image.astype(np.float64).mean()
    out_mean = out.astype(np.float64).mean()
    assert abs(original_mean - out_mean) < 40  # loose bound, error diffusion is approximate


@pytest.mark.parametrize("method", DIFFUSION_METHODS)
def test_error_diffusion_matches_apply_dispatch(gradient_image, method):
    assert np.array_equal(
        error_diffusion(gradient_image, PALETTE, method), apply(gradient_image, PALETTE, method)
    )


def test_error_diffusion_rejects_unknown_kernel(gradient_image):
    with pytest.raises(ValueError):
        error_diffusion(gradient_image, PALETTE, "bogus")


def test_ordered_matches_apply_dispatch(gradient_image):
    assert np.array_equal(ordered(gradient_image, PALETTE), apply(gradient_image, PALETTE, "ordered"))


def test_ordered_default_matrix_size_is_4x4(gradient_image):
    assert np.array_equal(ordered(gradient_image, PALETTE), ordered(gradient_image, PALETTE, matrix_size=4))


def test_ordered_matrix_sizes_differ(gradient_image):
    o2 = ordered_2x2(gradient_image, PALETTE)
    o4 = ordered(gradient_image, PALETTE)
    o8 = ordered_8x8(gradient_image, PALETTE)
    assert not np.array_equal(o2, o4)
    assert not np.array_equal(o4, o8)


def test_ordered_and_floyd_steinberg_differ_on_gradient(gradient_image):
    fs = error_diffusion(gradient_image, PALETTE, "floyd_steinberg")
    od = ordered(gradient_image, PALETTE)
    assert not np.array_equal(fs, od)


def test_random_dither_is_deterministic_with_seed(gradient_image):
    a = random_dither(gradient_image, PALETTE, seed=42)
    b = random_dither(gradient_image, PALETTE, seed=42)
    assert np.array_equal(a, b)


def test_random_dither_varies_with_seed(gradient_image):
    a = random_dither(gradient_image, PALETTE, seed=1)
    b = random_dither(gradient_image, PALETTE, seed=2)
    assert not np.array_equal(a, b)


def test_apply_none_is_plain_nearest_color(gradient_image):
    out = apply(gradient_image, PALETTE, "none")
    _assert_only_palette_colors(out, PALETTE)


def test_apply_rejects_unknown_method(gradient_image):
    with pytest.raises(ValueError):
        apply(gradient_image, PALETTE, "bogus")


@pytest.mark.parametrize("method", ALL_METHODS)
def test_zero_strength_matches_plain_nearest_color(gradient_image, method):
    out = apply(gradient_image, PALETTE, method, strength=0.0)
    none_out = apply(gradient_image, PALETTE, "none")
    np.testing.assert_array_equal(out, none_out)


# "random" isn't seeded by apply(), so two default calls aren't guaranteed
# to match each other regardless of strength.
DETERMINISTIC_METHODS = [m for m in ALL_METHODS if m != "random"]


@pytest.mark.parametrize("method", DETERMINISTIC_METHODS)
def test_full_strength_matches_default(gradient_image, method):
    out = apply(gradient_image, PALETTE, method, strength=1.0)
    default_out = apply(gradient_image, PALETTE, method)
    np.testing.assert_array_equal(out, default_out)


def test_apply_rejects_negative_strength(gradient_image):
    with pytest.raises(ValueError):
        apply(gradient_image, PALETTE, "floyd_steinberg", strength=-0.1)
