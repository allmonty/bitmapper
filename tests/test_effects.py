import numpy as np
import pytest

from bitmapper.effects import apply_scanlines


def test_scanlines_zero_is_a_no_op():
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = apply_scanlines(img, 0.0)
    np.testing.assert_array_equal(out, img)


def test_scanlines_one_makes_odd_rows_black():
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = apply_scanlines(img, 1.0)
    np.testing.assert_array_equal(out[1::2], 0)
    np.testing.assert_array_equal(out[0::2], img[0::2])


def test_scanlines_partial_strength_darkens_odd_rows_only():
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = apply_scanlines(img, 0.5)
    np.testing.assert_array_equal(out[0::2], img[0::2])
    assert (out[1::2] < img[1::2]).all()
    assert (out[1::2] > 0).all()


def test_scanlines_rejects_out_of_range_strength():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_scanlines(img, -0.1)
    with pytest.raises(ValueError):
        apply_scanlines(img, 1.1)
