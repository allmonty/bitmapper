import numpy as np
import pytest

from bitmapper.adjustments import adjust_contrast, adjust_gamma, adjust_saturation, apply


def test_contrast_noop_at_one():
    img = np.array([[[10, 50, 200]]], dtype=np.uint8)
    np.testing.assert_array_equal(adjust_contrast(img, 1.0), img)


def test_contrast_zero_collapses_to_mid_gray():
    img = np.array([[[10, 50, 200]]], dtype=np.uint8)
    out = adjust_contrast(img, 0.0)
    np.testing.assert_array_equal(out, [[[127, 127, 127]]])


def test_contrast_above_one_pushes_values_away_from_midgray():
    img = np.array([[[200, 200, 200]]], dtype=np.uint8)
    out = adjust_contrast(img, 2.0)
    assert out[0, 0, 0] > 200


def test_contrast_rejects_negative_amount():
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        adjust_contrast(img, -1.0)


def test_saturation_noop_at_one():
    img = np.array([[[10, 50, 200]]], dtype=np.uint8)
    np.testing.assert_array_equal(adjust_saturation(img, 1.0), img)


def test_saturation_zero_is_grayscale():
    img = np.array([[[10, 50, 200]]], dtype=np.uint8)
    out = adjust_saturation(img, 0.0)
    assert out[0, 0, 0] == out[0, 0, 1] == out[0, 0, 2]


def test_saturation_rejects_negative_amount():
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        adjust_saturation(img, -1.0)


def test_gamma_noop_at_one():
    img = np.array([[[10, 50, 200]]], dtype=np.uint8)
    np.testing.assert_array_equal(adjust_gamma(img, 1.0), img)


def test_gamma_above_one_brightens_midtones():
    img = np.array([[[100, 100, 100]]], dtype=np.uint8)
    out = adjust_gamma(img, 2.0)
    assert out[0, 0, 0] > 100


def test_gamma_rejects_non_positive():
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        adjust_gamma(img, 0.0)
    with pytest.raises(ValueError):
        adjust_gamma(img, -1.0)


def test_apply_all_defaults_is_a_noop():
    img = np.array([[[10, 50, 200]]], dtype=np.uint8)
    np.testing.assert_array_equal(apply(img), img)


def test_apply_chains_contrast_saturation_gamma():
    img = np.array([[[10, 50, 200]]], dtype=np.uint8)
    combined = apply(img, contrast=1.2, saturation=0.5, gamma=1.5)
    step1 = adjust_contrast(img, 1.2)
    step2 = adjust_saturation(step1, 0.5)
    step3 = adjust_gamma(step2, 1.5)
    np.testing.assert_array_equal(combined, step3)
