import numpy as np
import pytest

from bitmapper.palette_gen import generate_palette, kmeans, median_cut


@pytest.mark.parametrize("algorithm", ["median_cut", "kmeans"])
def test_generate_palette_returns_requested_count(algorithm, gradient_image):
    palette = generate_palette(gradient_image, n_colors=8, algorithm=algorithm)
    assert palette.shape == (8, 3)
    assert palette.dtype == np.uint8


@pytest.mark.parametrize("fn", [median_cut, kmeans])
def test_single_color_image_degenerates_gracefully(fn, solid_image):
    palette = fn(solid_image, n_colors=4)
    assert palette.shape == (4, 3)
    # every generated color should equal the single source color
    for color in palette:
        np.testing.assert_array_equal(color, (200, 50, 10))


@pytest.mark.parametrize("fn", [median_cut, kmeans])
def test_n_colors_one(fn, gradient_image):
    palette = fn(gradient_image, n_colors=1)
    assert palette.shape == (1, 3)


def test_unknown_algorithm_raises(gradient_image):
    with pytest.raises(ValueError):
        generate_palette(gradient_image, n_colors=4, algorithm="bogus")
