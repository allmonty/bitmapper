import numpy as np
import pytest

from bitmapper.color import luminance


def test_luminance_is_rec601_luma():
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]])
    result = luminance(colors)
    expected = np.array([76.245, 149.685, 29.07, 255.0, 0.0])
    np.testing.assert_allclose(result, expected)


def test_luminance_broadcasts_over_a_grid():
    grid = np.zeros((2, 3, 3), dtype=np.uint8)
    grid[0, 0] = (255, 0, 0)
    grid[1, 2] = (0, 0, 255)
    result = luminance(grid)
    assert result.shape == (2, 3)
    assert result[0, 0] == pytest.approx(76.245)
    assert result[1, 2] == pytest.approx(29.07)
