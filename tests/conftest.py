import numpy as np
import pytest


def random_image(size=40, seed=0):
    """Deterministic random RGB image, for pipeline-level tests that only
    need "some image" rather than a specific pattern.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


@pytest.fixture
def solid_image():
    return np.full((8, 8, 3), (200, 50, 10), dtype=np.uint8)


@pytest.fixture
def checkerboard_image():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[0::2, 0::2] = (255, 255, 255)
    img[1::2, 1::2] = (255, 255, 255)
    return img


@pytest.fixture
def gradient_image():
    ramp = np.linspace(0, 255, 16, dtype=np.uint8)
    row = np.stack([ramp, ramp, ramp], axis=1)  # (16, 3)
    img = np.tile(row, (16, 1, 1))
    return img
