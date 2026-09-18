import numpy as np

from bitmapper.grid import downsample, upscale


def test_downsample_average_even_blocks():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[0:2, 0:2] = (0, 0, 0)
    img[0:2, 2:4] = (100, 100, 100)
    img[2:4, 0:2] = (200, 200, 200)
    img[2:4, 2:4] = (50, 50, 50)

    out = downsample(img, grid_size=(2, 2), mode="average")

    assert out.shape == (2, 2, 3)
    np.testing.assert_array_equal(out[0, 0], (0, 0, 0))
    np.testing.assert_array_equal(out[0, 1], (100, 100, 100))
    np.testing.assert_array_equal(out[1, 0], (200, 200, 200))
    np.testing.assert_array_equal(out[1, 1], (50, 50, 50))


def test_downsample_average_mixed_block():
    img = np.array(
        [[[0, 0, 0], [10, 0, 0]], [[20, 0, 0], [30, 0, 0]]], dtype=np.uint8
    )
    out = downsample(img, grid_size=(1, 1), mode="average")
    assert out.shape == (1, 1, 3)
    np.testing.assert_array_equal(out[0, 0], (15, 0, 0))


def test_downsample_nearest_picks_block_center():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[1, 1] = (9, 9, 9)  # center of top-left 2x2 block
    img[1, 3] = (7, 7, 7)  # center of top-right 2x2 block

    out = downsample(img, grid_size=(2, 2), mode="nearest")

    np.testing.assert_array_equal(out[0, 0], (9, 9, 9))
    np.testing.assert_array_equal(out[0, 1], (7, 7, 7))


def test_downsample_rejects_unknown_mode():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    try:
        downsample(img, (1, 1), mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_upscale_replicates_blocks():
    grid = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)  # 1x2 grid
    out = upscale(grid, output_size=(4, 2))  # width=4, height=2

    assert out.shape == (2, 4, 3)
    # left half of every row is the first cell, right half the second
    np.testing.assert_array_equal(out[:, :2], np.broadcast_to([1, 2, 3], (2, 2, 3)))
    np.testing.assert_array_equal(out[:, 2:], np.broadcast_to([4, 5, 6], (2, 2, 3)))


def test_upscale_with_no_gap_is_unchanged():
    grid = np.full((2, 2, 3), 100, dtype=np.uint8)
    out = upscale(grid, output_size=(8, 8), gap_px=0)
    assert (out == 100).all()


def test_upscale_gap_draws_gutter_at_block_boundary():
    grid = np.full((2, 2, 3), 100, dtype=np.uint8)
    out = upscale(grid, output_size=(8, 8), gap_px=2, gap_color=(0, 0, 0))

    # boundary between the two blocks sits at column/row 4; gap_px=2 draws
    # columns/rows 3-4 as the gutter.
    np.testing.assert_array_equal(out[:, 3:5], 0)
    np.testing.assert_array_equal(out[3:5, :], 0)
    # interior of each block is untouched
    np.testing.assert_array_equal(out[0:3, 0:3], 100)
    np.testing.assert_array_equal(out[0:3, 5:8], 100)


def test_upscale_gap_does_not_touch_canvas_edges():
    grid = np.full((1, 1, 3), 100, dtype=np.uint8)
    out = upscale(grid, output_size=(8, 8), gap_px=2)
    assert (out == 100).all()  # single block, no interior boundary to gap


def test_upscale_gap_uses_custom_color():
    grid = np.full((2, 1, 3), 100, dtype=np.uint8)
    out = upscale(grid, output_size=(4, 8), gap_px=2, gap_color=(255, 0, 0))
    np.testing.assert_array_equal(out[3:5, :], np.broadcast_to([255, 0, 0], (2, 4, 3)))


def test_downsample_then_upscale_round_trip_is_blocky():
    img = np.random.default_rng(0).integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    grid = downsample(img, grid_size=(4, 4), mode="average")
    out = upscale(grid, output_size=(20, 20))

    assert out.shape == img.shape
    # each 5x5 block in the output must be perfectly flat (a single color)
    for i in range(0, 20, 5):
        for j in range(0, 20, 5):
            block = out[i : i + 5, j : j + 5]
            assert (block == block[0, 0]).all()
