import numpy as np
import pytest

from roi import crop_square_with_padding


def test_returns_requested_size(synthetic_board):
    crop = crop_square_with_padding(synthetic_board, 640, 360, 80)
    assert crop.shape == (80, 80, 3)


def test_crop_contains_the_source_pixels(synthetic_board):
    crop = crop_square_with_padding(synthetic_board, 640, 360, 80)
    expected = synthetic_board[320:400, 600:680]
    assert np.array_equal(crop, expected)


def test_crop_overlapping_the_border_is_zero_padded(synthetic_board):
    # Centre (10, 10) with half-size 40 puts 30 px outside the top-left corner.
    crop = crop_square_with_padding(synthetic_board, 10, 10, 80)

    assert crop.shape == (80, 80, 3)
    assert np.all(crop[:30, :] == 0), "top padding should be black"
    assert np.all(crop[:, :30] == 0), "left padding should be black"
    assert np.any(crop[30:, 30:] != 0), "real image data should survive"


def test_crop_overlapping_the_far_border_is_zero_padded(synthetic_board):
    height, width = synthetic_board.shape[:2]
    crop = crop_square_with_padding(synthetic_board, width - 10, height - 10, 80)

    assert crop.shape == (80, 80, 3)
    assert np.all(crop[50:, :] == 0)
    assert np.all(crop[:, 50:] == 0)


def test_odd_crop_size_is_honoured(synthetic_board):
    crop = crop_square_with_padding(synthetic_board, 640, 360, 81)
    assert crop.shape == (81, 81, 3)


def test_centre_fully_outside_the_image_raises(synthetic_board):
    with pytest.raises(ValueError, match="outside the image"):
        crop_square_with_padding(synthetic_board, -500, -500, 80)
