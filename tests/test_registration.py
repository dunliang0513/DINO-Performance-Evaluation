import cv2
import numpy as np

from registration import Registrar


def _warp(image, angle_degrees, scale, shift_x, shift_y):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_degrees, scale)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    return cv2.warpAffine(image, matrix, (width, height))


def test_recovers_a_known_transform(synthetic_board):
    """Warp the reference by a known amount; registration should undo it.

    This is the golden test for the whole registration path and needs no camera.
    """
    height, width = synthetic_board.shape[:2]
    warped = _warp(synthetic_board, angle_degrees=5.0, scale=1.03,
                   shift_x=15, shift_y=-9)

    registrar = Registrar(synthetic_board, match_scale=0.5)
    homography, inlier_count = registrar.register(warped)

    assert homography is not None, "registration should succeed on a modest warp"
    assert inlier_count >= 30

    recovered = cv2.warpPerspective(warped, homography, (width, height))

    # Compare away from the borders, where the warp legitimately loses data.
    region = (slice(150, height - 150), slice(250, width - 250))
    error = np.abs(
        recovered[region].astype(np.float64)
        - synthetic_board[region].astype(np.float64)
    ).mean()

    assert error < 15.0, f"mean absolute error {error:.2f} is too high"


def test_identity_registration_is_near_perfect(synthetic_board):
    registrar = Registrar(synthetic_board, match_scale=0.5)
    homography, inlier_count = registrar.register(synthetic_board.copy())

    assert homography is not None
    assert inlier_count >= 30
    # Registering an image against itself should be very close to the identity.
    assert np.allclose(homography / homography[2, 2], np.eye(3), atol=0.05)


def test_unregisterable_frame_returns_none(synthetic_board):
    """A flat grey frame has no SIFT keypoints, so registration must fail cleanly."""
    blank = np.full_like(synthetic_board, 127)

    registrar = Registrar(synthetic_board, match_scale=0.5)
    homography, inlier_count = registrar.register(blank)

    assert homography is None
    assert inlier_count == 0
