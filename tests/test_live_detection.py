import numpy as np

from live_detection import fit_preview


def _image(width, height):
    return np.zeros((height, width, 3), np.uint8)


def test_downscales_to_fit_the_box():
    preview = fit_preview(_image(1920, 1080), 1280, 720)
    assert preview.shape[:2] == (720, 1280)


def test_preserves_aspect_ratio_for_a_four_three_frame():
    """A 4:3 camera must not be stretched into a 16:9 box.

    The previous code passed the preview size to cv2.resize as an exact target,
    which distorted the frame and moved the ROI boxes away from where the
    screws really are.
    """
    preview = fit_preview(_image(640, 480), 1280, 720)
    height, width = preview.shape[:2]

    assert width / height == 640 / 480
    assert width <= 1280 and height <= 720


def test_never_upscales():
    """Enlarging a small frame invents detail and costs time for nothing."""
    original = _image(320, 240)
    preview = fit_preview(original, 1280, 720)

    assert preview.shape == original.shape


def test_constrained_by_whichever_dimension_binds_first():
    # Very wide frame: width is the limit, not height.
    preview = fit_preview(_image(3000, 500), 1000, 900)
    assert preview.shape[:2] == (166, 1000)

    # Very tall frame: height is the limit.
    preview = fit_preview(_image(500, 3000), 1000, 900)
    assert preview.shape[:2] == (900, 150)


def test_exact_fit_is_returned_untouched():
    original = _image(1280, 720)
    assert fit_preview(original, 1280, 720) is original
