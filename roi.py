"""Square ROI extraction with zero padding at image borders."""

import cv2


def crop_square_with_padding(image, center_x, center_y, crop_size):
    """Return a `crop_size` x `crop_size` BGR crop centred on (center_x, center_y).

    Regions falling outside the image are zero padded, so the returned array is
    always exactly the requested size. Raises ValueError only when the requested
    square lies entirely outside the image.
    """
    half = crop_size // 2
    image_height, image_width = image.shape[:2]

    x1 = center_x - half
    y1 = center_y - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    source_x1 = max(x1, 0)
    source_y1 = max(y1, 0)
    source_x2 = min(x2, image_width)
    source_y2 = min(y2, image_height)

    if source_x1 >= source_x2 or source_y1 >= source_y2:
        raise ValueError(
            f"ROI center ({center_x}, {center_y}) is outside the image."
        )

    crop = image[source_y1:source_y2, source_x1:source_x2]

    return cv2.copyMakeBorder(
        crop,
        source_y1 - y1,
        y2 - source_y2,
        source_x1 - x1,
        x2 - source_x2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
