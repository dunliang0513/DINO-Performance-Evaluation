import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_board():
    """A textured stand-in for a PCB, 1280x720 BGR.

    Deterministic (seeded) so test failures are reproducible. The bright
    rectangles give SIFT stable, distinctive keypoints; the noise floor stops
    large flat regions producing degenerate matches.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(40, 90, size=(720, 1280, 3), dtype=np.uint8)

    for index, x in enumerate(range(60, 1240, 110)):
        for jndex, y in enumerate(range(60, 700, 105)):
            width = 30 + ((index * 7 + jndex * 13) % 35)
            height = 18 + ((index * 11 + jndex * 5) % 25)
            shade = 150 + ((index * 23 + jndex * 17) % 100)
            cv2.rectangle(
                image,
                (x, y),
                (x + width, y + height),
                (int(shade), int(shade), int(shade)),
                -1,
            )

    return image


@pytest.fixture
def screw_points():
    """Four ROI centres, all comfortably inside the synthetic board."""
    return [(300, 200), (900, 200), (300, 520), (900, 520)]
