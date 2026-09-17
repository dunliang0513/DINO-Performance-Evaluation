"""Single source of truth for runtime configuration.

Every value may be overridden by an environment variable of the same name, so
switching camera, model or threshold never requires editing a script.
"""

import os


def _str(name, default):
    return os.environ.get(name, default)


def _int(name, default):
    return int(os.environ.get(name, default))


def _float(name, default):
    return float(os.environ.get(name, default))


def _bool(name, default):
    raw = os.environ.get(name)

    if raw is None:
        return default

    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Camera -----------------------------------------------------------------
# Default is the USB webcam; the Basler path is retained for the Orin rig.
CAMERA_BACKEND = _str("CAMERA_BACKEND", "usb")
# Index 2 is the external Logitech BRIO; index 0 is the laptop's built-in
# webcam. Verify with `v4l2-ctl --list-devices` after replugging - V4L2 index
# numbers are assigned in enumeration order and can shift between boots or
# USB ports, which would silently change which camera a capture came from.
USB_CAMERA_INDEX = _int("USB_CAMERA_INDEX", 2)
FRAME_WIDTH = _int("FRAME_WIDTH", 1920)
FRAME_HEIGHT = _int("FRAME_HEIGHT", 1080)
ROTATE_180 = _bool("ROTATE_180", False)
# Auto focus / exposure / white balance drift between the reference capture and
# inference, and that drift changes DINO embeddings MORE than a missing screw
# does. Measured on a Logitech BRIO: autofocus hunting left a live frame at 8%
# of the reference's sharpness, which alone pushed an intact ROI's score to
# 0.54 - the same score a genuinely removed screw produced. Locking these is
# not a nicety; without it the comparison is meaningless.
LOCK_CAMERA_CONTROLS = _bool("LOCK_CAMERA_CONTROLS", True)
# Fixed focus position, in the camera's own units. This is specific to the
# working distance between camera and board: re-run `python tune_camera.py`
# whenever either moves.
FOCUS_ABSOLUTE = _int("FOCUS_ABSOLUTE", 20)
# Fixed exposure time, in the camera's own units. Switching a UVC camera to
# manual exposure pins whatever value it happens to hold, which is usually far
# too dark -- on a BRIO it produced a mean brightness of 70 where auto gave 124.
# Reading the value back from auto mode does not help either: in Aperture
# Priority the camera also trims gain internally, so the reported exposure does
# not describe the resulting image. The value has to be measured against actual
# brightness, which is what `python tune_camera.py` does. Scene-dependent:
# re-run it whenever the lighting changes.
EXPOSURE_ABSOLUTE = _int("EXPOSURE_ABSOLUTE", 420)

BASLER_SERIAL = _str("BASLER_SERIAL", "")
BASLER_TIMEOUT_MS = _int("BASLER_TIMEOUT_MS", 2000)

# --- Paths ------------------------------------------------------------------
CONFIG_PATH = _str("CONFIG_PATH", "product_config.json")
REFERENCE_PATH = _str("REFERENCE_PATH", "my_photo-1.jpg")

# --- Registration -----------------------------------------------------------
MATCH_SCALE = _float("MATCH_SCALE", 0.25)
MIN_GOOD_MATCHES = _int("MIN_GOOD_MATCHES", 30)
REGISTRATION_INTERVAL = _int("REGISTRATION_INTERVAL", 3)
MAX_REGISTRATION_FAILURES = _int("MAX_REGISTRATION_FAILURES", 3)
LOWE_RATIO = _float("LOWE_RATIO", 0.7)
RANSAC_REPROJECTION_THRESHOLD = _float("RANSAC_REPROJECTION_THRESHOLD", 5.0)

# --- Detection --------------------------------------------------------------
DETECTION_INTERVAL = _int("DETECTION_INTERVAL", 2)
DEBOUNCE_COUNT = _int("DEBOUNCE_COUNT", 3)
# Calibration replaces hand-picked thresholds. Each ROI has its own normal
# range -- corner ROIs swing far more than central ones because homography
# residual is worst at the edges. Measured on a real board: a corner ROI varied
# 0.740-0.911 between frames (std 0.049) while a central one held 0.950-0.974
# (std 0.007). One global threshold cannot fit both, so each ROI gets a band
# derived from its own measured behaviour instead.
CALIBRATION_FRAMES = _int("CALIBRATION_FRAMES", 30)
# How many standard deviations below an ROI's own mean counts as a defect.
CALIBRATION_SIGMA = _float("CALIBRATION_SIGMA", 4.0)
# Floor on the measured spread. A very stable ROI would otherwise get an
# impossibly tight band that a single noisy frame could trip.
CALIBRATION_MIN_STD = _float("CALIBRATION_MIN_STD", 0.02)

# The ROI crop must be tight enough that the component dominates it. At 80 px
# on a 1080p board the screw is a small fraction of the crop, so removing it
# barely changes the embedding: measured d-prime 0.54, i.e. not separable. At
# 32 px the same defect separates cleanly at d-prime 5.9. Bigger is emphatically
# not safer here - at 120 px the ranking inverts.
DEFAULT_CROP_SIZE = _int("DEFAULT_CROP_SIZE", 32)
DEFAULT_SIMILARITY_THRESHOLD = _float("DEFAULT_SIMILARITY_THRESHOLD", 0.75)

# --- Model ------------------------------------------------------------------
MODEL_NAME = _str("MODEL_NAME", "dinov2-small")
POOLING = _str("POOLING", "cls")
DEVICE = _str("DEVICE", "cuda")
USE_FP16 = _bool("USE_FP16", True)

# --- Preview ----------------------------------------------------------------
PREVIEW_MAX_WIDTH = _int("PREVIEW_MAX_WIDTH", 1280)
PREVIEW_MAX_HEIGHT = _int("PREVIEW_MAX_HEIGHT", 720)
