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
DEFAULT_CROP_SIZE = _int("DEFAULT_CROP_SIZE", 80)
DEFAULT_SIMILARITY_THRESHOLD = _float("DEFAULT_SIMILARITY_THRESHOLD", 0.75)

# --- Model ------------------------------------------------------------------
MODEL_NAME = _str("MODEL_NAME", "dinov2-small")
POOLING = _str("POOLING", "cls")
DEVICE = _str("DEVICE", "cuda")
USE_FP16 = _bool("USE_FP16", True)

# --- Preview ----------------------------------------------------------------
PREVIEW_MAX_WIDTH = _int("PREVIEW_MAX_WIDTH", 1280)
PREVIEW_MAX_HEIGHT = _int("PREVIEW_MAX_HEIGHT", 720)
