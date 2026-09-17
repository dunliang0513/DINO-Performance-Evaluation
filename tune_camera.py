"""Find the sharpest fixed-focus setting for the current camera position.

Continuous autofocus is disabled during capture (see config.LOCK_CAMERA_CONTROLS)
because refocusing between the reference image and a live frame changes DINO
embeddings more than a missing component does. That means the focus position has
to be chosen once, deliberately, for the working distance in use.

Run this whenever the camera or the board moves, then put the reported value in
config.py or the FOCUS_ABSOLUTE environment variable -- and re-capture the
reference image afterwards, so both sides of the comparison share one focus.

Usage:
    python tune_camera.py
"""

import time

import cv2

import config
from camera_source import UsbCamera

FOCUS_VALUES = range(0, 256, 5)
EXPOSURE_VALUES = range(100, 900, 40)
# Mid-grey. Aiming here keeps detail in both the dark board and the bright
# metal screws, rather than crushing one to keep the other.
TARGET_BRIGHTNESS = 125.0
# Above roughly this fraction of saturated pixels, specular highlights on the
# screw heads start losing the detail the comparison depends on.
MAX_BLOWN_OUT_FRACTION = 0.005
SETTLE_SECONDS = 0.5
FRAMES_PER_STEP = 4


def sharpness(image):
    """Variance of the Laplacian: higher means more high-frequency detail."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(grey, cv2.CV_64F).var()


def brightness(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()


def blown_out_fraction(image):
    """Fraction of pixels at or near saturation."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float((grey > 250).mean())


def settle(camera, frames=FRAMES_PER_STEP):
    for _ in range(frames):
        camera.read()


def tune_exposure(camera):
    """Pick the exposure whose brightness lands nearest mid-grey.

    Note that reading the value back from automatic mode does NOT work: in
    Aperture Priority the camera also trims gain internally, so the exposure it
    reports does not reproduce the image it was producing. Brightness has to be
    measured directly.
    """
    print(f"\n{'exposure':>9} {'brightness':>11} {'blown out':>10}")

    best_exposure = None
    best_error = None

    for exposure in EXPOSURE_VALUES:
        camera.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
        time.sleep(SETTLE_SECONDS)
        settle(camera)

        ok, frame = camera.read()

        if not ok:
            continue

        level = brightness(frame)
        blown = blown_out_fraction(frame)
        error = abs(level - TARGET_BRIGHTNESS)
        marker = ""

        # Reject anything that is blowing out highlights, however close its
        # average brightness looks.
        if blown <= MAX_BLOWN_OUT_FRACTION and (
            best_error is None or error < best_error
        ):
            best_exposure, best_error = exposure, error
            marker = "  <-- best so far"

        print(f"{exposure:9d} {level:11.1f} {blown * 100:9.2f}%{marker}")

    return best_exposure


def main():
    # Open with locking disabled so this script owns the focus control.
    camera = UsbCamera(
        config.USB_CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT
    )
    camera.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    camera.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    print(f"Tuning {camera.name}")
    print("Point the camera at the board, then leave it still.\n")
    print(f"{'focus':>6} {'sharpness':>11}")

    best_focus = None
    best_score = -1.0
    best_exposure = None

    try:
        for focus in FOCUS_VALUES:
            camera.camera.set(cv2.CAP_PROP_FOCUS, focus)
            time.sleep(SETTLE_SECONDS)

            scores = []

            for _ in range(FRAMES_PER_STEP):
                ok, frame = camera.read()

                if ok:
                    scores.append(sharpness(frame))

            if not scores:
                continue

            # Take the best of several frames: a single frame can be spoiled by
            # motion or a rolling-shutter artefact.
            score = max(scores)
            marker = ""

            if score > best_score:
                best_focus, best_score = focus, score
                marker = "  <-- best so far"

            print(f"{focus:6d} {score:11.1f}{marker}")

        # Exposure is tuned once, at the focus that won -- a blurred frame
        # would bias the brightness reading.
        if best_focus is not None:
            camera.camera.set(cv2.CAP_PROP_FOCUS, best_focus)
            time.sleep(SETTLE_SECONDS)
            best_exposure = tune_exposure(camera)
    finally:
        camera.release()

    if best_focus is None:
        raise SystemExit("No frames were captured. Is the camera in use?")

    print()
    print(f"Sharpest focus: {best_focus}  (sharpness {best_score:.1f})")

    if best_exposure is not None:
        print(f"Best exposure:  {best_exposure}")

    print()
    print("Apply with:")
    print(f"    export FOCUS_ABSOLUTE={best_focus}")

    if best_exposure is not None:
        print(f"    export EXPOSURE_ABSOLUTE={best_exposure}")

    print("    # or edit them in config.py")
    print()
    print("Then RE-CAPTURE the reference and RE-CALIBRATE, so both sides of")
    print("the comparison share these settings:")
    print("    python capture_reference.py")
    print("    python calibrate.py")


if __name__ == "__main__":
    main()
