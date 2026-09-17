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
SETTLE_SECONDS = 0.5
FRAMES_PER_STEP = 4


def sharpness(image):
    """Variance of the Laplacian: higher means more high-frequency detail."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(grey, cv2.CV_64F).var()


def main():
    # Open with locking disabled so this script owns the focus control.
    camera = UsbCamera(
        config.USB_CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT
    )
    camera.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    print(f"Tuning {camera.name}")
    print("Point the camera at the board, then leave it still.\n")
    print(f"{'focus':>6} {'sharpness':>11}")

    best_focus = None
    best_score = -1.0

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
    finally:
        camera.release()

    if best_focus is None:
        raise SystemExit("No frames were captured. Is the camera in use?")

    print()
    print(f"Sharpest focus: {best_focus}  (sharpness {best_score:.1f})")
    print()
    print("Apply it with either:")
    print(f"    export FOCUS_ABSOLUTE={best_focus}")
    print(f"    # or edit FOCUS_ABSOLUTE in config.py")
    print()
    print("Then re-capture the reference so it shares this focus:")
    print("    python capture_reference.py")


if __name__ == "__main__":
    main()
