import time

import cv2

import config
from camera_source import create_camera

OUTPUT_PATH = config.REFERENCE_PATH

camera = create_camera()

print(f"Using camera: {camera.name}")
print("Press SPACE to save the reference image.")
print("Press Q to quit.")

saved_message_until = 0.0

try:
    while True:
        ok, frame = camera.read()

        if not ok:
            continue

        height, width = frame.shape[:2]
        preview_scale = min(
            config.PREVIEW_MAX_WIDTH / width,
            config.PREVIEW_MAX_HEIGHT / height,
            1.0
        )

        preview = cv2.resize(
            frame,
            (
                int(width * preview_scale),
                int(height * preview_scale)
            )
        )

        cv2.putText(
            preview,
            f"Camera: {camera.name}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            preview,
            f"Image: {width} x {height} | SPACE: Save | Q: Quit",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if time.monotonic() < saved_message_until:
            cv2.putText(
                preview,
                f"Saved: {OUTPUT_PATH}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.imshow("Reference Capture", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not cv2.imwrite(OUTPUT_PATH, frame):
                raise RuntimeError(
                    f"Could not save reference image: {OUTPUT_PATH}"
                )

            print(
                f"Saved {OUTPUT_PATH} at {width} x {height}."
            )
            saved_message_until = time.monotonic() + 2.0

        if key == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
