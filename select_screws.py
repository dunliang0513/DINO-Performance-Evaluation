import json
import cv2
from camera_source import create_camera
import config

CONFIG_PATH = config.CONFIG_PATH
REFERENCE_PATH = config.REFERENCE_PATH
CAPTURE_NEW_REFERENCE = False
DEFAULT_CROP_SIZE = config.DEFAULT_CROP_SIZE
DEFAULT_SIMILARITY_THRESHOLD = config.DEFAULT_SIMILARITY_THRESHOLD


def capture_reference():
    camera = create_camera()

    print(f"Using camera: {camera.name}")
    print("Press SPACE to save the reference image or Q to cancel.")

    saved = False

    try:
        while True:
            ok, frame = camera.read()

            if not ok:
                continue

            height, width = frame.shape[:2]
            preview_scale = min(
                1280 / width,
                800 / height,
                1.0
            )
            preview_width = int(width * preview_scale)
            preview_height = int(height * preview_scale)
            preview = cv2.resize(
                frame,
                (preview_width, preview_height)
            )

            cv2.imshow("Capture Reference", preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                cv2.imwrite(REFERENCE_PATH, frame)
                print(f"Saved reference image: {REFERENCE_PATH}")
                saved = True
                break

            if key == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if not saved:
        raise RuntimeError("Reference capture was cancelled.")


if CAPTURE_NEW_REFERENCE:
    capture_reference()

original_image = cv2.imread(REFERENCE_PATH)

if original_image is None:
    raise RuntimeError(
        f"Could not load the reference image: {REFERENCE_PATH}"
    )

points = []
display_image = original_image.copy()


def redraw_points():
    global display_image
    display_image = original_image.copy()

    for number, (x, y) in enumerate(points, start=1):
        cv2.circle(
            display_image,
            (x, y),
            20,
            (0, 0, 255),
            3
        )
        cv2.putText(
            display_image,
            str(number),
            (x + 22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Select Screws", display_image)


def undo_last_point():
    if not points:
        print("There is no screw point to undo.")
        return

    removed_point = points.pop()
    print(f"Removed last screw point: {removed_point}")
    redraw_points()


def reset_all_points():
    points.clear()
    print("All screw points were cleared.")
    redraw_points()


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Screw {len(points)}: ({x}, {y})")
        redraw_points()
    elif event == cv2.EVENT_RBUTTONDOWN:
        undo_last_point()


cv2.namedWindow("Select Screws", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Select Screws", on_mouse)
redraw_points()

print("Left click: Add a screw point")
print("Right click or U/Backspace: Undo the last point")
print("R: Clear all points and start again")
print("S or Enter: Save the configuration")
print("Q or Esc: Cancel without saving")

save_requested = False

while True:
    key = cv2.waitKey(20) & 0xFF

    if key in (ord("u"), 8, 127):
        undo_last_point()
    elif key == ord("r"):
        reset_all_points()
    elif key in (ord("s"), 10, 13):
        if not points:
            print("Select at least one screw before saving.")
            continue

        save_requested = True
        break
    elif key in (ord("q"), 27):
        break

cv2.destroyAllWindows()

if not save_requested:
    print("Selection cancelled. The existing configuration was not changed.")
    raise SystemExit(0)

print("All coordinates:", points)

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        existing_screws = json.load(config_file).get("screws", [])
except (FileNotFoundError, json.JSONDecodeError):
    existing_screws = []


def screw_setting(number, key, default):
    index = number - 1

    if index < len(existing_screws):
        return existing_screws[index].get(key, default)

    return default

config = {
    "reference_path": REFERENCE_PATH,
    "screws": [
        {
            "id": number,
            "center": [x, y],
            "crop_size": screw_setting(
                number,
                "crop_size",
                DEFAULT_CROP_SIZE
            ),
            "similarity_threshold": screw_setting(
                number,
                "similarity_threshold",
                DEFAULT_SIMILARITY_THRESHOLD
            )
        }
        for number, (x, y) in enumerate(points, start=1)
    ]
}

with open(CONFIG_PATH, "w", encoding="utf-8") as config_file:
    json.dump(config, config_file, indent=2)

print(f"Saved product configuration: {CONFIG_PATH}")
