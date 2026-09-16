import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor
from camera_source import create_camera
from trt_feature_extractor import TensorRTFeatureExtractor
import time

CONFIG_PATH = "product_config.json"
CAMERA_BACKEND = "basler"
ROTATE_180 = True
USB_CAMERA_INDEX = 4
USB_WIDTH = 1920
USB_HEIGHT = 1080
BASLER_SERIAL = ""
BASLER_TIMEOUT_MS = 2000
# MATCH_SCALE = 0.5
MATCH_SCALE = 0.25
MIN_GOOD_MATCHES = 30
REGISTRATION_INTERVAL = 3
MAX_REGISTRATION_FAILURES = 3

# DETECTION_INTERVAL = 5
# DEBOUNCE_COUNT = 5
DETECTION_INTERVAL = 2
DEBOUNCE_COUNT = 3

MODEL_NAME = "facebook/dinov2-small"
TENSORRT_ENGINE_PATH = "dinov2_small_features_fp16.engine"

with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
    product_config = json.load(config_file)

REFERENCE_PATH = product_config["reference_path"]
screws = product_config["screws"]

if not screws:
    raise RuntimeError("The product configuration contains no screws.")

SCREW_POINTS = [tuple(screw["center"]) for screw in screws]
SCREW_SIZES = [screw["crop_size"] for screw in screws]
SCREW_THRESHOLDS = [
    screw["similarity_threshold"] for screw in screws
]

reference = cv2.imread(REFERENCE_PATH)

if reference is None:
    raise RuntimeError("Could not load the reference image.")
    
device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    raise RuntimeError("TensorRT mode requires a CUDA device.")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
feature_model = TensorRTFeatureExtractor(
    TENSORRT_ENGINE_PATH,
    device=device
)


def crop_square_with_padding(image, center_x, center_y, crop_size):
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

    crop = image[
        source_y1:source_y2,
        source_x1:source_x2
    ]

    top = source_y1 - y1
    bottom = y2 - source_y2
    left = source_x1 - x1
    right = x2 - source_x2

    return cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )


def extract_center_feature(bgr_image):
    rgb_image = cv2.cvtColor(
        bgr_image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb_image)

    inputs = processor(
        images=pil_image,
        return_tensors="pt"
    )
    return feature_model.infer(inputs["pixel_values"])

reference_height, reference_width = reference.shape[:2]

reference_small = cv2.resize(
    reference,
    None,
    fx=MATCH_SCALE,
    fy=MATCH_SCALE
)

reference_gray = cv2.cvtColor(
    reference_small,
    cv2.COLOR_BGR2GRAY
)

sift = cv2.SIFT_create()

reference_keypoints, reference_descriptors = (
    sift.detectAndCompute(reference_gray, None)
)

index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)

matcher = cv2.FlannBasedMatcher(
    index_params,
    search_params
)

reference_features = []

for (x, y), crop_size in zip(SCREW_POINTS, SCREW_SIZES):
    reference_crop = crop_square_with_padding(
        reference,
        x,
        y,
        crop_size
    )

    feature = extract_center_feature(reference_crop)
    reference_features.append(feature)

print("Reference screw features are ready.")

reference_features = torch.cat(
    reference_features,
    dim=0
)

latest_scores = [None] * len(SCREW_POINTS)
screw_states = ["UNKNOWN"] * len(SCREW_POINTS)
low_counts = [0] * len(SCREW_POINTS)
high_counts = [0] * len(SCREW_POINTS)
frame_count = 0
last_homography_full = None
last_inlier_count = 0
registration_failures = 0

camera = create_camera(
    backend=CAMERA_BACKEND,
    usb_index=USB_CAMERA_INDEX,
    usb_width=USB_WIDTH,
    usb_height=USB_HEIGHT,
    basler_serial=BASLER_SERIAL,
    basler_timeout_ms=BASLER_TIMEOUT_MS,
    basler_width=USB_WIDTH,
    basler_height=USB_HEIGHT,
    rotate_180=ROTATE_180
)

print(f"Using camera: {camera.name}")

scale_matrix = np.array(
    [
        [MATCH_SCALE, 0, 0],
        [0, MATCH_SCALE, 0],
        [0, 0, 1]
    ],
    dtype=np.float64
)

inverse_scale_matrix = np.linalg.inv(scale_matrix)

print("Live registration started. Press Q to quit.")

frame_time_ms = 0.0
registration_time_ms = 0.0
dino_time_ms = 0.0
display_fps = 0.0
SMOOTHING = 0.1


def smooth_value(old_value, new_value):
    if old_value == 0:
        return new_value

    return (
        old_value * (1.0 - SMOOTHING)
        + new_value * SMOOTHING
    )

while True:

    frame_start = time.perf_counter()

    ok, frame = camera.read()

    if not ok:
        print("Could not read a camera frame.")
        break

    frame_count += 1

    registration_start = time.perf_counter()
    aligned_frame = None
    status = "Registration failed"
    status_color = (0, 0, 255)
    registration_updated = False
    registration_attempted = (
        last_homography_full is None
        or frame_count % REGISTRATION_INTERVAL == 0
    )

    if registration_attempted:
        frame_small = cv2.resize(
            frame,
            None,
            fx=MATCH_SCALE,
            fy=MATCH_SCALE
        )

        frame_gray = cv2.cvtColor(
            frame_small,
            cv2.COLOR_BGR2GRAY
        )

        frame_keypoints, frame_descriptors = (
            sift.detectAndCompute(frame_gray, None)
        )

        new_homography_full = None
        new_inlier_count = 0

        if frame_descriptors is not None:
            pairs = matcher.knnMatch(
                frame_descriptors,
                reference_descriptors,
                k=2
            )

            good_matches = []

            for pair in pairs:
                if len(pair) != 2:
                    continue

                first, second = pair

                if first.distance < 0.7 * second.distance:
                    good_matches.append(first)

            if len(good_matches) >= MIN_GOOD_MATCHES:
                frame_points = np.float32(
                    [
                        frame_keypoints[match.queryIdx].pt
                        for match in good_matches
                    ]
                ).reshape(-1, 1, 2)

                reference_points = np.float32(
                    [
                        reference_keypoints[match.trainIdx].pt
                        for match in good_matches
                    ]
                ).reshape(-1, 1, 2)

                homography_small, mask = cv2.findHomography(
                    frame_points,
                    reference_points,
                    cv2.RANSAC,
                    5.0
                )

                if homography_small is not None:
                    new_homography_full = (
                        inverse_scale_matrix
                        @ homography_small
                        @ scale_matrix
                    )
                    new_inlier_count = int(mask.sum())

        if new_homography_full is not None:
            last_homography_full = new_homography_full
            last_inlier_count = new_inlier_count
            registration_failures = 0
            registration_updated = True
        else:
            registration_failures += 1

            if registration_failures >= MAX_REGISTRATION_FAILURES:
                last_homography_full = None
                last_inlier_count = 0

    if last_homography_full is not None:
        aligned_frame = cv2.warpPerspective(
            frame,
            last_homography_full,
            (reference_width, reference_height)
        )

        if registration_updated:
            registration_mode = "Updated"
        elif registration_failures > 0:
            registration_mode = (
                f"Stale {registration_failures}/"
                f"{MAX_REGISTRATION_FAILURES}"
            )
        else:
            registration_mode = "Cached"

        status = (
            f"Registered | Inliers: {last_inlier_count} | "
            f"{registration_mode}"
        )
        status_color = (0, 255, 0)

    current_registration_ms = (
        time.perf_counter() - registration_start
    ) * 1000

    registration_time_ms = smooth_value(
        registration_time_ms,
        current_registration_ms
    )
    
    if aligned_frame is None:
        display = frame.copy()
    else:
        display = aligned_frame.copy()

        if frame_count % DETECTION_INTERVAL == 0:

            dino_start = time.perf_counter()

            query_crops = []

            for (x, y), crop_size in zip(
                SCREW_POINTS,
                SCREW_SIZES
            ):
                crop = crop_square_with_padding(
                    aligned_frame,
                    x,
                    y,
                    crop_size
                )

                crop_rgb = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2RGB
                )

                query_crops.append(
                    Image.fromarray(crop_rgb)
                )

            inputs = processor(
                images=query_crops,
                return_tensors="pt"
            )
            query_features = feature_model.infer(
                inputs["pixel_values"]
            )

            scores = F.cosine_similarity(
                reference_features,
                query_features,
                dim=1
            )

            latest_scores = scores.cpu().tolist()

            current_dino_ms = (
                time.perf_counter() - dino_start
            ) * 1000

            dino_time_ms = smooth_value(
                dino_time_ms,
                current_dino_ms
            )

            for index, score in enumerate(latest_scores):
                threshold = SCREW_THRESHOLDS[index]

                if score < threshold:
                    low_counts[index] += 1
                    high_counts[index] = 0

                    if low_counts[index] >= DEBOUNCE_COUNT:
                        screw_states[index] = "MISSING"
                        low_counts[index] = DEBOUNCE_COUNT
                else:
                    high_counts[index] += 1
                    low_counts[index] = 0

                    if high_counts[index] >= DEBOUNCE_COUNT:
                        screw_states[index] = "PRESENT"
                        high_counts[index] = DEBOUNCE_COUNT

        for number, ((x, y), crop_size, score, screw_state) in enumerate(
            zip(
                SCREW_POINTS,
                SCREW_SIZES,
                latest_scores,
                screw_states
            ),
            start=1
        ):
            index = number - 1
            half = crop_size // 2
            x1 = x - half
            y1 = y - half
            x2 = x + half
            y2 = y + half

            if score is None:
                color = (0, 255, 255)
                label = f"S{number}: WAIT"
            elif screw_state == "MISSING":
                color = (0, 0, 255)
                label = f"S{number}: MISSING {score:.2f}"

                if high_counts[index] > 0:
                    label += (
                        f" H{high_counts[index]}/{DEBOUNCE_COUNT}"
                    )
            elif screw_state == "PRESENT":
                color = (0, 255, 0)
                label = f"S{number}: PRESENT {score:.2f}"

                if low_counts[index] > 0:
                    label += (
                        f" L{low_counts[index]}/{DEBOUNCE_COUNT}"
                    )
            else:
                color = (0, 255, 255)
                pending_count = max(
                    low_counts[index],
                    high_counts[index]
                )
                label = (
                    f"S{number}: CHECK "
                    f"{pending_count}/{DEBOUNCE_COUNT}"
                )

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                color,
                3
            )

            cv2.putText(
                display,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

    cv2.putText(
        display,
        status,
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    current_frame_ms = (
        time.perf_counter() - frame_start
    ) * 1000

    frame_time_ms = smooth_value(
        frame_time_ms,
        current_frame_ms
    )

    if frame_time_ms > 0:
        display_fps = 1000.0 / frame_time_ms

    metrics_text = (
        f"FPS: {display_fps:.1f} | "
        f"Frame: {frame_time_ms:.1f} ms | "
        f"Reg: {registration_time_ms:.1f} ms | "
        f"DINO: {dino_time_ms:.1f} ms"
    )

    cv2.putText(
        display,
        metrics_text,
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # preview = cv2.resize(display, (960, 540))
    preview = cv2.resize(display, (1280, 720))
    cv2.imshow("Live Registration", preview)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
