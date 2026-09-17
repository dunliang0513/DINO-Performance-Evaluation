"""Live screw-presence detection over a camera stream.

Usage:
    python live_detection.py --model dinov2-small --pooling cls
"""

import argparse
import json
import time

import cv2
import numpy as np
import torch

import config
from backends import AVAILABLE_MODELS, get_backend
from camera_source import create_camera
from registration import Registrar
from roi import crop_square_with_padding

SMOOTHING = 0.1


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=config.MODEL_NAME, choices=AVAILABLE_MODELS
    )
    parser.add_argument(
        "--pooling", default=config.POOLING, choices=("cls", "mean")
    )
    parser.add_argument("--device", default=config.DEVICE)
    return parser.parse_args()


def smooth(previous, current):
    if previous == 0:
        return current
    return previous * (1.0 - SMOOTHING) + current * SMOOTHING


def synchronise(device):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_product():
    try:
        with open(config.CONFIG_PATH, "r", encoding="utf-8") as handle:
            product = json.load(handle)
    except FileNotFoundError:
        raise SystemExit(
            f"{config.CONFIG_PATH} not found. Run `python select_screws.py` "
            f"to define the screw positions first."
        )

    screws = product.get("screws", [])

    if not screws:
        raise SystemExit(
            f"{config.CONFIG_PATH} contains no screws. Re-run select_screws.py."
        )

    reference = cv2.imread(product["reference_path"])

    if reference is None:
        raise SystemExit(
            f"Could not load the reference image {product['reference_path']}. "
            f"Run `python capture_reference.py` first."
        )

    return reference, screws


def main():
    arguments = parse_arguments()
    reference, screws = load_product()

    points = [tuple(screw["center"]) for screw in screws]
    sizes = [screw["crop_size"] for screw in screws]
    thresholds = [screw["similarity_threshold"] for screw in screws]

    registrar = Registrar(reference)
    backend = get_backend(arguments.model, device=arguments.device)

    reference_crops = [
        crop_square_with_padding(reference, x, y, size)
        for (x, y), size in zip(points, sizes)
    ]
    reference_features = backend.embed(reference_crops, pooling=arguments.pooling)

    backend.warmup(len(points))
    print(f"Model: {backend.name} ({arguments.pooling}, dim {backend.dim})")

    camera = create_camera()
    print(f"Camera: {camera.name}")
    print("Press Q to quit.")

    scores = [None] * len(points)
    states = ["UNKNOWN"] * len(points)
    low_counts = [0] * len(points)
    high_counts = [0] * len(points)

    homography = None
    inlier_count = 0
    failures = 0
    frame_index = 0
    frame_ms = registration_ms = dino_ms = 0.0

    try:
        while True:
            frame_start = time.perf_counter()

            ok, frame = camera.read()

            if not ok:
                print("Could not read a camera frame.")
                break

            frame_index += 1

            registration_start = time.perf_counter()
            should_register = (
                homography is None
                or frame_index % config.REGISTRATION_INTERVAL == 0
            )

            if should_register:
                new_homography, new_inliers = registrar.register(frame)

                if new_homography is not None:
                    homography = new_homography
                    inlier_count = new_inliers
                    failures = 0
                else:
                    failures += 1

                    if failures >= config.MAX_REGISTRATION_FAILURES:
                        homography = None
                        inlier_count = 0

            registration_ms = smooth(
                registration_ms,
                (time.perf_counter() - registration_start) * 1000,
            )

            if homography is None:
                display = frame.copy()
                status = "Registration failed"
                status_colour = (0, 0, 255)
            else:
                aligned = registrar.warp(frame, homography)
                display = aligned.copy()
                status = f"Registered | Inliers: {inlier_count}"
                status_colour = (0, 255, 0)

                if frame_index % config.DETECTION_INTERVAL == 0:
                    synchronise(arguments.device)
                    dino_start = time.perf_counter()

                    crops = [
                        crop_square_with_padding(aligned, x, y, size)
                        for (x, y), size in zip(points, sizes)
                    ]
                    query_features = backend.embed(
                        crops, pooling=arguments.pooling
                    )

                    synchronise(arguments.device)
                    dino_ms = smooth(
                        dino_ms, (time.perf_counter() - dino_start) * 1000
                    )

                    # Both sides are unit-norm, so the dot product is the cosine.
                    scores = np.sum(
                        reference_features * query_features, axis=1
                    ).tolist()

                    for index, score in enumerate(scores):
                        if score < thresholds[index]:
                            low_counts[index] += 1
                            high_counts[index] = 0

                            if low_counts[index] >= config.DEBOUNCE_COUNT:
                                states[index] = "MISSING"
                                low_counts[index] = config.DEBOUNCE_COUNT
                        else:
                            high_counts[index] += 1
                            low_counts[index] = 0

                            if high_counts[index] >= config.DEBOUNCE_COUNT:
                                states[index] = "PRESENT"
                                high_counts[index] = config.DEBOUNCE_COUNT

                for number, ((x, y), size, score, state) in enumerate(
                    zip(points, sizes, scores, states), start=1
                ):
                    half = size // 2

                    if score is None:
                        colour = (0, 255, 255)
                        label = f"S{number}: WAIT"
                    elif state == "MISSING":
                        colour = (0, 0, 255)
                        label = f"S{number}: MISSING {score:.2f}"
                    elif state == "PRESENT":
                        colour = (0, 255, 0)
                        label = f"S{number}: PRESENT {score:.2f}"
                    else:
                        colour = (0, 255, 255)
                        label = f"S{number}: CHECK {score:.2f}"

                    cv2.rectangle(
                        display, (x - half, y - half), (x + half, y + half),
                        colour, 3,
                    )
                    cv2.putText(
                        display, label, (x - half, y - half - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2,
                    )

            frame_ms = smooth(
                frame_ms, (time.perf_counter() - frame_start) * 1000
            )
            fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0

            cv2.putText(
                display, status, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_colour, 2,
            )
            cv2.putText(
                display,
                f"{backend.name}/{arguments.pooling} | FPS: {fps:.1f} | "
                f"Frame: {frame_ms:.1f} ms | Reg: {registration_ms:.1f} ms | "
                f"DINO: {dino_ms:.1f} ms",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            preview = cv2.resize(
                display,
                (config.PREVIEW_MAX_WIDTH, config.PREVIEW_MAX_HEIGHT),
            )
            cv2.imshow("Live Detection", preview)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
