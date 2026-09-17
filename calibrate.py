"""Learn each ROI's normal similarity range from a known-good board.

Hand-picked thresholds do not work here, and not because they are hard to
guess. Each ROI has a genuinely different normal range: measured on a real
board, a corner ROI drifted 0.740-0.911 between consecutive frames while a
central one held 0.950-0.974. A single global number cannot fit both -- set it
low enough to spare the corner and it stops catching real defects elsewhere.

So instead of choosing a number, measure one. This captures many frames of a
board you know is complete, records each ROI's own mean and spread, and writes
them into the product config. Detection then flags an ROI only when it falls
CALIBRATION_SIGMA standard deviations below its OWN normal.

Run this with EVERY component present, after capture_reference.py and
select_screws.py. Move the board slightly during the run -- a few degrees of
rotation and a centimetre of shift -- so the measured spread reflects the pose
variation the line will actually see. Calibrating on a perfectly still board
produces a band too tight to survive normal handling.

Usage:
    python calibrate.py
"""

import json

import cv2
import numpy as np

import config
from backends import get_backend
from camera_source import create_camera
from registration import Registrar
from roi import crop_square_with_padding
from scoring import calibrated_threshold, score_rois


# An ROI whose baseline sits this far below the median is almost certainly not
# just a harder ROI. Measured on a real board, simulated pose change (3 degrees
# of rotation, 15 px of shift, 2% of scale) moved any ROI by at most 0.09,
# while a hand entering frame over an ROI cost 0.26.
SUSPECT_BASELINE_GAP = 0.15


def _warn_about_suspect_baselines(screws, scores):
    """Flag ROIs whose learned "normal" looks like it captured a problem.

    Calibration is trusting by nature: whatever it sees, it learns as normal.
    Calibrate with a screw missing and that ROI's baseline becomes the missing
    state, so the defect can never be flagged again. The same happens if a hand
    or its shadow crosses an ROI during the run.

    Comparing each baseline against the median across ROIs catches both,
    because a genuine defect or occlusion drops one ROI far below its peers.
    """
    means = np.array([screw["baseline_mean"] for screw in screws])
    median = float(np.median(means))

    suspects = [
        (screw, mean)
        for screw, mean in zip(screws, means)
        if median - mean > SUSPECT_BASELINE_GAP
    ]

    if not suspects:
        print("\nAll baselines are consistent with each other.")
        return

    print(
        f"\nWARNING: {len(suspects)} ROI(s) learned a baseline far below the "
        f"median of {median:.3f}:"
    )

    for screw, mean in suspects:
        print(f"    S{screw['id']}: {mean:.3f}")

    print(
        "\nThat usually means the component was missing during calibration, "
        "or something crossed the ROI -- a hand, or its shadow. Either way the "
        "problem has been learned as normal, and those ROIs will no longer "
        "report a defect.\n"
        "Re-run with every component present and nothing entering frame. To "
        "vary the pose, nudge the board from outside the camera's view."
    )


def main():
    with open(config.CONFIG_PATH, "r", encoding="utf-8") as handle:
        product = json.load(handle)

    screws = product.get("screws", [])

    if not screws:
        raise SystemExit(
            f"{config.CONFIG_PATH} has no screws. Run select_screws.py first."
        )

    reference = cv2.imread(product["reference_path"])

    if reference is None:
        raise SystemExit(
            f"Could not load {product['reference_path']}. "
            f"Run capture_reference.py first."
        )

    points = [tuple(screw["center"]) for screw in screws]
    sizes = [screw["crop_size"] for screw in screws]

    registrar = Registrar(reference)
    backend = get_backend(config.MODEL_NAME)
    reference_features = backend.embed(
        [
            crop_square_with_padding(reference, x, y, size)
            for (x, y), size in zip(points, sizes)
        ],
        pooling=config.POOLING,
    )

    camera = create_camera()
    print(f"Camera: {camera.name}")
    print(f"Model:  {backend.name}/{config.POOLING}")
    print(
        f"\nCapturing {config.CALIBRATION_FRAMES} frames of a COMPLETE board."
    )
    print("Move the board gently while this runs.\n")

    samples = []
    skipped = 0

    try:
        while len(samples) < config.CALIBRATION_FRAMES:
            ok, frame = camera.read()

            if not ok:
                continue

            homography, _ = registrar.register(frame)

            if homography is None:
                skipped += 1
                continue

            aligned = registrar.warp(frame, homography)
            crops = [
                crop_square_with_padding(aligned, x, y, size)
                for (x, y), size in zip(points, sizes)
            ]
            samples.append(
                score_rois(
                    reference_features,
                    backend.embed(crops, pooling=config.POOLING),
                )
            )
            print(
                f"  {len(samples)}/{config.CALIBRATION_FRAMES}", end="\r"
            )
    finally:
        camera.release()

    scores = np.array(samples)
    print(f"\nCaptured {len(scores)} frames ({skipped} unregisterable).\n")

    print(f"{'ROI':>5} {'mean':>7} {'std':>7} {'min':>7} {'threshold':>10}")

    for index, screw in enumerate(screws):
        column = scores[:, index]
        screw["baseline_mean"] = round(float(column.mean()), 4)
        screw["baseline_std"] = round(float(column.std()), 4)
        threshold = calibrated_threshold(
            column.mean(),
            column.std(),
            config.CALIBRATION_SIGMA,
            config.CALIBRATION_MIN_STD,
        )
        print(
            f"  S{screw['id']:<3d} {column.mean():7.3f} {column.std():7.3f} "
            f"{column.min():7.3f} {threshold:10.3f}"
        )

    _warn_about_suspect_baselines(screws, scores)

    product["calibration"] = {
        "frames": len(scores),
        "model": backend.name,
        "pooling": config.POOLING,
        "sigma": config.CALIBRATION_SIGMA,
    }

    with open(config.CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(product, handle, indent=2)

    print(f"\nWrote per-ROI baselines to {config.CONFIG_PATH}.")
    print("Detection will now use these instead of fixed thresholds.")


if __name__ == "__main__":
    main()
