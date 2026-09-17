"""SIFT + FLANN + RANSAC homography registration of a frame onto a reference."""

import cv2
import numpy as np

import config


class Registrar:
    """Registers frames onto a fixed reference image.

    Reference keypoints are computed once at construction. Matching runs at
    `match_scale` for speed, and the resulting homography is rescaled back to
    full resolution before being returned, so callers always work in full-image
    coordinates.
    """

    def __init__(
        self,
        reference,
        match_scale=None,
        min_good_matches=None,
        lowe_ratio=None,
        ransac_threshold=None,
    ):
        self.match_scale = (
            config.MATCH_SCALE if match_scale is None else match_scale
        )
        self.min_good_matches = (
            config.MIN_GOOD_MATCHES if min_good_matches is None
            else min_good_matches
        )
        self.lowe_ratio = (
            config.LOWE_RATIO if lowe_ratio is None else lowe_ratio
        )
        self.ransac_threshold = (
            config.RANSAC_REPROJECTION_THRESHOLD if ransac_threshold is None
            else ransac_threshold
        )

        self.reference_height, self.reference_width = reference.shape[:2]

        reference_small = cv2.resize(
            reference, None, fx=self.match_scale, fy=self.match_scale
        )
        reference_gray = cv2.cvtColor(reference_small, cv2.COLOR_BGR2GRAY)

        self.sift = cv2.SIFT_create()
        self.reference_keypoints, self.reference_descriptors = (
            self.sift.detectAndCompute(reference_gray, None)
        )

        if self.reference_descriptors is None:
            raise RuntimeError(
                "The reference image has no SIFT features. It is probably "
                "blank, out of focus, or badly exposed."
            )

        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50),
        )

        scale_matrix = np.array(
            [[self.match_scale, 0, 0], [0, self.match_scale, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        self.scale_matrix = scale_matrix
        self.inverse_scale_matrix = np.linalg.inv(scale_matrix)

    def register(self, frame):
        """Return (homography_full_resolution, inlier_count).

        The homography maps `frame` onto the reference. Returns (None, 0) when
        the frame cannot be registered.
        """
        frame_small = cv2.resize(
            frame, None, fx=self.match_scale, fy=self.match_scale
        )
        frame_gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        frame_keypoints, frame_descriptors = self.sift.detectAndCompute(
            frame_gray, None
        )

        if frame_descriptors is None or len(frame_descriptors) < 2:
            return None, 0

        pairs = self.matcher.knnMatch(
            frame_descriptors, self.reference_descriptors, k=2
        )

        good_matches = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < self.lowe_ratio * second.distance
        ]

        if len(good_matches) < self.min_good_matches:
            return None, 0

        frame_points = np.float32(
            [frame_keypoints[match.queryIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)

        reference_points = np.float32(
            [self.reference_keypoints[match.trainIdx].pt for match in good_matches]
        ).reshape(-1, 1, 2)

        homography_small, mask = cv2.findHomography(
            frame_points,
            reference_points,
            cv2.RANSAC,
            self.ransac_threshold,
        )

        if homography_small is None:
            return None, 0

        homography_full = (
            self.inverse_scale_matrix @ homography_small @ self.scale_matrix
        )

        return homography_full, int(mask.sum())

    def warp(self, frame, homography):
        """Warp `frame` onto the reference frame of view."""
        return cv2.warpPerspective(
            frame, homography, (self.reference_width, self.reference_height)
        )
