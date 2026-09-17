"""Per-ROI similarity scoring and verdict debouncing.

Extracted so the live detection path and the evaluation harness apply
*identical* decision logic. If these rules were duplicated, the benchmark would
gradually stop predicting what the deployed system actually does -- which is the
one failure that would make the whole evaluation worthless.
"""

import numpy as np

UNKNOWN = "UNKNOWN"
PRESENT = "PRESENT"
MISSING = "MISSING"


def score_rois(reference_features, query_features):
    """Cosine similarity per ROI, as a [N] float array.

    Both inputs are [N, D] and unit-norm -- the backends guarantee that at their
    boundary -- so the cosine reduces to a row-wise dot product.
    """
    reference_features = np.asarray(reference_features)
    query_features = np.asarray(query_features)

    if reference_features.shape != query_features.shape:
        raise ValueError(
            f"reference and query shape mismatch: "
            f"{reference_features.shape} vs {query_features.shape}"
        )

    return np.sum(reference_features * query_features, axis=1)


class Debouncer:
    """Hysteresis over per-ROI threshold decisions.

    A verdict only changes after `debounce_count` consecutive contrary
    detections, so a single noisy frame cannot toggle it. Counters clamp at
    `debounce_count` rather than growing without bound, so a long run of one
    verdict does not delay a later genuine flip.
    """

    def __init__(self, thresholds, debounce_count):
        self.thresholds = list(thresholds)
        self.debounce_count = debounce_count
        self.reset()

    def reset(self):
        """Clear all verdicts and counters.

        Called when registration is lost: a board swapped while registration is
        down must not inherit the previous board's verdicts.
        """
        count = len(self.thresholds)
        self.states = [UNKNOWN] * count
        self._low_counts = [0] * count
        self._high_counts = [0] * count

    def update(self, scores):
        """Fold one frame of scores in and return the current verdicts."""
        if len(scores) != len(self.thresholds):
            raise ValueError(
                f"expected {len(self.thresholds)} scores, got {len(scores)}"
            )

        for index, score in enumerate(scores):
            if score < self.thresholds[index]:
                self._low_counts[index] = min(
                    self._low_counts[index] + 1, self.debounce_count
                )
                self._high_counts[index] = 0

                if self._low_counts[index] >= self.debounce_count:
                    self.states[index] = MISSING
            else:
                self._high_counts[index] = min(
                    self._high_counts[index] + 1, self.debounce_count
                )
                self._low_counts[index] = 0

                if self._high_counts[index] >= self.debounce_count:
                    self.states[index] = PRESENT

        return list(self.states)
