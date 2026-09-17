import numpy as np
import pytest

from scoring import Debouncer, score_rois


def _unit(rows):
    features = np.array(rows, dtype=np.float32)
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def test_identical_embeddings_score_one():
    features = _unit([[1.0, 0.0], [0.6, 0.8]])
    scores = score_rois(features, features.copy())
    assert np.allclose(scores, 1.0, atol=1e-6)


def test_orthogonal_embeddings_score_zero():
    scores = score_rois(_unit([[1.0, 0.0]]), _unit([[0.0, 1.0]]))
    assert np.allclose(scores, 0.0, atol=1e-6)


def test_scores_are_per_roi_not_aggregated():
    reference = _unit([[1.0, 0.0], [1.0, 0.0]])
    query = _unit([[1.0, 0.0], [0.0, 1.0]])
    scores = score_rois(reference, query)
    assert scores.shape == (2,)
    assert scores[0] > 0.99 and abs(scores[1]) < 1e-6


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="shape"):
        score_rois(_unit([[1.0, 0.0]]), _unit([[1.0, 0.0], [0.0, 1.0]]))


# --- Debouncer -------------------------------------------------------------

def test_starts_unknown():
    debouncer = Debouncer(thresholds=[0.75, 0.75], debounce_count=3)
    assert debouncer.states == ["UNKNOWN", "UNKNOWN"]


def test_flips_to_present_only_after_enough_consecutive_detections():
    debouncer = Debouncer(thresholds=[0.75], debounce_count=3)

    assert debouncer.update([0.9]) == ["UNKNOWN"]
    assert debouncer.update([0.9]) == ["UNKNOWN"]
    assert debouncer.update([0.9]) == ["PRESENT"], "should flip on the third"


def test_flips_to_missing_only_after_enough_consecutive_detections():
    debouncer = Debouncer(thresholds=[0.75], debounce_count=3)

    assert debouncer.update([0.1]) == ["UNKNOWN"]
    assert debouncer.update([0.1]) == ["UNKNOWN"]
    assert debouncer.update([0.1]) == ["MISSING"]


def test_a_single_contrary_frame_does_not_flip_an_established_state():
    """The whole point of hysteresis: one noisy frame must not toggle a verdict."""
    debouncer = Debouncer(thresholds=[0.75], debounce_count=3)
    for _ in range(3):
        debouncer.update([0.9])
    assert debouncer.states == ["PRESENT"]

    assert debouncer.update([0.1]) == ["PRESENT"], "one bad frame flipped it"
    assert debouncer.update([0.9]) == ["PRESENT"]


def test_sustained_contrary_evidence_does_flip_an_established_state():
    debouncer = Debouncer(thresholds=[0.75], debounce_count=3)
    for _ in range(3):
        debouncer.update([0.9])

    debouncer.update([0.1])
    debouncer.update([0.1])
    assert debouncer.update([0.1]) == ["MISSING"]


def test_oscillating_scores_hold_the_current_state():
    debouncer = Debouncer(thresholds=[0.75], debounce_count=3)
    for _ in range(3):
        debouncer.update([0.9])

    for score in [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]:
        assert debouncer.update([score]) == ["PRESENT"]


def test_counters_cannot_run_away():
    """Counters clamp at debounce_count, so a long run cannot delay a later flip."""
    debouncer = Debouncer(thresholds=[0.75], debounce_count=3)
    for _ in range(50):
        debouncer.update([0.9])

    debouncer.update([0.1])
    debouncer.update([0.1])
    assert debouncer.update([0.1]) == ["MISSING"], (
        "a 50-frame PRESENT run should still flip after exactly 3 contrary frames"
    )


def test_rois_are_debounced_independently():
    debouncer = Debouncer(thresholds=[0.75, 0.75], debounce_count=2)
    debouncer.update([0.9, 0.1])
    assert debouncer.update([0.9, 0.1]) == ["PRESENT", "MISSING"]


def test_per_roi_thresholds_are_honoured():
    """ROI 0 is lenient, ROI 1 strict; the same score must decide differently."""
    debouncer = Debouncer(thresholds=[0.30, 0.95], debounce_count=1)
    assert debouncer.update([0.5, 0.5]) == ["PRESENT", "MISSING"]


def test_reset_clears_state_and_counters():
    debouncer = Debouncer(thresholds=[0.75], debounce_count=3)
    for _ in range(3):
        debouncer.update([0.9])
    assert debouncer.states == ["PRESENT"]

    debouncer.reset()
    assert debouncer.states == ["UNKNOWN"]
    # Counters must clear too, or one post-reset frame would flip it straight back.
    assert debouncer.update([0.9]) == ["UNKNOWN"]


def test_wrong_number_of_scores_raises():
    debouncer = Debouncer(thresholds=[0.75, 0.75], debounce_count=3)
    with pytest.raises(ValueError, match="2"):
        debouncer.update([0.9])


# --- Calibration -----------------------------------------------------------

from scoring import calibrated_threshold, thresholds_from_config


def test_a_jittery_roi_earns_a_wider_band_than_a_steady_one():
    """The whole point of calibrating instead of hand-picking.

    Measured on a real board: a corner ROI ranged 0.740-0.911 (std 0.049)
    while a central one held 0.950-0.974 (std 0.007). One global threshold
    cannot fit both, so each must be judged against its own spread.
    """
    jittery = calibrated_threshold(0.847, 0.049, sigma=4.0, minimum_std=0.02)
    steady = calibrated_threshold(0.963, 0.007, sigma=4.0, minimum_std=0.02)

    assert jittery < steady, "the noisier ROI must get the more forgiving band"
    assert jittery < 0.740, "must not flag the corner ROI's own worst frame"


def test_minimum_std_stops_an_absurdly_tight_band():
    """A near-zero spread would otherwise make one noisy frame a defect."""
    without_floor = 0.99 - 4.0 * 0.001
    with_floor = calibrated_threshold(0.99, 0.001, sigma=4.0, minimum_std=0.02)

    assert with_floor < without_floor
    assert with_floor == pytest.approx(0.99 - 4.0 * 0.02)


def test_calibrated_baselines_are_preferred_over_fixed_thresholds():
    screws = [
        {"similarity_threshold": 0.75, "baseline_mean": 0.90,
         "baseline_std": 0.02},
    ]
    thresholds, calibrated = thresholds_from_config(screws, 4.0, 0.02)

    assert calibrated == 1
    assert thresholds[0] == pytest.approx(0.90 - 4.0 * 0.02)


def test_uncalibrated_rois_fall_back_to_their_fixed_threshold():
    """Adding a 17th ROI must not silently break the 16 already calibrated."""
    screws = [
        {"similarity_threshold": 0.75, "baseline_mean": 0.90,
         "baseline_std": 0.02},
        {"similarity_threshold": 0.60},
    ]
    thresholds, calibrated = thresholds_from_config(screws, 4.0, 0.02)

    assert calibrated == 1, "only one ROI has a baseline"
    assert thresholds[1] == 0.60
