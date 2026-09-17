import numpy as np

from calibrate import SUSPECT_BASELINE_GAP, _warn_about_suspect_baselines


def _screws(means):
    return [
        {"id": index + 1, "baseline_mean": mean}
        for index, mean in enumerate(means)
    ]


def test_flags_a_baseline_that_learned_a_missing_component(capsys):
    """Calibration trusts whatever it sees.

    Run it with a screw missing and that ROI's baseline becomes the missing
    state -- after which the defect can never be reported. The same happens if
    a hand or shadow crosses an ROI mid-run.
    """
    means = [0.92, 0.93, 0.68, 0.91, 0.94]
    _warn_about_suspect_baselines(_screws(means), np.zeros((30, 5)))

    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "S3" in output, "the low ROI must be named"
    assert "S1" not in output, "healthy ROIs must not be flagged"


def test_stays_quiet_when_every_baseline_agrees(capsys):
    _warn_about_suspect_baselines(
        _screws([0.92, 0.93, 0.89, 0.91, 0.94]), np.zeros((30, 5))
    )

    output = capsys.readouterr().out
    assert "WARNING" not in output
    assert "consistent" in output


def test_ordinary_pose_variation_does_not_trip_the_warning():
    """Measured on a real board: 3 degrees of rotation, 15 px of shift and 2%
    of scale moved any ROI by at most 0.09. A hand crossing an ROI cost 0.26.
    The gap must sit between those, or it either cries wolf on normal
    handling or misses real occlusion.
    """
    assert 0.09 < SUSPECT_BASELINE_GAP < 0.26


def test_flags_several_bad_rois_at_once(capsys):
    _warn_about_suspect_baselines(
        _screws([0.92, 0.69, 0.93, 0.67, 0.91]), np.zeros((30, 5))
    )

    output = capsys.readouterr().out
    assert "2 ROI(s)" in output
    assert "S2" in output and "S4" in output
