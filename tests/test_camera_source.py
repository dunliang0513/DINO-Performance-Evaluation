import cv2
import numpy as np
import pytest

import camera_source
import config


class FakeCapture:
    """Stand-in for cv2.VideoCapture that records the calls made to it."""

    def __init__(self, index, opened=True, granted=(1920, 1080), refuse=False):
        self.index = index
        self.opened = opened
        self.refuse = refuse
        self.granted_width, self.granted_height = granted
        self.calls = []
        self.released = False
        self.read_count = 0

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.calls.append((prop, value))
        return not self.refuse

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.granted_width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.granted_height)
        return 0.0

    def read(self):
        self.read_count += 1
        frame = np.zeros((self.granted_height, self.granted_width, 3), np.uint8)
        return True, frame

    def release(self):
        self.released = True


@pytest.fixture
def fake_capture(monkeypatch):
    created = {}

    def factory(index, *args, **kwargs):
        capture = FakeCapture(index, **created.get("kwargs", {}))
        created["capture"] = capture
        return capture

    monkeypatch.setattr(camera_source.cv2, "VideoCapture", factory)
    return created


def test_sets_mjpg_fourcc(fake_capture):
    camera_source.UsbCamera(0, 1920, 1080)
    capture = fake_capture["capture"]

    props = [prop for prop, _ in capture.calls]
    assert cv2.CAP_PROP_FOURCC in props, (
        "MJPG must be requested or the webcam may drop to ~5 fps at 1080p"
    )


def test_checks_opened_before_configuring(fake_capture):
    fake_capture["kwargs"] = {"opened": False}

    with pytest.raises(RuntimeError, match="Could not open"):
        camera_source.UsbCamera(3, 1920, 1080)

    # The capture must not have been configured after failing to open.
    assert fake_capture["capture"].calls == []


def test_warns_when_resolution_is_not_granted(fake_capture, capsys):
    fake_capture["kwargs"] = {"granted": (1280, 720)}

    camera = camera_source.UsbCamera(0, 1920, 1080)

    assert camera.width == 1280
    assert camera.height == 720
    assert "1280x720" in capsys.readouterr().out


def test_flushes_warmup_frames(fake_capture):
    camera_source.UsbCamera(0, 1920, 1080, warmup_frames=5)
    assert fake_capture["capture"].read_count == 5


def test_locks_focus_exposure_and_white_balance(fake_capture):
    """Auto controls drift between reference capture and inference.

    Measured on a BRIO: autofocus hunting left a live frame at 8% of the
    reference's sharpness, which alone dropped an intact ROI to 0.54 -- the
    same score a genuinely removed screw produced. If these are not pinned,
    the similarity comparison measures camera state, not screw presence.
    """
    camera_source.UsbCamera(0, 1920, 1080)
    props = dict(fake_capture["capture"].calls)

    assert props.get(cv2.CAP_PROP_AUTOFOCUS) == 0, "autofocus must be disabled"
    assert props.get(cv2.CAP_PROP_FOCUS) == config.FOCUS_ABSOLUTE
    assert cv2.CAP_PROP_AUTO_EXPOSURE in props, "exposure must be pinned"
    assert props.get(cv2.CAP_PROP_AUTO_WB) == 0, "white balance must be pinned"


def test_control_locking_can_be_disabled(fake_capture, monkeypatch):
    monkeypatch.setattr(config, "LOCK_CAMERA_CONTROLS", False)

    camera_source.UsbCamera(0, 1920, 1080)
    props = dict(fake_capture["capture"].calls)

    assert cv2.CAP_PROP_AUTOFOCUS not in props


def test_warns_but_survives_a_camera_that_refuses_the_controls(
    fake_capture, capsys
):
    """A webcam without manual focus is still usable, just less reliable."""
    fake_capture["kwargs"] = {"refuse": True}

    camera = camera_source.UsbCamera(0, 1920, 1080)

    assert camera.name, "construction must still succeed"
    assert "refused to lock" in capsys.readouterr().out


def test_pins_an_explicit_exposure_value(fake_capture):
    """Manual exposure without a value is worse than leaving it automatic.

    Switching a UVC camera to manual mode pins whatever exposure it happens to
    be holding. On a BRIO that produced mean brightness 70 where auto gave 124
    -- dark enough to change every embedding. Setting the mode is not enough;
    the value has to be set too.
    """
    camera_source.UsbCamera(0, 1920, 1080)
    props = dict(fake_capture["capture"].calls)

    assert props.get(cv2.CAP_PROP_EXPOSURE) == config.EXPOSURE_ABSOLUTE
