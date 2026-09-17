import cv2
import numpy as np
import pytest

import camera_source


class FakeCapture:
    """Stand-in for cv2.VideoCapture that records the calls made to it."""

    def __init__(self, index, opened=True, granted=(1920, 1080)):
        self.index = index
        self.opened = opened
        self.granted_width, self.granted_height = granted
        self.calls = []
        self.released = False
        self.read_count = 0

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.calls.append((prop, value))
        return True

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
