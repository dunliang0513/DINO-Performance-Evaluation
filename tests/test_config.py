import importlib

import pytest


@pytest.fixture(autouse=True)
def restore_config():
    """Reload config after each test so overrides never leak into other modules."""
    yield
    import config
    importlib.reload(config)


def test_defaults_target_the_usb_webcam():
    import config
    importlib.reload(config)

    assert config.CAMERA_BACKEND == "usb"
    # The external BRIO, not the laptop's built-in webcam at index 0.
    assert config.USB_CAMERA_INDEX == 2
    # The 180-degree rotation was a Basler mounting artifact, not a scene property.
    assert config.ROTATE_180 is False


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("CAMERA_BACKEND", "basler")
    monkeypatch.setenv("USB_CAMERA_INDEX", "4")
    monkeypatch.setenv("ROTATE_180", "true")
    monkeypatch.setenv("MATCH_SCALE", "0.5")

    import config
    importlib.reload(config)

    assert config.CAMERA_BACKEND == "basler"
    assert config.USB_CAMERA_INDEX == 4
    assert config.ROTATE_180 is True
    assert config.MATCH_SCALE == 0.5


def test_bool_parsing_accepts_common_spellings(monkeypatch):
    for raw, expected in [("1", True), ("yes", True), ("ON", True),
                          ("0", False), ("false", False), ("no", False)]:
        monkeypatch.setenv("ROTATE_180", raw)
        import config
        importlib.reload(config)
        assert config.ROTATE_180 is expected, f"{raw!r} should parse to {expected}"
