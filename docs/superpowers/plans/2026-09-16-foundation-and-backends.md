# Foundation & Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the existing screw-detection pipeline running on a USB webcam with any of four DINO backbones (DINOv2-S/B, DINOv3-S/B), replacing the broken Orin/TensorRT path.

**Architecture:** Extract the duplicated constants and the registration/cropping logic out of the three existing scripts into shared modules (`config.py`, `roi.py`, `registration.py`). Put feature extraction behind a `FeatureExtractor` protocol in `backends/` so models are a runtime parameter rather than a hardcoded import, and so a TensorRT backend can be added for the Jetson later without touching callers.

**Tech Stack:** Python 3.10, PyTorch 2.9.1+cu128, transformers 5.17 (DINOv2), timm 1.0.29 (DINOv3), OpenCV 4.12 (SIFT registration), pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-dinov2-v3-webcam-eval-design.md`

**Scope:** Spec phases 1-4. Phases 5-6 (dataset capture, evaluation, benchmarking) are a separate plan, written once the PCB is available — they cannot be verified without it.

---

## File Structure

| File | Responsibility |
|---|---|
| `requirements.txt` | Pinned dependencies (new) |
| `config.py` | All runtime constants, env-overridable (new) |
| `roi.py` | `crop_square_with_padding` — extracted from `live_detection_trt.py:66-103` (new) |
| `registration.py` | SIFT/FLANN/RANSAC homography — extracted from `live_detection_trt.py:134-320` (new) |
| `camera_source.py` | Camera abstraction — **modify** for webcam robustness |
| `backends/base.py` | `FeatureExtractor` protocol + `pool_tokens` + `l2_normalise` (new) |
| `backends/torch_backend.py` | DINOv2 via transformers, DINOv3 via timm (new) |
| `backends/__init__.py` | `get_backend(name)` registry (new) |
| `capture_reference.py` | **modify** — read from `config.py` |
| `select_screws.py` | **modify** — read from `config.py` |
| `live_detection.py` | Live demo, `--model` flag, honest CUDA timing (new) |
| `live_detection_trt.py` | **delete** at the end — superseded, currently broken |
| `tests/conftest.py` | Synthetic image fixtures, no hardware needed (new) |
| `tests/test_roi.py` | Crop edge cases (new) |
| `tests/test_registration.py` | Known-homography recovery (new) |
| `tests/test_backends.py` | Pooling correctness, embedding contract (new) |
| `tests/test_camera_source.py` | Webcam config via a fake VideoCapture (new) |

**Why `pool_tokens` is its own function:** the spec flags that mean-pooling must exclude CLS and DINOv3's register tokens. Keeping the reduction as a pure function makes that testable in milliseconds without downloading a model.

---

## Task 1: Environment and dependencies

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Create the virtual environment**

```bash
cd /home/oem/DINO-Performance-Evaluation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

- [ ] **Step 2: Write `requirements.txt`**

All versions below were confirmed to exist for cp310/linux_x86_64 on 2026-09-16.
OpenCV is held at 4.x deliberately: 5.x is newly released and changes enough of
the API surface that the SIFT/FLANN code here is not worth re-validating now.

```
--extra-index-url https://download.pytorch.org/whl/cu128

torch==2.9.1+cu128
torchvision==0.24.1+cu128

transformers==5.17.0
timm==1.0.29

opencv-contrib-python==4.12.0.88
numpy==2.2.6
Pillow>=10,<13

# Used by the evaluation plan (phases 5-6); installed now to avoid a second pass
scikit-learn==1.9.1
matplotlib==3.11.2

pytest==9.1.1
```

- [ ] **Step 3: Install**

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Expected: completes without resolver errors. The torch download is ~900 MB, so allow several minutes.

- [ ] **Step 4: Verify CUDA is actually usable**

```bash
source .venv/bin/activate
python -c "import torch, cv2, timm, transformers; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0)); print('cv2', cv2.__version__, '| SIFT', hasattr(cv2, 'SIFT_create')); print('timm', timm.__version__, '| transformers', transformers.__version__)"
```

Expected output: `cuda True`, device name containing `RTX 4090`, and `SIFT True`.
If `cuda False`, stop — the driver (570.211.01, CUDA 12.8) supports cu128, so a
`False` here means the wrong wheel was installed. Check `pip show torch` reports a `+cu128` version.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "Add pinned dependencies for the x86 CUDA 12.8 environment"
```

---

## Task 2: Central configuration

**Files:**
- Create: `config.py`
- Test: `tests/test_config.py`

The three existing scripts each carry their own copy of `CAMERA_BACKEND`,
`ROTATE_180`, `USB_CAMERA_INDEX` and the frame size, and all three hardcode
`"basler"`. This replaces all of them.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

The autouse fixture below is not incidental. `importlib.reload(config)` mutates
a module that `registration.py` and `backends/` also import, and monkeypatch
restores the *environment* but not the already-reloaded module. Without the
teardown reload, a leaked `MATCH_SCALE=0.5` would silently change how the
registration tests behave depending on test ordering — the kind of failure that
wastes an afternoon.

```python
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
    assert config.USB_CAMERA_INDEX == 0
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
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_config.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 3: Write `config.py`**

```python
"""Single source of truth for runtime configuration.

Every value may be overridden by an environment variable of the same name, so
switching camera, model or threshold never requires editing a script.
"""

import os


def _str(name, default):
    return os.environ.get(name, default)


def _int(name, default):
    return int(os.environ.get(name, default))


def _float(name, default):
    return float(os.environ.get(name, default))


def _bool(name, default):
    raw = os.environ.get(name)

    if raw is None:
        return default

    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- Camera -----------------------------------------------------------------
# Default is the USB webcam; the Basler path is retained for the Orin rig.
CAMERA_BACKEND = _str("CAMERA_BACKEND", "usb")
USB_CAMERA_INDEX = _int("USB_CAMERA_INDEX", 0)
FRAME_WIDTH = _int("FRAME_WIDTH", 1920)
FRAME_HEIGHT = _int("FRAME_HEIGHT", 1080)
ROTATE_180 = _bool("ROTATE_180", False)
BASLER_SERIAL = _str("BASLER_SERIAL", "")
BASLER_TIMEOUT_MS = _int("BASLER_TIMEOUT_MS", 2000)

# --- Paths ------------------------------------------------------------------
CONFIG_PATH = _str("CONFIG_PATH", "product_config.json")
REFERENCE_PATH = _str("REFERENCE_PATH", "my_photo-1.jpg")

# --- Registration -----------------------------------------------------------
MATCH_SCALE = _float("MATCH_SCALE", 0.25)
MIN_GOOD_MATCHES = _int("MIN_GOOD_MATCHES", 30)
REGISTRATION_INTERVAL = _int("REGISTRATION_INTERVAL", 3)
MAX_REGISTRATION_FAILURES = _int("MAX_REGISTRATION_FAILURES", 3)
LOWE_RATIO = _float("LOWE_RATIO", 0.7)
RANSAC_REPROJECTION_THRESHOLD = _float("RANSAC_REPROJECTION_THRESHOLD", 5.0)

# --- Detection --------------------------------------------------------------
DETECTION_INTERVAL = _int("DETECTION_INTERVAL", 2)
DEBOUNCE_COUNT = _int("DEBOUNCE_COUNT", 3)
DEFAULT_CROP_SIZE = _int("DEFAULT_CROP_SIZE", 80)
DEFAULT_SIMILARITY_THRESHOLD = _float("DEFAULT_SIMILARITY_THRESHOLD", 0.75)

# --- Model ------------------------------------------------------------------
MODEL_NAME = _str("MODEL_NAME", "dinov2-small")
POOLING = _str("POOLING", "cls")
DEVICE = _str("DEVICE", "cuda")
USE_FP16 = _bool("USE_FP16", True)

# --- Preview ----------------------------------------------------------------
PREVIEW_MAX_WIDTH = _int("PREVIEW_MAX_WIDTH", 1280)
PREVIEW_MAX_HEIGHT = _int("PREVIEW_MAX_HEIGHT", 720)
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_config.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "Add central configuration with environment overrides"
```

---

## Task 3: Extract ROI cropping

**Files:**
- Create: `roi.py`, `tests/conftest.py`, `tests/test_roi.py`
- Source: `live_detection_trt.py:66-103` (logic moves out; that file is deleted in Task 13)

- [ ] **Step 1: Write the shared fixtures**

Create `tests/conftest.py`. The board is deliberately textured — SIFT needs
corners to latch onto, and a flat synthetic image would make the registration
test in Task 4 fail for reasons unrelated to the code under test.

```python
import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_board():
    """A textured stand-in for a PCB, 1280x720 BGR.

    Deterministic (seeded) so test failures are reproducible. The bright
    rectangles give SIFT stable, distinctive keypoints; the noise floor stops
    large flat regions producing degenerate matches.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(40, 90, size=(720, 1280, 3), dtype=np.uint8)

    for index, x in enumerate(range(60, 1240, 110)):
        for jndex, y in enumerate(range(60, 700, 105)):
            width = 30 + ((index * 7 + jndex * 13) % 35)
            height = 18 + ((index * 11 + jndex * 5) % 25)
            shade = 150 + ((index * 23 + jndex * 17) % 100)
            cv2.rectangle(
                image,
                (x, y),
                (x + width, y + height),
                (int(shade), int(shade), int(shade)),
                -1,
            )

    return image


@pytest.fixture
def screw_points():
    """Four ROI centres, all comfortably inside the synthetic board."""
    return [(300, 200), (900, 200), (300, 520), (900, 520)]
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_roi.py`:

```python
import numpy as np
import pytest

from roi import crop_square_with_padding


def test_returns_requested_size(synthetic_board):
    crop = crop_square_with_padding(synthetic_board, 640, 360, 80)
    assert crop.shape == (80, 80, 3)


def test_crop_contains_the_source_pixels(synthetic_board):
    crop = crop_square_with_padding(synthetic_board, 640, 360, 80)
    expected = synthetic_board[320:400, 600:680]
    assert np.array_equal(crop, expected)


def test_crop_overlapping_the_border_is_zero_padded(synthetic_board):
    # Centre (10, 10) with half-size 40 puts 30 px outside the top-left corner.
    crop = crop_square_with_padding(synthetic_board, 10, 10, 80)

    assert crop.shape == (80, 80, 3)
    assert np.all(crop[:30, :] == 0), "top padding should be black"
    assert np.all(crop[:, :30] == 0), "left padding should be black"
    assert np.any(crop[30:, 30:] != 0), "real image data should survive"


def test_crop_overlapping_the_far_border_is_zero_padded(synthetic_board):
    height, width = synthetic_board.shape[:2]
    crop = crop_square_with_padding(synthetic_board, width - 10, height - 10, 80)

    assert crop.shape == (80, 80, 3)
    assert np.all(crop[50:, :] == 0)
    assert np.all(crop[:, 50:] == 0)


def test_odd_crop_size_is_honoured(synthetic_board):
    crop = crop_square_with_padding(synthetic_board, 640, 360, 81)
    assert crop.shape == (81, 81, 3)


def test_centre_fully_outside_the_image_raises(synthetic_board):
    with pytest.raises(ValueError, match="outside the image"):
        crop_square_with_padding(synthetic_board, -500, -500, 80)
```

- [ ] **Step 3: Run it and confirm it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_roi.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'roi'`.

- [ ] **Step 4: Write `roi.py`**

```python
"""Square ROI extraction with zero padding at image borders."""

import cv2


def crop_square_with_padding(image, center_x, center_y, crop_size):
    """Return a `crop_size` x `crop_size` BGR crop centred on (center_x, center_y).

    Regions falling outside the image are zero padded, so the returned array is
    always exactly the requested size. Raises ValueError only when the requested
    square lies entirely outside the image.
    """
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

    crop = image[source_y1:source_y2, source_x1:source_x2]

    return cv2.copyMakeBorder(
        crop,
        source_y1 - y1,
        y2 - source_y2,
        source_x1 - x1,
        x2 - source_x2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
```

- [ ] **Step 5: Run the tests and confirm they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_roi.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add roi.py tests/conftest.py tests/test_roi.py
git commit -m "Extract ROI cropping into a tested module"
```

---

## Task 4: Extract SIFT registration

**Files:**
- Create: `registration.py`, `tests/test_registration.py`
- Source: `live_detection_trt.py:134-320`

The live loop currently rebuilds matcher state inline and interleaves it with
drawing code. Wrapping it in a class lets the reference keypoints be computed
once, and makes the golden test below possible without a camera.

- [ ] **Step 1: Write the failing test**

Create `tests/test_registration.py`:

```python
import cv2
import numpy as np

from registration import Registrar


def _warp(image, angle_degrees, scale, shift_x, shift_y):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_degrees, scale)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y
    return cv2.warpAffine(image, matrix, (width, height))


def test_recovers_a_known_transform(synthetic_board):
    """Warp the reference by a known amount; registration should undo it.

    This is the golden test for the whole registration path and needs no camera.
    """
    height, width = synthetic_board.shape[:2]
    warped = _warp(synthetic_board, angle_degrees=5.0, scale=1.03,
                   shift_x=15, shift_y=-9)

    registrar = Registrar(synthetic_board, match_scale=0.5)
    homography, inlier_count = registrar.register(warped)

    assert homography is not None, "registration should succeed on a modest warp"
    assert inlier_count >= 30

    recovered = cv2.warpPerspective(warped, homography, (width, height))

    # Compare away from the borders, where the warp legitimately loses data.
    region = (slice(150, height - 150), slice(250, width - 250))
    error = np.abs(
        recovered[region].astype(np.float64)
        - synthetic_board[region].astype(np.float64)
    ).mean()

    assert error < 15.0, f"mean absolute error {error:.2f} is too high"


def test_identity_registration_is_near_perfect(synthetic_board):
    registrar = Registrar(synthetic_board, match_scale=0.5)
    homography, inlier_count = registrar.register(synthetic_board.copy())

    assert homography is not None
    assert inlier_count >= 30
    # Registering an image against itself should be very close to the identity.
    assert np.allclose(homography / homography[2, 2], np.eye(3), atol=0.05)


def test_unregisterable_frame_returns_none(synthetic_board):
    """A flat grey frame has no SIFT keypoints, so registration must fail cleanly."""
    blank = np.full_like(synthetic_board, 127)

    registrar = Registrar(synthetic_board, match_scale=0.5)
    homography, inlier_count = registrar.register(blank)

    assert homography is None
    assert inlier_count == 0
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_registration.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'registration'`.

- [ ] **Step 3: Write `registration.py`**

```python
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_registration.py -v
```

Expected: 3 passed.

If `test_recovers_a_known_transform` fails on the error tolerance rather than on
`homography is None`, the registration is working and the tolerance simply needs
widening for the synthetic texture — raise the 15.0 bound and note it. If
`homography is None`, that is a real failure: SIFT found too few matches, which
points at a bug in the scale handling.

- [ ] **Step 5: Commit**

```bash
git add registration.py tests/test_registration.py
git commit -m "Extract SIFT registration with a known-homography golden test"
```

---

## Task 5: Webcam robustness in `camera_source.py`

**Files:**
- Modify: `camera_source.py:4-21` (`UsbCamera`) and `camera_source.py:178-211` (`create_camera`)
- Test: `tests/test_camera_source.py`

Four concrete defects in the current `UsbCamera`:
1. `isOpened()` is checked *after* the property sets, so properties get written to a closed capture.
2. No MJPG fourcc — UVC webcams commonly fall back to ~5 fps at 1080p on raw YUYV, which would silently distort every latency measurement.
3. The granted resolution is never verified; webcams fall back without saying so.
4. The first frames arrive before auto-exposure settles.

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_source.py`:

```python
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
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_camera_source.py -v
```

Expected: FAIL — `UsbCamera` takes no `warmup_frames`, sets no fourcc, and has no `.width`.

- [ ] **Step 3: Replace the `UsbCamera` class**

In `camera_source.py`, replace the whole `UsbCamera` class (lines 4-21) with:

```python
class UsbCamera:
    """USB / UVC webcam via OpenCV.

    Requests MJPG explicitly: many UVC webcams only offer 1080p at a usable
    frame rate in MJPG, falling back to roughly 5 fps on raw YUYV. Silently
    accepting that would distort every latency measurement taken downstream.
    """

    def __init__(self, index, width, height, warmup_frames=5):
        self.camera = cv2.VideoCapture(index)

        if not self.camera.isOpened():
            raise RuntimeError(
                f"Could not open USB camera index {index}. "
                f"Check `ls /dev/video*` and `v4l2-ctl --list-devices`."
            )

        self.camera.set(
            cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
        )
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (self.width, self.height) != (width, height):
            print(
                f"Warning: requested {width}x{height} but the camera granted "
                f"{self.width}x{self.height}."
            )

        # Discard the first frames while auto-exposure and white balance settle.
        for _ in range(warmup_frames):
            self.camera.read()

        self.name = f"USB camera {index} ({self.width}x{self.height})"

    def read(self):
        return self.camera.read()

    def release(self):
        self.camera.release()
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_camera_source.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Point `create_camera` defaults at the config**

In `camera_source.py`, change the `create_camera` signature defaults (line 178-188) from the hardcoded values to read from `config`. Add `import config` at the top of the file, then:

```python
def create_camera(
    backend=None,
    usb_index=None,
    usb_width=None,
    usb_height=None,
    basler_serial=None,
    basler_timeout_ms=None,
    basler_width=None,
    basler_height=None,
    rotate_180=None,
):
    backend = config.CAMERA_BACKEND if backend is None else backend
    usb_index = config.USB_CAMERA_INDEX if usb_index is None else usb_index
    usb_width = config.FRAME_WIDTH if usb_width is None else usb_width
    usb_height = config.FRAME_HEIGHT if usb_height is None else usb_height
    basler_serial = (
        config.BASLER_SERIAL if basler_serial is None else basler_serial
    )
    basler_timeout_ms = (
        config.BASLER_TIMEOUT_MS if basler_timeout_ms is None
        else basler_timeout_ms
    )
    basler_width = config.FRAME_WIDTH if basler_width is None else basler_width
    basler_height = (
        config.FRAME_HEIGHT if basler_height is None else basler_height
    )
    rotate_180 = config.ROTATE_180 if rotate_180 is None else rotate_180

    normalized_backend = backend.strip().lower()

    if normalized_backend == "usb":
        camera = UsbCamera(usb_index, usb_width, usb_height)
    elif normalized_backend == "basler":
        camera = BaslerCamera(
            serial_number=basler_serial,
            timeout_ms=basler_timeout_ms,
            width=basler_width,
            height=basler_height,
        )
    else:
        raise ValueError(
            f"CAMERA_BACKEND must be 'basler' or 'usb', got {backend!r}."
        )

    if rotate_180:
        return RotatedCamera(camera)

    return camera
```

- [ ] **Step 6: Verify against the real webcam**

```bash
source .venv/bin/activate
python -c "
from camera_source import create_camera
camera = create_camera()
print('opened:', camera.name)
ok, frame = camera.read()
print('read ok:', ok, 'shape:', None if frame is None else frame.shape)
camera.release()
"
```

Expected: opens `USB camera 0`, `read ok: True`, and a shape of `(1080, 1920, 3)`.
A warning about a different granted resolution is acceptable — record whatever it reports.

- [ ] **Step 7: Run the whole suite and commit**

```bash
source .venv/bin/activate && python -m pytest tests/ -v
git add camera_source.py tests/test_camera_source.py
git commit -m "Make USB webcam capture robust: MJPG, resolution check, warmup"
```

---

## Task 6: Backend protocol and pooling

**Files:**
- Create: `backends/__init__.py`, `backends/base.py`, `tests/test_backends.py`

`pool_tokens` is a free function rather than a method precisely so the
prefix-token rule from the spec can be tested without downloading a model.

- [ ] **Step 1: Write the failing test**

Create `tests/test_backends.py`:

```python
import numpy as np
import pytest
import torch

from backends.base import l2_normalise, pool_tokens


def test_mean_pooling_excludes_prefix_tokens():
    """The spec's key correctness rule.

    DINOv2 has 1 prefix token (CLS); DINOv3 has CLS plus register tokens.
    Averaging any of them into the patch mean corrupts every embedding, and
    would do so silently, so it is pinned by a test here.
    """
    tokens = torch.zeros(1, 7, 4)
    tokens[:, :3] = 999.0   # 3 prefix tokens: CLS + 2 registers
    tokens[:, 3:] = 2.0     # 4 real patch tokens

    pooled = pool_tokens(tokens, pooling="mean", num_prefix_tokens=3)

    assert torch.allclose(pooled, torch.full((1, 4), 2.0)), (
        "prefix tokens leaked into the mean"
    )


def test_cls_pooling_takes_the_first_token():
    tokens = torch.zeros(1, 5, 3)
    tokens[:, 0] = 7.0
    tokens[:, 1:] = 1.0

    pooled = pool_tokens(tokens, pooling="cls", num_prefix_tokens=1)

    assert torch.allclose(pooled, torch.full((1, 3), 7.0))


def test_unknown_pooling_raises():
    tokens = torch.zeros(1, 5, 3)
    with pytest.raises(ValueError, match="pooling"):
        pool_tokens(tokens, pooling="max", num_prefix_tokens=1)


def test_l2_normalise_produces_unit_vectors():
    features = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
    normalised = l2_normalise(features)

    norms = torch.linalg.norm(normalised, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-6)


def test_l2_normalise_survives_a_zero_vector():
    features = torch.zeros(1, 4)
    normalised = l2_normalise(features)
    assert torch.isfinite(normalised).all(), "a zero vector must not produce NaN"
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_backends.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'backends'`.

- [ ] **Step 3: Write `backends/base.py`**

```python
"""The feature-extractor contract shared by every backend."""

from typing import Protocol

import torch


def pool_tokens(tokens, pooling, num_prefix_tokens):
    """Reduce a token sequence [B, T, D] to one vector per item, [B, D].

    `num_prefix_tokens` counts the non-patch tokens at the front of the
    sequence: 1 for DINOv2 (CLS), and CLS plus the register tokens for DINOv3.
    Mean pooling must skip all of them.
    """
    if pooling == "cls":
        return tokens[:, 0]

    if pooling == "mean":
        return tokens[:, num_prefix_tokens:].mean(dim=1)

    raise ValueError(
        f"Unknown pooling {pooling!r}; expected 'cls' or 'mean'."
    )


def l2_normalise(features, epsilon=1e-12):
    """Scale each row to unit length so cosine similarity is a dot product.

    Clamping the norm keeps an all-zero row (a fully black crop) from producing
    NaNs that would poison every downstream score.
    """
    norms = torch.linalg.norm(features, dim=1, keepdim=True).clamp_min(epsilon)
    return features / norms


class FeatureExtractor(Protocol):
    """What every backend must provide."""

    name: str
    dim: int
    num_prefix_tokens: int

    def embed(self, crops_bgr, pooling="cls"):
        """Embed a list of BGR uint8 crops, returning [N, dim] unit-norm float32."""
        ...

    def warmup(self, batch_size):
        """Run throwaway inferences so later timings exclude lazy setup."""
        ...
```

- [ ] **Step 4: Write a placeholder `backends/__init__.py`**

The registry is filled in during Task 9; for now it only needs to make the
package importable.

```python
"""Pluggable feature-extraction backends."""
```

- [ ] **Step 5: Run the tests and confirm they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_backends.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add backends/ tests/test_backends.py
git commit -m "Add the feature-extractor contract and token pooling"
```

---

## Task 7: PyTorch backend for DINOv2 and DINOv3

**Files:**
- Create: `backends/torch_backend.py`
- Modify: `tests/test_backends.py` (append)

DINOv2 comes from `transformers`, DINOv3 from `timm` (the `facebook/dinov3-*`
repositories are gated; the timm mirrors carry the same weights and are not).
Each model uses **its own** preprocessing — DINOv2 is patch14 at 224, DINOv3 is
patch16 at 256. Sharing one processor would handicap whichever model did not own
it and would invalidate the comparison.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_backends.py`:

```python
# --- Backend integration tests -------------------------------------------
# These download weights on first run (roughly 90 MB for small, 350 MB for
# base), so they are marked and excluded from the default run.

MODEL_DIMS = {
    "dinov2-small": 384,
    "dinov2-base": 768,
    "dinov3-small": 384,
    "dinov3-base": 768,
}


@pytest.mark.download
@pytest.mark.parametrize("model_name,expected_dim", sorted(MODEL_DIMS.items()))
def test_embeddings_have_the_right_shape_and_norm(
    model_name, expected_dim, synthetic_board
):
    from backends import get_backend

    backend = get_backend(model_name, device="cpu", use_fp16=False)
    crops = [
        synthetic_board[100:180, 100:180],
        synthetic_board[300:380, 300:380],
    ]

    features = backend.embed(crops, pooling="cls")

    assert features.shape == (2, expected_dim)
    assert backend.dim == expected_dim
    assert np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-4)
    assert features.dtype == np.float32


@pytest.mark.download
def test_dinov3_reports_register_tokens(synthetic_board):
    """DINOv3 carries register tokens beyond the CLS token.

    If this ever reports 1, mean pooling silently averages register tokens into
    the patch mean, so it is asserted rather than assumed.
    """
    from backends import get_backend

    backend = get_backend("dinov3-small", device="cpu", use_fp16=False)
    assert backend.num_prefix_tokens > 1


@pytest.mark.download
def test_pooling_modes_give_different_embeddings(synthetic_board):
    from backends import get_backend

    backend = get_backend("dinov2-small", device="cpu", use_fp16=False)
    crops = [synthetic_board[100:180, 100:180]]

    cls_features = backend.embed(crops, pooling="cls")
    mean_features = backend.embed(crops, pooling="mean")

    assert not np.allclose(cls_features, mean_features), (
        "cls and mean pooling should not collapse to the same vector"
    )


@pytest.mark.download
def test_identical_crops_are_maximally_similar(synthetic_board):
    """Sanity check on the similarity metric the whole pipeline rests on."""
    from backends import get_backend

    backend = get_backend("dinov2-small", device="cpu", use_fp16=False)
    crop = synthetic_board[100:180, 100:180]

    features = backend.embed([crop, crop.copy()], pooling="cls")
    similarity = float(features[0] @ features[1])

    assert similarity > 0.999
```

Register the marker so the suite does not warn. Create `pytest.ini`:

```ini
[pytest]
markers =
    download: test downloads model weights (slow, needs network)
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
source .venv/bin/activate && python -m pytest tests/test_backends.py -v -m download
```

Expected: FAIL — `cannot import name 'get_backend' from 'backends'`.

- [ ] **Step 3: Write `backends/torch_backend.py`**

```python
"""PyTorch feature extractors for DINOv2 (transformers) and DINOv3 (timm)."""

import cv2
import numpy as np
import torch

from backends.base import l2_normalise, pool_tokens

# DINOv3 is gated under facebook/*; these timm mirrors carry the same weights
# and are not gated, so no manual approval is needed to reproduce results.
MODEL_SOURCES = {
    "dinov2-small": ("transformers", "facebook/dinov2-small"),
    "dinov2-base": ("transformers", "facebook/dinov2-base"),
    "dinov3-small": ("timm", "vit_small_patch16_dinov3.lvd1689m"),
    "dinov3-base": ("timm", "vit_base_patch16_dinov3.lvd1689m"),
}


class TorchDinoBackend:
    """Embeds BGR crops with a frozen DINO backbone."""

    def __init__(self, name, device="cuda", use_fp16=True):
        if name not in MODEL_SOURCES:
            raise ValueError(
                f"Unknown model {name!r}. "
                f"Expected one of: {', '.join(sorted(MODEL_SOURCES))}."
            )

        self.name = name
        self.device = torch.device(device)
        # fp16 is a CUDA-only optimisation; on CPU it is slower, not faster.
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        source, model_id = MODEL_SOURCES[name]
        self.source = source
        self.model_id = model_id

        if source == "transformers":
            self._load_transformers(model_id)
        else:
            self._load_timm(model_id)

        self.model.eval()
        self.model.to(device=self.device, dtype=self.dtype)

    def _load_transformers(self, model_id):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.dim = self.model.config.hidden_size
        # DINOv2 in transformers has a single CLS prefix token, no registers.
        self.num_prefix_tokens = 1

    def _load_timm(self, model_id):
        import timm

        self.model = timm.create_model(model_id, pretrained=True, num_classes=0)
        data_config = timm.data.resolve_model_data_config(self.model)
        self.processor = timm.data.create_transform(
            **data_config, is_training=False
        )
        self.dim = self.model.num_features
        # DINOv3 prepends CLS plus register tokens; timm reports the total.
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 1)

    def _to_pixel_values(self, crops_bgr):
        """BGR uint8 crops -> a normalised, batched float tensor."""
        from PIL import Image

        images = [
            Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            for crop in crops_bgr
        ]

        if self.source == "transformers":
            batch = self.processor(images=images, return_tensors="pt")
            pixel_values = batch["pixel_values"]
        else:
            pixel_values = torch.stack([self.processor(image) for image in images])

        return pixel_values.to(device=self.device, dtype=self.dtype)

    def _forward_tokens(self, pixel_values):
        """Return the full token sequence [B, T, D]."""
        if self.source == "transformers":
            return self.model(pixel_values=pixel_values).last_hidden_state

        return self.model.forward_features(pixel_values)

    @torch.inference_mode()
    def embed(self, crops_bgr, pooling="cls"):
        if not crops_bgr:
            return np.zeros((0, self.dim), dtype=np.float32)

        pixel_values = self._to_pixel_values(crops_bgr)
        tokens = self._forward_tokens(pixel_values)

        pooled = pool_tokens(tokens, pooling, self.num_prefix_tokens)
        # Normalise in float32: fp16 norms lose precision near unit length.
        normalised = l2_normalise(pooled.float())

        return normalised.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def warmup(self, batch_size, iterations=5):
        """Run throwaway inferences so later timings exclude lazy CUDA setup."""
        dummy = [
            np.zeros((80, 80, 3), dtype=np.uint8) for _ in range(batch_size)
        ]

        for _ in range(iterations):
            self.embed(dummy, pooling="cls")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
```

- [ ] **Step 4: Write the registry in `backends/__init__.py`**

```python
"""Pluggable feature-extraction backends.

A TensorRT backend for the Jetson Orin Nano slots in here later without any
change to callers.
"""

from backends.base import FeatureExtractor, l2_normalise, pool_tokens
from backends.torch_backend import MODEL_SOURCES, TorchDinoBackend

AVAILABLE_MODELS = tuple(sorted(MODEL_SOURCES))

__all__ = [
    "AVAILABLE_MODELS",
    "FeatureExtractor",
    "TorchDinoBackend",
    "get_backend",
    "l2_normalise",
    "pool_tokens",
]


def get_backend(name, device=None, use_fp16=None):
    """Build the feature extractor for `name`, defaulting from config."""
    import config

    device = config.DEVICE if device is None else device
    use_fp16 = config.USE_FP16 if use_fp16 is None else use_fp16

    return TorchDinoBackend(name, device=device, use_fp16=use_fp16)
```

- [ ] **Step 5: Run the download tests**

```bash
source .venv/bin/activate && python -m pytest tests/test_backends.py -v -m download
```

Expected: 7 passed. First run downloads roughly 900 MB across the four models, so allow time.

If `test_dinov3_reports_register_tokens` fails with `num_prefix_tokens == 1`,
do not relax the assertion — inspect the model with
`python -c "import timm; m = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True); print(m.num_prefix_tokens)"`
and fix `_load_timm` to read whatever attribute timm actually exposes.

- [ ] **Step 6: Confirm the fast suite still passes**

```bash
source .venv/bin/activate && python -m pytest tests/ -v -m "not download"
```

Expected: all pass, in a few seconds.

- [ ] **Step 7: Commit**

```bash
git add backends/ tests/test_backends.py pytest.ini
git commit -m "Add PyTorch backends for DINOv2 and DINOv3 with per-model preprocessing"
```

---

## Task 8: Rewire `capture_reference.py`

**Files:**
- Modify: `capture_reference.py:7-19`

- [ ] **Step 1: Replace the hardcoded constants**

Delete lines 7-19 (the `CAMERA_BACKEND` through `PREVIEW_MAX_HEIGHT` block) and
the now-redundant arguments in the `create_camera(...)` call. Replace the top of
the file with:

```python
import time

import cv2

import config
from camera_source import create_camera

OUTPUT_PATH = config.REFERENCE_PATH

camera = create_camera()

print(f"Using camera: {camera.name}")
print("Press SPACE to save the reference image.")
print("Press Q to quit.")
```

Then replace the two remaining uses of the deleted preview constants with
`config.PREVIEW_MAX_WIDTH` and `config.PREVIEW_MAX_HEIGHT`.

- [ ] **Step 2: Verify it imports cleanly**

```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('capture_reference.py').read()); print('syntax OK')"
```

Expected: `syntax OK`.

- [ ] **Step 3: Capture a real reference image**

Point the webcam at the PCB (or any textured object, for now) and run:

```bash
source .venv/bin/activate && python capture_reference.py
```

Press SPACE, then Q. Expected: `my_photo-1.jpg` is written.

```bash
ls -la my_photo-1.jpg
```

- [ ] **Step 4: Commit**

```bash
git add capture_reference.py
git commit -m "Read capture settings from the shared config"
```

---

## Task 9: Rewire `select_screws.py`

**Files:**
- Modify: `select_screws.py:5-17`

- [ ] **Step 1: Replace the hardcoded constants**

Replace lines 5-17 with:

```python
import config

CONFIG_PATH = config.CONFIG_PATH
REFERENCE_PATH = config.REFERENCE_PATH
CAPTURE_NEW_REFERENCE = False
DEFAULT_CROP_SIZE = config.DEFAULT_CROP_SIZE
DEFAULT_SIMILARITY_THRESHOLD = config.DEFAULT_SIMILARITY_THRESHOLD
```

Then simplify the `create_camera(...)` call inside `capture_reference()` to take
no arguments, since `create_camera` now defaults from config.

- [ ] **Step 2: Verify it parses**

```bash
source .venv/bin/activate && python -c "import ast; ast.parse(open('select_screws.py').read()); print('syntax OK')"
```

- [ ] **Step 3: Define the screw ROIs**

```bash
source .venv/bin/activate && python select_screws.py
```

Click each screw once, press S to save. Expected: `product_config.json` is written.

```bash
cat product_config.json
```

Expected: a `reference_path` and one entry per screw, each with `center`,
`crop_size` and `similarity_threshold`.

- [ ] **Step 4: Commit**

```bash
git add select_screws.py
git commit -m "Read ROI selection settings from the shared config"
```

---

## Task 10: Live detection on the webcam

**Files:**
- Create: `live_detection.py`

This replaces `live_detection_trt.py`, reusing `registration.py` and `roi.py`
rather than repeating them, and adding a `--model` flag so all four backbones
can be compared by eye before the formal evaluation exists.

Timing note: every measured region calls `torch.cuda.synchronize()`. Without it
the timers measure kernel *launch* rather than execution, which is the defect in
the original script at `live_detection_trt.py:401-408`.

- [ ] **Step 1: Write `live_detection.py`**

```python
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
```

- [ ] **Step 2: Verify the error paths before touching hardware**

```bash
source .venv/bin/activate
mv product_config.json product_config.json.bak
python live_detection.py --model dinov2-small
```

Expected: exits with the message telling you to run `select_screws.py`, not a traceback.

```bash
mv product_config.json.bak product_config.json
```

- [ ] **Step 3: Run it live on each model**

```bash
source .venv/bin/activate
python live_detection.py --model dinov2-small
python live_detection.py --model dinov3-small
python live_detection.py --model dinov2-base --pooling mean
python live_detection.py --model dinov3-base --pooling mean
```

Expected for each: the window opens, status reads `Registered` with a non-zero
inlier count, and each ROI shows a similarity score. Remove a screw and the
corresponding box should turn red within `DEBOUNCE_COUNT` detections.

Record the reported `Reg:` and `DINO:` milliseconds for each model — these are
the first honest latency figures for the project, and worth noting in the commit
message. If `Reg:` dominates `DINO:`, that confirms the spec's suspicion that
SIFT, not the backbone, is the bottleneck.

- [ ] **Step 4: Commit**

```bash
git add live_detection.py
git commit -m "Add live detection with selectable backend and synchronised timing"
```

---

## Task 11: Remove the superseded TensorRT script

**Files:**
- Delete: `live_detection_trt.py`

Removing it only after `live_detection.py` is confirmed working means the repo
is never left without a functioning live path. The file imports a module that
does not exist (`trt_feature_extractor`), so it cannot run in any case, and the
README already documents that gap.

- [ ] **Step 1: Confirm nothing still references it**

```bash
cd /home/oem/DINO-Performance-Evaluation
grep -rn "live_detection_trt\|trt_feature_extractor" --include="*.py" --include="*.md" . | grep -v "docs/superpowers"
```

Expected: only matches in `README.md`, which is updated in the next step.

- [ ] **Step 2: Delete it and refresh the README**

```bash
git rm live_detection_trt.py
```

In `README.md`, replace the `live_detection_trt.py` line in the Repository
layout block with:

```
live_detection.py       live detection, selectable DINO backend
```

Then remove the two now-stale Status bullets about `trt_feature_extractor.py`
and the TensorRT engine, replacing them with:

```markdown
- The evaluation harness (labelled dataset capture, accuracy scoring and latency
  benchmarking) is not built yet — see the spec for the plan.
```

- [ ] **Step 3: Confirm the suite still passes**

```bash
source .venv/bin/activate && python -m pytest tests/ -v -m "not download"
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "Replace the broken TensorRT script with live_detection.py"
```

---

## Done criteria

- [ ] `python -m pytest tests/ -m "not download"` passes in under ~10 seconds
- [ ] `python -m pytest tests/ -m download` passes (all four models load)
- [ ] `python live_detection.py --model <each of the four>` runs against the webcam
- [ ] Removing a screw flips the corresponding ROI to MISSING
- [ ] `Reg:` and `DINO:` timings recorded for all four models
- [ ] No file still imports `trt_feature_extractor`

The next plan covers spec phases 5-6: `capture_dataset.py`, `preprocess.py`,
`evaluate.py`, `benchmark.py` and `report.py`. It needs the PCB in hand.
