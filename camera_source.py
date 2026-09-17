import cv2

import config


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

        if config.LOCK_CAMERA_CONTROLS:
            self._lock_controls()

        # Discard the first frames while auto-exposure and white balance settle.
        for _ in range(warmup_frames):
            self.camera.read()

        self.name = f"USB camera {index} ({self.width}x{self.height})"

    def _lock_controls(self):
        """Pin focus, exposure and white balance so captures stay comparable.

        A reference image and a live frame are only comparable if the camera
        treats them the same way. Continuous autofocus in particular will
        refocus between the two, and the resulting blur shifts embeddings more
        than the defect being looked for.

        Cameras that do not expose a given control simply return False, which
        is reported rather than raised: a webcam without manual focus is still
        usable, just less reliable.
        """
        settings = [
            ("autofocus", cv2.CAP_PROP_AUTOFOCUS, 0),
            ("focus", cv2.CAP_PROP_FOCUS, config.FOCUS_ABSOLUTE),
            ("auto exposure", cv2.CAP_PROP_AUTO_EXPOSURE, 1),
            ("auto white balance", cv2.CAP_PROP_AUTO_WB, 0),
        ]

        refused = [
            name
            for name, prop, value in settings
            if not self.camera.set(prop, value)
        ]

        if refused:
            print(
                f"Warning: this camera refused to lock {', '.join(refused)}. "
                f"Reference and live frames may not stay comparable."
            )

    def read(self):
        return self.camera.read()

    def release(self):
        self.camera.release()


class BaslerCamera:
    def __init__(
        self,
        serial_number="",
        timeout_ms=2000,
        width=None,
        height=None
    ):
        try:
            from pypylon import genicam, pylon
        except ImportError as error:
            raise RuntimeError(
                "Basler mode requires pypylon. Install it with: "
                "python -m pip install pypylon"
            ) from error

        self.pylon = pylon
        self.genicam = genicam
        self.timeout_ms = timeout_ms
        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()

        if not devices:
            raise RuntimeError("No Basler camera was found.")

        selected_device = None

        if serial_number:
            for device in devices:
                if device.GetSerialNumber() == serial_number:
                    selected_device = device
                    break

            if selected_device is None:
                available = ", ".join(
                    device.GetSerialNumber() for device in devices
                )
                raise RuntimeError(
                    f"Basler serial {serial_number} was not found. "
                    f"Available serials: {available}"
                )
        else:
            selected_device = devices[0]

        self.camera = pylon.InstantCamera(
            factory.CreateDevice(selected_device)
        )
        self.camera.Open()

        self._configure_roi(width, height)

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = (
            pylon.OutputBitAlignment_MsbAligned
        )

        self.camera.StartGrabbing(
            pylon.GrabStrategy_LatestImageOnly
        )

        device_info = self.camera.GetDeviceInfo()
        self.name = (
            f"{device_info.GetModelName()} "
            f"({device_info.GetSerialNumber()})"
        )

    @staticmethod
    def _nearest_valid_value(node, requested_value):
        minimum = int(node.Min)
        maximum = int(node.Max)
        increment = max(int(node.Inc), 1)
        clamped = min(max(int(requested_value), minimum), maximum)
        return minimum + ((clamped - minimum) // increment) * increment

    def _configure_roi(self, width, height):
        genicam = self.genicam

        if genicam.IsWritable(self.camera.OffsetX):
            self.camera.OffsetX.Value = int(self.camera.OffsetX.Min)

        if genicam.IsWritable(self.camera.OffsetY):
            self.camera.OffsetY.Value = int(self.camera.OffsetY.Min)

        if width is not None and genicam.IsWritable(self.camera.Width):
            self.camera.Width.Value = self._nearest_valid_value(
                self.camera.Width,
                width
            )

        if height is not None and genicam.IsWritable(self.camera.Height):
            self.camera.Height.Value = self._nearest_valid_value(
                self.camera.Height,
                height
            )

        if width is not None and genicam.IsWritable(self.camera.OffsetX):
            centered_x = int(self.camera.OffsetX.Max) // 2
            self.camera.OffsetX.Value = self._nearest_valid_value(
                self.camera.OffsetX,
                centered_x
            )

        if height is not None and genicam.IsWritable(self.camera.OffsetY):
            centered_y = int(self.camera.OffsetY.Max) // 2
            self.camera.OffsetY.Value = self._nearest_valid_value(
                self.camera.OffsetY,
                centered_y
            )

    def read(self):
        grab_result = None

        try:
            grab_result = self.camera.RetrieveResult(
                self.timeout_ms,
                self.pylon.TimeoutHandling_Return
            )

            if grab_result is None or not grab_result.GrabSucceeded():
                return False, None

            converted = self.converter.Convert(grab_result)
            frame = converted.GetArray().copy()
            return True, frame
        finally:
            if grab_result is not None:
                grab_result.Release()

    def release(self):
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()

        if self.camera.IsOpen():
            self.camera.Close()


class RotatedCamera:
    def __init__(self, camera):
        self.camera = camera
        self.name = f"{camera.name} | rotated 180 degrees"

    def read(self):
        ok, frame = self.camera.read()

        if not ok or frame is None:
            return ok, frame

        return True, cv2.rotate(frame, cv2.ROTATE_180)

    def release(self):
        self.camera.release()


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
