"""Webcam capture via AVFoundation. Returns JPEG bytes — never writes to disk.

See docs/learnings.md for why this uses AVCapturePhotoOutput with explicit
JPEG codec and a 2-second sensor settle. Don't shorten without testing on a
cold camera.
"""

# pyobjc frameworks expose Objective-C classes via runtime bridging; pyright
# can't see them. Silence those attribute errors for this module only.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import time
from io import BytesIO
from threading import Event
from typing import Any

import objc
import AVFoundation as AVF  # type: ignore[import-not-found]
from Foundation import NSDate, NSObject, NSRunLoop  # type: ignore[import-not-found]
from PIL import Image


_AUTHORIZED = 3
_DENIED = 2
_RESTRICTED = 1


def _list_video_devices() -> list[Any]:
    types = [
        "AVCaptureDeviceTypeBuiltInWideAngleCamera",
        "AVCaptureDeviceTypeExternal",
        "AVCaptureDeviceTypeDeskViewCamera",
    ]
    resolved = [getattr(AVF, t) for t in types if getattr(AVF, t, None) is not None]
    session = AVF.AVCaptureDeviceDiscoverySession.discoverySessionWithDeviceTypes_mediaType_position_(
        resolved, AVF.AVMediaTypeVideo, 0
    )
    devs = list(session.devices())
    if not devs:
        devs = list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo))
    return devs


def _pick_default_device(devices: list[Any]) -> Any:
    builtin_type = getattr(AVF, "AVCaptureDeviceTypeBuiltInWideAngleCamera", None)
    if builtin_type is not None:
        for d in devices:
            if d.deviceType() == builtin_type:
                return d
    return devices[0]


def _pump_until(predicate, timeout_s: float, step_s: float = 0.05) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(step_s))
    return predicate()


def ensure_camera_authorized(timeout_s: float = 60.0) -> None:
    """Ensure camera access is granted. Triggers the macOS prompt if needed."""
    status = AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo)
    if status == _AUTHORIZED:
        return
    if status in (_DENIED, _RESTRICTED):
        raise PermissionError(
            "Camera permission denied. Grant it in System Settings → "
            "Privacy & Security → Camera, for the terminal you're running from."
        )

    event = Event()
    granted_box = {"value": False}

    def _handler(granted):
        granted_box["value"] = bool(granted)
        event.set()

    AVF.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVF.AVMediaTypeVideo, _handler
    )

    deadline = time.time() + timeout_s
    while not event.is_set() and time.time() < deadline:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))

    if not granted_box["value"]:
        raise PermissionError("Camera permission not granted.")


class _PhotoDelegate(NSObject):
    def initWithEvent_(self, event):
        self = objc.super(_PhotoDelegate, self).init()
        if self is None:
            return None
        self._event = event
        self.data = None
        self.error_str = None
        return self

    def captureOutput_didFinishProcessingPhoto_error_(self, output, photo, error):
        try:
            if error is not None:
                self.error_str = str(error)
            else:
                self.data = photo.fileDataRepresentation()
        finally:
            self._event.set()


def capture_webcam(
    *,
    device_index: int | None = None,
    settle_s: float = 2.0,
    max_width: int = 800,
    jpeg_quality: int = 75,
) -> bytes:
    """Grab one webcam frame as JPEG bytes, downsampled.

    Captures via AVCapturePhotoOutput at the camera's native res, then resizes
    with Pillow before returning. The native frame is ~720p / 240 KB; the VLM
    doesn't need that, especially when we're already sending 1–3 screen images.
    """
    ensure_camera_authorized()
    devices = _list_video_devices()
    if not devices:
        raise RuntimeError("No video capture devices found")
    device = devices[device_index] if device_index is not None else _pick_default_device(devices)

    input_, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
    if input_ is None:
        raise RuntimeError(f"Failed to create camera input: {err}")

    session = AVF.AVCaptureSession.alloc().init()
    session.setSessionPreset_(AVF.AVCaptureSessionPresetHigh)
    if not session.canAddInput_(input_):
        raise RuntimeError("Cannot add camera input to session")
    session.addInput_(input_)

    photo_output = AVF.AVCapturePhotoOutput.alloc().init()
    if not session.canAddOutput_(photo_output):
        raise RuntimeError("Cannot add photo output to session")
    session.addOutput_(photo_output)

    session.startRunning()
    if not _pump_until(lambda: session.isRunning(), timeout_s=5.0):
        raise RuntimeError("Capture session did not start")
    _pump_until(lambda: False, timeout_s=settle_s)

    fmt = {AVF.AVVideoCodecKey: AVF.AVVideoCodecTypeJPEG}
    settings = AVF.AVCapturePhotoSettings.photoSettingsWithFormat_(fmt)

    event = Event()
    delegate = _PhotoDelegate.alloc().initWithEvent_(event)
    photo_output.capturePhotoWithSettings_delegate_(settings, delegate)

    _pump_until(event.is_set, timeout_s=12.0)
    session.stopRunning()

    if delegate.error_str is not None:
        raise RuntimeError(f"Capture failed: {delegate.error_str}")
    if delegate.data is None:
        raise RuntimeError("Timed out waiting for photo callback")

    img = Image.open(BytesIO(bytes(delegate.data))).convert("RGB")
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()
