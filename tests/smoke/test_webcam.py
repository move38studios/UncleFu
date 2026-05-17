"""Smoke test: grab one webcam frame and save as JPEG.

Uses AVCapturePhotoOutput with explicit JPEG codec. We start the session,
wait for `isRunning` plus a 2s sensor-settle window, then snap one frame
and write its native JPEG bytes to disk (no Pillow round-trip — that's how
the first attempt produced a black image: HEIC bytes mis-decoded).

Flags:
  --list           list video devices and exit
  --device IDX     pick device by 0-based index from --list (default: built-in)
  --settle SEC     seconds to wait after the session starts (default: 2.0)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from threading import Event

import objc
import AVFoundation as AVF
from Foundation import NSDate, NSObject, NSRunLoop


def _list_video_devices():
    """All current video devices (built-in, external, DeskView, etc.)."""
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


def _ensure_camera_authorized(timeout_s: float = 60.0) -> None:
    Authorized, Denied, Restricted = 3, 2, 1
    status = AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo)
    if status == Authorized:
        return
    if status in (Denied, Restricted):
        raise RuntimeError(
            "Camera permission denied. Grant it in System Settings → "
            "Privacy & Security → Camera, for the terminal you're running this from."
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
        raise RuntimeError("Camera permission not granted (timed out or denied).")


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


def _pick_default_device(devices):
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


def capture_webcam(
    out_path: Path,
    *,
    device_index: int | None = None,
    settle_s: float = 2.0,
) -> str:
    _ensure_camera_authorized()
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
    # Wait for the session to actually be running.
    if not _pump_until(lambda: session.isRunning(), timeout_s=3.0):
        raise RuntimeError("Capture session did not start")
    # Sensor / auto-exposure settle.
    _pump_until(lambda: False, timeout_s=settle_s)

    # Force JPEG output explicitly. Default codec may be HEIC on modern macOS,
    # which round-tripped through Pillow gave us a black image last time.
    fmt = {AVF.AVVideoCodecKey: AVF.AVVideoCodecTypeJPEG}
    settings = AVF.AVCapturePhotoSettings.photoSettingsWithFormat_(fmt)

    event = Event()
    delegate = _PhotoDelegate.alloc().initWithEvent_(event)
    photo_output.capturePhotoWithSettings_delegate_(settings, delegate)

    _pump_until(event.is_set, timeout_s=5.0)
    session.stopRunning()

    if delegate.error_str is not None:
        raise RuntimeError(f"Capture failed: {delegate.error_str}")
    if delegate.data is None:
        raise RuntimeError("Timed out waiting for photo callback")

    # delegate.data is the raw JPEG NSData. Write straight to disk.
    out_path.write_bytes(bytes(delegate.data))
    return str(device.localizedName())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="list video devices and exit")
    parser.add_argument("--device", type=int, default=None, help="device index from --list")
    parser.add_argument("--settle", type=float, default=2.0, help="seconds for sensor settle")
    parser.add_argument("--out", type=Path, default=Path("/tmp/sc_webcam.jpg"))
    args = parser.parse_args()

    _ensure_camera_authorized()
    devices = _list_video_devices()

    if args.list:
        for i, d in enumerate(devices):
            print(f"[{i}] {d.localizedName()}  type={d.deviceType()}")
        return 0

    t0 = time.perf_counter()
    name = capture_webcam(args.out, device_index=args.device, settle_s=args.settle)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    size_kb = args.out.stat().st_size / 1024
    print(f"OK webcam ({name}) -> {args.out} ({size_kb:.1f} KB, {elapsed_ms:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
