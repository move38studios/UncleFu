"""Screen capture. Returns JPEG bytes for each display — never writes to disk.

`capture_screens` returns one JPEG per physical display, primary first.
`capture_screen` is a convenience for tests / smoke checks that want just
the primary.
"""

from __future__ import annotations

import io

import mss
from PIL import Image


def capture_screens(
    *, max_width: int = 960, jpeg_quality: int = 65, max_displays: int = 3
) -> list[bytes]:
    """Capture every physical display, return JPEG bytes per display.

    mss exposes monitors as: monitors[0] = combined virtual desktop,
    monitors[1..N] = each physical display. We skip [0] and cap the count
    at `max_displays` so a six-monitor setup doesn't blow up our inference time.
    """
    out: list[bytes] = []
    with mss.MSS() as sct:
        physical = sct.monitors[1 : 1 + max_displays]
        for monitor in physical:
            raw = sct.grab(monitor)
            img = Image.frombytes("RGB", raw.size, raw.rgb)
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            out.append(buf.getvalue())
    return out


def capture_screen(*, max_width: int = 1280, jpeg_quality: int = 80) -> bytes:
    """Capture the primary display only. Used by smoke tests."""
    screens = capture_screens(max_width=max_width, jpeg_quality=jpeg_quality, max_displays=1)
    if not screens:
        raise RuntimeError("No displays found")
    return screens[0]
