"""Smoke test: grab one screen frame via mss and save as JPEG."""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

from PIL import Image
import mss


def capture_screen(out_path: Path, *, max_width: int = 1280, jpeg_quality: int = 80) -> None:
    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.rgb)

    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)

    img.save(out_path, format="JPEG", quality=jpeg_quality)


def main() -> int:
    out = Path("/tmp/sc_screen.jpg")
    t0 = time.perf_counter()
    capture_screen(out)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    size_kb = out.stat().st_size / 1024
    print(f"OK screen -> {out} ({size_kb:.1f} KB, {elapsed_ms:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
