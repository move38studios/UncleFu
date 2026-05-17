"""Measure VLM latency + output quality at different input image sizes.

Question: does pre-downsampling the JPEG meaningfully speed up Gemma's
inference, or does Gemma's processor re-resize internally anyway? If
smaller-in = faster-out, we should drop the default capture width in
sensors/screen_sensor.py and capture/webcam.py.

Run:
    uv run python tests/smoke/test_downsample_probe.py

Captures one screen + (if webcam permission has been granted) one
webcam frame, then for each input width [1280, 640, 320, 160] runs
the production WEBCAM_PROMPT and SCREEN_PROMPT, measures warm-call
latency, prints the response so we can eyeball quality drop.

What we expect to see in the output:
- If durations are flat across widths → Gemma's processor normalizes,
  pre-resizing is wasted work; keep defaults.
- If durations drop materially at smaller sizes → switch defaults and
  measure where quality starts to break.
"""

from __future__ import annotations

import io
import statistics
import sys
import time
from pathlib import Path

import mss
from PIL import Image

from unclefu.sensors.prompts import SCREEN_PROMPT, WEBCAM_PROMPT
from unclefu.vlm.client import MlxVlmClient


WIDTHS = [1280, 640, 320, 160]
WARMUPS = 1   # one warmup call to load + warm the prefill cache
TRIALS = 3    # then this many measured calls per width


def _capture_screen_full() -> Image.Image:
    """Native-resolution screen capture (no downsampling). We'll resize
    afterwards per-width to compare."""
    with mss.MSS() as sct:
        raw = sct.grab(sct.monitors[1])
        return Image.frombytes("RGB", raw.size, raw.rgb)


def _capture_webcam_full() -> Image.Image | None:
    """Webcam capture — uses production AVFoundation path. Returns None
    if the camera isn't authorised (smoke test still useful with just
    the screen path in that case)."""
    try:
        from unclefu.capture.webcam import capture_webcam, ensure_camera_authorized
        ensure_camera_authorized()
        jpeg = capture_webcam()
        return Image.open(io.BytesIO(jpeg)).convert("RGB")
    except Exception as e:
        print(f"  (webcam unavailable: {type(e).__name__}: {e})")
        return None


def _resize(img: Image.Image, width: int) -> bytes:
    """Resize to the target width keeping aspect ratio, encode as JPEG-80."""
    if img.width != width:
        ratio = width / img.width
        sized = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
    else:
        sized = img
    buf = io.BytesIO()
    sized.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _time_calls(
    vlm: MlxVlmClient, *, prompt: str, jpeg: bytes, user_text: str,
) -> tuple[list[float], str]:
    """Run WARMUPS + TRIALS calls. Return latencies (just the trials)
    and the last response text."""
    last_text = ""
    for _ in range(WARMUPS):
        result = vlm.chat(
            system=prompt, user_text=user_text, images_jpeg=[jpeg],
            max_tokens=200, temperature=0.1,
        )
        last_text = result.text
    times: list[float] = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        result = vlm.chat(
            system=prompt, user_text=user_text, images_jpeg=[jpeg],
            max_tokens=200, temperature=0.1,
        )
        times.append(time.perf_counter() - t0)
        last_text = result.text
    return times, last_text


def _probe(label: str, img: Image.Image, vlm: MlxVlmClient,
           prompt: str, user_text: str) -> None:
    print(f"\n{'='*80}")
    print(f"{label}  (source: {img.width}x{img.height})")
    print("=" * 80)
    for w in WIDTHS:
        if w > img.width:
            print(f"\n--- width {w}: SKIPPED (larger than source) ---")
            continue
        jpeg = _resize(img, w)
        kb = len(jpeg) // 1024
        print(f"\n--- width {w}  ({kb} KB) ---")
        try:
            times, text = _time_calls(vlm, prompt=prompt, jpeg=jpeg, user_text=user_text)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}")
            continue
        med = statistics.median(times)
        mn, mx = min(times), max(times)
        print(f"  latency: median={med:.2f}s  min={mn:.2f}s  max={mx:.2f}s  ({TRIALS} trials)")
        print(f"  response: {text.strip()[:200]}")


def main() -> int:
    print("Loading VLM…")
    vlm = MlxVlmClient()
    if not vlm.wait_ready(timeout_s=600):
        print("FAIL: VLM did not become ready", file=sys.stderr)
        return 2
    print("  ready.\n")

    screen = _capture_screen_full()
    print(f"Captured screen at {screen.width}x{screen.height}")
    webcam = _capture_webcam_full()
    if webcam is not None:
        print(f"Captured webcam at {webcam.width}x{webcam.height}")

    _probe("SCREEN", screen, vlm, SCREEN_PROMPT, "Describe this screen capture.")
    if webcam is not None:
        _probe("WEBCAM", webcam, vlm, WEBCAM_PROMPT, "Describe this webcam frame.")

    print("\n" + "=" * 80)
    print("Interpretation guide:")
    print("- If latencies are roughly flat across widths → Gemma re-resizes")
    print("  internally; pre-downsampling is wasted CPU on our end. Keep defaults.")
    print("- If smaller widths are materially faster → switch the production")
    print("  capture width to the smallest size where the response still")
    print("  describes the content accurately.")
    print("=" * 80)
    vlm.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
