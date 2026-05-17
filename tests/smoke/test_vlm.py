"""Smoke test of the production sensors against the in-process mlx-vlm.

Captures via the production code paths and routes through the live
MlxVlmClient (spawns a worker, downloads the Gemma model on first
run). Run after:
    uv run python tests/smoke/test_screen.py
    uv run python tests/smoke/test_webcam.py
"""

from __future__ import annotations

import json
import sys
import time

from unclefu.sensors.screen_sensor import ScreenSensor
from unclefu.sensors.webcam_sensor import WebcamSensor
from unclefu.vlm.client import MlxVlmClient, VLMError


def main() -> int:
    print("loading VLM (Gemma 4 E4B via mlx-vlm)…")
    vlm = MlxVlmClient()
    if not vlm.wait_ready(timeout_s=600):
        print("FAIL: VLM model did not become ready within 10 min", file=sys.stderr)
        return 2

    print("\n=== webcam sensor ===")
    t0 = time.perf_counter()
    try:
        snap = WebcamSensor(vlm=vlm).tick()
    except VLMError as e:
        print(f"webcam FAIL: {e}", file=sys.stderr)
        return 2
    print(f"  {int((time.perf_counter()-t0)*1000)}ms  {snap.description}")
    print(f"  structured: {json.dumps(snap.structured, indent=2)}")

    print("\n=== screen sensor (display 1) ===")
    t0 = time.perf_counter()
    try:
        snap = ScreenSensor(vlm=vlm, display_idx=1).tick()
    except VLMError as e:
        print(f"screen FAIL: {e}", file=sys.stderr)
        return 2
    print(f"  {int((time.perf_counter()-t0)*1000)}ms  {snap.description}")
    print(f"  structured: {json.dumps(snap.structured, indent=2)}")

    vlm.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
