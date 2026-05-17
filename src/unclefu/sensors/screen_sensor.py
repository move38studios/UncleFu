"""Screen sensor: one per display."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import mss
from PIL import Image

from ..vlm.client import MlxVlmClient, parse_json_response
from ..vlm.schema import SensorObservation, SensorSnapshot


def _capture_one_screen(display_idx: int, *, max_width: int, jpeg_quality: int) -> bytes:
    """Capture exactly one physical display by index (1-based for mss)."""
    with mss.MSS() as sct:
        monitors = sct.monitors
        if display_idx < 1 or display_idx >= len(monitors):
            raise RuntimeError(
                f"display_idx={display_idx} out of range (1..{len(monitors)-1})"
            )
        raw = sct.grab(monitors[display_idx])
        img = Image.frombytes("RGB", raw.size, raw.rgb)
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    return buf.getvalue()


@dataclass
class ScreenSensor:
    vlm: MlxVlmClient
    display_idx: int                       # 1-based, matching mss.monitors[N]
    # Composed at session start by the Runner from BASE_SCREEN_PROMPT +
    # the focus classifier's `screen_context`. Runner mutates on
    # "Change focus…" for mid-session updates.
    system_prompt: str = ""
    source: str = ""
    default_interval_s: float = 12.0
    max_width: int = 960
    jpeg_quality: int = 65

    def __post_init__(self) -> None:
        if not self.source:
            self.source = f"screen_{self.display_idx - 1}"

    def tick(self) -> SensorSnapshot:
        t0 = time.time()
        jpeg = _capture_one_screen(
            self.display_idx, max_width=self.max_width, jpeg_quality=self.jpeg_quality
        )
        result = self.vlm.chat(
            system=self.system_prompt,
            user_text="Describe this screen capture.",
            images_jpeg=[jpeg],
            max_tokens=200,
        )
        parsed = parse_json_response(result.text)
        obs = SensorObservation.model_validate(parsed)
        return SensorSnapshot(
            ts=t0,
            source=self.source,
            description=obs.description,
            structured={"confidence": obs.confidence},
            cycle_ms=int((time.time() - t0) * 1000),
        )


def discover_screen_sensors(
    *, vlm: MlxVlmClient, system_prompt: str = "",
    max_displays: int = 3, **kwargs,
) -> list[ScreenSensor]:
    """Build one ScreenSensor per physical display visible to mss, capped."""
    with mss.MSS() as sct:
        count = max(0, len(sct.monitors) - 1)  # monitors[0] is the virtual desktop
    n = min(count, max_displays)
    return [
        ScreenSensor(vlm=vlm, display_idx=i + 1, system_prompt=system_prompt, **kwargs)
        for i in range(n)
    ]
