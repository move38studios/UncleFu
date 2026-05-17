"""Webcam sensor: capture → tiny VLM call → parsed SensorObservation."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..capture.webcam import capture_webcam
from ..vlm.client import MlxVlmClient, parse_json_response
from ..vlm.schema import SensorObservation, SensorSnapshot


@dataclass
class WebcamSensor:
    vlm: MlxVlmClient
    # Composed at session start by the Runner from BASE_WEBCAM_PROMPT +
    # the focus classifier's `webcam_context`. The Runner mutates this
    # in place on "Change focus…" so mid-session re-classification
    # takes effect on the next tick.
    system_prompt: str
    source: str = "webcam"
    default_interval_s: float = 30.0

    def tick(self) -> SensorSnapshot:
        t0 = time.time()
        jpeg = capture_webcam()
        result = self.vlm.chat(
            system=self.system_prompt,
            user_text="Describe this webcam frame.",
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
