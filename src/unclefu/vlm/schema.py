"""Schemas for the decoupled architecture.

Sensors return a `SensorObservation` (description + confidence). The
Director returns a `Decision`. Nothing else is structured — early
versions had wide per-sensor schemas (posture/face_distance/alertness
enums, app/stuck/risky booleans) but nothing was consuming the
structured fields at runtime, just the natural-language description.
Keeping them around cost us prompt tokens, generation time, and parse
failures for no gain. See docs/decisions.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


# Director-side enums.
Urgency = Literal["low", "medium", "high"]

# Expressions the Director may pick to flavor the menu bar icon when speaking.
# `idle` and `talking` are runtime-driven (default state + auto-on-during-playback);
# the rest are the model's emotive register. Keep this set small so a small VLM
# can pick reliably.
Expression = Literal[
    "idle",
    "talking",
    "disapproving",
    "concerned",
    "smirk",
    "approving",
    "alarmed",
]

# Sensor-side confidence. Director gates on this — low-confidence
# observations don't trigger drift calls. Three buckets is enough; more
# granularity (numeric 0-100, "very low / low / med / high / very high")
# isn't worth the prompt complexity for a small model.
Confidence = Literal["low", "medium", "high"]


# ---- What sensors return ----------------------------------------------


class SensorObservation(BaseModel):
    """One sensor's reading of one frame. Description is the only
    semantic payload; confidence tells the Director whether to act on it."""

    description: str = Field(min_length=3, max_length=400)
    confidence: Confidence


# ---- Generic snapshot record (what lives in the sensor_snapshot table) ----


@dataclass(frozen=True)
class SensorSnapshot:
    """One observation from one sensor at one time.

    `structured` is a free-form dict — used today only to carry the
    confidence value (`{"confidence": "high"}`) and, for error rows,
    `{"exc_type": ..., "exc_message": ...}`. Pre-simplification this
    column carried per-sensor JSON; the column is kept for forward
    flexibility, not to be a god-object.
    """

    ts: float
    source: str               # "webcam", "screen_0", "screen_1", "audio", ...
    description: str          # the human-readable "what's happening"
    structured: dict          # confidence + (for errors) exc info
    cycle_ms: int

    @property
    def is_error(self) -> bool:
        return self.description == "" and "exc_type" in self.structured

    @property
    def confidence(self) -> Confidence:
        """Confidence the sensor expressed in its own observation. Defaults
        to 'high' for old rows that pre-date the field — we trust them
        unless explicitly told otherwise."""
        c = self.structured.get("confidence")
        if c in ("low", "medium", "high"):
            return c  # type: ignore[return-value]
        return "high"


# ---- Director (Speaker) decision schema ----


class Decision(BaseModel):
    """What the Director returns each tick."""

    should_speak: bool
    urgency: Urgency
    message: str | None = None
    reason: str = Field(min_length=1, max_length=80)
    # Director-picked emotive register for the menu bar icon. Only meaningful
    # when should_speak=true; ignored otherwise. Model may omit; default `idle`.
    expression: Expression = "idle"
