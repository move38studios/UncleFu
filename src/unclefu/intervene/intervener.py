"""Decides whether to actually speak this cycle, given the Director's request
plus our own throttles and dedup.

The Director (LLM) has its own self-restraint in the prompt, but trusting it
alone is fragile. We enforce:
- a hard minimum gap between any two speeches
- a longer cooldown after a recent speech, override-able by `urgency=high`
- no exact line ever spoken twice in a session
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ..personalities import Personality
from ..tts.speaker import Speaker
from ..vlm.schema import Decision


class Outcome(str, Enum):
    SPOKE = "spoke"
    MODEL_DECLINED = "model_declined"
    NO_MESSAGE = "no_message"
    THROTTLED_MIN = "throttled_min"
    THROTTLED_COOLDOWN = "throttled_cooldown"
    DEDUPED = "deduped"


@dataclass
class Intervener:
    speaker: Speaker
    personality: Personality
    min_gap_s: float = 60.0
    post_speech_cooldown_s: float = 90.0
    _last_spoke_at: float = 0.0
    _spoken_lines: set[str] = field(default_factory=set)
    _spoken_log: deque[tuple[float, str]] = field(default_factory=lambda: deque(maxlen=200))

    def maybe_speak(
        self, decision: Decision, *, now: float, force: bool = False,
    ) -> Outcome:
        """Maybe speak this decision.

        force=True bypasses the throttle gates (min_gap + cooldown) so an
        explicit user action — the "Talk to me" button — can speak immediately
        regardless of when the Director last fired. Per-session dedup still
        applies; a forced repeat of an already-spoken line still returns
        DEDUPED. `_last_spoke_at` is still updated so the Director doesn't
        fire right on top of the click.
        """
        if not decision.should_speak:
            return Outcome.MODEL_DECLINED
        msg = (decision.message or "").strip()
        if not msg:
            return Outcome.NO_MESSAGE

        if not force:
            gap = now - self._last_spoke_at
            if self._last_spoke_at > 0 and gap < self.min_gap_s:
                return Outcome.THROTTLED_MIN
            # post-speech cooldown applies ONLY to urgency=low lines. The
            # Director should reach for `low` for soft wellness nudges
            # (posture, water, "you stood up"); those are the ones we
            # genuinely want to feel rare. Drift (medium) and risk (high)
            # both bypass the cooldown — when the user is off-task or
            # about to do something destructive, the min_gap floor is
            # all we want gating Uncle.
            if (
                decision.urgency == "low"
                and self._last_spoke_at > 0
                and gap < self.post_speech_cooldown_s
            ):
                return Outcome.THROTTLED_COOLDOWN

        normalized = _normalize(msg)
        if normalized in self._spoken_lines:
            return Outcome.DEDUPED

        self.speaker.say(msg, voice=self.personality.voice)
        self._last_spoke_at = now
        self._spoken_lines.add(normalized)
        self._spoken_log.append((now, msg))
        return Outcome.SPOKE

    def spoken_count(self) -> int:
        return len(self._spoken_log)


def _normalize(s: str) -> str:
    out = " ".join(s.split()).lower()
    return out.rstrip(".!?…").strip()
