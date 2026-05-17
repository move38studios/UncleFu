"""Intervener throttle + dedup logic over Decision objects."""

from __future__ import annotations

from dataclasses import dataclass, field

from unclefu.intervene.intervener import Intervener, Outcome
from unclefu.personalities import get as get_personality
from unclefu.vlm.schema import Decision


@dataclass
class _RecordingSpeaker:
    spoken: list[tuple[str, str | None]] = field(default_factory=list)

    def say(self, text: str, *, voice: str | None = None) -> None:
        self.spoken.append((text, voice))

    def stop(self) -> None:
        return

    def is_speaking(self) -> bool:
        return False


def _mk(**kwargs) -> tuple[Intervener, _RecordingSpeaker]:
    sp = _RecordingSpeaker()
    iv = Intervener(speaker=sp, personality=get_personality("uncle_fu"), **kwargs)
    return iv, sp


def _d(message: str = "hey, sit up", urgency: str = "low", should_speak: bool = True) -> Decision:
    return Decision(should_speak=should_speak, urgency=urgency, message=message, reason="test")


def test_speaks_first_time():
    iv, sp = _mk()
    out = iv.maybe_speak(_d(), now=100.0)
    assert out is Outcome.SPOKE
    assert sp.spoken == [("hey, sit up", "Uncle_Fu")]


def test_model_declined_means_silent():
    iv, sp = _mk()
    out = iv.maybe_speak(
        Decision(should_speak=False, urgency="low", message=None, reason="x"),
        now=100.0,
    )
    assert out is Outcome.MODEL_DECLINED
    assert sp.spoken == []


def test_empty_message_is_noop():
    iv, sp = _mk()
    out = iv.maybe_speak(_d(message="   "), now=100.0)
    assert out is Outcome.NO_MESSAGE
    assert sp.spoken == []


def test_throttle_blocks_within_min_gap():
    iv, sp = _mk(min_gap_s=90.0, post_speech_cooldown_s=180.0)
    iv.maybe_speak(_d("first"), now=100.0)
    out = iv.maybe_speak(_d("second"), now=100.0 + 30.0)
    assert out is Outcome.THROTTLED_MIN


def test_low_urgency_waits_for_cooldown():
    """Cooldown gates `low` only — soft wellness nudges feel rare."""
    iv, sp = _mk(min_gap_s=90.0, post_speech_cooldown_s=180.0)
    iv.maybe_speak(_d("first"), now=100.0)
    out = iv.maybe_speak(_d("second", urgency="low"), now=100.0 + 100.0)
    assert out is Outcome.THROTTLED_COOLDOWN


def test_medium_urgency_bypasses_cooldown_but_not_min_gap():
    """Drift detection (medium) shouldn't wait for the soft-nudge cooldown
    to clear — if the user is on YouTube, Uncle should be able to fire
    again past the min gap."""
    iv, sp = _mk(min_gap_s=90.0, post_speech_cooldown_s=180.0)
    iv.maybe_speak(_d("first"), now=100.0)
    # Within min_gap → blocked.
    out = iv.maybe_speak(_d("drift", urgency="medium"), now=100.0 + 30.0)
    assert out is Outcome.THROTTLED_MIN
    # Past min_gap but still inside cooldown → SPEAKS (this is the new behavior).
    out = iv.maybe_speak(_d("drift now", urgency="medium"), now=100.0 + 100.0)
    assert out is Outcome.SPOKE


def test_high_urgency_overrides_cooldown_but_not_min_gap():
    iv, sp = _mk(min_gap_s=90.0, post_speech_cooldown_s=180.0)
    iv.maybe_speak(_d("first"), now=100.0)
    out = iv.maybe_speak(_d("urgent", urgency="high"), now=100.0 + 30.0)
    assert out is Outcome.THROTTLED_MIN
    out = iv.maybe_speak(_d("urgent now", urgency="high"), now=100.0 + 100.0)
    assert out is Outcome.SPOKE


def test_dedup_blocks_identical_line_later_in_session():
    iv, sp = _mk(min_gap_s=90.0, post_speech_cooldown_s=180.0)
    iv.maybe_speak(_d("Hey, shoulders."), now=100.0)
    out = iv.maybe_speak(_d("  hey, shoulders!  ", urgency="medium"), now=100.0 + 1000.0)
    assert out is Outcome.DEDUPED


def test_different_lines_after_cooldown_speak():
    iv, sp = _mk(min_gap_s=90.0, post_speech_cooldown_s=180.0)
    iv.maybe_speak(_d("first thing", urgency="low"), now=100.0)
    out = iv.maybe_speak(_d("totally different remark", urgency="low"), now=100.0 + 200.0)
    assert out is Outcome.SPOKE


# ── force=True bypass for click-to-talk ───────────────────────────────────


def test_force_bypasses_min_gap():
    """The 'Talk to me' button must not be silenced by the 60 s throttle —
    the user explicitly asked the character to speak."""
    iv, sp = _mk(min_gap_s=90.0, post_speech_cooldown_s=180.0)
    iv.maybe_speak(_d("first"), now=100.0)
    out = iv.maybe_speak(_d("second"), now=100.0 + 5.0, force=True)
    assert out is Outcome.SPOKE
    assert sp.spoken[-1] == ("second", "Uncle_Fu")


def test_force_bypasses_cooldown():
    iv, sp = _mk(min_gap_s=90.0, post_speech_cooldown_s=180.0)
    iv.maybe_speak(_d("first"), now=100.0)
    out = iv.maybe_speak(
        _d("second", urgency="low"), now=100.0 + 100.0, force=True,
    )
    assert out is Outcome.SPOKE


def test_force_still_respects_dedup():
    """Even with force=True, never repeat an already-spoken line in the
    same session — that's the stale-line uninstall risk."""
    iv, sp = _mk(min_gap_s=90.0, post_speech_cooldown_s=180.0)
    iv.maybe_speak(_d("Aiya. Posture."), now=100.0)
    out = iv.maybe_speak(_d("aiya. posture!"), now=100.0 + 2.0, force=True)
    assert out is Outcome.DEDUPED


def test_force_updates_last_spoke_at():
    """A forced click still resets the throttle clock for the Director,
    so Uncle Fu doesn't talk over his own click line a second later."""
    iv, sp = _mk(min_gap_s=90.0, post_speech_cooldown_s=180.0)
    iv.maybe_speak(_d("click line"), now=200.0, force=True)
    # 30 s later, a normal (non-forced) speech should still be throttled.
    out = iv.maybe_speak(_d("director line"), now=200.0 + 30.0)
    assert out is Outcome.THROTTLED_MIN
