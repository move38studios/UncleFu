"""MutableSpeaker tests and pure helpers around the character/menubar layer.

The rumps-based MenuBarApp itself is not exercised — it needs a real run
loop. The helpers it relies on (Personality.emoji_for / .sprite_path) are
covered in test_personalities.py.
"""

from __future__ import annotations

from unclefu.personalities import get as get_personality
from unclefu.tts.speaker import MutableSpeaker, NullSpeaker


def test_mutable_speaker_starts_unmuted_and_calls_inner():
    captured = []

    class _S:
        def say(self, text, *, voice=None): captured.append((text, voice))
        def stop(self): pass
        def is_speaking(self): return False

    s = MutableSpeaker(_S())
    s.say("hi", voice="Uncle_Fu")
    assert captured == [("hi", "Uncle_Fu")]


def test_mutable_speaker_swallows_when_muted():
    captured = []

    class _S:
        def say(self, text, *, voice=None): captured.append(text)
        def stop(self): pass
        def is_speaking(self): return False

    s = MutableSpeaker(_S(), muted=True)
    s.say("hi")
    assert captured == []

    s.muted = False
    s.say("hello")
    assert captured == ["hello"]


def test_mutable_speaker_around_null_speaker():
    s = MutableSpeaker(NullSpeaker())
    s.say("x")
    s.muted = True
    s.say("y")
    assert not s.is_speaking()


# ── personality icon resolution ───────────────────────────────────────────


def test_personality_emoji_falls_back_to_global_default():
    """Personalities without overrides should still resolve every expression."""
    fu = get_personality("uncle_fu")
    # 'idle' isn't overridden by uncle_fu yet — comes from the global default.
    assert fu.emoji_for("idle") == "🧙‍♂️"
    assert fu.emoji_for("talking") == "💬"
    assert fu.emoji_for("disapproving") == "😤"
    assert fu.emoji_for("alarmed") == "😱"
