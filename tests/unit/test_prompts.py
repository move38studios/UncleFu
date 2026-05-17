from __future__ import annotations

from unclefu.director.prompts import (
    build_director_system_prompt,
    format_recent_snapshots_block,
    format_recent_speech_block,
    format_relative,
)
from unclefu.personalities import get as get_personality


def test_format_relative_seconds():
    assert format_relative(0) == "0s"
    assert format_relative(45) == "45s"


def test_format_relative_minutes():
    assert format_relative(60) == "1m"
    assert format_relative(75) == "1m15s"


def test_format_relative_hours():
    assert format_relative(3600) == "1h00m"
    assert format_relative(3720) == "1h02m"


def test_empty_snapshot_block_says_so():
    block = format_recent_snapshots_block("Webcam", [])
    assert "no observations yet" in block


def test_snapshot_block_lists_in_order():
    block = format_recent_snapshots_block(
        "Webcam",
        [(20.0, "high", "first"), (40.0, "low", "second")],
    )
    assert block.index("first") < block.index("second")
    assert "20s" in block
    assert "40s" in block
    # Confidence annotation is on the line so the Director can read it.
    assert "[conf=high]" in block
    assert "[conf=low]" in block


def test_speech_block_when_empty():
    assert "haven't said anything" in format_recent_speech_block([])


def test_speech_block_lists_lines():
    block = format_recent_speech_block([(30.0, "shoulders, friend"), (90.0, "still upright?")])
    assert "shoulders, friend" in block
    assert "still upright?" in block


def test_director_system_prompt_includes_personality_and_format():
    p = get_personality("uncle_fu")
    prompt = build_director_system_prompt(p)
    assert "Uncle Fu" in prompt
    assert p.prompt_fragment.strip().splitlines()[0] in prompt
    assert "should_speak" in prompt
    # The expression field must be taught in the system prompt so the model
    # knows to emit it.
    assert "expression" in prompt
    assert "disapproving" in prompt
    assert "alarmed" in prompt
    # Focus drift must be the headline rule.
    assert "DRIFT FROM FOCUS" in prompt or "drift from focus" in prompt.lower()
    assert "USER'S STATED FOCUS" in prompt
    # Confidence gating must be taught.
    assert "conf=low" in prompt
    assert "confidence" in prompt.lower()
    # No double braces left over from .format()
    assert "{{" not in prompt
    assert "}}" not in prompt
