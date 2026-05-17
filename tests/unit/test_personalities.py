from __future__ import annotations

import pytest

from unclefu.personalities import (
    DEFAULT_PERSONALITY,
    EXPRESSIONS,
    PERSONALITIES,
    get,
)


def test_uncle_fu_is_registered_and_default():
    assert set(PERSONALITIES) == {"uncle_fu"}
    assert DEFAULT_PERSONALITY == "uncle_fu"


def test_personality_has_required_fields():
    for key, p in PERSONALITIES.items():
        assert p.key == key
        assert p.display_name
        assert p.voice
        assert len(p.prompt_fragment) >= 20


def test_uncle_fu_uses_uncle_fu_voice():
    assert get("uncle_fu").voice == "Uncle_Fu"


def test_get_raises_on_unknown():
    with pytest.raises(ValueError):
        get("nope")


def test_emoji_fallback_covers_every_expression():
    """Every personality must have an emoji (default or override) for every
    expression — otherwise the menu bar would show an empty title at runtime."""
    for p in PERSONALITIES.values():
        for expr in EXPRESSIONS:
            assert p.emoji_for(expr), f"{p.key} missing emoji for {expr}"


def test_uncle_fu_has_click_lines():
    fu = get("uncle_fu")
    # Need enough variety that the dedup'd talk-to-me item keeps working
    # across a session.
    assert len(fu.click_lines) >= 10
    # No empties.
    assert all(line.strip() for line in fu.click_lines)
    # No duplicates within the set.
    assert len(set(fu.click_lines)) == len(fu.click_lines)


def test_uncle_fu_has_focus_aware_click_lines():
    """Focus mode is the product; at least a few click lines should
    nudge the user back to the task they signed up for, even without
    knowing the specific focus text."""
    fu = get("uncle_fu")
    blob = " ".join(fu.click_lines).lower()
    # Look for any of the focus-y vocab the lines should land.
    assert any(w in blob for w in ("focus", "task", "supposed to", "promise"))


def test_sprite_path_returns_none_when_asset_missing():
    """Until real PNGs land, sprite_path() returns None for every expression
    and the menu bar uses the emoji fallback. Once art arrives this test
    can be flipped or removed."""
    fu = get("uncle_fu")
    for expr in EXPRESSIONS:
        # Either None (no asset yet) or an existing file — never a stale path.
        result = fu.sprite_path(expr)
        if result is not None:
            assert result.is_file()
