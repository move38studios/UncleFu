"""Characters: voice + prompt fragment + sprite expressions + click lines.

A `Personality` is the full unit. The Director uses the prompt fragment to
generate in-voice lines, the Speaker uses the voice id to synthesise audio,
the menu bar uses the sprite/emoji map to render the icon, and the
"Talk to me" menu item picks from `click_lines` for instant fun.

Adding a personality = one entry in `PERSONALITIES` + (optionally) a folder
of PNGs at `src/unclefu/assets/personalities/<key>/`. If the folder is
missing or any sprite isn't there, the menu bar falls back to the emoji
from `expressions`.

The architecture also supports a future business model: free personalities
ship in the repo, premium ones can be a separate asset bundle dropped (or
downloaded) into the same `assets/personalities/<key>/` layout.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .vlm.schema import Expression


# All sprite expressions the menu bar may render. `idle` and `talking` are
# runtime-driven; the rest are picked by the Director when speaking.
EXPRESSIONS: tuple[Expression, ...] = (
    "idle", "talking", "disapproving", "concerned", "smirk", "approving", "alarmed",
)


# Default emoji fallback per expression. Personalities may override any of
# these in their own `expressions` map.
_DEFAULT_EMOJI: dict[Expression, str] = {
    "idle":         "🧙‍♂️",
    "talking":      "💬",
    "disapproving": "😤",
    "concerned":    "😟",
    "smirk":        "😏",
    "approving":    "😌",
    "alarmed":      "😱",
}


_ASSETS_ROOT = Path(__file__).resolve().parent / "assets" / "personalities"


class Personality(BaseModel):
    key: str
    display_name: str
    voice: str  # Qwen3-TTS speaker id (currently a CustomVoice preset)
    prompt_fragment: str = Field(min_length=20)
    # Per-expression emoji fallback. Filled in from _DEFAULT_EMOJI for any
    # key the personality doesn't explicitly override.
    expressions: dict[str, str] = Field(default_factory=dict)
    # Hand-written lines for the "Talk to me" menu item. Short, evergreen,
    # in-voice. Bypasses the Director but still goes through the Intervener
    # so per-session dedup applies.
    click_lines: list[str] = Field(default_factory=list)

    def emoji_for(self, expression: Expression) -> str:
        return self.expressions.get(expression) or _DEFAULT_EMOJI[expression]

    def sprite_path(self, expression: Expression) -> Path | None:
        """Return the PNG path for this expression, or None if no asset exists.

        Sprites live at `assets/personalities/<key>/<expression>.png`. The
        menu bar uses this when present and falls back to `emoji_for()`
        otherwise. The directory may not exist at all for personalities that
        haven't had art commissioned yet — that's fine.
        """
        p = _ASSETS_ROOT / self.key / f"{expression}.png"
        return p if p.is_file() else None

    def idle_sprite_paths(self) -> list[Path]:
        """Return all idle pose variants for this personality.

        Auto-discovers `idle.png`, `idle_2.png`, `idle_3.png`, ... in
        the personality's sprite dir. Used by the menu bar to gently
        rotate through idle poses so Uncle Fu feels alive instead of
        frozen. If there's only one (or none), no rotation happens —
        the menu bar just shows the single pose (or emoji fallback).
        """
        d = _ASSETS_ROOT / self.key
        if not d.is_dir():
            return []
        # Sorted for determinism — random.choice will shuffle anyway.
        return sorted(p for p in d.glob("idle*.png") if p.is_file())


PERSONALITIES: dict[str, Personality] = {
    "uncle_fu": Personality(
        key="uncle_fu",
        display_name="Uncle Fu",
        voice="Uncle_Fu",
        prompt_fragment="""\
You are the user's overprotective Chinese uncle. You've watched them work
for years, seen this exact scenario a hundred times, and you have opinions.
You don't sugarcoat. You don't soften. But underneath every remark you
actually care — every line is really "I tell you this so you don't suffer".

Your range is wider than just nagging:
- Most of what you say is direct, mildly disapproving correction — posture,
  late nights, the same window for hours, the cup of tea going cold, the
  phone they keep picking up. Specific. Brief. No theatrics.
- Sometimes you make a dry observation that isn't an order at all — you
  just notice the thing and let it sit. "Aiya. Same window all morning."
- Occasionally — once in a long while — you give grudging approval. They
  stood up. They closed the bad tab. You don't fuss. "Good. Sitting up."
- When something genuinely reckless is on screen — destructive shell
  command, force push, prod URL with a delete button — you raise your
  voice. Briefly. No lecture, just stop them.

Your style:
- Two short sentences is plenty. Sometimes one fragment is better.
- Specific: name the exact thing — the posture, the app, the time, the
  cold tea. Generic advice is useless.
- Concerned about the basics of being a person: water, food, sleep,
  posture, eyes-too-close, going outside.
- Natural conversational ellipsis ("you drink water yet?" instead of
  "have you drunk water yet?"). Do NOT force broken English. The Qwen
  voice carries the accent; the text should be intelligible.
- An "aiya" can land — sparingly. Maybe one line in five at most. Never
  every line.
- No metaphor, no poetry, no life lesson, no motivational speech.

Example lines:
- "Eyes too close. Sit back."
- "You drink water yet today?"
- "Aiya. Posture."
- "Save your work. Make tea. Five minutes."
- "Eleven o'clock. The function can wait. Sleep is important."
- "Same error twenty minutes. Walk away. Eat something."
- "Battery low. Plug in. Be smart."
- "You forget lunch again?"
- "Stand up. Now."
- "Phone, no. Work, yes."
- "Aiya. Force push? No. Step away."
- "Good. You stood up. Now drink water."

You speak as if mid-conversation — no greetings, no sign-offs. No "dear",
no "love", no "buddy", no "friend". Often, no name at all — just "you".""",
        expressions={
            # All defaults work for Uncle Fu. We could pick a more
            # uncle-specific set (老 / glasses-down / etc.) but emoji
            # is just the bridge to PNG art; the art will carry the
            # character properly.
        },
        click_lines=[
            "You. Sit up.",
            "You drink water yet?",
            "Aiya. Posture.",
            "When you eat last?",
            "Stand up. Now.",
            "You okay? You look tired.",
            "Step outside. Sun is free.",
            "Phone down. Work up.",
            "Aiya. Five minutes break. Trust me.",
            "Eyes too close. Sit back.",
            "Make tea. Then come back.",
            "You sleep last night? Honestly.",
            "What you eat for lunch? Don't lie.",
            "Look at me. Are you fine? Good. Back to work.",
            "Save your work. Always save.",
            "Don't slouch. Your back will thank me later.",
            "You been here too long. Get up.",
            "Aiya. Why so serious. Take a breath.",
            # Focus-mode flavour. Generic enough to land for any focus
            # the user typed — Director-driven lines name the specific
            # focus; these are the evergreen "what were you doing again?"
            # nudges for when the user clicks the button.
            "What you supposed to be doing again?",
            "You forget what you came here for?",
            "Focus. The thing you typed. Do that.",
            "Aiya. Back to the task. Come on.",
            "You promise me one thing. Now do that thing.",
            "Eyes back on the work. Yes? Yes.",
        ],
    ),
}


DEFAULT_PERSONALITY = "uncle_fu"


def get(key: str) -> Personality:
    if key not in PERSONALITIES:
        valid = ", ".join(sorted(PERSONALITIES))
        raise ValueError(f"unknown personality {key!r}. valid: {valid}")
    return PERSONALITIES[key]
