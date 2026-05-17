"""One-shot per-session classifier: given the user's stated focus, decide
which sensors are relevant AND generate short focus-specific context
paragraphs that the sensors splice into their per-call prompts.

Called once at session start (and again on "Change focus…"). The result:
- gates whether the screen sensor runs (webcam is always on)
- gates whether the focus is even an intelligible statement (re-prompt
  on gibberish)
- provides per-sensor "what should I be looking for" context that gets
  prepended to the base sensor prompt, so sensor observations carry a
  hint about what on-task looks like for THIS focus

Webcam is ALWAYS on as a design decision — it carries wellness signals
(posture, alertness, presence) that are valuable regardless of focus,
and "user is at the desk during their piano practice" is a real drift
signal. Per-focus context tunes WHERE the sensor pays attention; it
never disables wellness coverage.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..vlm.client import MlxVlmClient, VLMError, parse_json_response


CLASSIFIER_PROMPT = """You are processing a user's stated focus for a Uncle Fu focus session. Uncle Fu is a focus companion — it watches the user via webcam + screen sensors and speaks up when they drift from what they said they'd do.

The user just typed this as their focus:
  "{focus}"

Return JSON with FOUR fields:

1) `valid`: is this an intelligible focus statement?
   - TRUE if it's a plausible thing a real person could be sitting down to
     focus on — even if short or casual. "writing code", "cooking",
     "reading", "thinking about the proposal", "studying", and vague
     things like "work" or "stuff" all pass.
   - FALSE if it's clearly nonsensical: random keyboard mash
     ("dsfalkj", "asdfasdf"), pure punctuation/symbols, a single
     character, or obvious test input ("test", "asdf", "x", ".").
   - When in doubt, lean TRUE.

2) `screen`: is the SCREEN sensor worth running for this focus?
   Only meaningful if valid=true. If valid=false, set screen=true.
   - screen=true → COMPUTER activity. Writing code, studying online,
     reading PDFs, watching videos, designing in an app, Zoom call,
     browsing docs, writing in a Google Doc / Notion / Obsidian,
     reviewing PRs, emailing, online practice tests, watching a movie.
   - screen=false → OFF-COMPUTER physical activity. Tuning an instrument,
     cooking, exercise, reading a paper book, meditation, drawing on
     paper, journaling on paper, walking, in-person conversation, nap.
   When in doubt, lean true.

3) `webcam_context`: ONE concise sentence (max 200 chars) describing
   what the webcam should look for given THIS focus. The webcam sensor
   ALREADY watches for posture, presence, alertness — your job is to add
   one focus-specific framing on top. The sensor will splice your
   sentence into its prompt.

   Examples:
   - focus="writing the auth migration" → "User should appear to be at
     the desk, looking at the screen."
   - focus="tuning my piano" → "User should ideally be AWAY from the
     desk (at the piano); seeing them at the desk is a drift signal."
   - focus="cooking dinner" → "User should be away from the desk
     (in the kitchen); presence at the desk is a drift signal."
   - focus="going for a run" → "User should not be present at the desk.
     Empty desk is the on-task state."
   - focus="reading 'War and Peace' (paperback)" → "User may be at the
     desk reading a paper book, or in a chair away from the screen.
     Either is on-task."

4) `screen_context`: ONE concise sentence (max 200 chars) describing
   what the screen sensor should look for given THIS focus. Only generate
   this if screen=true. If screen=false, set screen_context to an empty
   string "".

   Examples:
   - focus="writing the auth migration" → "On-task content: code editor
     with auth-related files, terminal, related docs, AI assistant
     working on auth."
   - focus="designing the landing page in Figma" → "On-task content:
     Figma window with a landing-page design open."
   - focus="watching the new Marvel movie" → "On-task content: a video
     player showing a movie. Other windows or web browsing is drift."
   - focus="studying for the LSAT online" → "On-task content: LSAT
     practice material, test prep website, related PDFs."

Keep the context sentences SHORT and DIRECTIONAL. They're hints for a
small VLM, not full instructions. Mention "drift signal" or "on-task
content" when it helps the sensor know what's at stake.

Return ONE JSON object, nothing else. No prose, no markdown fences.

{{
  "valid": true | false,
  "screen": true | false,
  "webcam_context": "<one sentence, max 200 chars>",
  "screen_context": "<one sentence, max 200 chars>" | ""
}}
"""


class Relevance(BaseModel):
    """Per-session sensor relevance + focus validity + per-sensor context.

    Webcam is always on (architecture-level decision). `valid=false` means
    the caller should re-prompt the user. `webcam_context` and
    `screen_context` are short paragraphs the sensors splice into their
    per-call prompts to bias their observations toward the focus.
    """

    valid: bool
    screen: bool
    webcam_context: str = Field(default="", max_length=240)
    screen_context: str = Field(default="", max_length=240)

    @property
    def webcam(self) -> bool:
        return True

    @property
    def reason(self) -> str:
        """Synthetic display field — what gets shown in startup print
        and debug logs. Built from the contexts since we no longer
        carry a separate `reason` field (the contexts ARE the reason)."""
        if not self.valid:
            return "invalid focus"
        if self.screen and self.screen_context:
            return self.screen_context
        if not self.screen and self.webcam_context:
            return self.webcam_context
        return "screen on" if self.screen else "screen off"

    @classmethod
    def placeholder(cls) -> "Relevance":
        """Pre-classification stand-in. Conservative: screen sensor off
        (we'd rather not run a screen sensor unsupervised by focus
        context for the first few seconds — Director can wait). Webcam
        runs with its base prompt (no focus context). The real
        Relevance replaces this once `classify_focus` returns on the
        background thread."""
        return cls(
            valid=True, screen=False,
            webcam_context="", screen_context="",
        )


def classify_focus(vlm: MlxVlmClient, focus: str) -> Relevance:
    """Classify a focus string. Returns a Relevance object.

    On any failure (VLM error, malformed JSON, schema mismatch) we fall
    back to `valid=true, screen=True` with empty contexts — the safe
    defaults. We'd rather start the session with both sensors on and no
    focus framing than reject a real focus because of a transient
    classifier failure.
    """
    cleaned = (focus or "").strip()
    if not cleaned:
        return Relevance(valid=False, screen=True, webcam_context="", screen_context="")

    try:
        result = vlm.chat(
            system=CLASSIFIER_PROMPT.format(focus=cleaned),
            user_text="Classify the focus above. Reply with JSON only.",
            images_jpeg=(),
            max_tokens=300,  # bumped — contexts are longer than just {screen, reason}
            temperature=0.0,  # classification — pick the most likely answer
        )
        parsed = parse_json_response(result.text)
        return Relevance.model_validate(parsed)
    except (VLMError, Exception):
        return Relevance(
            valid=True, screen=True,
            webcam_context="", screen_context="",
        )
