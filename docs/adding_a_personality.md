# Adding a personality

A personality is a self-contained drop-in: one entry in `PERSONALITIES` plus an optional asset directory. The architecture guarantees no special-cased code anywhere — get the entry right and the menu bar, Director, TTS, and click-to-talk all just work.

This doc is the **starting point**. It walks the full flow at a high level and links to the deep docs for sprites + voice when you need them.

---

## What you're building

A `Personality` ([src/unclefu/personalities.py](../src/unclefu/personalities.py)) has five things:

| Field | What it is | Required? |
|---|---|---|
| `key` | snake_case identifier, e.g. `british_mom` | yes |
| `display_name` | human-readable, e.g. `"British Mom"` | yes |
| `voice` | a Qwen3-TTS speaker id (preset name or path to a baked `.npy`) | yes |
| `prompt_fragment` | the character's voice + behaviour, injected into the Director's system prompt | yes (≥20 chars) |
| `click_lines` | hand-written quips for the "Talk to me" menu item | recommended |
| `expressions` | per-expression emoji overrides for the menu bar | optional |

Sprite art (PNGs) lives separately under `src/unclefu/assets/personalities/<key>/`. If any sprite is missing, the menu bar falls back to the emoji from `expressions` (or the global default). This means you can ship a working personality with **zero sprite art on day one**.

---

## Minimum viable personality (no art, no voice baking)

The fastest path: a CustomVoice preset + emoji fallback + a few good lines. Three steps.

### 1. Pick a Qwen3-TTS preset speaker

The CustomVoice model we ship comes with 9 preset speakers (run `uv run python tests/smoke/test_qwen_voices.py` to A/B listen):

| Speaker | Notes |
|---|---|
| `Ryan` | Dynamic English male, rhythm-forward (en-native) |
| `Aiden` | Sunny American male (en-native) |
| `Vivian` | Chinese-native female |
| `Serena` | Chinese-native female |
| `Uncle_Fu` | Chinese-native male — used by the `uncle_fu` personality |
| `Dylan` | Chinese-native male |
| `Eric` | Chinese-native male |
| `Ono_Anna` | Japanese-native female |
| `Sohee` | Korean-native female |

Only `Ryan` and `Aiden` read English cleanly; the others sound accented. That accent can *be* the character (it's Uncle Fu's whole thing) — but if you want a non-accented voice for a personality whose accent isn't part of the bit, you'll need to bake a custom voice (see [voice_design.md](voice_design.md)).

### 2. Write the prompt fragment

This is the most important step. It's injected into the Director's system prompt as the character's voice. **Describe what the character *does*, not what they *sound like***.

**Good:** *"You roast the user because you care. Name the specific thing on screen. Tell them what to do about it."*

**Bad:** *"You speak cryptically, with strange word choices and odd rhythm."* — the model will dutifully produce poetry like *"the wise words are buffering"*. See [learnings.md](learnings.md) for the full saga.

A useful structure (mirrored from `uncle_fu`):

```
You are the user's [archetype]. You've [past relationship to them]…

Your range:
- Most of what you say is [tone A] — [specifics]
- Sometimes [tone B] — [specifics]
- Occasionally [tone C — approval/grudging compliment]
- When something genuinely [risky/urgent], [response]

Your style:
- [Length] — usually [N] sentences max
- [Specificity rule]
- [Vocabulary do/don't]
- [Catchphrases / phrasings]
- No metaphor, no poetry, no life lesson.

Example lines:
- "..."
- "..."
  [12–18 concrete examples]

You speak as if mid-conversation — no greetings, no sign-offs.
```

Worked examples + explicit forbidden phrases beat abstract rules. "No 'siren song', no 'chorus', no 'wisdom buffering'" reads weird in a prompt but is the most reliable lever the model responds to.

### 3. Write the click lines

10–20 short, evergreen, in-voice lines for the "Talk to me" menu item. These bypass the Director (instant) but still go through dedup (no hearing the same line twice in a session).

Mix four buckets:

- Posture / presence: *"You. Sit up."*
- Wellness: *"You drink water yet?"*
- Mild scold: *"Phone down. Work up."*
- Focus nudge (generic — works for any user-typed focus): *"What you supposed to be doing again?"*

Keep each one ≤8 words. Lines that depend on a specific event ("you just opened Reddit") belong in the Director prompt, not here — click lines fire whenever the user asks, regardless of state.

### 4. Drop the entry into `PERSONALITIES`

```python
# src/unclefu/personalities.py
PERSONALITIES: dict[str, Personality] = {
    "uncle_fu": Personality(...),
    "british_mom": Personality(
        key="british_mom",
        display_name="British Mom",
        voice="Ryan",  # closest en-native preset; bake a real one later
        prompt_fragment="""\
You are the user's mother. British, warm, slightly tired of them …
""",
        click_lines=[
            "Love, you've been at it ages. Tea?",
            "Sit up properly.",
            ...
        ],
    ),
}
```

### 5. Try it

```bash
uv run python -m unclefu --cli --personality british_mom --focus "test"
# or in the menu-bar app:
uv run python -m unclefu --personality british_mom
```

The menu bar will use the global emoji defaults until sprite art exists. That's intentional — ship the voice + prompt first, commission art later.

---

## Adding sprite art

Once the personality has voice + lines that land, commission seven sprite PNGs and drop them at:

```
src/unclefu/assets/personalities/british_mom/
├── idle.png            # default state — character watching
├── talking.png         # while TTS is playing
├── disapproving.png    # nag / scold
├── concerned.png       # soft worry, late-night
├── smirk.png           # dry observation, mild tease
├── approving.png       # grudging compliment
└── alarmed.png         # risky action, urgency=high
```

You can also drop `idle_2.png`, `idle_3.png`, … — the menu bar auto-rotates between them every ~20 s for a sign-of-life effect.

**Spec:** 44 × 44 px, transparent background, distinct silhouette at small sizes. Full pipeline (model choice, prompts, chroma-key workflow, post-process) is in [sprite_design.md](sprite_design.md).

---

## Baking a custom voice

Only needed if none of the 9 CustomVoice presets fit your character — e.g. you want an en-native female and Qwen's only en-native female would be inappropriate, or you want a specific celebrity-ish voice.

Two paths, both produce the same artifact (`assets/voices/<key>.npy`):

- **Voice clone** (`Qwen3-TTS-Base`): 3-second reference clip + matching text → speaker embedding.
- **Voice design** (`Qwen3-TTS-VoiceDesign`): natural-language description → speaker embedding.

Full workflow + the `tools/bake_voice.py` script in [voice_design.md](voice_design.md). Once baked, set `voice="assets/voices/british_mom.npy"` instead of a preset name; everything else is identical.

---

## Premium personalities (planned)

The same drop-in shape supports premium personality packs as a future business model. A premium pack is a tarball containing:

- The `assets/personalities/<key>/` sprite directory
- The `assets/voices/<key>.npy` baked voice
- A small JSON describing the `Personality` fields (prompt fragment, click lines, etc.)

A future loader will discover and register packs from `~/Library/Application Support/UncleFu/personalities/<key>/`. No code paths today special-case any one personality — the architecture is ready; the loader, licensing, and distribution aren't built yet.

---

## Checklist

- [ ] Voice picked (preset or baked)
- [ ] Prompt fragment with WHAT-TO-SAY structure + ≥10 example lines + explicit forbidden phrasings
- [ ] 10–20 click lines, mixed across posture / wellness / scold / focus-nudge
- [ ] Entry added to `PERSONALITIES` in `src/unclefu/personalities.py`
- [ ] `uv run python -m unclefu --cli --personality <key> --focus "test"` runs and speaks
- [ ] (Optional) Seven sprite PNGs in `assets/personalities/<key>/`
- [ ] (Optional) Baked voice at `assets/voices/<key>.npy`
