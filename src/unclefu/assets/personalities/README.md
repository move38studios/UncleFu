# Personality assets

Each personality lives in its own directory keyed by `Personality.key`. The
menu bar looks for PNGs at `<key>/<expression>.png`; missing files fall back
to the emoji defined in `Personality.expressions` (or the global default in
`src/unclefu/personalities.py`).

## Expression taxonomy

Seven sprites per character. `idle` and `talking` are runtime-driven; the
other five are picked by the Director when speaking. Keep the set in sync
with the `Expression` literal in `vlm/schema.py`.

| File | When it shows |
|---|---|
| `idle.png` | default — character at rest, watching |
| `talking.png` | during TTS playback |
| `disapproving.png` | nag / scold / "fix it" |
| `concerned.png` | soft worry, late-night, "are you ok" |
| `smirk.png` | dry observation, mild tease |
| `approving.png` | grudging compliment |
| `alarmed.png` | risky action, "what are you doing" |

## File spec

- PNG, transparent background.
- 44 × 44 px (22 pt @ 2x retina). The menu bar height is fixed; rumps will
  scale anything larger but the result looks fuzzy.
- Distinct silhouette at small sizes — the character has to read from
  across the screen, not just up close.

## Business model note

Free personalities (currently: `uncle_fu`) ship in the repo. Future premium
personalities can be a drop-in: a `<key>/` directory + a `PERSONALITIES`
entry. The same fallback path works for partial bundles (e.g. premium
character with only the five Director-picked expressions, falling back to
shared idle/talking sprites).
