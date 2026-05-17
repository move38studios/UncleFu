# CLAUDE.md

Working notes for Claude (and us). Keep it tight; if something gets long, push the detail into `docs/`.

## What this project is

Uncle Fu is a local-only macOS menu-bar focus companion. At session start the user types what they're focusing on; a character (Uncle Fu today, more later) sits in the menu bar and speaks up when (a) the screen drifts off-focus or (b) wellness/risk signals trigger — posture collapse, late-night, destructive shell commands. The menu bar icon is the character's face; it changes expression to match the line being spoken. Spec is in [README.md](README.md). Architectural choices live in [docs/decisions.md](docs/decisions.md). Gotchas live in [docs/learnings.md](docs/learnings.md) — read this before debugging anything pyobjc / mlx-vlm / TCC-related; you will hit a thing that's in there.

## Stack

- **Python 3.12** (not 3.14 — wheel coverage is patchy for pyobjc/pillow).
- **uv** for env + deps + running. Always `uv run <cmd>`, never bare `python`.
- **pyobjc** (only the framework packages we actually use — `AVFoundation`, `Cocoa`, `Quartz`, `CoreMediaIO`). Never `pyobjc` meta-package.
- **mss** for screen capture, **mlx-vlm** for in-process VLM (Gemma 4 E4B), **pydantic** for schemas, **pillow** for image manipulation when needed.

## Principles

### Types

- **pyright in basic mode.** Strict mode is too noisy with pyobjc's untyped surface. Targets: zero pyright errors in `src/unclefu/`; `tests/` may relax.
- **Annotate everything in `src/`.** Functions, methods, dataclasses. `from __future__ import annotations` is fine.
- **Pydantic at every boundary.** Anything coming over the wire (VLM responses), off disk (settings, session log), or across module boundaries with non-trivial shape goes through a pydantic model. Pure-function internals don't need it.

### Tests

- **pytest under `tests/unit/`** for pure-function code (history buffers, schema helpers, prompt formatting, throttle/dedup logic). Aim for *decent* coverage — every branch in the Intervener, edge cases in time formatting, schema round-trips. Don't chase 100%.
- **`tests/smoke/`** are manual integration tests that need hardware (camera, screens) and download/load the real MLX models. They live as runnable scripts (`uv run python tests/smoke/test_webcam.py`). Pytest ignores them by default.
- A test that requires shipping new code to be deleted/refactored to pass is a *good* test. A test that locks in implementation details is not.

### Code style

- **Short, no apology.** No big preambles in functions, no narrating comments. Names carry the meaning. See the [main system instructions](https://docs.claude.com/en/docs/claude-code/memory#claude-md) on comments — apply them here.
- **Don't pre-abstract.** Three similar lines beats a clever helper. We will know what the right abstraction is after building 80% of the app, not before.
- **One way to do a thing.** Don't add config knobs we can't justify with a real use case.

### Privacy — the moat

This is the whole product positioning. **Treat any new network call to a non-localhost host as a release-blocking bug.** Frames live in RAM, never on disk (except smoke-test scratch files in `/tmp/`). No telemetry, no analytics, no auto-update phoning home.

## Running

```
uv run python -m unclefu                                    # menu-bar mode; modal asks for focus
uv run python -m unclefu --focus "writing the migration"    # skip modal with --focus
uv run python -m unclefu --cli --focus "studying"           # headless terminal mode (--focus required)
uv run python -m unclefu --debug                            # per-session text debug log
uv run python tests/smoke/test_webcam.py                        # one-shot manual smoke test
uv run pytest                                                   # unit tests only
uv run pyright src/                                             # typecheck
```

## TTS

- **Qwen3-TTS only.** No `say` fallback. Default model:
  `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit` (~2 GB on disk,
  ~4 GB peak RAM during synth, ~5-7 s per one-sentence line on
  M-series). Loaded on a background thread at startup via `mlx-audio`;
  output played through `afplay`. We pass `repetition_penalty=1.3`
  to discourage tail-babble. **Do NOT downgrade to the 0.6B variant**
  — it has unreliable EOS and truncates lines mid-word; see
  [docs/learnings.md](docs/learnings.md) for the full saga.
  [docs/voice_design.md](docs/voice_design.md) covers the voice-baking
  plan for future personalities.

## Conventions

- New module lives under `src/unclefu/<area>/`. Keep `__init__.py` empty unless there's a real reason.
- VLM is `mlx-community/gemma-4-e4b-it-4bit` loaded once at startup into a dedicated worker thread (`MlxVlmClient` in `vlm/client.py`); sensors + Director share it through a request/response queue. No external server.
- **No constrained decoding.** `mlx-vlm.generate()` is plain free generation; constrained / JSON-schema decoding on Gemma 4 was ~10× slower in benchmarks. Put the JSON shape in the prompt; validate with pydantic on receipt.
- **One concern per LLM call.** Sensors describe one source. Director decides whether to speak (and which expression to show). Don't fold these back together "for efficiency" — folding vision + decision into one call is slow and fragile.
- When you hit a new gotcha — pyobjc, mlx-vlm, macOS permissions, venvstacks — append it to [docs/learnings.md](docs/learnings.md). Future Claude will thank present Claude.

## Module map (current)

```
src/unclefu/
├── __main__.py              CLI
├── capture/                 raw frame grabs (webcam, screens)
├── sensors/                 per-source observers (webcam_sensor.py, screen_sensor.py)
├── director/                Director text-only LLM call + real-world context
├── intervene/intervener.py  throttle + dedup + dispatch to TTS
├── tts/speaker.py           Qwen3-TTS via mlx-audio
├── personalities.py         characters: voice + prompt fragment + sprites + click lines
├── assets/personalities/    sprite directories, one per character (PNG-first, emoji fallback)
├── storage/                 SessionLog (SQLite) + DebugLog (per-session text)
├── runtime/                 PeriodicThread, Runner (orchestrates threads)
├── ui/menubar.py            rumps app — icon swapping + click-to-talk menu
└── vlm/                     MlxVlmClient (in-process Gemma 4 via mlx-vlm) + shared schema (incl. Expression)
```

## Focus mode is the product

The user types what they're focusing on at session start. The Director
gets that as the first block of every cycle and uses it as the headline
signal: drift from focus = speak; wellness/risk signals fire
independently regardless of focus.

Don't add "is this generally productive?" logic to the Director prompt.
The contract is concrete — user said X, screen shows not-X, Uncle speaks.
Generic productivity inference is out of scope.

## The character is the metric

No internal score, no tier system, no numeric state. Signals (posture,
alertness, screen content, real-world context) go straight into the
Director's prompt; the Director decides when and how to speak and emits
an `expression` (`disapproving` / `concerned` / `smirk` / `approving` /
`alarmed`) that drives the menu bar icon. Don't add a middleman layer.
