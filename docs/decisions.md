# Decisions

Load-bearing engineering choices with the reasoning, so they don't get relitigated. README is the product spec; this file is the *why* behind the architecture.

---

## Packaging: venvstacks layered runtime

The .app ships a layered Python runtime built with [venvstacks](https://venvstacks.lmstudio.ai/) — three stacked virtualenvs (cpython runtime / framework with MLX + transformers / application with rumps + pyobjc) chained via `sitecustomize.py`. A small Mach-O C launcher (`Contents/MacOS/UncleFu`) dlopens libpython and runs `-m unclefu` in-process.

Why venvstacks specifically:

- MLX ships as a namespace package and uses ad-hoc dynamic linking that confuses the older bundling tools. venvstacks was built specifically to package MLX-based apps; it's what LM Studio uses internally.
- Layered design means the heavy framework layer (~500 MB) is cacheable separately from the small application layer that actually changes between builds.
- Native LaunchServices requires a Mach-O `CFBundleExecutable` — a shell-script launcher won't be granted menu-bar UI access on Sequoia. Hence the C launcher.

Why the layers live under `Contents/Resources/` and not `Contents/Frameworks/`: codesign treats *every* directory under `Frameworks/` as a signable bundle, and venvstacks layers contain JSON metadata + headers + Python source that fail that validation. `Resources/` has no such constraint; the layers run identically because all internal paths are `@executable_path`-relative.

The .app is ad-hoc signed (`codesign -s -`). That's the minimum bar for macOS TCC to persist permission grants. Real Developer ID signing + notarization is on the roadmap.

## VLM: in-process via mlx-vlm

`MlxVlmClient` owns one loaded Gemma 4 model on a dedicated worker thread (MLX streams are thread-local — same constraint as mlx-audio; see [learnings.md](learnings.md)). Sensors and Director submit `_Request` objects to a queue; the worker pulls, generates, returns via a per-request response queue. Callers feel like they're calling a synchronous function.

Key properties:

- **One model, many callers.** Webcam, screens, Director all share the same loaded Gemma — ~3 GB resident regardless of how many sensors run.
- **Text-only via the same path.** Director sends `images_jpeg=()`; mlx-vlm handles `image=[]` cleanly (no images = text-only generation).
- **No HTTP at runtime.** The only network call is the first-launch `huggingface_hub.snapshot_download` of model weights to `~/.cache/huggingface/`. After that the app works fully offline.

In-process inference also means a self-contained app — no separate model server to start, no two-app onboarding, no HTTP envelope overhead.

## Focus mode is the product

Every session starts with the user typing what they're focusing on. The Director gets that as the first block of every per-cycle user message and uses it as the headline signal:

- **Drift from focus → speak.** Screen content stopped matching the stated focus. Uncle Fu names both: *"Reddit. You said you were debugging auth."*
- **Wellness or risk → speak.** Posture collapse, eyes-too-close, late-night, destructive shell command. Independent of focus — fires regardless.

A small VLM can't reliably judge "is this productive work?" in general. It *can* judge "is this screen aligned with the stated focus?" That sharper contract is the whole product.

Implementation:
- `session.focus TEXT NOT NULL` — no focus, no session.
- `rumps.Window` blocks on launch. Cancel = quit. "Change focus…" menu item re-pops the modal mid-session.
- `--focus` flag required in `--cli` mode.
- Focus is NOT shown in the menu bar dropdown by design — the ritual is in the typing, not the staring.

What this is NOT: a Pomodoro timer, a site blocker, or a productivity tracker. There are no session lengths, nothing is blocked, and the report card is the character's verdict — not a chart.

## The character is the metric — no score

No numeric state, no tier system, nothing on the menu bar except the character's face. Signals (posture, alertness, screen content, real-world context like time of day and battery) go straight into the Director's prompt; the Director decides when and how to speak and emits an `expression` that drives the menu bar icon.

A score layer between inputs and decision is a middleman the model can replace by reading the inputs directly. It also dilutes the product — "personality in your menu bar" is a different thing from "wellness dashboard with a number on it."

## Expression taxonomy: seven sprites, five Director choices

| Expression | Who picks | When |
|---|---|---|
| `idle` | runtime (default) | not speaking |
| `talking` | runtime (auto) | TTS playing |
| `disapproving` | Director | nag / scold / "fix it" |
| `concerned` | Director | soft worry, late-night, tiredness |
| `smirk` | Director | dry observation, mild tease |
| `approving` | Director | grudging compliment |
| `alarmed` | Director | risky action, urgency=high |

The model picks `expression` alongside `message` in the same JSON. No second call, no separate classifier.

Why this exact count: each one maps to a tonal register the Director prompt already needs. Fewer than five collapses meaningfully different tones together (a scold and a soft worry look the same). More than seven and a small VLM/LLM starts confusing them — `disappointed` vs `disapproving` vs `concerned` is too fine a distinction.

The taxonomy is locked in `vlm/schema.py` as `Expression: Literal[...]`. Adding or removing one is a coordinated change: schema enum + prompt examples + sprite filenames + emoji fallback map.

## PNG-first sprites with emoji fallback

The menu bar prefers PNG sprites at `assets/personalities/<key>/<expression>.png` but falls back to an emoji per expression if any sprite is missing. This decouples the *system* (PNG-ready) from the *content* (sprites can land later without blocking other work).

Each personality also gets to override individual emoji in `Personality.expressions` if the global defaults don't fit.

## User-initiated speech bypasses the Intervener throttle

`Intervener.maybe_speak()` takes a `force=False` kwarg. The Director path always uses the default (`force=False`) so the min-gap + cooldown gates apply. The "Talk to me" menu button passes `force=True` — the user explicitly asked, so the throttle (which exists to keep the *automated* Director from over-talking) doesn't apply.

Two things `force=True` still respects:

- **Dedup.** A forced repeat of an already-spoken line returns `DEDUPED`. Repetition kills the character; a button-press doesn't override that.
- **`_last_spoke_at` update.** A forced speech still resets the throttle clock, so the Director doesn't immediately fire on top of a click line.

Any future user-driven speech (hotkey, settings "preview voice", etc.) takes the same path.

## Decoupled sensors + Director

- **Sensors** are dumb observers. One per source. Each does one thing — look at its input, describe it. Tiny prompt, tiny output, fast call (<8 s on Gemma 4 E4B).
- **Director** is a text-only LLM call. Reads recent snapshots from SQLite + focus + real-world context + recent speech log, decides whether to speak, what to say, and which expression to show.

Folding vision + decision into one multi-image call is slow (image budget grows with sensors) and brittle (every problem and every fix lives in the same prompt). The independent-component shape is faster end-to-end *and* easier to debug — a stuck sensor or a wrong call is contained.

Each component runs on its own cadence (webcam ~30 s, screens ~12 s, Director ~20 s) in its own thread.

## Free-form `screen.content`, not an enum

Sensors return `screen.content` as free-form text like *"YouTube — coding tutorial"*, *"GitHub PR #42"*, *"Google Slides — sprint review"*. We tried an `app_category` enum for canonicalization; in practice the model kept emitting reasonable-but-out-of-enum values (`"presentation"`, `"design tool"`) and validation failures cost more than the enum was worth.

Posture / distance / alertness are still constrained enums — those are a small fixed set and the model handles them fine.

## History feedback: narratives only

Every cycle the Director sees the last 4 narratives. We *don't* feed back past structured signals — the model should re-read posture / distance / alertness from the fresh image each cycle, otherwise it gets sticky on improvements.

## Premium personalities as the business model

A personality is a self-contained drop-in: a `PERSONALITIES` entry + an `assets/personalities/<key>/` directory. No code paths special-case any one personality.

This shape makes "ship one free, sell more" a future business model without rearchitecting:

- Free: Uncle Fu (in repo).
- Premium (planned): British Mom, Chaos Goblin, Tech Bro, etc. — each a downloadable directory containing sprites, a voice embedding (`.npy`), and a small JSON describing the prompt fragment and click lines.

Implications carried into the design today:

- No personality-specific code outside `personalities.py` and the assets dir.
- Voice loading (Qwen3-TTS) already supports either a preset speaker name or a path to a `.npy` embedding (see [voice_design.md](voice_design.md)).
- The Personality model is plain pydantic — a future "load from disk" loader is just JSON → `Personality(...)`.

Licensing, payment, and distribution are not built yet. The architecture just doesn't paint itself into a corner that would prevent them.

## Python + rumps, not Electron / Tauri / Swift

UI surface is tiny — a menu bar icon + a few modals. No good reason to ship Chromium for that. Swift is a future investment, not an MVP cost. The whole app fits in plain Python on top of `rumps` (which is a thin wrapper over PyObjC / AppKit).

## `AVCapturePhotoOutput`, not `AVCaptureVideoDataOutput`

Photo output doesn't require a `dispatch_queue_t`. PyObjC 12 doesn't ship a clean way to create one. Video data output is the "correct" pattern for analysis loops in native code, but for a one-frame-every-30-seconds workload, the simplicity of photo output wins. Revisit if we ever need higher frame rates or finer control over exposure.

## Threads, not asyncio

`PeriodicThread` per sensor + per Director. SQLite (WAL) handles concurrent writes fine. AVFoundation stays on the webcam thread; the main thread runs the rumps / NSApplication run loop. Mixing asyncio with the run-loop pumping AVFoundation needs would be more complex than the threaded version.

## Qwen3-TTS, not macOS `say`

Qwen3-TTS via [mlx-audio](https://github.com/Blaizzy/mlx-audio) — `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit`. The quality gap to a neural TTS is too large to ignore; the *one thing* that decides whether a user keeps the app running is whether they enjoy hearing it speak.

Costs we accept: ~2 GB on disk, ~4 GB peak RAM during synth, ~5–7 s per one-sentence line on M-series. We pass `repetition_penalty=1.3` to discourage tail-babble. No fallback to `say` — keeping a fallback path produces two slightly-different in-product experiences; better to stay silent and log if Qwen fails to load (same as `--mute`).

The 1.7B variant specifically, not the 0.6B — the smaller model has unreliable EOS and truncates English lines mid-word. See [learnings.md](learnings.md) for the full saga.

## One personality for now: Uncle Fu

`Uncle_Fu` is the only Qwen3-TTS preset speaker that genuinely fits the overprotective-Chinese-uncle persona out of the box — the others are mostly Chinese/JP/KR-native and sound accented when reading English. Shipping one tight voice + character pairing where they pull in the same direction beats shipping four where three feel mismatched.

Future personalities (British Mom, Tech Bro, Chaos Goblin) come back via baked speaker embeddings — plan in [voice_design.md](voice_design.md).

## SQLite, not JSONL files

Indexed time queries ("snapshots from the last hour"), atomic appends without file locking under threaded sensor loads (WAL mode), and an obvious place for the future report-card / journal views to live. Each sensor snapshot's source-specific fields are stored as JSON in the `structured` column so we can reconstruct it via `WebcamStructured.model_validate(...)` / `ScreenStructured.model_validate(...)`.

## Smoke tests live in `tests/smoke/`, not `tests/`

They require hardware (camera, screens) and download/load real MLX models. `tests/unit/` is for pure-function code (intervener throttle, schema, prompt formatting). Smoke tests run manually from a real terminal; pytest skips them by default.

## No OpenCV, no MediaPipe

The VLM does the vision work. OpenCV would just be ~90 MB of native binary to call `VideoCapture(0).read()` — for which we already use a 50-line pyobjc / AVFoundation wrapper.
