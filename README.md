<div align="center">

<img src="assets/icon.png" alt="Uncle Fu" width="200" />

# Uncle Fu

**Your overprotective Chinese uncle, living in your menu bar.**
**He watches you work. When you drift, he tells you.**

[![macOS](https://img.shields.io/badge/macOS-15%20Sequoia-blue?logo=apple)](https://www.apple.com/macos/macos-sequoia/)
[![Apple Silicon](https://img.shields.io/badge/silicon-M--series-purple)](https://support.apple.com/116943)
[![Python](https://img.shields.io/badge/python-3.12-yellow?logo=python)](https://www.python.org/)
[![License: PolyForm NC](https://img.shields.io/badge/license-PolyForm--NC--1.0.0-green)](LICENSE)
[![Local only](https://img.shields.io/badge/runs-100%25%20local-success)](#privacy)

</div>

---

You launch the app. It asks you what you're focusing on. You type something — *"writing the auth migration"*, *"studying for the LSAT"*, *"finishing the slides"*. A little character moves into your menu bar and watches.

If you stay on task, he stays quiet.

If you slide into YouTube, scroll Reddit, get sucked into a Wikipedia rabbit hole — or if your posture collapses, or it's 1 AM and you look exhausted, or you're about to `git push --force` — he speaks up. In voice. Naming both the drift and the thing you promised:

> *"aiya. YouTube? You said you were writing the migration."*

> *"eyes too close. sit back."*

> *"force push? no. step away."*

Everything runs locally. Nothing leaves your Mac. Not a single frame ever touches a cloud.

<div align="center">

[![Watch the walkthrough](https://img.youtube.com/vi/HiLTkGs5z14/maxresdefault.jpg)](https://www.youtube.com/watch?v=HiLTkGs5z14)

*▶ Watch the walkthrough (7 min)*

</div>

---

## Quick start

### 1. Build the app

```bash
git clone https://github.com/<you>/unclefu
cd unclefu
uv sync
uv run python packaging/build.py        # ~10 min first time
cp -R dist/UncleFu.app /Applications/
open /Applications/UncleFu.app          # right-click → Open the first time (ad-hoc signed)
```

Requires `uv` (`brew install uv`) and `pipx` (`brew install pipx`) on the build machine.

### 2. First launch

Uncle Fu walks you through:

1. **System check** — Apple Silicon, macOS 15+, 16 GB RAM, 8 GB free disk.
2. **Permissions gate** — one prompt for **Camera** (posture), one for **Screen Recording** (drift detection). Both can be revoked any time in System Settings.
3. **Model download** — first launch fetches ~7 GB of weights (Gemma 4 VLM + Qwen3-TTS) into `~/.cache/huggingface/`. Menu bar shows progress.
4. **Focus prompt** — *"What are you focusing on for this session?"* Type a sentence, hit Start.

You'll then see Uncle Fu's face in the menu bar. Click it any time for a hand-written quip; pick **Change focus…** to switch context mid-session.

### 3. Headless / debug

```bash
uv run python -m unclefu --cli --focus "studying for the LSAT"   # terminal mode, no menu bar
uv run python -m unclefu --debug                                  # per-session debug log under ~/Library/.../UncleFu/debug/
uv run pytest                                                     # unit tests
```

See `uv run python -m unclefu --help` for all flags (sensor cadences, throttles, model overrides).

---

## What you get

Three loops running independently in the background:

- **Webcam sensor** (~every 30 s) — describes your posture, presence, alertness.
- **Screen sensor** (one per display, ~every 12 s) — describes what's on screen.
- **Director** (~every 20 s, text-only) — reads recent observations + your stated focus + real-world context (time of day, battery, input rate), and decides whether to speak.

The Director has two independent reasons to speak:

1. **Drift from focus.** Screen stopped matching what you said. → *"Reddit. You said you were debugging auth."*
2. **Wellness or risk.** Posture collapse, late-night, destructive shell command. → fires regardless of focus.

The menu bar swaps Uncle Fu's expression to match the line — `disapproving`, `concerned`, `smirk`, `approving`, `alarmed` — then settles back to `idle`. Click the icon for an instant hand-written quip that bypasses the throttle.

---

## Privacy

This is the whole moat. Lean on it everywhere.

- **All inference is in-process MLX.** No remote API calls, ever. The VLM (Gemma 4 E4B) and TTS (Qwen3-TTS) both run on your Apple Silicon GPU.
- **Frames never touch disk.** Captured in RAM, sent to the VLM, dropped.
- **No telemetry, no analytics, no account, no auto-update phoning home.**
- **Open source.** Verify it yourself: every network call is in the code, and there are zero non-`huggingface.co` ones outside the first-launch model download.

Model weights are fetched from HuggingFace on first launch only; cached locally afterward. After that, Uncle Fu works fully offline.

---

## Architecture

```
                 ┌─────────────────────────┐
                 │  Focus modal at launch  │   "What are you focusing on?"
                 │     (rumps.Window)      │   → stored on session row
                 └─────────────┬───────────┘
                               │ focus string
                               ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Webcam Sensor  │  │ Screen 0 Sensor │  │ Screen 1 Sensor │
│  thread, ~30s   │  │  thread, ~12s   │  │  thread, ~12s   │
│  1 image → JSON │  │  1 image → JSON │  │  1 image → JSON │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌────────────────────────────────────────────────────┐
    │      sensor_snapshot  (SQLite, WAL, thread-safe)   │
    └────────────────────────────────────────────────────┘
                              ▲
              reads recent    │ snapshots (text only, no images)
                              │
                   ┌──────────┴──────────┐
                   │      Director       │   text-only LLM call
                   │  thread, ~20s       │   reads focus + observations
                   │  Decision JSON      │   + real-world context +
                   │  (msg + expression) │   recent speech log
                   └──────────┬──────────┘
                              │
                   ┌──────────┴──────────┐         ┌──────────────┐
                   │     Intervener      │────────►│   Menu bar   │
                   │  throttle + dedup   │ icon    │  swaps sprite│
                   │  → Qwen3-TTS        │ swap    │  to match    │
                   │  (personality voice)│         │  expression  │
                   └─────────────────────┘         └──────────────┘
```

Deeper writeups live in `docs/`:
- [`adding_a_personality.md`](docs/adding_a_personality.md) — ship a new character (start here)
- [`decisions.md`](docs/decisions.md) — the load-bearing architectural choices
- [`learnings.md`](docs/learnings.md) — pyobjc / mlx-vlm / TCC gotchas worth knowing before you debug them yourself
- [`voice_design.md`](docs/voice_design.md) — how future personalities will get their own baked voices
- [`sprite_design.md`](docs/sprite_design.md) — the seven-expression art pipeline
- [`ideas.md`](docs/ideas.md) — parking lot for future features

---

## Roadmap

The character system is built so each character is a self-contained unit — a `Personality` entry + an asset directory at `src/unclefu/assets/personalities/<key>/`. Free personalities ship in the repo. Planned premium personality packs (British Mom, Chaos Goblin, Concerned Friend, …) drop in as additional directories. Walkthrough for shipping one is in [`docs/adding_a_personality.md`](docs/adding_a_personality.md).

Other things on the list:

- End-of-session report card (lines spoken, focus held, screenshot-friendly)
- "Shut up for an hour" global hotkey
- Settings UI for cadences + sensitivity + mute hours
- Real Developer ID signing + notarization (so TCC grants persist across rebuilds)
- Voice baking pipeline for new personalities — see [`docs/voice_design.md`](docs/voice_design.md)

---

## Requirements

| | |
|---|---|
| **OS** | macOS 15 Sequoia or later |
| **Chip** | Apple Silicon (M1+) — MLX is arm64-only |
| **RAM** | 16 GB minimum (peaks at ~7 GB with both models loaded) |
| **Disk** | 8 GB free for model weights |
| **Permissions** | Camera, Screen Recording |

Uncle Fu's preflight will refuse to start (with a clear message) if any of these aren't met.

---

## Tech stack

Python 3.12, `uv` for env management, [`venvstacks`](https://github.com/lmstudio-ai/venvstacks) for shipping the layered Python runtime inside the `.app`.

| Concern | Library |
|---|---|
| VLM | `mlx-vlm` running Gemma 4 E4B 4-bit |
| TTS | `mlx-audio` running Qwen3-TTS 1.7B CustomVoice 4-bit |
| Menu bar shell | `rumps` (on PyObjC / AppKit) |
| Webcam capture | `pyobjc-framework-AVFoundation` |
| Screen capture | `mss` |
| Image encoding | `Pillow` |
| Schema validation | `pydantic` |
| Storage | `sqlite3` (stdlib, WAL) |
| Packaging | `venvstacks` + Mach-O C launcher + ad-hoc `codesign` |

**No OpenCV.** Uncle Fu doesn't process video — it grabs one frame every 10–30 s per source and asks the VLM what it sees. The VLM does the vision work.

---

## License

[**PolyForm Noncommercial License 1.0.0**](LICENSE).

Plain English: you can read, fork, modify, share, and run Uncle Fu for any **noncommercial** purpose — personal use, education, research, charitable / public-sector organizations. You **cannot** sell Uncle Fu or a derivative as a product, bundle it into a commercial offering, or use it inside a for-profit business without a separate commercial license from the copyright holder.

If you want to use Uncle Fu commercially, reach out.

---

<div align="center">

*"aiya. close the tab. do the thing."*

</div>
