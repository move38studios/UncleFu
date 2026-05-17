# Uncle Fu Lite — speculative v2

A radically simplified version of Uncle Fu that any Mac user can run, not just the ones with 24 GB and a fan. **Speculative — not committed.** Numbers and model choices are best-guess based on current (mid-2026) MLX ecosystem.

The current Uncle Fu loads Gemma 4 E4B (~3 GB resident) + Qwen3-TTS 1.7B (~4 GB peak during synth), needs the GPU continuously, and effectively gates the product to M-series Pro / Max with 16+ GB. The "lite" version targets:

- **Any Apple Silicon Mac**, including the base 8 GB Air
- **No noticeable battery drain** — Neural Engine + tiny CPU subprocesses, no continuous GPU
- **Instant install** — total weight model footprint ~1 GB vs ~7 GB today
- **Sub-second reactions** instead of the 20-second Director tick

The core promise stays the same: *you type a focus, a character lives in your menu bar, it speaks up when you drift or when wellness/risk signals fire.* What changes is how the perception + speech layers get there.

---

## Stack comparison

| Layer | Current (heavy) | Lite (speculative) |
|---|---|---|
| Screen perception | mlx-vlm + Gemma 4 E4B, ~5.2 GB on disk, ~3 GB RAM, ~2.3 s/call | macOS Accessibility APIs + Vision OCR — **0 MB models**, ~100 ms/call |
| Webcam perception | mlx-vlm same Gemma | Vision framework face / pose, on Neural Engine — **0 MB models**, ~20 ms/call |
| Director (decide + write line) | Gemma 4 E4B via mlx-vlm | Tiny MLX LLM (~700 MB) — see below |
| TTS | Qwen3-TTS 1.7B, ~2 GB disk, ~4 GB peak RAM, ~5-7 s/line | macOS `say` (or AVSpeechSynthesizer) — **0 MB models**, ~0 latency |
| **Total models on disk** | **~7 GB** | **~700 MB** |
| **Steady-state RAM** | **~7 GB** | **~1 GB** |
| **Eligible Mac** | M-series 16 GB+ | Any Apple Silicon, including 8 GB Air |

---

## Perception: tiered, cheap, fast

Three tiers, each answering a different granularity of "what's the user doing?" Cheaper tiers run first; pricier ones are only consulted when the cheap signal is ambiguous.

### Tier 1 — Accessibility (free, always-on)

macOS Accessibility APIs give you the macro signal in near-zero time:

- Frontmost app name (`NSWorkspace.frontmostApplication`)
- Window title of the focused window
- Browser URL (AppleScript / Accessibility for Safari, Chrome, Arc, …)

This alone gets you ~70 % of "drifting from focus": *"Chrome — YouTube — RustConf 2024 keynote"* matched against the user's focus string handles the obvious cases. The drift detector at this tier is essentially `if focus_keywords.isdisjoint(window_title_tokens): suspicious`.

### Tier 2 — Vision OCR (cheap, on demand)

When Tier 1 is ambiguous — Slack, Notion, Figma, Linear, anything Electron-y where the window title is generic and the actual content lives in a Canvas — capture the focused window and run Apple's `VNRecognizeTextRequest`. Runs on the Neural Engine in ~50–200 ms with no external deps and excellent accuracy on screen text (no Tesseract DPI / font headaches). Gives you the actual words on the page, code in the editor, terminal command being typed, dialog text — the semantic detail the AX tree misses.

Triggered selectively (every Nth Tier-1 cycle, or when Tier 1 says "Slack" and we want to know which channel), not constantly.

### Tier 3 — Tiny LLM as Director (escalation only)

For genuinely ambiguous cases — *"OCR says 'study guide' but is this THEIR study guide?"* — a tiny on-device LLM gets the structured Tier-1 + Tier-2 output and decides whether to speak. **Most of its calls won't need a model verdict** — Tier 1 + Tier 2 + simple rules cover ~90 % of decisions deterministically. The LLM is for the long tail.

The LLM's other job is writing the line itself. Even when Tiers 1 + 2 already decided "speak now about Reddit", a small LLM in the personality's voice produces a better line than a template lookup: *"aiya. Reddit? You said you were debugging auth."*

---

## The tiny LLM: which model

Researched against the current MLX ecosystem (mid-2026). For a 1-sentence in-character generation task, anything ≥1 B follows persona reliably; below that, voice slips.

| Model | Disk (4-bit) | Peak RAM | Verdict |
|---|---|---|---|
| `mlx-community/Qwen3-1.7B-4bit` | ~970 MB | ~1.5 GB | **Primary pick.** Best persona adherence at sub-1 GB. Multilingual covers the "aiya" register naturally. Sub-second per sentence on M-series. |
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | ~695 MB | ~1.1 GB | **Go-smaller pick.** Very style-stable on short outputs. Safer for 8 GB Air. |
| `mlx-community/Qwen3-0.6B-4bit` | ~380 MB | ~700 MB | Decent. Persona drifts under stress (long prompt + many examples). Worth probing. |
| `mlx-community/Qwen2.5-0.5B-Instruct-4bit` | ~280 MB | ~500 MB | Borderline — follows simple templates, but persona slips often enough to feel off-character. Only viable with very strict prompting and few-shot examples. |
| `mlx-community/SmolLM2-360M-Instruct-4bit` | ~210 MB | ~400 MB | Same caveat as above; voice slips frequently. Probably too small. |

Sub-second latency on all of them for a one-sentence output. The MLX model also stays loaded across calls (same pattern as today's `MlxVlmClient`), so there's no per-call load cost.

### Apple Foundation Models — tier 0 if available

macOS 26+ ships a ~3 B on-device LLM as part of Apple Intelligence, exposed via [`apple-fm-sdk`](https://github.com/apple/python-apple-fm-sdk). Per-process RAM cost is **near zero** because the model is already resident in OS memory. Hard requirements: macOS 26.0+, Apple Intelligence enabled (English, M1+, AI not disabled in Settings).

**If the user qualifies**, this is the right Director path — it's the most resource-efficient option that exists. Otherwise fall through to the bundled MLX model. The detection logic is one feature probe at startup; the rest of the code path is identical.

---

## Voice: macOS `say`

`say` (or `AVSpeechSynthesizer` directly via PyObjC) costs near zero RAM with sub-100 ms latency. The new Sequoia neural voices (Ava, Tessa, the downloadable Premium tier) are materially better than the 2010-era voices people remember.

**The catch is character-coupling.** Uncle Fu's accent IS the character — no system voice does "overprotective Chinese uncle". So Lite ships **different personalities**, picked because they ride well on the available system voices:

- *"Dry British Dad"* on `Daniel` or `Oliver`
- *"Tech Bro"* on `Aaron` or `Tom`
- *"British Mom"* on `Tessa` or `Karen`
- *"Concerned Friend"* on `Samantha` or `Ava`

The Qwen-voiced original Uncle Fu remains the premium experience (and / or a future premium personality pack); Lite picks characters that work with what `say` already has.

---

## Webcam

Same Apple Vision framework: `VNDetectFaceRectanglesRequest` for presence, `VNDetectFaceLandmarksRequest` for posture / eye-distance estimation. All Neural-Engine accelerated, ~20–50 ms per frame, no model download required. Lower fidelity than a VLM reading the frame (no "you look tired" judgement), but enough for the wellness signals that fire deterministically anyway (face missing > N minutes, eyes too close to camera, sustained slouch).

---

## End-to-end resource profile

| Metric | Current | Lite |
|---|---|---|
| Models on disk | ~7 GB | ~700 MB (Qwen3-1.7B) or 0 (Apple FM) |
| Steady-state RAM | ~7 GB | ~1 GB (Qwen3-1.7B) or ~100 MB (Apple FM) |
| Reaction latency | ~20 s (Director tick) | sub-second (Tier-1 events fire as they happen) |
| GPU duty | Continuous | Bursty, Neural Engine only |
| Battery | Material drain on a 14" MacBook Pro | Imperceptible on an Air |
| Fan | Spins | Doesn't |
