# Voice design plan

How we'll go from "Uncle_Fu only" to "per-personality bespoke voices, all baked into the repo".

---

## Current state

- TTS: Qwen3-TTS via mlx-audio, model `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit`.
- One personality (`uncle_fu`) using the `Uncle_Fu` preset speaker.
- Qwen3-TTS has three variants, only the first two are on mlx-community right now:
  - **CustomVoice** — 9 fixed preset speakers (what we use today). Easy. No baking.
  - **Base** — voice cloning from a 3-second reference clip + matching text. Outputs speaker embeddings we can save and reuse.
  - **VoiceDesign** — natural-language voice description ("warm British male, 40, mellow") → speaker embeddings. Not on mlx-community yet; only HuggingFace transformers with Flash-Attention.

The artifact in both Base and VoiceDesign cases is the same shape: a small tensor of **speaker embeddings**. That's the unit we ship per personality.

---

## Where we're going

```
src/unclefu/assets/voices/
├── british_mom.npy    # ~50–200 KB tensor of speaker embedding
├── chaos_goblin.npy
├── tech_bro.npy
└── (audio_refs/        ← optional, only if we want re-baking later)
```

At runtime, `QwenSpeaker.say()` loads the embedding file matching the active personality, passes it to Qwen3-TTS as a `voice_clone_prompt` (or equivalent argument once we wire the Base variant), generates audio, plays via `afplay`. Same code path as today; just a different "voice" source than the CustomVoice preset.

No network needed at run time. No reference audio files in the running app. Just the embeddings.

---

## How we bake a new voice

We'll write **one** tooling script: `tools/bake_voice.py` (not a runtime path; lives outside `src/unclefu/`). It supports two modes.

### Mode A — voice clone (works today, once the Base variant lands on mlx-community)

```
uv run python tools/bake_voice.py clone \
    --name british_mom \
    --reference path/to/3s_clip.wav \
    --reference-text "Hello love, the kettle's just boiled." \
    --out src/unclefu/assets/voices/british_mom.npy
```

Internals:
1. Load `Qwen3-TTS-12Hz-1.7B-Base` (any quant; baking is offline, slow is fine).
2. Call the Base model's `extract_speaker_embedding(audio, text)` (or equivalent — name TBC from the qwen-tts Python API).
3. Save the resulting tensor as `.npy`.

Requirements for the reference clip:
- 3–10 seconds, clean (no music, no overlap, no fan noise).
- Matches the gender / age / language we want.
- Reference text must be exactly what's spoken in the clip.

Where reference audio comes from: any creative-commons-licensed reading, a friend who consents, our own recordings. Audio files don't need to live in the repo; only the embedding does.

### Mode B — voice design (waits on mlx-community VoiceDesign port)

```
uv run python tools/bake_voice.py design \
    --name british_mom \
    --description "Warm British female, 40s, mellow, slightly maternal, mid-tempo." \
    --out src/unclefu/assets/voices/british_mom.npy
```

Internals:
1. Load `Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
2. Pass the description; model emits a speaker embedding.
3. Save as `.npy`.

Mode B is the prize — fully synthetic, no reference recording needed, fully reproducible from a prompt. We just need the mlx-audio team (or someone) to port the VoiceDesign architecture, **or** we stand up a one-off HuggingFace transformers run-once script on a machine with enough GPU/CPU and just bake the embeddings outside MLX. Either way, the result is the same `.npy` we'd ship.

---

## Loading + runtime

`QwenSpeaker` grows a tiny extension: if `voice` looks like a path to a `.npy` file (or matches a key in a `voices/` registry), load the embedding and pass to `generate_audio` as `voice_clone_prompt` (the kwarg `qwen-tts` exposes for "use this baked voice"). If it doesn't, fall back to treating `voice` as a CustomVoice preset name — same as today.

That keeps the existing code path untouched and lets us mix-and-match: `Uncle_Fu` preset for the `uncle_fu` personality, baked embedding for `british_mom`, both live in the same `PERSONALITIES` dict.

Per-personality wiring:

```python
PERSONALITIES["british_mom"] = Personality(
    key="british_mom",
    display_name="British Mom",
    voice="assets/voices/british_mom.npy",   # baked embedding
    prompt_fragment="…",
)
```

---

## Sequence

1. Watch mlx-community for either the **Base** quant (voice clone) or **VoiceDesign** quant. Whichever lands first wins.
2. Once **Base** is available: pick one or two reference clips → bake → commit `.npy` → add personality entries → wire the personality switcher in the menu bar.
3. Once **VoiceDesign** is available: extend `tools/bake_voice.py` with the `design` subcommand; iterate on prompts until each personality has a distinctive voice; commit those `.npy` files.
4. Long term: if we end up with many personalities, consider a tiny "voice market" CSV mapping persona → reference clip → description, and a CI step that re-bakes on description change.

---

## What we DO NOT want

- Reference audio files in the repo. They're large, license-fraught, and unnecessary at run time.
- Per-personality re-loads at run time. The Qwen model is loaded once; the speaker embedding is a tensor we attach per call.
- Cloud-anything. The whole TTS layer stays local — that's the privacy promise.
