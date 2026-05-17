"""Generate the same English phrase across all 9 CustomVoice presets for A/B listening.

Loads `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit` once and synthesises
each voice. Writes WAVs to /tmp/sc_qwen_voices/voice_<Speaker>.wav.

Run:
    uv run python tests/smoke/test_qwen_voices.py

First run downloads ~2 GB. Subsequent runs hit the HF cache.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model


MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"

# (speaker name, one-line description) — pulled from the model card
SPEAKERS: list[tuple[str, str]] = [
    ("Ryan",      "Dynamic English male, rhythm-forward (en native)"),
    ("Aiden",     "Sunny American male (en native)"),
    ("Vivian",    "Bright young female (zh native, will sound accented in English)"),
    ("Serena",    "Warm gentle young female (zh native)"),
    ("Uncle_Fu",  "Seasoned mellow male (zh native)"),
    ("Dylan",     "Youthful Beijing male (zh native)"),
    ("Eric",      "Lively Chengdu male (zh-Sichuan native)"),
    ("Ono_Anna",  "Playful Japanese female (ja native)"),
    ("Sohee",     "Warm Korean female (ko native)"),
]

PHRASE = (
    "Hey. You've been heads-down for forty minutes. "
    "Quick stretch and a sip of water — back in five."
)

OUT = Path("/tmp/sc_qwen_voices")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Wipe stale files from previous runs so the folder is just this batch.
    for p in OUT.glob("voice_*.wav"):
        p.unlink()

    print(f"Downloading / locating {MODEL_ID} …")
    t0 = time.time()
    model_dir = Path(snapshot_download(repo_id=MODEL_ID))
    print(f"  model dir: {model_dir}  ({int(time.time()-t0)}s)")

    print("Loading model into MLX …")
    t0 = time.time()
    model = load_model(model_dir)
    print(f"  loaded in {time.time()-t0:.1f}s")

    summary: list[tuple[str, float, int]] = []  # (speaker, gen_seconds, kb)
    for speaker, desc in SPEAKERS:
        print(f"\n→ {speaker:<10} — {desc}")
        t0 = time.time()
        generate_audio(
            text=PHRASE,
            model=model,
            voice=speaker,
            lang_code="en",
            output_path=str(OUT),
            file_prefix=f"voice_{speaker}",
            audio_format="wav",
            save=True,
            verbose=False,
            play=False,
        )
        elapsed = time.time() - t0
        # generate_audio writes voice_<name>_000.wav by default
        candidates = sorted(OUT.glob(f"voice_{speaker}*.wav"))
        if not candidates:
            print(f"   ⚠ no file produced for {speaker}")
            continue
        produced = candidates[0]
        # Normalise filename so it's easy to scan in Finder
        final = OUT / f"voice_{speaker}.wav"
        if produced != final:
            if final.exists():
                final.unlink()
            shutil.move(str(produced), str(final))
        kb = final.stat().st_size // 1024
        summary.append((speaker, elapsed, kb))
        print(f"   {elapsed:.1f}s, {kb} KB → {final.name}")

    print("\n────  Summary  ─────────────────────────")
    print(f"{'Voice':<10}  {'Time':>6}  {'Size':>7}")
    for speaker, elapsed, kb in summary:
        print(f"{speaker:<10}  {elapsed:>5.1f}s  {kb:>5} KB")

    print(f"\nFiles written to: {OUT}")
    # Open the folder in Finder.
    subprocess.run(["open", str(OUT)], check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
