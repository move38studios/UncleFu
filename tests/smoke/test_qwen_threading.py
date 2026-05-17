"""Isolate the MLX 'no Stream(gpu, 0) in current thread' error.

Runs Qwen3-TTS synthesis on a worker thread (the same place QwenSpeaker
calls it from in production) with three different stream-context attempts.
Whichever one succeeds is what QwenSpeaker should use.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model


MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
TEXT = "Eyes too close. Sit back."
OUT = Path("/tmp/sc_qwen_threading")


def _run_synth(label: str, fn) -> None:
    print(f"\n=== {label} ===")
    try:
        fn()
        print(f"   ✅ {label} succeeded")
    except Exception as e:
        print(f"   ❌ {label} FAILED: {type(e).__name__}: {e}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Downloading / locating model …")
    model_dir = Path(snapshot_download(repo_id=MODEL_ID))
    print(f"  {model_dir}")
    print("Loading model on main thread …")
    t0 = time.time()
    model = load_model(model_dir)
    print(f"  loaded in {time.time()-t0:.1f}s")

    # --- baseline: synth on the main thread (should always work) ---

    _run_synth("main thread, no context", lambda: generate_audio(
        text=TEXT, model=model, voice="Uncle_Fu", lang_code="en",
        output_path=str(OUT), file_prefix="main", audio_format="wav",
        save=True, verbose=False, play=False,
    ))

    # --- on a worker thread, four variants ---

    def synth_no_context():
        generate_audio(
            text=TEXT, model=model, voice="Uncle_Fu", lang_code="en",
            output_path=str(OUT), file_prefix="worker_nocontext",
            audio_format="wav", save=True, verbose=False, play=False,
        )

    def synth_with_mx_gpu():
        with mx.stream(mx.gpu):  # type: ignore[arg-type]
            generate_audio(
                text=TEXT, model=model, voice="Uncle_Fu", lang_code="en",
                output_path=str(OUT), file_prefix="worker_mx_gpu",
                audio_format="wav", save=True, verbose=False, play=False,
            )

    def synth_with_default_stream():
        with mx.stream(mx.default_stream(mx.gpu)):  # type: ignore[arg-type]
            generate_audio(
                text=TEXT, model=model, voice="Uncle_Fu", lang_code="en",
                output_path=str(OUT), file_prefix="worker_default_stream",
                audio_format="wav", save=True, verbose=False, play=False,
            )

    def synth_with_set_default_device():
        mx.set_default_device(mx.gpu)  # type: ignore[arg-type]
        generate_audio(
            text=TEXT, model=model, voice="Uncle_Fu", lang_code="en",
            output_path=str(OUT), file_prefix="worker_set_default",
            audio_format="wav", save=True, verbose=False, play=False,
        )

    def synth_on_cpu():
        with mx.stream(mx.cpu):  # type: ignore[arg-type]
            generate_audio(
                text=TEXT, model=model, voice="Uncle_Fu", lang_code="en",
                output_path=str(OUT), file_prefix="worker_cpu",
                audio_format="wav", save=True, verbose=False, play=False,
            )

    for label, fn in [
        ("worker thread, no context", synth_no_context),
        ("worker thread, with mx.stream(mx.gpu)", synth_with_mx_gpu),
        ("worker thread, with mx.stream(mx.default_stream(mx.gpu))",
         synth_with_default_stream),
        ("worker thread, after mx.set_default_device(mx.gpu)",
         synth_with_set_default_device),
        ("worker thread, with mx.stream(mx.cpu)", synth_on_cpu),
    ]:
        t = threading.Thread(target=lambda f=fn, l=label: _run_synth(l, f))
        t.start()
        t.join()

    print(f"\nFiles written to: {OUT}")
    print("Check which 'worker_*.wav' files exist — those are the variants that succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
