"""Text-to-speech: Qwen3-TTS via mlx-audio. Output played through `afplay`.

We do NOT use macOS `say` anymore. Qwen3-TTS sounds dramatically better and
supports voice cloning + voice design (see docs/voice_design.md).

Architecture: ONE dedicated worker thread owns the Qwen model and runs all
synthesis. MLX state (streams, devices) is thread-local; loading on one
thread and synthesising on another raises `RuntimeError: There is no
Stream(gpu, 0) in current thread.` Keeping load + synth on the same thread
sidesteps that entirely.

`say()` posts a (text, voice) item to a queue, returns immediately. The
worker drains the queue, synthesises, and spawns an `afplay` subprocess
to play. `say()` calls supersede in-flight playback (latest-wins). `stop()`
kills any current playback and flushes the queue.
"""

from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Protocol


DEFAULT_QWEN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
DEFAULT_VOICE = "Uncle_Fu"

# Repetition penalty passed to Qwen3-TTS. The model's default is 1.05; we
# bump it to 1.3 to discourage "code code coden coden" tail-babble that
# the 0.6B variant produced when the generation cap was raised. 1.3 is
# the same range mlx-audio internally uses for ICL (voice-clone) calls.
# See docs/learnings.md for the EOS / cap saga.
_REPETITION_PENALTY = 1.3

# Sentinel pushed onto the queue to ask the worker to exit.
_SHUTDOWN = object()


def _wav_duration_seconds(path: Path) -> float:
    """Read the WAV header and return its duration in seconds. 0 on error.

    Used only for diagnostics — we want to know how long we *expected*
    afplay to run, so a short afplay elapsed time can be classified as
    'killed early' vs 'audio file was already short'.
    """
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            return frames / rate if rate > 0 else 0.0
    except Exception:
        return 0.0


class Speaker(Protocol):
    """A thing that can speak a string aloud. May be a no-op."""

    def say(self, text: str, *, voice: str | None = None) -> None: ...
    def stop(self) -> None: ...
    def is_speaking(self) -> bool: ...


class NullSpeaker:
    """No-op speaker. Used with --mute and in tests."""

    def say(self, text: str, *, voice: str | None = None) -> None:
        return

    def stop(self) -> None:
        return

    def is_speaking(self) -> bool:
        return False


class MutableSpeaker:
    """Wraps a Speaker with a runtime `muted` flag.

    Lets the menu bar toggle mute without restarting the runner.
    """

    def __init__(self, inner: Speaker, *, muted: bool = False) -> None:
        self._inner = inner
        self.muted = muted

    def say(self, text: str, *, voice: str | None = None) -> None:
        if self.muted:
            return
        self._inner.say(text, voice=voice)

    def stop(self) -> None:
        self._inner.stop()

    def is_speaking(self) -> bool:
        return self._inner.is_speaking()


class QwenSpeaker:
    """Qwen3-TTS via mlx-audio on a single dedicated worker thread."""

    def __init__(self, *, model_id: str = DEFAULT_QWEN_MODEL) -> None:
        if shutil.which("afplay") is None:
            raise RuntimeError("`afplay` not on PATH — this only runs on macOS")
        self._model_id = model_id
        self._tmpdir = Path(tempfile.mkdtemp(prefix="sc-qwen-"))
        self._queue: queue.Queue = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._stopping = threading.Event()
        self._model_ready = threading.Event()
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="qwen-worker",
        )
        self._worker.start()

    # ---- public ----

    def say(self, text: str, *, voice: str | None = None) -> None:
        if not text.strip():
            return
        if self._stopping.is_set():
            return
        # Latest-wins semantics: kill any in-flight playback and drop queued
        # work — we only want to speak the most recent line. The Intervener
        # already enforces throttling above us, but if something does sneak
        # past, we still don't want lines to pile up.
        with self._proc_lock:
            self._kill_proc_locked(reason="say()")
        self._drain_queue()
        self._queue.put((text, voice or DEFAULT_VOICE))

    def stop(self) -> None:
        with self._proc_lock:
            self._kill_proc_locked(reason="stop()")
        self._drain_queue()

    def shutdown(self) -> None:
        """Stop playback, signal the worker to exit, and join."""
        self._stopping.set()
        self.stop()
        self._queue.put(_SHUTDOWN)
        self._worker.join(timeout=2.0)

    def is_speaking(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def is_ready(self) -> bool:
        return self._model_ready.is_set()

    # ---- worker ----

    def _run(self) -> None:
        try:
            from huggingface_hub import snapshot_download
            from mlx_audio.tts.utils import load_model
            from ..vlm.download_progress import (
                KNOWN_MODEL_SIZES_BYTES, PROGRESS, hf_cache_path_for,
            )
            PROGRESS.register(
                key="tts",
                label="Qwen3 TTS",
                cache_dir=hf_cache_path_for(self._model_id),
                target_bytes=KNOWN_MODEL_SIZES_BYTES.get(
                    self._model_id, 2_100_000_000,
                ),
            )
            PROGRESS.mark_started("tts")
            path = Path(snapshot_download(repo_id=self._model_id))
            PROGRESS.mark_done("tts")
            model = load_model(path)
        except BaseException as e:
            print(
                f"[QwenSpeaker] model load failed: {type(e).__name__}: {e}",
                file=sys.stderr, flush=True,
            )
            self._model_ready.set()
            return
        self._model_ready.set()

        from mlx_audio.tts.generate import generate_audio

        while not self._stopping.is_set():
            item = self._queue.get()
            if item is _SHUTDOWN:
                break
            text, voice = item

            for p in self._tmpdir.glob("out*.wav"):
                try:
                    p.unlink()
                except OSError:
                    pass

            try:
                # join_audio=True asks mlx-audio to concatenate any segment
                # outputs into a single file. Without it, longer lines end
                # up split across out_000.wav, out_001.wav, ... and we'd
                # only ever play the first segment.
                #
                # repetition_penalty kwarg flows through `**kwargs` to
                # Qwen3TTS.generate(...) (signature default is 1.05; we
                # raise it to discourage the tail-babble described in
                # docs/learnings.md).
                generate_audio(
                    text=text,
                    model=model,
                    voice=voice,
                    lang_code="en",
                    output_path=str(self._tmpdir),
                    file_prefix="out",
                    audio_format="wav",
                    save=True,
                    verbose=False,
                    play=False,
                    join_audio=True,
                    repetition_penalty=_REPETITION_PENALTY,
                )
            except Exception as e:
                print(
                    f"[QwenSpeaker] synth failed: {type(e).__name__}: {e}",
                    file=sys.stderr, flush=True,
                )
                continue

            wavs = sorted(self._tmpdir.glob("out*.wav"))
            if not wavs:
                continue
            if len(wavs) > 1:
                print(
                    f"[QwenSpeaker] {len(wavs)} segments produced; "
                    f"playing only the joined file (others: {[w.name for w in wavs[1:]]})",
                    file=sys.stderr, flush=True,
                )
            # When join_audio=True, mlx-audio writes a single joined file.
            # We prefer the joined one if both exist.
            joined = [w for w in wavs if "joined" in w.name.lower()]
            to_play = joined[0] if joined else wavs[0]
            expected_s = _wav_duration_seconds(to_play)
            with self._proc_lock:
                if self._stopping.is_set():
                    return
                self._kill_proc_locked(reason="worker:before-spawn")
                self._proc = subprocess.Popen(
                    ["afplay", str(to_play)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._diag(f"spawned afplay pid={self._proc.pid} "
                           f"file={to_play.name} expected={expected_s:.2f}s "
                           f"text={text!r}")
                # Daemon monitor: wait for afplay to exit and log how long
                # it actually ran + its exit code. Lets us distinguish a
                # killed afplay (non-zero code, short elapsed) from a
                # natural finish (code 0, elapsed ≈ expected) from a
                # truncated audio file (code 0, elapsed < expected).
                threading.Thread(
                    target=self._watch_proc,
                    args=(self._proc, expected_s),
                    name="afplay-watch",
                    daemon=True,
                ).start()

    # ---- helpers ----

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _kill_proc_locked(self, *, reason: str = "?") -> None:
        if self._proc is None:
            return
        # Only log when we actually interrupt a live process — kills against
        # an already-finished afplay are no-ops and would just be noise.
        if self._proc.poll() is None:
            self._diag(f"KILL afplay pid={self._proc.pid} reason={reason}")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def _diag(self, msg: str) -> None:
        """Stderr breadcrumb for TTS lifecycle, tagged with a monotonic
        timestamp so spawn/kill pairs are easy to correlate."""
        print(f"[QwenSpeaker {time.monotonic():9.3f}] {msg}",
              file=sys.stderr, flush=True)

    def _watch_proc(self, proc: subprocess.Popen, expected_s: float) -> None:
        """Block until `proc` exits, then log how long it actually ran.

        Runs on a short-lived daemon thread, one per afplay invocation.
        Lets the diag log distinguish 'afplay was killed' from 'afplay
        finished naturally but audio was shorter than the text suggested'.
        """
        t0 = time.monotonic()
        try:
            proc.wait()
        except Exception as e:
            self._diag(f"watch: wait() failed pid={proc.pid}: {e}")
            return
        elapsed = time.monotonic() - t0
        gap = expected_s - elapsed
        flag = ""
        if proc.returncode != 0:
            flag = "  ⚠ non-zero exit"
        elif expected_s > 0 and gap > 0.3:
            flag = f"  ⚠ ended {gap:.2f}s early (audio shorter than expected)"
        self._diag(
            f"afplay pid={proc.pid} exited code={proc.returncode} "
            f"elapsed={elapsed:.2f}s expected={expected_s:.2f}s{flag}"
        )
