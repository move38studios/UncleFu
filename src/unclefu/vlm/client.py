"""In-process VLM client for Gemma 4 E4B via `mlx-vlm`.

~2.3 s warm per single-image call on M-series with `gemma-4-e4b-it-4bit`
(~100-token JSON output). No HTTP, no separate server — the only network
call is the first-launch `huggingface_hub.snapshot_download` of weights
to `~/.cache/huggingface/`. See [docs/decisions.md](../../docs/decisions.md).

Thread-affinity rule (from `docs/learnings.md` — MLX streams are
thread-local): a single worker thread owns the loaded model and is the
only thread that calls `generate()`. Callers (sensors, Director) on
arbitrary threads submit `_Request` objects to a queue and block on a
per-call response queue. Cheap synchronization, zero risk of GPU stream
misuse, naturally serializes VLM calls (which is what you want — only
one mx.gpu generation at a time anyway).
"""

from __future__ import annotations

import io
import json
import queue
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


DEFAULT_MODEL = "mlx-community/gemma-4-e4b-it-4bit"


class VLMError(RuntimeError):
    """Anything that went wrong calling or parsing the VLM."""


@dataclass
class ChatResult:
    text: str                       # the message.content the model returned
    user_text: str                  # whatever text content we sent (for debug logs)
    prompt_tokens: int | None       # always None for in-process mlx-vlm (kept for log shape compat)
    completion_tokens: int | None   # always None for in-process mlx-vlm


@dataclass
class _Request:
    system: str
    user_text: str
    images_jpeg: list[bytes]
    max_tokens: int
    temperature: float
    response_q: "queue.Queue[ChatResult | Exception]" = field(
        default_factory=lambda: queue.Queue(maxsize=1)
    )


_SHUTDOWN = object()  # sentinel pushed onto _req_q to ask the worker to exit


class MlxVlmClient:
    """Thin wrapper around mlx-vlm with the producer/consumer contract sensors expect.

    Construction is non-blocking — the worker spawns immediately and starts
    downloading + loading the model in the background. Use `wait_ready()`
    or `is_ready` if you need to know when the first call won't be slow,
    or just call `chat()` and let it block on the response.
    """

    def __init__(self, *, model_id: str = DEFAULT_MODEL) -> None:
        self._model_id = model_id
        self._req_q: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._failed = threading.Event()
        self._fail_reason: str = ""
        self._stopping = threading.Event()
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="vlm-worker",
        )
        self._worker.start()

    # ---- public API ----

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and not self._failed.is_set()

    def wait_ready(self, timeout_s: float | None = None) -> bool:
        """Block until the worker has loaded the model (or failed).
        Returns True if ready, False if failed or timed out."""
        if not self._ready.wait(timeout=timeout_s):
            return False
        return not self._failed.is_set()

    def chat(
        self,
        *,
        system: str,
        user_text: str,
        images_jpeg: Sequence[bytes] = (),
        max_tokens: int = 500,
        temperature: float = 0.1,
        timeout_s: float = 180.0,
    ) -> ChatResult:
        """Submit one chat request and block until the response comes back.

        Safe to call from any thread. The worker serializes all calls.
        Timeout includes model load wait on the first call — first sensor
        cycle after launch will block until Gemma is in memory.
        """
        if self._failed.is_set():
            raise VLMError(f"VLM worker failed at load: {self._fail_reason}")
        if self._stopping.is_set():
            raise VLMError("VLM client is shutting down")
        req = _Request(
            system=system,
            user_text=user_text,
            images_jpeg=list(images_jpeg),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._req_q.put(req)
        try:
            result = req.response_q.get(timeout=timeout_s)
        except queue.Empty as e:
            raise VLMError(f"VLM call timed out after {timeout_s}s") from e
        if isinstance(result, Exception):
            raise result
        return result

    def shutdown(self) -> None:
        """Signal the worker to stop and wait briefly for it to drain."""
        self._stopping.set()
        self._req_q.put(_SHUTDOWN)
        self._worker.join(timeout=2.0)

    # ---- worker ----

    def _run(self) -> None:
        # Imports deferred to the worker thread — failing here is reported
        # cleanly via _failed instead of crashing the whole app on import.
        try:
            from huggingface_hub import snapshot_download
            from mlx_vlm import generate, load  # type: ignore[import-not-found]
            from mlx_vlm.prompt_utils import apply_chat_template  # type: ignore[import-not-found]
            from PIL import Image as PILImage
        except ImportError as e:
            self._fail_reason = f"missing dep: {e}"
            print(f"[MlxVlmClient] {self._fail_reason}",
                  file=sys.stderr, flush=True)
            self._failed.set()
            self._ready.set()
            return

        try:
            from .download_progress import (
                KNOWN_MODEL_SIZES_BYTES, PROGRESS, hf_cache_path_for,
            )
            PROGRESS.register(
                key="vlm",
                label="Gemma 4 VLM",
                cache_dir=hf_cache_path_for(self._model_id),
                target_bytes=KNOWN_MODEL_SIZES_BYTES.get(
                    self._model_id, 5_500_000_000,
                ),
            )
            PROGRESS.mark_started("vlm")
            t0 = time.time()
            path = snapshot_download(repo_id=self._model_id)
            dl_s = time.time() - t0
            PROGRESS.mark_done("vlm")
            t0 = time.time()
            model, processor = load(path)
            load_s = time.time() - t0
            print(
                f"[MlxVlmClient] model ready: download={dl_s:.1f}s "
                f"load={load_s:.1f}s id={self._model_id}",
                file=sys.stderr, flush=True,
            )
        except BaseException as e:
            self._fail_reason = f"{type(e).__name__}: {e}"
            print(f"[MlxVlmClient] model load failed: {self._fail_reason}",
                  file=sys.stderr, flush=True)
            self._failed.set()
            self._ready.set()
            return

        self._ready.set()

        # Cache the model config — apply_chat_template needs it on every call.
        config = model.config

        while not self._stopping.is_set():
            item = self._req_q.get()
            if item is _SHUTDOWN:
                break
            req: _Request = item  # type: ignore[assignment]
            try:
                images: list[Any] = [
                    PILImage.open(io.BytesIO(jpeg)).convert("RGB")
                    for jpeg in req.images_jpeg
                ]
                messages = [
                    {"role": "system", "content": req.system},
                    {"role": "user", "content": req.user_text},
                ]
                formatted: str = apply_chat_template(  # type: ignore[arg-type,assignment]
                    processor, config, messages, num_images=len(images),
                )
                # mlx-vlm `image=` is typed as `str | List[str]` but accepts
                # PIL.Image objects too at runtime. Empty list = text-only.
                output = generate(
                    model,
                    processor,  # type: ignore[arg-type]
                    formatted,
                    image=images,  # type: ignore[arg-type]
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    verbose=False,
                )
                # mlx-vlm 0.5.x returns a plain str; defensive fallback for
                # versions that returned an object with .text.
                text = output if isinstance(output, str) else getattr(
                    output, "text", str(output)
                )
                req.response_q.put(ChatResult(
                    text=text,
                    user_text=req.user_text,
                    prompt_tokens=None,
                    completion_tokens=None,
                ))
            except Exception as e:
                req.response_q.put(VLMError(
                    f"VLM call failed: {type(e).__name__}: {e}"
                ))


def parse_json_response(text: str) -> dict:
    """Strip markdown fences if present and json.loads. Raises VLMError on failure."""
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.endswith("```"):
            t = t[:-3]
    try:
        return json.loads(t.strip())
    except json.JSONDecodeError as e:
        raise VLMError(f"non-JSON response: {text[:300]!r}") from e
