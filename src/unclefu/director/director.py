"""The Director — text-only LLM call deciding whether to speak.

Inputs: user-stated focus, recent snapshots per source (text only),
recent speech log, real-world context. Output: validated Decision
(incl. expression). No images — Gemma 4 handles text-only via the
same MlxVlmClient we use for sensors, just with `images_jpeg=()`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..personalities import Personality
from ..vlm.client import MlxVlmClient, parse_json_response
from ..vlm.schema import Decision, SensorSnapshot
from .context import RealWorldContext, format_for_prompt as format_realworld
from .prompts import (
    build_director_system_prompt,
    format_recent_snapshots_block,
    format_recent_speech_block,
)


@dataclass
class DirectorCall:
    decision: Decision
    user_text: str
    raw_response: str
    call_ms: int


def _snapshots_block_for(
    label: str, snaps: list[SensorSnapshot], now: float
) -> str:
    # Pass the sensor's self-reported confidence through to the Director
    # so it can gate on it: low-confidence observations shouldn't trigger
    # drift calls (Gemma wasn't sure what it was looking at).
    items = [(now - s.ts, s.confidence, s.description) for s in snaps]
    return format_recent_snapshots_block(label, items)


def call_director(
    *,
    personality: Personality,
    vlm: MlxVlmClient,
    focus: str,
    session_seconds: float,
    recent_by_source: dict[str, list[SensorSnapshot]],
    recent_speech: list[tuple[float, str]],
    realworld: RealWorldContext,
    now: float | None = None,
) -> DirectorCall:
    if now is None:
        now = time.time()

    blocks: list[str] = [
        f"USER'S STATED FOCUS for this session:\n  \"{focus}\"",
        format_realworld(realworld),
    ]

    # Webcam first, then screens in stable order.
    if "webcam" in recent_by_source:
        blocks.append(_snapshots_block_for("Webcam", recent_by_source["webcam"], now))
    for source in sorted(k for k in recent_by_source if k.startswith("screen_")):
        blocks.append(
            _snapshots_block_for(source.replace("_", " ").title(), recent_by_source[source], now)
        )

    blocks.append(format_recent_speech_block(recent_speech))
    blocks.append(f"Session length so far: {int(session_seconds // 60)} min")
    blocks.append(
        "Decide: should you speak now? Reply with JSON only.\n"
        "Drift check is POSITIVE matching: do the observations plausibly\n"
        "support the user actually doing the stated focus right now? If not,\n"
        "speak — even if what they're doing looks productive in some other\n"
        "context. 'Useful but unrelated' is still drift. If the focus is\n"
        "off-the-computer (piano, cooking, exercise, reading paper book) and\n"
        "they're at the desk doing computer things — that is drift. Wellness\n"
        "signals (posture, late night, risky action) fire independently."
    )

    user_text = "\n\n".join(blocks)

    t0 = time.time()
    result = vlm.chat(
        system=build_director_system_prompt(personality),
        user_text=user_text,
        images_jpeg=(),                # text-only
        max_tokens=200,
        temperature=0.4,               # slightly higher — we want some character in the lines
    )
    call_ms = int((time.time() - t0) * 1000)

    parsed = parse_json_response(result.text)
    decision = Decision.model_validate(parsed)
    return DirectorCall(
        decision=decision,
        user_text=user_text,
        raw_response=result.text,
        call_ms=call_ms,
    )
