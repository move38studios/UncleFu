"""Per-session human-readable debug log.

Layout: one text file per session at the configured path. Header at the top
(personality, model, interval settings), then sections appended as things
happen:
- SNAPSHOT <source>  — one per sensor tick (or error)
- DIRECTOR           — one per Director tick (or error)

Concurrent writes are protected by an internal lock; the runner has several
threads.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TextIO

from ..director.director import DirectorCall
from ..personalities import Personality
from ..vlm.schema import SensorSnapshot


_BAR = "─" * 78


@dataclass
class DebugLog:
    path: Path
    _fh: TextIO
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        personality: Personality,
        model: str,
        webcam_interval_s: float,
        screen_interval_s: float,
        director_interval_s: float,
    ) -> "DebugLog":
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = path.open("w", encoding="utf-8")
        fh.write("Uncle Fu debug log\n")
        fh.write(f"started:     {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(
            f"personality: {personality.display_name} ({personality.key}) "
            f"voice={personality.voice}\n"
        )
        fh.write(f"model:       {model}\n")
        fh.write(
            f"intervals:   webcam={webcam_interval_s}s  "
            f"screens={screen_interval_s}s  director={director_interval_s}s\n"
        )
        fh.flush()
        return cls(path=path, _fh=fh)

    def write_snapshot(self, snap: SensorSnapshot) -> None:
        when = datetime.fromtimestamp(snap.ts).isoformat(timespec="seconds")
        with self._lock:
            self._fh.write(f"\n{_BAR}\n")
            self._fh.write(f"SNAPSHOT  {snap.source}  @ {when}  ({snap.cycle_ms}ms)\n")
            if snap.is_error:
                self._fh.write(
                    f"  ERROR {snap.structured.get('exc_type')}: "
                    f"{snap.structured.get('exc_message')}\n"
                )
            else:
                self._fh.write(f"  description: {snap.description}\n")
                self._fh.write(
                    f"  structured:  "
                    f"{json.dumps(snap.structured, ensure_ascii=False)}\n"
                )
            self._fh.flush()

    def write_director_call(
        self,
        *,
        ts: float,
        call: DirectorCall,
        outcome: str,
    ) -> None:
        when = datetime.fromtimestamp(ts).isoformat(timespec="seconds")
        d = call.decision
        with self._lock:
            self._fh.write(f"\n{_BAR}\n")
            self._fh.write(f"DIRECTOR  @ {when}  ({call.call_ms}ms)\n")
            self._fh.write(
                f"  urgency={d.urgency} expression={d.expression}  "
                f"outcome={outcome}\n"
            )
            self._fh.write("\n  --- user message ---\n  ")
            self._fh.write(call.user_text.replace("\n", "\n  "))
            self._fh.write("\n\n  --- raw response ---\n  ")
            self._fh.write(call.raw_response.replace("\n", "\n  "))
            self._fh.write("\n\n  --- decision ---\n  ")
            self._fh.write(
                json.dumps(d.model_dump(), indent=2, ensure_ascii=False)
                .replace("\n", "\n  ")
            )
            self._fh.write("\n")
            self._fh.flush()

    def write_director_error(self, *, ts: float, exc: BaseException) -> None:
        when = datetime.fromtimestamp(ts).isoformat(timespec="seconds")
        with self._lock:
            self._fh.write(f"\n{_BAR}\n")
            self._fh.write(f"DIRECTOR  @ {when}  ERROR {type(exc).__name__}: {exc}\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.write(f"\n{_BAR}\nclosed: {datetime.now().isoformat(timespec='seconds')}\n")
            finally:
                self._fh.close()
