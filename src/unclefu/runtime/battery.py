"""Battery-aware policy: pause or quit when the laptop is unplugged.

Uncle Fu is GPU-heavy (continuous VLM + TTS inference on Apple
Silicon). Running on battery for any length of time drains fast —
real measurements suggest ~25-40% per hour depending on cadence.

Policy buckets (with `--no-battery-check` to override entirely):

  on AC                       → normal
  on battery & pct ≥ 30       → warn once in the menu bar; keep running
  on battery & pct < 30       → PAUSE: stop sensor + Director threads,
                                speak one farewell line, character → 💤
  on battery & pct < 15       → QUIT: log the session, exit cleanly

A monitor thread polls every ~30 s. State transitions debounce so a
flickering pmset reading doesn't flap the policy.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from ..director.context import _read_battery


# Thresholds. Conservative: pause well before the user is in trouble.
PAUSE_PCT = 30
QUIT_PCT = 15
POLL_INTERVAL_S = 30.0


class PowerPolicy(str, Enum):
    """What the BatteryMonitor wants the Runner to do."""

    NORMAL = "normal"          # on AC, or first cycle while we read
    WARN = "warn"              # on battery, pct ≥ PAUSE_PCT
    PAUSE = "pause"            # on battery, PAUSE_PCT > pct ≥ QUIT_PCT
    QUIT = "quit"              # on battery, pct < QUIT_PCT


@dataclass
class BatteryState:
    """What the monitor exposes for the menu bar / Runner to read."""

    pct: int | None = None     # 0-100 or None if no battery (desktop)
    on_ac: bool = True         # plugged in
    policy: PowerPolicy = PowerPolicy.NORMAL


@dataclass
class BatteryMonitor:
    """Background thread that polls pmset every POLL_INTERVAL_S seconds
    and drives policy transitions. The caller (Runner) wires its
    callbacks to actually pause / resume / quit."""

    on_pause: Callable[[BatteryState], None]
    on_resume: Callable[[BatteryState], None]
    on_quit: Callable[[BatteryState], None]
    on_warn: Callable[[BatteryState], None] | None = None  # one-shot notice
    enabled: bool = True
    state: BatteryState = field(default_factory=BatteryState)
    _stopping: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _warned_at_pct: int | None = None  # so we only warn once per AC→battery transition

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="battery-monitor", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        # Tick once immediately so policy is correct on first read,
        # then settle into the poll interval.
        while not self._stopping.is_set():
            self._tick()
            if self._stopping.wait(timeout=POLL_INTERVAL_S):
                return

    def _tick(self) -> None:
        if not self.enabled:
            return
        pct, on_ac = _read_battery()
        prev_policy = self.state.policy
        new_policy = self._decide(pct, on_ac)
        self.state = BatteryState(pct=pct, on_ac=on_ac, policy=new_policy)
        if new_policy == prev_policy:
            return
        # Edge transitions only:
        if new_policy == PowerPolicy.WARN:
            if self.on_warn is not None and self._warned_at_pct != (pct or 0):
                self._warned_at_pct = pct or 0
                self.on_warn(self.state)
        elif new_policy == PowerPolicy.PAUSE:
            self.on_pause(self.state)
        elif new_policy == PowerPolicy.QUIT:
            self.on_quit(self.state)
        elif new_policy == PowerPolicy.NORMAL:
            # We came back to AC. If we were paused, resume.
            if prev_policy in (PowerPolicy.PAUSE, PowerPolicy.WARN):
                self.on_resume(self.state)
                self._warned_at_pct = None

    @staticmethod
    def _decide(pct: int | None, on_ac: bool) -> PowerPolicy:
        if on_ac or pct is None:
            return PowerPolicy.NORMAL
        if pct < QUIT_PCT:
            return PowerPolicy.QUIT
        if pct < PAUSE_PCT:
            return PowerPolicy.PAUSE
        return PowerPolicy.WARN
