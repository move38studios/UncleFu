"""Real-world context for the Director: time, day, battery, input activity.

`ContextGatherer` holds state across calls because input event rates need a
baseline (we sample monotonic event counters and divide by elapsed time).
The Runner owns one instance and the Director queries it each tick.
"""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime

import Quartz  # type: ignore[import-not-found]


# CGEventType constants. pyobjc exposes these but pyright doesn't see them
# reliably, so we hard-code the values from CGEventTypes.h.
_KEY_DOWN = 10
_LEFT_MOUSE_DOWN = 1
_RIGHT_MOUSE_DOWN = 3
_OTHER_MOUSE_DOWN = 25
_HID_SYSTEM_STATE = 1   # kCGEventSourceStateHIDSystemState


@dataclass(frozen=True)
class RealWorldContext:
    local_dt: datetime
    battery_pct: int | None       # None if no battery (desktop) or pmset failed
    on_ac_power: bool
    input_events_per_min: float | None  # None on first sample (no baseline yet)

    @property
    def weekday(self) -> str:
        return self.local_dt.strftime("%A")

    @property
    def date_iso(self) -> str:
        return self.local_dt.strftime("%Y-%m-%d")

    @property
    def hhmm(self) -> str:
        return self.local_dt.strftime("%H:%M")

    @property
    def clock_str(self) -> str:
        """Both 24h and 12h forms so the model doesn't have to do AM/PM math.

        Small VLMs/LLMs are surprisingly bad at this: given 21:50 they will
        sometimes say "almost 8pm". Giving both forms removes the trap.
        """
        return self.local_dt.strftime("%H:%M (%-I:%M %p)")

    @property
    def time_of_day(self) -> str:
        h = self.local_dt.hour
        if 5 <= h < 12:
            return "morning"
        if 12 <= h < 17:
            return "afternoon"
        if 17 <= h < 21:
            return "evening"
        if 21 <= h < 24:
            return "late evening"
        return "middle of the night"  # 0–5

    @property
    def is_weekend(self) -> bool:
        return self.local_dt.weekday() >= 5


@dataclass
class ContextGatherer:
    """Stateful: keeps last input counter + ts to compute per-minute rate."""

    _prev_input_total: int = 0
    _prev_ts: float | None = None

    def gather(self, *, now: float | None = None) -> RealWorldContext:
        ts = now if now is not None else time.time()
        local_dt = datetime.fromtimestamp(ts)
        pct, on_ac = _read_battery()
        rate = self._input_rate_per_minute(ts)
        return RealWorldContext(
            local_dt=local_dt,
            battery_pct=pct,
            on_ac_power=on_ac,
            input_events_per_min=rate,
        )

    def _input_rate_per_minute(self, now_ts: float) -> float | None:
        total = _read_input_counter()
        if self._prev_ts is None or total < self._prev_input_total:
            self._prev_input_total = total
            self._prev_ts = now_ts
            return None
        dt = now_ts - self._prev_ts
        if dt <= 0:
            return None
        delta = total - self._prev_input_total
        self._prev_input_total = total
        self._prev_ts = now_ts
        return (delta / dt) * 60.0


def _read_battery() -> tuple[int | None, bool]:
    """Return (percent, on_ac_power). pct=None if no internal battery."""
    try:
        out = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, True
    on_ac = "AC Power" in out
    m = re.search(r"(\d+)%", out)
    pct = int(m.group(1)) if m else None
    return pct, on_ac


def _read_input_counter() -> int:
    """Monotonic counter of keyDown + mouseDown events since boot.

    Reads HID-system-level event counts. Does not capture key contents; just a
    count. No Input Monitoring permission needed in our testing — if Apple
    starts demanding one on a future macOS, we'll see counts of 0 and the
    rate will read 0 (graceful).
    """
    try:
        return sum(
            int(Quartz.CGEventSourceCounterForEventType(_HID_SYSTEM_STATE, t))
            for t in (_KEY_DOWN, _LEFT_MOUSE_DOWN, _RIGHT_MOUSE_DOWN, _OTHER_MOUSE_DOWN)
        )
    except Exception:
        return 0


def format_for_prompt(ctx: RealWorldContext) -> str:
    weekend_tag = " (weekend)" if ctx.is_weekend else ""
    lines = [
        "Real-world context:",
        f"  date: {ctx.weekday}, {ctx.date_iso}{weekend_tag}",
        f"  time: {ctx.clock_str} — {ctx.time_of_day}",
    ]
    if ctx.battery_pct is not None:
        state = "plugged in" if ctx.on_ac_power else "on battery"
        lines.append(f"  battery: {ctx.battery_pct}% ({state})")
    if ctx.input_events_per_min is not None:
        lines.append(
            f"  input rate: {ctx.input_events_per_min:.0f} key/click events/min "
            f"({_input_label(ctx.input_events_per_min)})"
        )
    return "\n".join(lines)


def _input_label(rate_per_min: float) -> str:
    if rate_per_min < 5:
        return "essentially idle — likely reading, watching, or away from input"
    if rate_per_min < 30:
        return "light input — could be reading + occasional click"
    if rate_per_min < 120:
        return "moderately active"
    return "heavily active — typing or rapid-fire input"
