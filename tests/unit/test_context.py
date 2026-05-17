from __future__ import annotations

from datetime import datetime

from unclefu.director.context import (
    RealWorldContext,
    _input_label,
    format_for_prompt,
)


def _ctx(
    *,
    dt: datetime,
    battery_pct: int | None = None,
    on_ac_power: bool = True,
    input_events_per_min: float | None = None,
) -> RealWorldContext:
    return RealWorldContext(
        local_dt=dt,
        battery_pct=battery_pct,
        on_ac_power=on_ac_power,
        input_events_per_min=input_events_per_min,
    )


def test_weekday_and_iso():
    c = _ctx(dt=datetime(2026, 5, 13, 19, 30))
    assert c.weekday == "Wednesday"
    assert c.date_iso == "2026-05-13"
    assert c.hhmm == "19:30"


def test_clock_str_includes_both_24h_and_12h():
    c = _ctx(dt=datetime(2026, 5, 13, 21, 50))
    assert "21:50" in c.clock_str
    assert "9:50 PM" in c.clock_str

    morning = _ctx(dt=datetime(2026, 5, 13, 9, 5))
    assert "09:05" in morning.clock_str
    assert "9:05 AM" in morning.clock_str


def test_time_of_day_buckets():
    assert _ctx(dt=datetime(2026, 5, 13, 7, 0)).time_of_day == "morning"
    assert _ctx(dt=datetime(2026, 5, 13, 14, 0)).time_of_day == "afternoon"
    assert _ctx(dt=datetime(2026, 5, 13, 19, 0)).time_of_day == "evening"
    assert _ctx(dt=datetime(2026, 5, 13, 22, 0)).time_of_day == "late evening"
    assert _ctx(dt=datetime(2026, 5, 13, 3, 0)).time_of_day == "middle of the night"


def test_is_weekend():
    # 2026-05-16 is a Saturday
    assert _ctx(dt=datetime(2026, 5, 16, 12, 0)).is_weekend
    # 2026-05-13 is a Wednesday
    assert not _ctx(dt=datetime(2026, 5, 13, 12, 0)).is_weekend


def test_format_full_block():
    out = format_for_prompt(_ctx(
        dt=datetime(2026, 5, 16, 23, 15),
        battery_pct=42, on_ac_power=False,
        input_events_per_min=150.0,
    ))
    assert "Real-world context:" in out
    assert "Saturday" in out
    assert "(weekend)" in out
    assert "23:15" in out
    assert "11:15 PM" in out
    assert "late evening" in out
    assert "42% (on battery)" in out
    assert "150" in out
    assert "heavily active" in out


def test_format_omits_missing_fields():
    out = format_for_prompt(_ctx(
        dt=datetime(2026, 5, 13, 10, 0),
        battery_pct=None,        # e.g. desktop
        input_events_per_min=None,  # first sample
    ))
    assert "Wednesday" in out
    assert "10:00" in out
    assert "battery" not in out
    assert "input rate" not in out


def test_input_label_buckets():
    assert "idle" in _input_label(0).lower()
    assert "light" in _input_label(15).lower()
    assert "moderate" in _input_label(60).lower()
    assert "heavily" in _input_label(200).lower()
