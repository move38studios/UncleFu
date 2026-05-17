from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from unclefu.storage.session_log import SessionLog, error_snapshot
from unclefu.vlm.schema import Decision, SensorSnapshot


def _snap(ts: float, source: str = "webcam") -> SensorSnapshot:
    return SensorSnapshot(
        ts=ts, source=source, description=f"snap from {source}",
        structured={"posture": "upright", "x": ts}, cycle_ms=2500,
    )


def test_session_log_creates_tables(tmp_path: Path):
    db = tmp_path / "test.db"
    log = SessionLog.open(db, started_at=1000.0, focus="testing")
    log.close(ended_at=1010.0)
    conn = sqlite3.connect(str(db))
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"session", "sensor_snapshot", "decision"} <= tables


def test_session_log_requires_non_empty_focus(tmp_path: Path):
    """Focus mode is the product — no focus, no session."""
    import pytest
    db = tmp_path / "test.db"
    with pytest.raises(ValueError):
        SessionLog.open(db, started_at=1000.0, focus="")
    with pytest.raises(ValueError):
        SessionLog.open(db, started_at=1000.0, focus="   ")


def test_focus_is_persisted_and_trimmed(tmp_path: Path):
    db = tmp_path / "test.db"
    log = SessionLog.open(
        db, started_at=1000.0,
        focus="  writing the auth migration  ",
        personality="uncle_fu",
    )
    log.close(ended_at=1010.0)
    conn = sqlite3.connect(str(db))
    f = conn.execute("SELECT focus FROM session").fetchone()[0]
    conn.close()
    assert f == "writing the auth migration"


def test_record_snapshot_and_query(tmp_path: Path):
    db = tmp_path / "test.db"
    log = SessionLog.open(db, started_at=1000.0, focus="testing", personality="uncle_fu")
    log.record_snapshot(_snap(1001.0, "webcam"))
    log.record_snapshot(_snap(1002.0, "screen_0"))
    log.record_snapshot(_snap(1003.0, "webcam"))
    log.close(ended_at=1010.0)

    log2 = SessionLog.open(db, started_at=2000.0, focus="testing", personality="uncle_fu")
    # New session; should see no rows for it.
    assert log2.recent_snapshots("webcam", limit=10) == []
    log2.close(ended_at=2001.0)

    # Reopen the first session: most-recent first.
    conn = sqlite3.connect(str(db))
    sid = conn.execute(
        "SELECT id FROM session WHERE personality='uncle_fu' ORDER BY id LIMIT 1"
    ).fetchone()[0]
    rows = list(conn.execute(
        "SELECT ts, source FROM sensor_snapshot WHERE session_id=? ORDER BY ts",
        (sid,),
    ))
    conn.close()
    assert rows == [(1001.0, "webcam"), (1002.0, "screen_0"), (1003.0, "webcam")]


def test_error_snapshot_round_trip(tmp_path: Path):
    db = tmp_path / "test.db"
    log = SessionLog.open(db, started_at=1000.0, focus="testing")
    err = error_snapshot(
        ts=1500.0, source="webcam",
        exc_type="VLMError", exc_message="ReadTimeout",
        cycle_ms=120000,
    )
    log.record_snapshot(err)
    log.close(ended_at=1600.0)

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT exc_type, exc_message, description FROM sensor_snapshot"
    ).fetchone()
    conn.close()
    assert row == ("VLMError", "ReadTimeout", "")


def test_recent_snapshots_excludes_errors(tmp_path: Path):
    db = tmp_path / "test.db"
    log = SessionLog.open(db, started_at=1000.0, focus="testing")
    log.record_snapshot(_snap(1001.0, "webcam"))
    log.record_snapshot(error_snapshot(
        ts=1002.0, source="webcam", exc_type="VLMError",
        exc_message="x", cycle_ms=0,
    ))
    log.record_snapshot(_snap(1003.0, "webcam"))
    out = log.recent_snapshots("webcam", limit=10)
    log.close(ended_at=1010.0)
    assert [s.ts for s in out] == [1003.0, 1001.0]


def test_record_and_query_decision(tmp_path: Path):
    db = tmp_path / "test.db"
    log = SessionLog.open(db, started_at=1000.0, focus="testing", personality="uncle_fu")
    d_spoke = Decision(
        should_speak=True, urgency="medium",
        message="bro, take a breath", reason="direct_address",
        expression="disapproving",
    )
    d_silent = Decision(
        should_speak=False, urgency="low", message=None, reason="nothing notable",
    )
    log.record_decision(ts=1001.0, decision=d_spoke, outcome="spoke", call_ms=2400)
    log.record_decision(ts=1021.0, decision=d_silent, outcome="model_declined", call_ms=2100)
    lines = log.recent_spoken_lines(limit=5)
    latest = log.latest_decision()
    log.close(ended_at=1041.0)
    assert lines == [(1001.0, "bro, take a breath")]
    assert latest is not None
    ts, outcome, msg, expression = latest
    assert ts == 1021.0
    assert outcome == "model_declined"
    assert msg is None
    assert expression == "idle"  # default on the silent decision


def test_personality_recorded_on_session(tmp_path: Path):
    db = tmp_path / "test.db"
    log = SessionLog.open(db, started_at=1000.0, focus="testing", personality="uncle_fu")
    log.close(ended_at=1010.0)
    conn = sqlite3.connect(str(db))
    p = conn.execute("SELECT personality FROM session").fetchone()[0]
    conn.close()
    assert p == "uncle_fu"


def test_structured_json_roundtrips(tmp_path: Path):
    db = tmp_path / "test.db"
    log = SessionLog.open(db, started_at=1000.0, focus="testing")
    log.record_snapshot(_snap(1001.0, "screen_0"))
    log.close(ended_at=1010.0)
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT structured FROM sensor_snapshot").fetchone()
    conn.close()
    parsed = json.loads(row[0])
    assert parsed["posture"] == "upright"
