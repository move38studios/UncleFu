"""SQLite session log for the decoupled architecture.

Three tables:
- session       : one row per process invocation.
- sensor_snapshot : one row per sensor tick (webcam, screen_0, ...). Errors
  land here too, as a degenerate row (description="", structured contains exc info).
- decision      : one row per Director tick.

Stored at `~/Library/Application Support/UncleFu/sessions.db` by default.
Privacy: only structured fields + text descriptions go to disk. No images.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..vlm.schema import Decision, SensorSnapshot


def default_db_path() -> Path:
    base = Path.home() / "Library" / "Application Support" / "UncleFu"
    base.mkdir(parents=True, exist_ok=True)
    return base / "sessions.db"


_SCHEMA = """
-- One-time legacy cleanup: tables from previous phases that no longer exist.
-- IF EXISTS makes these no-ops once we're past them.
DROP TABLE IF EXISTS observation;
DROP TABLE IF EXISTS cycle_error;

CREATE TABLE IF NOT EXISTS session (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    personality TEXT,
    focus       TEXT NOT NULL,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS sensor_snapshot (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES session(id),
    ts          REAL NOT NULL,
    source      TEXT NOT NULL,
    description TEXT NOT NULL,
    structured  TEXT NOT NULL,   -- JSON of WebcamStructured / ScreenStructured / etc.
    cycle_ms    INTEGER NOT NULL,
    exc_type    TEXT,            -- non-null iff this row represents an error
    exc_message TEXT
);
CREATE INDEX IF NOT EXISTS sensor_snapshot_session_source_ts
    ON sensor_snapshot(session_id, source, ts);
CREATE INDEX IF NOT EXISTS sensor_snapshot_ts
    ON sensor_snapshot(ts);

CREATE TABLE IF NOT EXISTS decision (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES session(id),
    ts          REAL NOT NULL,
    should_speak INTEGER NOT NULL,     -- 0/1
    urgency     TEXT NOT NULL,
    message     TEXT,                  -- null if should_speak=false
    reason      TEXT NOT NULL,
    expression  TEXT NOT NULL,         -- Director-picked icon expression
    outcome     TEXT NOT NULL,         -- intervener outcome ("spoke", "throttled_min", ...)
    call_ms     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS decision_session_ts
    ON decision(session_id, ts);
CREATE INDEX IF NOT EXISTS decision_spoken
    ON decision(outcome) WHERE outcome = 'spoke';
"""


# The columns the current schema expects each table to carry. If a table
# exists but is missing one of these, that means the DB was created by an
# older Uncle Fu schema and we need to wipe-and-recreate. Pre-1.0; we
# don't bother with proper migration tooling.
_EXPECTED_COLS: dict[str, set[str]] = {
    "session": {"focus"},
    "decision": {"expression"},
}


def _migrate_legacy_schema(conn: sqlite3.Connection) -> None:
    """One-shot pre-1.0 migration: if a table exists with an older shape,
    drop it (plus anything that FKs into it) so the CREATE statements in
    `_SCHEMA` can rebuild a fresh copy. Idempotent; cheap when there's
    nothing to do."""
    needs_full_wipe = False
    for table, required in _EXPECTED_COLS.items():
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if not rows:
            continue  # table doesn't exist yet — CREATE will handle it
        present = {r[1] for r in rows}
        if not required <= present:
            needs_full_wipe = True
            break
    if needs_full_wipe:
        # session is FK-referenced by sensor_snapshot and decision, so any
        # session reset has to drop both of them too. Cheaper to just wipe
        # the lot than to play schema Tetris.
        for t in ("decision", "sensor_snapshot", "session"):
            conn.execute(f"DROP TABLE IF EXISTS {t}")


@dataclass
class SessionLog:
    db_path: Path
    session_id: int
    _conn: sqlite3.Connection

    @classmethod
    def open(
        cls,
        db_path: Path | None = None,
        *,
        started_at: float,
        focus: str,
        personality: str | None = None,
        notes: str = "",
    ) -> "SessionLog":
        if not focus or not focus.strip():
            raise ValueError("session focus is required (no focus, no session)")
        path = db_path if db_path is not None else default_db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        _migrate_legacy_schema(conn)
        conn.executescript(_SCHEMA)
        cur = conn.execute(
            "INSERT INTO session(started_at, personality, focus, notes) VALUES (?, ?, ?, ?)",
            (started_at, personality, focus.strip(), notes),
        )
        sid = cur.lastrowid
        if sid is None:
            raise RuntimeError("Failed to create session row")
        return cls(db_path=path, session_id=sid, _conn=conn)

    # ---- writes ----

    def record_snapshot(self, snap: SensorSnapshot) -> None:
        self._conn.execute(
            """
            INSERT INTO sensor_snapshot(
                session_id, ts, source, description, structured, cycle_ms,
                exc_type, exc_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.session_id, snap.ts, snap.source, snap.description,
                json.dumps(snap.structured), snap.cycle_ms,
                snap.structured.get("exc_type"),
                snap.structured.get("exc_message"),
            ),
        )

    def record_decision(
        self,
        *,
        ts: float,
        decision: Decision,
        outcome: str,
        call_ms: int,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO decision(
                session_id, ts, should_speak,
                urgency, message, reason, expression, outcome, call_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.session_id, ts,
                1 if decision.should_speak else 0,
                decision.urgency, decision.message, decision.reason,
                decision.expression, outcome, call_ms,
            ),
        )

    # ---- reads ----

    def recent_snapshots(self, source: str, *, limit: int) -> list[SensorSnapshot]:
        rows = self._conn.execute(
            """
            SELECT ts, source, description, structured, cycle_ms
            FROM sensor_snapshot
            WHERE session_id=? AND source=? AND exc_type IS NULL
            ORDER BY ts DESC LIMIT ?
            """,
            (self.session_id, source, limit),
        ).fetchall()
        return [
            SensorSnapshot(
                ts=row[0], source=row[1], description=row[2],
                structured=json.loads(row[3]), cycle_ms=row[4],
            )
            for row in rows
        ]

    def latest_snapshot(self, source: str) -> SensorSnapshot | None:
        rows = self.recent_snapshots(source, limit=1)
        return rows[0] if rows else None

    def latest_decision(self) -> tuple[float, str, str | None, str] | None:
        """Most recent decision row: (ts, outcome, last_message_or_none, expression).

        `outcome` reflects whether the line was actually spoken (vs. throttled
        etc.). `expression` is the Director's icon pick for that line — only
        meaningful if it was spoken.
        """
        row = self._conn.execute(
            """
            SELECT ts, outcome, message, expression FROM decision
            WHERE session_id=? ORDER BY ts DESC LIMIT 1
            """,
            (self.session_id,),
        ).fetchone()
        if row is None:
            return None
        return row[0], row[1], row[2], row[3]

    def recent_spoken_lines(self, *, limit: int) -> list[tuple[float, str]]:
        rows = self._conn.execute(
            """
            SELECT ts, message FROM decision
            WHERE session_id=? AND outcome='spoke' AND message IS NOT NULL
            ORDER BY ts DESC LIMIT ?
            """,
            (self.session_id, limit),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def close(self, *, ended_at: float) -> None:
        try:
            self._conn.execute(
                "UPDATE session SET ended_at=? WHERE id=?",
                (ended_at, self.session_id),
            )
        finally:
            self._conn.close()


@contextmanager
def open_session(
    db_path: Path | None = None,
    *,
    started_at: float,
    focus: str,
    personality: str | None = None,
    notes: str = "",
):
    log = SessionLog.open(
        db_path, started_at=started_at, focus=focus,
        personality=personality, notes=notes,
    )
    try:
        yield log
    finally:
        import time as _time
        log.close(ended_at=_time.time())


def error_snapshot(*, ts: float, source: str, exc_type: str, exc_message: str, cycle_ms: int) -> SensorSnapshot:
    """Build a SensorSnapshot that represents an error."""
    return SensorSnapshot(
        ts=ts,
        source=source,
        description="",
        structured={"exc_type": exc_type, "exc_message": exc_message},
        cycle_ms=cycle_ms,
    )
