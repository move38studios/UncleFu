from __future__ import annotations

from pathlib import Path

from unclefu.director.director import DirectorCall
from unclefu.personalities import get as get_personality
from unclefu.storage.debug_log import DebugLog
from unclefu.vlm.schema import Decision, SensorSnapshot


def test_debug_log_writes_header_snapshot_director(tmp_path: Path):
    p = tmp_path / "debug.log"
    log = DebugLog.open(
        p,
        personality=get_personality("uncle_fu"),
        model="gemma-4-e4b-it-mlx",
        webcam_interval_s=30.0,
        screen_interval_s=12.0,
        director_interval_s=20.0,
    )

    snap = SensorSnapshot(
        ts=1747_000_000.0, source="screen_0",
        description="VS Code with main.py on screen, no errors visible.",
        structured={"content": "VS Code — main.py"},
        cycle_ms=4200,
    )
    log.write_snapshot(snap)

    err = SensorSnapshot(
        ts=1747_000_020.0, source="webcam",
        description="",
        structured={"exc_type": "VLMError", "exc_message": "ReadTimeout"},
        cycle_ms=0,
    )
    log.write_snapshot(err)

    call = DirectorCall(
        decision=Decision(
            should_speak=True, urgency="medium",
            message="hey, sit up", reason="posture_drift",
            expression="disapproving",
        ),
        user_text="Webcam: ...\nScreen 0: ...",
        raw_response='{"should_speak": true, ...}',
        call_ms=1800,
    )
    log.write_director_call(ts=1747_000_030.0, call=call, outcome="spoke")
    log.close()

    body = p.read_text()
    assert "Uncle Fu" in body
    assert "SNAPSHOT  screen_0" in body
    assert "VS Code — main.py" in body
    assert "SNAPSHOT  webcam" in body
    assert "ERROR VLMError: ReadTimeout" in body
    assert "DIRECTOR" in body
    assert "urgency=medium" in body
    assert "expression=disapproving" in body
    assert "hey, sit up" in body
