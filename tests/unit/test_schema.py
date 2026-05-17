"""Schema tests post-simplification: only SensorObservation and Decision."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from unclefu.vlm.schema import Decision, SensorObservation, SensorSnapshot


def test_sensor_observation_roundtrips():
    payload = {"description": "User upright at desk.", "confidence": "high"}
    s = SensorObservation.model_validate(payload)
    assert s.description == "User upright at desk."
    assert s.confidence == "high"
    assert s.model_dump() == payload


def test_sensor_observation_rejects_bad_confidence():
    with pytest.raises(ValidationError):
        SensorObservation.model_validate(
            {"description": "x" * 10, "confidence": "kinda_sure"}
        )


def test_sensor_observation_rejects_too_short_description():
    with pytest.raises(ValidationError):
        SensorObservation.model_validate({"description": "a", "confidence": "high"})


def test_decision_accepts_silent_and_speaking():
    silent = Decision.model_validate({
        "should_speak": False, "urgency": "low",
        "message": None, "reason": "nothing notable",
    })
    assert silent.message is None
    # expression defaults to "idle" when the model omits it.
    assert silent.expression == "idle"
    speaking = Decision.model_validate({
        "should_speak": True, "urgency": "medium",
        "message": "hey, sit up", "reason": "posture_drift",
        "expression": "disapproving",
    })
    assert speaking.message == "hey, sit up"
    assert speaking.expression == "disapproving"


def test_decision_rejects_unknown_expression():
    with pytest.raises(ValidationError):
        Decision.model_validate({
            "should_speak": True, "urgency": "medium",
            "message": "x", "reason": "y", "expression": "grumpy_cat",
        })


def test_sensor_snapshot_dataclass_basic():
    s = SensorSnapshot(
        ts=1.0, source="webcam", description="ok",
        structured={"confidence": "high"}, cycle_ms=2000,
    )
    assert s.source == "webcam"
    assert not s.is_error
    assert s.confidence == "high"


def test_sensor_snapshot_is_error_when_exc_info_present():
    s = SensorSnapshot(
        ts=1.0, source="webcam", description="",
        structured={"exc_type": "VLMError", "exc_message": "timeout"},
        cycle_ms=0,
    )
    assert s.is_error


def test_sensor_snapshot_confidence_defaults_high_when_missing():
    """Old rows pre-confidence have no entry — trust them."""
    s = SensorSnapshot(
        ts=1.0, source="webcam", description="legacy row",
        structured={}, cycle_ms=2000,
    )
    assert s.confidence == "high"


def test_sensor_snapshot_confidence_defaults_high_on_garbage_value():
    """Defensive — random string in confidence shouldn't break consumers."""
    s = SensorSnapshot(
        ts=1.0, source="webcam", description="x" * 10,
        structured={"confidence": "kinda"}, cycle_ms=2000,
    )
    assert s.confidence == "high"
