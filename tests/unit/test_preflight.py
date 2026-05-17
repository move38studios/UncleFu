"""Preflight checks are pure functions wrapping platform/subprocess
calls. We test the aggregator and the summary formatter without
mocking subprocess (the real ones run on every CI Mac); for the
specific check functions we just test that they return a CheckResult.
"""

from __future__ import annotations

from unclefu.preflight import (
    CheckResult,
    PreflightResult,
    _check_arch,
    _check_disk_free,
    _check_macos_version,
    _check_ram,
    run_preflight,
)


def test_check_result_basic():
    c = CheckResult(name="x", ok=True, detail="all good")
    assert c.ok
    assert c.detail == "all good"


def test_preflight_aggregate_ok_when_all_pass():
    pre = PreflightResult(checks=[
        CheckResult("a", True, "ok"),
        CheckResult("b", True, "ok"),
    ])
    assert pre.ok
    assert pre.failures == []


def test_preflight_aggregate_fails_on_any_failure():
    pre = PreflightResult(checks=[
        CheckResult("a", True, "ok"),
        CheckResult("b", False, "uh oh"),
    ])
    assert not pre.ok
    assert len(pre.failures) == 1
    assert pre.failures[0].name == "b"


def test_summary_for_modal_lists_failures():
    pre = PreflightResult(checks=[
        CheckResult("Apple Silicon", False, "x86_64"),
        CheckResult("RAM", False, "8 GB (need 16 GB+)"),
        CheckResult("macOS version", True, "macOS 14.5"),
    ])
    s = pre.summary_for_modal()
    assert "Apple Silicon" in s
    assert "x86_64" in s
    assert "RAM" in s
    assert "8 GB" in s
    # Passing check shouldn't appear.
    assert "macOS 14.5" not in s


def test_summary_when_all_passed():
    pre = PreflightResult(checks=[CheckResult("a", True, "ok")])
    assert "All checks passed" in pre.summary_for_modal()


def test_each_check_returns_a_result():
    """Sanity — each individual check runs to completion and returns
    a CheckResult, even on the test machine (whose specifics we don't
    know in CI). We don't assert ok/not-ok since the test machine
    might be anywhere."""
    for fn in (_check_arch, _check_macos_version, _check_ram, _check_disk_free):
        result = fn()
        assert isinstance(result, CheckResult)
        assert result.name
        assert result.detail


def test_run_preflight_returns_all_four_checks():
    pre = run_preflight()
    names = {c.name for c in pre.checks}
    assert names == {"Apple Silicon", "macOS version", "RAM", "Disk free"}
