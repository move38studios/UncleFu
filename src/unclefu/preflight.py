"""Hardware + OS sanity checks run before any heavy init.

Uncle Fu needs:
- Apple Silicon (MLX is arm64-only)
- macOS 14+ (current mlx-vlm assumes Sonoma-or-later vDSP / Metal APIs)
- 16 GB RAM (we hit ~7 GB peak with both models loaded; 8 GB would
  swap badly)
- 8 GB free disk in the HuggingFace cache partition (Gemma ~5.2 GB +
  Qwen ~2 GB + slack for refs/blobs/safetensors hardlinks)

If any check fails, we show a clean modal explaining what's missing,
then exit with code 1. No half-started state, no scary stack trace.

Failing checks are recoverable — user upgrades macOS / frees disk /
plugs in more RAM (lol). The modal text states the specific problem.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


MIN_MACOS = (15, 0)            # Sequoia (MLX ≥ 0.29.2 requires it)
MIN_RAM_GB = 16
MIN_FREE_DISK_GB = 8           # models total ~7 GB; this is download budget + slack


@dataclass(frozen=True)
class CheckResult:
    """Result of one individual check. `ok=True` means we pass."""

    name: str            # short label, e.g. "Apple Silicon"
    ok: bool
    detail: str          # what we found (e.g. "M2 Pro" or "x86_64")


@dataclass(frozen=True)
class PreflightResult:
    """Aggregate result. `ok` is True iff every check passed."""

    checks: list[CheckResult]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok]

    def summary_for_modal(self) -> str:
        """Human-readable failure summary suitable for an NSAlert body."""
        if self.ok:
            return "All checks passed."
        lines = ["Uncle Fu can't run on this machine:\n"]
        for c in self.failures:
            lines.append(f"  ✗ {c.name}: {c.detail}")
        lines.append(
            "\nUncle Fu needs Apple Silicon, macOS 14+, 16 GB RAM, "
            "and 8 GB free disk for model weights."
        )
        return "\n".join(lines)


def _check_arch() -> CheckResult:
    arch = platform.machine()
    return CheckResult(
        name="Apple Silicon",
        ok=arch == "arm64",
        detail=arch if arch else "unknown",
    )


def _check_macos_version() -> CheckResult:
    """Parse `sw_vers -productVersion` for accuracy — platform.mac_ver()
    can lie under Rosetta or in some sandboxed contexts."""
    try:
        out = subprocess.run(
            ["sw_vers", "-productVersion"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        parts = out.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        version = (major, minor)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return CheckResult(name="macOS version", ok=False, detail="couldn't read")
    ok = version >= MIN_MACOS
    return CheckResult(
        name="macOS version",
        ok=ok,
        detail=f"macOS {major}.{minor} (need {MIN_MACOS[0]}.{MIN_MACOS[1]}+)",
    )


def _check_ram() -> CheckResult:
    """Total physical memory via sysctl. Failure here doesn't auto-fail
    the preflight — we report it. Some users may want to try with 8 GB."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        bytes_ = int(out)
        gb = bytes_ // (1024 ** 3)
    except (OSError, subprocess.SubprocessError, ValueError):
        return CheckResult(name="RAM", ok=False, detail="couldn't read")
    ok = gb >= MIN_RAM_GB
    return CheckResult(
        name="RAM",
        ok=ok,
        detail=f"{gb} GB (need {MIN_RAM_GB} GB+)",
    )


def _check_disk_free() -> CheckResult:
    """Free space in the HuggingFace cache partition. We don't actually
    require that ~/.cache/huggingface exist — shutil.disk_usage on the
    parent (~/.cache or ~) works fine."""
    cache_root = Path(os.environ.get("HF_HOME", "")) if os.environ.get("HF_HOME") else Path.home() / ".cache" / "huggingface"
    probe = cache_root if cache_root.exists() else cache_root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free_bytes = shutil.disk_usage(str(probe)).free
        free_gb = free_bytes // (1024 ** 3)
    except OSError:
        return CheckResult(name="Disk free", ok=False, detail="couldn't read")
    ok = free_gb >= MIN_FREE_DISK_GB
    return CheckResult(
        name="Disk free",
        ok=ok,
        detail=f"{free_gb} GB free in {probe} (need {MIN_FREE_DISK_GB} GB+)",
    )


def run_preflight() -> PreflightResult:
    """Run every check and return the aggregate. Cheap (~10 ms total)."""
    return PreflightResult(checks=[
        _check_arch(),
        _check_macos_version(),
        _check_ram(),
        _check_disk_free(),
    ])
