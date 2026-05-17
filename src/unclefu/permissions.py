"""Camera + Screen Recording permission gate.

Runs after preflight (machine specs ok) and BEFORE any model worker
spawns. Doing it earlier matters: a permission denial mid-download
leaves sensors that silently fail and the focus modal looping with no
clear cause.

macOS TCC quirks worth knowing (Sequoia 15+):

1. **Camera dialog needs a live NSRunLoop.**
   AVCaptureDevice.requestAccessForMediaType_completionHandler_ posts
   the TCC alert through the main run loop. If the main thread blocks
   on a threading.Event waiting for the completion handler, the loop
   stops ticking, the dialog never paints, and the handler never fires
   — a deadlock that the user experiences as a beach ball. Fix: pump
   NSRunLoop in 100 ms slices while waiting.

2. **Screen Recording REQUIRES a process restart after grant.**
   `CGPreflightScreenCaptureAccess()` is hardcoded by Apple to return
   False for the entire lifetime of the process that called
   `CGRequestScreenCaptureAccess()` — even after the user toggles the
   permission on in System Settings. The only way to observe a fresh
   "granted" state is to quit and relaunch. So our flow is:
     - request → show modal "open Settings, then Quit & Relaunch"
     - terminate the app
     - on next launch, preflight returns True and the gate sails through

3. **Unsigned .app entries can be silently dropped from TCC.**
   At minimum the bundle needs an ad-hoc signature (`codesign -s -`),
   otherwise TCC may refuse to persist anything. Each ad-hoc rebuild
   produces a *different* signature, so the user must re-grant after
   every rebuild during development. Real Developer ID + notarisation
   is the only way to keep grants stable across releases.

4. **The system permission prompt fires AT MOST ONCE per bundle id.**
   If the user denied previously (or hit Don't Allow), calling
   requestAccess again returns immediately with `denied` and shows no
   UI. The only recovery path is System Settings → Privacy & Security
   → toggle the app for that category. We detect that case and open
   the right Settings pane for them.

5. **Bundle identity matters.** macOS keys TCC entries to
   CFBundleIdentifier. When running unpackaged (`uv run python -m
   unclefu`) the parent process inherits permissions (Terminal /
   Ghostty), which is why dev mode works without explicit prompts.
   When running from /Applications/UncleFu.app, the bundle id
   `dev.unclefu.app` gets its own TCC entries.

Status enum values come from AVCaptureDevice's
AVAuthorizationStatus enum (0=not_determined, 1=restricted,
2=denied, 3=authorized).
"""

from __future__ import annotations

# pyright: reportAttributeAccessIssue=false

import subprocess
import time
from enum import Enum


class PermissionStatus(str, Enum):
    AUTHORIZED = "authorized"
    DENIED = "denied"
    NOT_DETERMINED = "not_determined"
    RESTRICTED = "restricted"     # MDM / parental controls; user can't fix


# ── camera ────────────────────────────────────────────────────────────


# AVAuthorizationStatus values from AVFoundation/AVCaptureDevice.h.
_AV_AUTH_MAP: dict[int, PermissionStatus] = {
    0: PermissionStatus.NOT_DETERMINED,
    1: PermissionStatus.RESTRICTED,
    2: PermissionStatus.DENIED,
    3: PermissionStatus.AUTHORIZED,
}


def camera_status() -> PermissionStatus:
    """Inspect TCC for camera access. Doesn't prompt."""
    try:
        import AVFoundation  # type: ignore[import-not-found]
        val = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
            AVFoundation.AVMediaTypeVideo,
        )
        return _AV_AUTH_MAP.get(int(val), PermissionStatus.NOT_DETERMINED)
    except Exception:
        return PermissionStatus.NOT_DETERMINED


def request_camera(timeout_s: float = 60.0) -> PermissionStatus:
    """Trigger the camera permission prompt (no-op if not_determined is
    not the current state — macOS won't re-prompt). Pumps NSRunLoop in
    100 ms slices while waiting so the TCC dialog can paint and the
    completion handler can be delivered.

    Must run on the main thread. Calling from a worker thread is
    undefined: AVFoundation expects the run loop to be the main one.
    """
    status = camera_status()
    if status is not PermissionStatus.NOT_DETERMINED:
        # macOS won't re-prompt; whatever is set is what we have.
        return status

    try:
        import AVFoundation  # type: ignore[import-not-found]
        from Foundation import NSDate, NSRunLoop  # type: ignore[import-not-found]
    except ImportError:
        return PermissionStatus.NOT_DETERMINED

    # Use a list as a one-shot bucket — closures can't rebind names
    # without `nonlocal`, but list mutation is fine.
    done: list[bool] = []

    def _completion(granted: bool) -> None:
        done.append(bool(granted))

    AVFoundation.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        AVFoundation.AVMediaTypeVideo, _completion,
    )

    # Pump the run loop in 100 ms slices so AVFoundation can paint the
    # dialog and dispatch the callback. Without this, the main thread
    # is asleep, the dialog never appears, and the user sees a beach
    # ball. Status (not the bool from the handler) is the source of
    # truth — the handler value can disagree with the persisted TCC
    # entry in edge cases.
    loop = NSRunLoop.currentRunLoop()
    deadline = time.time() + timeout_s
    while not done and time.time() < deadline:
        loop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))
    return camera_status()


# ── screen recording ──────────────────────────────────────────────────


def screen_status() -> PermissionStatus:
    """Inspect screen recording TCC. CGPreflightScreenCaptureAccess
    returns True iff granted; False covers both denied and not_determined.
    Since the prompt is harmless to fire (no-op if denied), we don't
    bother distinguishing here."""
    try:
        import Quartz  # type: ignore[import-not-found]
        return (
            PermissionStatus.AUTHORIZED
            if Quartz.CGPreflightScreenCaptureAccess()
            else PermissionStatus.NOT_DETERMINED
        )
    except Exception:
        return PermissionStatus.NOT_DETERMINED


def trigger_screen_prompt() -> bool:
    """Fire CGRequestScreenCaptureAccess to register the bundle in TCC.

    Returns True if the call succeeded (the bundle is now visible in
    System Settings → Privacy & Security → Screen Recording), False on
    error. Does NOT mean the user granted — preflight for this process
    will keep returning False until restart. The caller must show a
    'Quit & Relaunch' modal and exit.
    """
    if screen_status() is PermissionStatus.AUTHORIZED:
        return True
    try:
        import Quartz  # type: ignore[import-not-found]
        Quartz.CGRequestScreenCaptureAccess()
        return True
    except Exception:
        return False


# ── opening System Settings ───────────────────────────────────────────


# URL schemes that jump straight to the relevant Privacy pane.
# Verified on macOS 15 Sequoia.
_SETTINGS_URLS: dict[str, str] = {
    "camera": "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera",
    "screen": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
}


def open_settings_for(perm: str) -> None:
    """Open System Settings to the right Privacy pane. `perm` is one of
    'camera' or 'screen'. Silently no-ops on unknown values."""
    url = _SETTINGS_URLS.get(perm)
    if not url:
        return
    try:
        subprocess.run(["open", url], check=False, timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass


# ── gate: the orchestrator main() calls ───────────────────────────────


def _force_quit() -> None:
    """Terminate the process via NSApplication so the run loop unwinds
    cleanly. sys.exit from here would leave the rumps event loop
    half-running on Sequoia."""
    try:
        from AppKit import NSApplication  # type: ignore[import-not-found]
        NSApplication.sharedApplication().terminate_(None)
    except Exception:
        pass
    # Belt-and-suspenders — if terminate_ didn't unwind for any reason.
    import os
    os._exit(0)


def run_permission_gate(personality_name: str = "Uncle Fu") -> bool:
    """Block until Camera + Screen Recording are both authorised, or
    walk the user to System Settings + exit so they can relaunch.

    Returns True only when both permissions are AUTHORIZED. Returns
    False (or terminates) in every other path; caller should exit(0)
    on False.

    Flow (Sequoia-correct):

      1. Both already authorised → return True (no UI).
      2. Otherwise show one explanatory modal listing what we need and
         why. Continue / Quit.
      3. Camera first (in-process, doesn't need restart):
         - request_camera() pumps NSRunLoop while waiting for the
           system dialog → user grants → status flips → continue.
         - If still not authorised (denied or "Don't Allow"), open
           Settings → Privacy → Camera and exit.
      4. Screen Recording (REQUIRES restart):
         - trigger_screen_prompt() registers the bundle in TCC.
         - Show "Open Settings, toggle on, then Quit & Relaunch"
           modal that opens the right pane.
         - Force-quit; on next launch preflight returns True and
           the gate sails through.

      Crucially, camera is handled BEFORE screen recording — we want
      to finish the in-process branch cleanly before throwing away
      the process for the screen-recording restart.

    Must be called AFTER NSApplication.sharedApplication() exists +
    activateIgnoringOtherApps so the modals take keyboard focus.
    """
    import rumps  # type: ignore[import-not-found]

    cam = camera_status()
    scr = screen_status()
    if cam is PermissionStatus.AUTHORIZED and scr is PermissionStatus.AUTHORIZED:
        return True

    # Single explanatory modal listing only what we actually need.
    # Pretending we still need a permission that's already granted
    # would confuse the user.
    needs: list[str] = []
    if cam is not PermissionStatus.AUTHORIZED:
        needs.append(
            "  • Camera — to read your posture, presence, and whether "
            "you've stepped away. All frames stay on the Mac."
        )
    if scr is not PermissionStatus.AUTHORIZED:
        needs.append(
            "  • Screen Recording — to notice when what's on screen "
            "drifts from what you said you'd focus on. All captures "
            "stay on the Mac."
        )
    msg = (
        f"{personality_name} needs a couple of permissions before he "
        f"can watch over you:\n\n" + "\n\n".join(needs) +
        "\n\nClicking Continue will trigger the macOS permission "
        "prompts. You can revoke any time in System Settings → "
        "Privacy & Security."
    )
    resp = rumps.alert(
        title=f"{personality_name} needs your permission",
        message=msg,
        ok="Continue",
        cancel="Quit",
    )
    if resp != 1:  # rumps: 1 = ok, 0 = cancel
        return False

    # ── 1. Camera ─────────────────────────────────────────────────────
    if cam is not PermissionStatus.AUTHORIZED:
        cam = request_camera()  # pumps NSRunLoop; safe on main thread
        if cam is not PermissionStatus.AUTHORIZED:
            # Either denied just now or denied previously (macOS won't
            # re-prompt — the request returns the existing status).
            # Walk them to Settings; relaunch path.
            rumps.alert(
                title=f"{personality_name} needs camera access",
                message=(
                    f"macOS didn't grant camera access. Open System "
                    f"Settings → Privacy & Security → Camera, toggle "
                    f"{personality_name} on, then re-launch."
                ),
                ok="Open System Settings",
            )
            open_settings_for("camera")
            return False

    # ── 2. Screen Recording (restart required) ────────────────────────
    if scr is not PermissionStatus.AUTHORIZED:
        trigger_screen_prompt()  # registers bundle in TCC
        # On Sequoia, preflight for THIS process is hardcoded to keep
        # returning False even after the user toggles the permission
        # on. The only way to observe a fresh "granted" is to quit and
        # relaunch. So we tell the user, open Settings, and exit.
        rumps.alert(
            title=f"{personality_name} needs screen access",
            message=(
                f"macOS requires {personality_name} to restart after "
                f"you grant screen recording.\n\n"
                f"1. In System Settings → Privacy & Security → "
                f"Screen Recording, toggle {personality_name} on.\n"
                f"2. Re-launch {personality_name}.\n\n"
                f"(Camera access is already set — this is just for "
                f"the screen.)"
            ),
            ok="Open System Settings & Quit",
        )
        open_settings_for("screen")
        _force_quit()
        # Unreachable, but the type checker doesn't know that.
        return False

    # Both authorised (camera just granted, screen already was) →
    # safe to proceed without restart.
    return True
