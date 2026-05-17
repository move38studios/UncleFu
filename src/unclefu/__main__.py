"""CLI entrypoint: `uv run python -m unclefu`."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Pre-warm `transformers` on the main thread before any worker spawns.
# Both MlxVlmClient and QwenSpeaker spin up worker threads that import
# `from transformers import AutoTokenizer`; if they race the import lock
# the loser sees a partial `transformers.__init__` and crashes with
# `cannot import name 'AutoTokenizer'`. Importing here serializes the
# heavy init on the main thread (~1-2 s) so both worker threads find
# the module fully constructed. See docs/learnings.md.
from transformers import AutoTokenizer as _AutoTokenizer  # type: ignore[import-not-found] # noqa: F401

from .director.focus_classifier import Relevance, classify_focus
from .personalities import DEFAULT_PERSONALITY, PERSONALITIES, get as get_personality
from .preflight import run_preflight
from .runtime.runner import build_and_start, run as run_headless
from .storage.debug_log import DebugLog
from .storage.session_log import default_db_path
from .tts.speaker import (
    DEFAULT_QWEN_MODEL,
    MutableSpeaker,
    NullSpeaker,
    QwenSpeaker,
    Speaker,
)
from .ui.menubar import MenuBarApp
from .vlm.client import DEFAULT_MODEL, MlxVlmClient


def _debug_root() -> Path:
    return default_db_path().parent / "debug"


def main() -> int:
    parser = argparse.ArgumentParser(prog="unclefu")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HuggingFace VLM model id loaded via mlx-vlm "
                             f"(default: {DEFAULT_MODEL})")
    parser.add_argument("--personality", default=DEFAULT_PERSONALITY,
                        choices=sorted(PERSONALITIES.keys()),
                        help=f"voice / character (default: {DEFAULT_PERSONALITY})")
    parser.add_argument("--mute", action="store_true",
                        help="start muted. menu bar lets you unmute at runtime.")
    parser.add_argument("--voice", default=None,
                        help="override the personality's default Qwen3-TTS speaker id "
                             "(CustomVoice preset name, e.g. Aiden, Ryan, Vivian)")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL,
                        help="mlx-community Qwen3-TTS model id to load")
    parser.add_argument("--webcam-interval", type=float, default=30.0,
                        help="seconds between webcam snapshots (default: 30)")
    parser.add_argument("--screen-interval", type=float, default=12.0,
                        help="seconds between screen snapshots per display (default: 12)")
    parser.add_argument("--director-interval", type=float, default=20.0,
                        help="seconds between Director decisions (default: 20)")
    parser.add_argument("--min-gap", type=float, default=60.0,
                        help="hard minimum seconds between any two spoken lines (default: 60)")
    parser.add_argument("--cooldown", type=float, default=90.0,
                        help="additional cooldown after speech for low urgency only (default: 90)")
    parser.add_argument("--debug-log", default=None, type=Path,
                        help="path for the human-readable per-cycle debug log")
    parser.add_argument("--debug", action="store_true",
                        help=f"shortcut: auto debug-log under {_debug_root()}")
    parser.add_argument("--cli", action="store_true",
                        help="headless mode — no menu bar, per-cycle stdout output. "
                             "Use for debugging or when you don't want the icon.")
    parser.add_argument("--focus", default=None,
                        help="what you're focusing on this session (e.g. "
                             "'writing the auth migration'). Required in --cli "
                             "mode; menu-bar mode prompts via modal if omitted.")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="bypass the system requirements check "
                             "(arch / macOS / RAM / disk). Don't unless "
                             "you know what you're doing.")
    parser.add_argument("--no-battery-check", action="store_true",
                        help="run even when on battery, regardless of "
                             "charge level. Default: pause < 30%%, quit "
                             "< 15%%.")
    args = parser.parse_args()

    # Pre-flight: refuse to start on machines that can't run this.
    # Cheap (~10 ms); fails fast with a modal explaining what's missing.
    if not args.skip_preflight:
        pre = run_preflight()
        if not pre.ok:
            msg = pre.summary_for_modal()
            print(msg, file=sys.stderr)
            if not args.cli:
                # Modal for menu-bar users — they wouldn't see stderr.
                try:
                    from AppKit import NSApplication  # type: ignore[import-not-found]
                    import rumps  # type: ignore[import-not-found]
                    NSApplication.sharedApplication().setActivationPolicy_(1)
                    NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
                    rumps.alert(title="Uncle Fu can't run here", message=msg)
                except Exception:
                    pass
            return 1

    debug_log_path = args.debug_log
    if args.debug and debug_log_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_log_path = _debug_root() / f"session_{ts}.log"

    personality = get_personality(args.personality)
    if args.voice:
        personality = personality.model_copy(update={"voice": args.voice})

    debug_log = None
    if debug_log_path is not None:
        debug_log = DebugLog.open(
            debug_log_path,
            personality=personality,
            model=args.model,
            webcam_interval_s=args.webcam_interval,
            screen_interval_s=args.screen_interval,
            director_interval_s=args.director_interval,
        )

    if args.cli:
        focus = (args.focus or "").strip()
        if not focus:
            parser.error("--cli mode requires --focus 'what you're working on'")
        # NullSpeaker only with --cli --mute together (skip Qwen load
        # entirely in fully-silent headless mode).
        inner: Speaker
        if args.mute:
            inner = NullSpeaker()
        else:
            inner = QwenSpeaker(model_id=args.qwen_model)
        mutable = MutableSpeaker(inner, muted=args.mute)
        vlm = MlxVlmClient(model_id=args.model)
        # In CLI we DO block on classification — no menu bar to show a
        # loading state, and headless users tail the terminal anyway.
        # Also gives us a chance to error out cleanly on gibberish before
        # spawning threads.
        print(f"classifying focus (waits for Gemma to be ready)…", flush=True)
        relevance = classify_focus(vlm, focus)
        if not relevance.valid:
            parser.error(
                f"--focus rejected by classifier ({relevance.reason!r}). "
                f"Try a more concrete focus."
            )
        print(
            f"  → screen sensor {'ON' if relevance.screen else 'OFF'}  "
            f"({relevance.reason})",
            flush=True,
        )
        try:
            run_headless(
                personality=personality,
                speaker=mutable,
                vlm=vlm,
                focus=focus,
                relevance=relevance,
                webcam_interval_s=args.webcam_interval,
                screen_interval_s=args.screen_interval,
                director_interval_s=args.director_interval,
                min_gap_s=args.min_gap,
                post_speech_cooldown_s=args.cooldown,
                debug_log=debug_log,
                battery_check=not args.no_battery_check,
            )
        except KeyboardInterrupt:
            print("\nstopped.")
        return 0

    # Menu bar mode (default).
    #
    # Boot sequence:
    #   1. NSApplication accessory mode (so modals + the menu bar work)
    #   2. Permission gate (camera + screen recording) — blocks until
    #      authorised or user gives up. We do this BEFORE any worker
    #      spawns: a permission prompt mid-download caused the focus
    #      modal to loop and pegged the GPU.
    #   3. Construct workers (vlm, speaker) — downloads start now in
    #      background threads.
    #   4. Hand a runner_factory closure to MenuBarApp; the app owns
    #      the focus modal + classifier + runner build. Menu bar
    #      appears immediately and shows download progress.
    #
    # Activation gotcha: a Python script launched from a terminal inherits
    # a Prohibited activation policy by default — modals appear but
    # keystrokes still go to the launching terminal (Ghostty, iTerm, …).
    # We bump to Accessory (menu-bar app, no Dock icon) and
    # activateIgnoringOtherApps so prompts steal focus the way users
    # expect. rumps.App.run() does the same activate call later for its
    # own menu interactions.
    from AppKit import NSApplication  # type: ignore[import-not-found]
    _NS_APP_POLICY_ACCESSORY = 1  # NSApplicationActivationPolicyAccessory
    nsapp = NSApplication.sharedApplication()
    nsapp.setActivationPolicy_(_NS_APP_POLICY_ACCESSORY)
    nsapp.activateIgnoringOtherApps_(True)

    # Permission gate. Returns False if the user denies / quits — we
    # exit cleanly without spinning up any model workers.
    from .permissions import run_permission_gate
    if not run_permission_gate(personality_name=personality.display_name):
        print("permissions not granted. bye.", flush=True)
        return 0

    # Workers can now spawn safely — TCC will allow capture without
    # any further prompts. Both report .is_ready=False until their
    # respective MLX models finish loading; the menu bar's BOOT phase
    # polls these flags via the refresh tick.
    vlm = MlxVlmClient(model_id=args.model)

    inner = QwenSpeaker(model_id=args.qwen_model)
    mutable = MutableSpeaker(inner, muted=args.mute)

    # Closure that knows how to build the runner once the user has
    # picked a focus. MenuBarApp calls this from
    # _collect_focus_and_start_session with the placeholder Relevance;
    # the runner's apply_relevance() then re-shapes prompts + screen
    # sensors when the background classifier finishes.
    def _runner_factory(focus: str, relevance: Relevance):
        return build_and_start(
            personality=personality,
            speaker=mutable,
            vlm=vlm,
            focus=focus,
            relevance=relevance,
            webcam_interval_s=args.webcam_interval,
            screen_interval_s=args.screen_interval,
            director_interval_s=args.director_interval,
            min_gap_s=args.min_gap,
            post_speech_cooldown_s=args.cooldown,
            debug_log=debug_log,
            battery_check=not args.no_battery_check,
        )

    app = MenuBarApp(
        vlm=vlm,
        mutable_speaker=mutable,
        personality=personality,
        runner_factory=_runner_factory,
    )
    try:
        app.run()
    finally:
        vlm.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
