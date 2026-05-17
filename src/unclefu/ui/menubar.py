"""Menu bar shell.

rumps.App owns the main thread (which keeps NSApplication's run loop alive
for AVFoundation). The app boots into a lifecycle state machine so the
menu bar is visible immediately even before model downloads / focus
collection are done.

Boot phases (driven by the 1 Hz refresh tick):

    BOOT          downloads + model loads in flight
                  title: '📥 Gemma 4 VLM 23% (1.2/5.2 GB)'
                  menu : Quit only
       │
       ▼          (both workers report .is_ready)
    AWAITING_FOCUS
                  title: '🧙 ready'
                  pops the focus modal on the main thread
       │
       ▼          (modal returns valid focus)
    CLASSIFYING
                  title: '⏳ classifying focus…'
                  background classifier runs; on result we build the
                  runner via runner_factory and stash the handle
       │
       ▼
    RUNNING       full menu + character icon + Director loop alive
                  icon state machine: idle ↔ talking ↔ <expression>

If a sprite PNG exists at assets/personalities/<key>/<expression>.png we
use it as the menu bar icon. Otherwise the personality's emoji fallback
is rendered as the title.
"""

# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import random
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import rumps  # type: ignore[import-not-found]

from ..director.focus_classifier import Relevance, classify_focus
from ..intervene.intervener import Intervener, Outcome
from ..personalities import Personality
from ..runtime.runner import RunnerHandle
from ..tts.speaker import MutableSpeaker
from ..vlm.client import MlxVlmClient
from ..vlm.schema import Decision, Expression


# Boot lifecycle phases. String values so they show up sensibly in logs
# and debug prints.
_PHASE_BOOT = "boot"
_PHASE_AWAITING_FOCUS = "awaiting_focus"
_PHASE_CLASSIFYING = "classifying"
_PHASE_RUNNING = "running"


_REFRESH_S = 1.0           # icon state machine + last-line refresh cadence
_EXPRESSION_HOLD_S = 4.0   # how long to hold the Director's expression after a line
_IDLE_SWAP_S = 20.0        # cadence for rotating between idle poses (gives life
                           # without competing with real expression changes)
# Set of valid expression names — used to defensively coerce a stray DB
# value to "idle" rather than trust whatever string came back.
_EXPRESSION_NAMES: frozenset[str] = frozenset({
    "idle", "talking", "disapproving", "concerned", "smirk", "approving", "alarmed",
})


def ask_focus(
    personality: Personality,
    *,
    initial: str = "",
    rejection: str | None = None,
) -> str | None:
    """Block on a modal asking the user what they're focusing on this session.

    Returns the stripped focus string, or None if the user cancelled (which
    the caller should treat as "don't start a session"). If `rejection` is
    given, it's prepended to the prompt — used on re-prompt after the
    classifier marked the previous attempt as invalid.

    Must be called AFTER a rumps.App / NSApplication has been instantiated,
    so the modal has an app context to attach to.
    """
    base_msg = (
        f"{personality.display_name} is going to keep you on task.\n\n"
        "What are you focusing on for this session?\n"
        "(e.g. 'writing the auth migration', 'studying for the LSAT')"
    )
    prompt_msg = f"{rejection}\n\n{base_msg}" if rejection else base_msg
    window = rumps.Window(
        message=prompt_msg,
        title=f"Uncle Fu — focus check",
        default_text=initial,
        ok="Start",
        cancel="Quit",
        dimensions=(360, 80),
    )
    response = window.run()
    if not response.clicked:
        return None
    text = (response.text or "").strip()
    if not text:
        # Re-prompt once if they hit OK with an empty box.
        return ask_focus(personality, initial="")
    return text


@dataclass
class MenuBarApp:
    """Menu-bar shell with a 4-phase boot lifecycle.

    Constructed before model downloads complete. Takes the workers
    (vlm + speaker) directly + a factory that knows how to build a
    runner once focus is collected. handle starts None and is filled
    in by `_start_session` after the user picks a focus.
    """

    # ── required: the pieces we need to boot ──────────────────────────
    vlm: MlxVlmClient
    mutable_speaker: MutableSpeaker
    personality: Personality
    # Called once at the end of the boot sequence to build the runner.
    # The callable captures all the per-session knobs (intervals, db
    # path, etc.) that __main__ knows about.
    runner_factory: Callable[[str, Relevance], RunnerHandle]

    # ── filled in as we transition through phases ─────────────────────
    handle: RunnerHandle | None = None
    _phase: str = _PHASE_BOOT
    _app: rumps.App = None  # type: ignore[assignment]
    _talk_item: rumps.MenuItem = None  # type: ignore[assignment]
    _last_line_item: rumps.MenuItem = None  # type: ignore[assignment]
    _mute_item: rumps.MenuItem = None  # type: ignore[assignment]
    _timer: rumps.Timer = None  # type: ignore[assignment]
    _lock: threading.Lock = None  # type: ignore[assignment]
    # Icon state machine for the RUNNING phase. Director decisions /
    # click-to-talk requests stage an expression + line into
    # `_pending_*`; the rising edge of speaker is_speaking() flushes
    # the staged values onto the visible UI. This delay is intentional:
    # the user shouldn't see the icon flip seconds before they hear
    # Uncle Fu. Falling edge starts the post-speech hold.
    _expression: Expression = "idle"
    _expression_hold_until: float = 0.0
    _pending_expression: Expression | None = None
    _pending_line: str | None = None
    _was_speaking: bool = False
    _last_seen_decision_ts: float | None = None
    # Idle pose rotation pool.
    _idle_pool: list[Path] = field(default_factory=list)
    _current_idle_path: Path | None = None
    _idle_swap_at: float = 0.0
    # Set when AWAITING_FOCUS is in flight so we don't pop the modal
    # twice if the refresh tick fires while it's already on screen.
    _focus_modal_open: bool = False

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._idle_pool = self.personality.idle_sprite_paths()

        # CFBundleName is "Uncle Fu" in the packaged app; rumps wants
        # a name for the NSStatusItem. Use the personality's name for
        # consistency when running unpackaged.
        self._app = rumps.App(
            self.personality.display_name,
            quit_button=None,  # type: ignore[arg-type]
        )

        # Boot-phase menu: nothing actionable yet, just a Quit. Full
        # menu is wired by _promote_to_running() once we have a handle.
        self._app.menu = [
            rumps.MenuItem("Quit", callback=self._on_quit),
        ]

        # MUST hold a reference — without it Python GCs the Timer wrapper
        # and the underlying NSTimer eventually stops firing, leaving the
        # menu bar frozen on the last-seen icon.
        self._timer = rumps.Timer(self._refresh, _REFRESH_S)
        self._timer.start()

    def _promote_to_running(self) -> None:
        """Called by _start_session once the handle exists. Swaps the
        boot-mode menu out for the full running menu and sets the
        initial character icon."""
        assert self.handle is not None
        self._apply_icon("idle")

        self._talk_item = rumps.MenuItem(
            f"👋 Talk to me, {self.personality.display_name}",
            callback=self._on_talk_to_me,
        )
        self._last_line_item = rumps.MenuItem("(no line spoken yet)")
        self._last_line_item.set_callback(None)
        self._mute_item = rumps.MenuItem(
            "Mute" if not self.mutable_speaker.muted else "Unmute",
            callback=self._on_toggle_mute,
        )

        self._app.menu.clear()  # wipe the boot-mode Quit
        self._app.menu = [
            self._talk_item,
            None,
            self._last_line_item,
            None,
            self._mute_item,
            rumps.MenuItem("Change focus…", callback=self._on_change_focus),
            None,
            rumps.MenuItem("Open debug folder", callback=self._open_debug),
            rumps.MenuItem("Open DB folder", callback=self._open_db),
            None,
            rumps.MenuItem("Quit", callback=self._on_quit),
        ]
        self._phase = _PHASE_RUNNING

    # ---- icon state ----

    def _apply_icon(self, expression: Expression) -> None:
        """Set the menu bar icon, preferring a PNG sprite when available.

        Idle has a special path: if the personality has multiple idle
        sprites (idle.png, idle_2.png, …) we rotate between them every
        _IDLE_SWAP_S seconds for a sign-of-life effect. All other
        expressions render the single corresponding sprite.

        Sprite path bypasses rumps' `App.icon` setter because that setter
        unconditionally calls `image.setSize_((20, 20))` (rumps.py line
        127), forcing every NSImage to 20pt logical = 40px retina. Our
        88×88 sprites scaled to 40px lose the per-expression detail
        (eyes, hand position) — making it look like the icon never
        changes. We construct the NSImage at 22pt logical = 44px retina
        (a clean 2:1 downsample from 88), set it via the same
        NSStatusItem path rumps uses (`_nsapp.setStatusBarIcon`), and
        keep the public `_icon` / `_icon_nsimage` attrs in sync so
        future calls into rumps still see consistent state.
        """
        sprite = self._pick_idle_sprite() if expression == "idle" else (
            self.personality.sprite_path(expression)
        )
        if sprite is not None:
            from AppKit import NSImage  # type: ignore[import-not-found]
            nsimg = NSImage.alloc().initByReferencingFile_(str(sprite))
            nsimg.setSize_((22, 22))
            self._app._icon = str(sprite)
            self._app._icon_nsimage = nsimg
            nsapp = getattr(self._app, "_nsapp", None)
            if nsapp is not None:
                try:
                    nsapp.setStatusBarIcon()
                except Exception:
                    # rumps hasn't fully started; setter is silently
                    # idempotent until then. Next refresh tick re-applies.
                    pass
            self._app.title = ""
        else:
            self._app.icon = None
            self._app.title = self.personality.emoji_for(expression)

    def _pick_idle_sprite(self) -> Path | None:
        """Choose an idle sprite from the personality's pool, rotating
        every _IDLE_SWAP_S seconds. Single-sprite pool (or none) → no
        rotation. Returns None if no idle sprites at all (caller falls
        back to emoji)."""
        if not self._idle_pool:
            # Fall back to the canonical idle.png if it exists (for
            # personalities that haven't populated a pool).
            return self.personality.sprite_path("idle")
        now = time.time()
        if self._current_idle_path is None or now >= self._idle_swap_at:
            # Re-roll. random.choice is fine; we accept the chance of
            # picking the same one twice in a row — it's subtle enough
            # that strict no-repeat machinery isn't worth the complexity.
            self._current_idle_path = random.choice(self._idle_pool)
            self._idle_swap_at = now + _IDLE_SWAP_S
        return self._current_idle_path

    def _refresh(self, _sender) -> None:
        # ── BOOT: workers still loading models ────────────────────────
        if self._phase == _PHASE_BOOT:
            if self._workers_ready():
                self._phase = _PHASE_AWAITING_FOCUS
                # Fall through to AWAITING_FOCUS branch this same tick.
            else:
                self._app.icon = None
                self._app.title = self._loading_title()
                return

        # ── AWAITING_FOCUS: models ready, pop the focus modal ─────────
        if self._phase == _PHASE_AWAITING_FOCUS:
            self._app.icon = None
            self._app.title = f"🧙 ready"
            if not self._focus_modal_open:
                self._focus_modal_open = True
                # ask_focus blocks. While it's up the refresh tick won't
                # fire again. On return we transition to CLASSIFYING.
                try:
                    self._collect_focus_and_start_session()
                finally:
                    self._focus_modal_open = False
            return

        # ── CLASSIFYING: background classifier running ────────────────
        if self._phase == _PHASE_CLASSIFYING:
            # _start_session sets handle and self._phase = RUNNING when
            # classification completes. Until then, show a loading state.
            if self.handle is not None and not self.handle.runner._classify_pending:
                self._promote_to_running()
            else:
                self._app.icon = None
                self._app.title = "⏳ classifying focus…"
                return

        # ── RUNNING: full menu + character icon + Director loop ───────
        assert self.handle is not None
        now = time.time()
        speaking = self._speaker_is_speaking()

        # Battery monitor asked us to quit (battery critical) — do it
        # cleanly from the main thread.
        if self.handle.runner.quit_requested:
            self.handle.shutdown()
            rumps.quit_application()
            return

        # If the background classifier rejected a mid-session focus
        # change, pop the focus modal here (on the main thread).
        rejection = self.handle.runner._focus_rejection
        if rejection is not None:
            self.handle.runner._focus_rejection = None  # clear before showing
            self._handle_focus_rejection(rejection)
            return

        # Paused on battery → show a dim sleep indicator instead of the
        # character. Character animation, click-to-talk, etc. all keep
        # working — only the sensors + Director are paused.
        if self.handle.runner.paused:
            self._app.icon = None
            pct = self.handle.runner.battery.pct if self.handle.runner.battery else "?"
            self._app.title = f"💤 {pct}%"
            return

        # ── stage: new Director decision? ────────────────────────────────
        # Don't apply yet — defer to the speaker rising edge so the icon
        # change is in sync with the audio. TTS synth takes 5-7 s after
        # the Director returns; without this delay the icon flashed
        # `disapproving`, then idled, then briefly showed `talking`
        # when audio finally started. Felt broken.
        latest = self.handle.log.latest_decision()
        if latest is not None:
            ts, outcome, msg, expression = latest
            if (
                self._last_seen_decision_ts != ts
                and outcome == Outcome.SPOKE.value
                and msg
            ):
                self._last_seen_decision_ts = ts
                self._pending_expression = (
                    expression if expression in _EXPRESSION_NAMES else "idle"
                )  # type: ignore[assignment]
                self._pending_line = msg

        # ── apply: speaker rising edge → swap to pending expression ──────
        if speaking and not self._was_speaking:
            if self._pending_expression is not None:
                self._expression = self._pending_expression
                self._pending_expression = None
            if self._pending_line is not None:
                self._last_line_item.title = f'"{self._pending_line[:80]}"'
                self._pending_line = None
            # Reset the hold timer — we'll restart it on the falling edge.
            self._expression_hold_until = 0.0

        # ── apply: speaker falling edge → start the post-speech hold ─────
        if not speaking and self._was_speaking:
            if self._expression != "idle":
                self._expression_hold_until = now + _EXPRESSION_HOLD_S

        self._was_speaking = speaking

        # ── render ───────────────────────────────────────────────────────
        with self._lock:
            if speaking:
                # Show the expression (not the generic 💬 talking emoji)
                # while audio plays — the character's mood is the point.
                # Fall back to "talking" if we somehow have no expression
                # (e.g. click-to-talk fired before pending was staged).
                self._apply_icon(
                    self._expression if self._expression != "idle" else "talking"
                )
            elif now < self._expression_hold_until and self._expression != "idle":
                self._apply_icon(self._expression)
            else:
                self._expression = "idle"
                self._apply_icon("idle")

    def _speaker_is_speaking(self) -> bool:
        try:
            return self.mutable_speaker.is_speaking()
        except Exception:
            return False

    # ---- boot-phase helpers ----

    def _workers_ready(self) -> bool:
        """True once both model workers have loaded their MLX models.
        Safe to call before handle exists — we read directly from the
        workers we hold."""
        if not self.vlm.is_ready:
            return False
        # MutableSpeaker wraps the actual QwenSpeaker; the readiness
        # flag lives on the inner. NullSpeaker (--mute --cli only) is
        # always considered ready since it has no model to load.
        inner = getattr(self.mutable_speaker, "_inner", self.mutable_speaker)
        return bool(getattr(inner, "is_ready", True))

    def _loading_title(self) -> str:
        """Boot-phase title. Download progress on first launch
        ('📥 Gemma 4 VLM 23% (1.2/5.2 GB)'); generic '⏳ loading…' once
        files are present but the model object is still being constructed."""
        from ..vlm.download_progress import PROGRESS
        s = PROGRESS.status()
        if s["phase"] == "downloading":
            label = s["active_label"] or "models"
            return (
                f"📥 {label} {s['pct']}% "
                f"({s['current_gb']:.1f}/{s['target_gb']:.1f} GB)"
            )
        return "⏳ loading models…"

    def _collect_focus_and_start_session(self) -> None:
        """Pop the focus modal on the main thread. On submit, build the
        runner via runner_factory (with placeholder Relevance) and spawn
        a background classifier. On classifier completion runner-side
        machinery calls apply_relevance and flips _classify_pending,
        which the refresh tick picks up to promote to RUNNING."""
        focus = ask_focus(self.personality)
        if focus is None:
            # User picked Quit on the focus modal.
            print("no focus, no session. bye.", flush=True)
            rumps.quit_application()
            return

        # Build the runner with a placeholder Relevance — sensors won't
        # have focus context yet; the background classifier corrects
        # both on completion (apply_relevance recomposes prompts +
        # spins screen sensors up/down as needed).
        self.handle = self.runner_factory(focus, Relevance.placeholder())
        self._phase = _PHASE_CLASSIFYING

        # Background classifier: if it rejects, set _focus_rejection so
        # the running-phase refresh tick re-pops the modal. If it
        # succeeds, apply_relevance flips _classify_pending and the
        # refresh tick promotes to RUNNING.
        def _classify_in_background() -> None:
            try:
                rel = classify_focus(self.vlm, focus)
            except Exception as e:
                print(f"  ✗ classifier crashed: {type(e).__name__}: {e}",
                      flush=True)
                assert self.handle is not None
                self.handle.runner._classify_pending = False
                return
            assert self.handle is not None
            self.handle.runner._classify_pending = False
            if not rel.valid:
                print(f"  ✗ classifier rejected focus {focus!r}: {rel.reason}",
                      flush=True)
                self.handle.runner._focus_rejection = (
                    f"'{focus[:60]}' wasn't recognised ({rel.reason}). "
                    f"Try a more concrete focus."
                )
                return
            self.handle.runner.apply_relevance(focus, rel)

        threading.Thread(
            target=_classify_in_background,
            name="initial-classifier", daemon=True,
        ).start()

    # ---- callbacks ----

    def _on_talk_to_me(self, _sender) -> None:
        """Speak a random hand-written line for this personality.

        Bypasses both the Director and the Intervener throttle — the user
        explicitly asked the character to speak, so respect that immediately.
        Per-session dedup still applies (no hearing the same line twice in
        a session); we keep trying random lines until one is fresh or we
        exhaust the click_lines list.
        """
        assert self.handle is not None
        lines = self.personality.click_lines
        if not lines:
            return
        intervener: Intervener = self.handle.runner.intervener
        # Try every line once at most — if all are used, fall silent rather
        # than repeat. With ~18 click lines per personality that's plenty of
        # button mashing before going stale.
        candidates = lines.copy()
        random.shuffle(candidates)
        for line in candidates:
            decision = Decision(
                should_speak=True,
                urgency="medium",
                message=line,
                reason="click_to_talk",
                expression="smirk",
            )
            outcome = intervener.maybe_speak(decision, now=time.time(), force=True)
            if outcome is Outcome.SPOKE:
                # Stage the expression and line — the refresh loop will
                # apply them on the speaker's rising edge so the icon
                # changes in sync with the audio (~5-7 s from now).
                self._pending_expression = "smirk"
                self._pending_line = line
                return
            if outcome is Outcome.DEDUPED:
                continue  # try a different line
            # NO_MESSAGE / MODEL_DECLINED shouldn't happen with our hand-built
            # Decision, but bail rather than loop forever.
            return

    def _on_toggle_mute(self, sender: rumps.MenuItem) -> None:
        self.mutable_speaker.muted = not self.mutable_speaker.muted
        sender.title = "Unmute" if self.mutable_speaker.muted else "Mute"
        if self.mutable_speaker.muted:
            self.mutable_speaker.stop()

    def _on_change_focus(self, _sender) -> None:
        # Wired in _promote_to_running so handle is guaranteed non-None.
        assert self.handle is not None
        new = ask_focus(self.personality, initial=self.handle.runner.focus)
        if new is None:
            return  # user cancelled; keep the existing focus
        self.handle.runner.set_focus(new)

    def _handle_focus_rejection(self, rejection: str) -> None:
        """Pop the focus modal with the rejection reason; loop until the
        user gives a valid focus or cancels (in which case the session
        continues with the placeholder Relevance — same as before they
        clicked OK on the rejected one)."""
        assert self.handle is not None
        rejection_msg: str | None = (
            f"Hmm, {self.personality.display_name} didn't catch that.\n"
            f"{rejection}"
        )
        attempt_focus = self.handle.runner.focus
        while True:
            new = ask_focus(
                self.personality,
                initial=attempt_focus,
                rejection=rejection_msg,
            )
            if new is None:
                # User cancelled — session keeps running with the
                # placeholder Relevance. They can still recover via
                # "Change focus…" later.
                return
            # set_focus blocks on classify; if invalid it logs and
            # returns without applying. We loop until valid or cancel.
            new_rel = classify_focus(self.handle.runner.vlm, new)
            if new_rel.valid:
                self.handle.runner.apply_relevance(new, new_rel)
                return
            attempt_focus = new
            rejection_msg = (
                f"Hmm, {self.personality.display_name} still didn't catch that.\n"
                f"'{new[:60]}' wasn't recognised ({new_rel.reason}). "
                f"Try something more concrete."
            )

    def _open_debug(self, _sender) -> None:
        assert self.handle is not None
        path = self.handle.debug_log.path.parent if self.handle.debug_log else None
        if path is None:
            rumps.notification(
                title="Uncle Fu",
                subtitle="No debug log this session",
                message="Re-run with --debug to enable.",
            )
            return
        subprocess.run(["open", str(path)], check=False)

    def _open_db(self, _sender) -> None:
        assert self.handle is not None
        subprocess.run(["open", str(self.handle.log.db_path.parent)], check=False)

    def _on_quit(self, _sender) -> None:
        if self.handle is not None:
            self.handle.shutdown()
        rumps.quit_application()

    def run(self) -> None:
        try:
            self._app.run()
        finally:
            if self.handle is not None:
                self.handle.shutdown()
