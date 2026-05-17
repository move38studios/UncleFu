"""Top-level orchestrator.

Spawns one thread per sensor, plus one thread for the Director. They share:
- SessionLog (SQLite, WAL, thread-safe)
- a stdout printer for live output
- the Intervener for TTS dispatch

Shutdown: Ctrl-C → KeyboardInterrupt in main thread → stop all PeriodicThreads.
SQLite connection closes in main thread on exit.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..capture.webcam import ensure_camera_authorized
from ..director.context import ContextGatherer
from ..director.director import call_director
from ..director.focus_classifier import Relevance, classify_focus
from ..intervene.intervener import Intervener, Outcome
from ..personalities import Personality
from ..sensors.screen_sensor import ScreenSensor, discover_screen_sensors
from ..sensors.webcam_sensor import WebcamSensor
from ..storage.debug_log import DebugLog
from ..storage.session_log import SessionLog, default_db_path, error_snapshot
from ..tts.speaker import Speaker
from ..vlm.client import DEFAULT_MODEL, MlxVlmClient, VLMError
from ..vlm.schema import Decision, SensorSnapshot
from .battery import BatteryMonitor, BatteryState, PowerPolicy
from .thread import PeriodicThread

from Foundation import NSDate, NSRunLoop  # type: ignore[import-not-found]


HISTORY_PER_SOURCE_FOR_DIRECTOR = 4
RECENT_SPEECH_FOR_DIRECTOR = 6


@dataclass
class Runner:
    personality: Personality
    speaker: Speaker
    vlm: MlxVlmClient
    log: SessionLog
    session_start: float
    intervener: Intervener
    focus: str
    relevance: Relevance
    # The always-on webcam sensor instance; Runner mutates its
    # `system_prompt` on focus change so its next tick uses fresh
    # focus context.
    webcam_sensor: WebcamSensor | None = None
    webcam_interval_s: float = 30.0
    screen_interval_s: float = 12.0
    director_interval_s: float = 20.0
    debug_log: DebugLog | None = None
    context: ContextGatherer = field(default_factory=ContextGatherer)
    _print_lock: threading.Lock = field(default_factory=threading.Lock)
    # Screen sensor threads are managed dynamically — they spin up or down
    # when the focus changes. Webcam + Director threads stay always-on
    # and are tracked by RunnerHandle.
    _screen_threads: list[PeriodicThread] = field(default_factory=list)
    # Held so set_focus can mutate prompts in place.
    _screen_sensors: list[ScreenSensor] = field(default_factory=list)
    # True between session start and the first apply_relevance() call.
    # Menu bar uses this to keep the "⏳ classifying…" state up so the
    # user doesn't see a blank icon while the background classifier runs.
    _classify_pending: bool = True
    # Set by the background classifier when it rejects the focus.
    # Menu bar's refresh tick (on the main thread) sees this, pops the
    # focus modal with the rejection message, calls set_focus on the
    # user's revised input, and clears this back to None. Lets us avoid
    # depending on macOS notification permissions for recovery UX.
    _focus_rejection: str | None = None
    # Battery state — None until the BatteryMonitor has reported.
    # Menu bar reads this to show 🔋 / 💤 indicators.
    battery: BatteryState | None = None
    # True while paused (on battery + below PAUSE_PCT). Sensor + Director
    # threads stop firing meaningful work; menu bar shows 💤. Resumes
    # when back on AC.
    paused: bool = False
    # Set when the menu bar should quit the app (battery critical).
    # Main thread's refresh tick picks this up and triggers handle.shutdown.
    quit_requested: bool = False

    # ---- sensor loops ----

    def _run_webcam(self, sensor: WebcamSensor) -> None:
        if self.paused:
            return
        try:
            snap = sensor.tick()
        except Exception as e:
            self._handle_sensor_error(sensor.source, e)
            return
        self._record_snapshot(snap)

    def _run_screen(self, sensor: ScreenSensor) -> None:
        if self.paused:
            return
        try:
            snap = sensor.tick()
        except Exception as e:
            self._handle_sensor_error(sensor.source, e)
            return
        self._record_snapshot(snap)

    def _record_snapshot(self, snap: SensorSnapshot) -> None:
        self.log.record_snapshot(snap)
        self._print(
            f"[{_when(snap.ts)}] sensor:{snap.source:<10} {snap.cycle_ms:5d}ms  "
            f"{snap.description[:120]}"
        )
        if self.debug_log is not None:
            self.debug_log.write_snapshot(snap)

    def _handle_sensor_error(self, source: str, exc: BaseException) -> None:
        snap = error_snapshot(
            ts=time.time(), source=source,
            exc_type=type(exc).__name__, exc_message=str(exc),
            cycle_ms=0,
        )
        self.log.record_snapshot(snap)
        self._print(f"[{_when(snap.ts)}] sensor:{source:<10}  ERROR {type(exc).__name__}: {exc}")
        if self.debug_log is not None:
            self.debug_log.write_snapshot(snap)

    # ---- director loop ----

    def _run_director(self) -> None:
        if self.paused:
            return
        ts = time.time()
        recent_by_source: dict[str, list[SensorSnapshot]] = {}
        for source in self._known_sources():
            snaps = self.log.recent_snapshots(source, limit=HISTORY_PER_SOURCE_FOR_DIRECTOR)
            if snaps:
                recent_by_source[source] = snaps

        realworld = self.context.gather(now=ts)

        if not recent_by_source:
            # Nothing to think about yet — sensors haven't produced anything.
            self._print(f"[{_when(ts)}] director: skipping (no sensor data yet)")
            return

        recent_speech = [
            (ts - t, msg) for t, msg in self.log.recent_spoken_lines(limit=RECENT_SPEECH_FOR_DIRECTOR)
        ]

        try:
            call = call_director(
                personality=self.personality,
                vlm=self.vlm,
                focus=self.focus,
                session_seconds=ts - self.session_start,
                recent_by_source=recent_by_source,
                recent_speech=recent_speech,
                realworld=realworld,
                now=ts,
            )
        except (VLMError, RuntimeError) as e:
            self._print(f"[{_when(ts)}] director  ERROR {type(e).__name__}: {e}")
            if self.debug_log is not None:
                self.debug_log.write_director_error(ts=ts, exc=e)
            return

        decision = call.decision
        outcome = self.intervener.maybe_speak(decision, now=time.time())

        self.log.record_decision(
            ts=ts, decision=decision, outcome=outcome.value, call_ms=call.call_ms,
        )
        if self.debug_log is not None:
            self.debug_log.write_director_call(
                ts=ts, call=call, outcome=outcome.value,
            )

        self._print(_format_director_line(ts, decision, outcome, call.call_ms))

    # ---- helpers ----

    def _known_sources(self) -> list[str]:
        rows = self.log._conn.execute(
            "SELECT DISTINCT source FROM sensor_snapshot WHERE session_id=?",
            (self.log.session_id,),
        ).fetchall()
        return [r[0] for r in rows]

    # ---- screen sensor lifecycle ----

    def start_screen_sensors(self) -> None:
        """Idempotent: discover physical displays, spin up one
        PeriodicThread per display, each carrying the current composed
        screen-sensor system prompt (base + focus context). Webcam +
        Director are not touched."""
        if self._screen_threads:
            return  # already running
        from ..sensors.prompts import BASE_SCREEN_PROMPT, compose_sensor_prompt
        screen_prompt = compose_sensor_prompt(
            BASE_SCREEN_PROMPT, self.relevance.screen_context
        )
        self._screen_sensors = discover_screen_sensors(
            vlm=self.vlm, system_prompt=screen_prompt,
            default_interval_s=self.screen_interval_s,
        )
        for s in self._screen_sensors:
            t = PeriodicThread(
                name=s.source, interval_s=self.screen_interval_s,
                fn=lambda s=s: self._run_screen(s),
            )
            t.start()
            self._screen_threads.append(t)

    def stop_screen_sensors(self) -> None:
        """Idempotent: stop and join screen sensor threads. Safe to call
        when none are running. Existing snapshots in the DB are untouched."""
        if not self._screen_threads:
            return
        for t in self._screen_threads:
            t.stop()
        for t in self._screen_threads:
            t.join(timeout=2.0)
        self._screen_threads.clear()
        self._screen_sensors.clear()

    # ---- focus management ----

    def apply_relevance(self, focus: str, relevance: Relevance) -> None:
        """Apply an ALREADY-CLASSIFIED Relevance to the running session.

        Recomposes per-sensor system prompts, adjusts screen-sensor
        threads, writes the (possibly changed) focus to SQLite. Does
        NOT call the classifier — caller is expected to have done that.
        Used by:
        - the post-startup background classification (no perceived freeze)
        - set_focus(), after it classifies
        """
        cleaned = focus.strip()
        if not cleaned:
            raise ValueError("focus cannot be empty")
        focus_changed = (cleaned != self.focus)
        self.focus = cleaned
        self.relevance = relevance
        self._classify_pending = False
        if focus_changed:
            self.log._conn.execute(
                "UPDATE session SET focus=? WHERE id=?",
                (cleaned, self.log.session_id),
            )
        self._print(
            f"focus settled → {cleaned!r}  "
            f"screen={relevance.screen} ({relevance.reason})"
        )
        # Recompose sensor prompts in place so they take effect on the
        # next sensor tick without restarting threads.
        from ..sensors.prompts import (
            BASE_SCREEN_PROMPT, BASE_WEBCAM_PROMPT, compose_sensor_prompt,
        )
        if self.webcam_sensor is not None:
            self.webcam_sensor.system_prompt = compose_sensor_prompt(
                BASE_WEBCAM_PROMPT, relevance.webcam_context
            )
        # Update existing screen sensors (if any). start_screen_sensors
        # will compose fresh prompts for newly-spun-up sensors.
        new_screen_prompt = compose_sensor_prompt(
            BASE_SCREEN_PROMPT, relevance.screen_context
        )
        for s in self._screen_sensors:
            s.system_prompt = new_screen_prompt
        if relevance.screen:
            self.start_screen_sensors()
        else:
            self.stop_screen_sensors()

    def set_focus(self, new_focus: str) -> None:
        """Public mid-session focus change: classifies via Gemma, then
        applies. Blocks for ~4 s during classification."""
        cleaned = new_focus.strip()
        if not cleaned:
            raise ValueError("focus cannot be empty")
        new_rel = classify_focus(self.vlm, cleaned)
        if not new_rel.valid:
            self._print(
                f"focus change to {cleaned!r} rejected by classifier "
                f"({new_rel.reason}); keeping {self.focus!r}"
            )
            return
        self.apply_relevance(cleaned, new_rel)

    def _print(self, line: str) -> None:
        with self._print_lock:
            print(line, flush=True)

    # ---- battery policy callbacks (called by BatteryMonitor thread) ----

    def on_battery_warn(self, state: BatteryState) -> None:
        """First time we notice we're on battery (above PAUSE_PCT).
        Update the state for the menu bar; no behavior change yet."""
        self.battery = state
        self._print(
            f"⚠ on battery at {state.pct}% — Uncle Fu is GPU-heavy. "
            f"Will pause at {30}% and quit at {15}%."
        )

    def on_battery_pause(self, state: BatteryState) -> None:
        """Battery dropped below PAUSE threshold. Stop processing,
        speak a farewell line, leave the menu bar visible so the user
        can quit cleanly or plug in to resume."""
        self.battery = state
        if self.paused:
            return
        self.paused = True
        self._print(
            f"💤 battery {state.pct}% — pausing inference until you "
            f"plug in. Use 'Quit' to stop entirely."
        )
        try:
            self.speaker.say(
                "Battery low. I am taking a nap. Plug in if you want me back.",
                voice=self.personality.voice,
            )
        except Exception:
            pass

    def on_battery_resume(self, state: BatteryState) -> None:
        """Plugged back in (or battery climbed above PAUSE_PCT).
        Un-pause and let the sensor + Director threads start producing
        again on their next tick."""
        self.battery = state
        if not self.paused:
            return
        self.paused = False
        self._print(f"🔌 back on AC ({state.pct or '?'}%) — resuming.")

    def on_battery_quit(self, state: BatteryState) -> None:
        """Battery critical. Set the quit_requested flag for the menu
        bar's main-thread refresh to pick up and trigger shutdown."""
        self.battery = state
        self._print(
            f"🪫 battery {state.pct}% — too low to continue. Quitting."
        )
        try:
            self.speaker.say(
                "Battery dying. I am going now. Save your work.",
                voice=self.personality.voice,
            )
        except Exception:
            pass
        self.quit_requested = True


def _when(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _format_director_line(
    ts: float, decision: Decision, outcome: Outcome, call_ms: int
) -> str:
    spoke_marker = {
        Outcome.SPOKE: "🔊",
        Outcome.MODEL_DECLINED: "  ",
        Outcome.NO_MESSAGE: "  ",
        Outcome.THROTTLED_MIN: "🤐",
        Outcome.THROTTLED_COOLDOWN: "🤐",
        Outcome.DEDUPED: "♻️ ",
    }[outcome]
    base = f"[{_when(ts)}] director {spoke_marker} call={call_ms}ms"
    if decision.should_speak and decision.message:
        verb = "spoke" if outcome == Outcome.SPOKE else f"wanted ({outcome.value})"
        base += (
            f"\n   🗯  [{decision.urgency} · {decision.expression} · "
            f"{decision.reason}] {verb}: {decision.message}"
        )
    return base


@dataclass
class RunnerHandle:
    """Lifecycle handle for the background sensor + director threads.

    Threads are started by `build_and_start`. The main thread is free to do
    something else (pump NSRunLoop directly, or run rumps.App.run()) and
    then call `shutdown()` on exit.
    """

    runner: Runner
    log: SessionLog
    speaker: Speaker
    vlm: MlxVlmClient
    debug_log: DebugLog | None
    threads: list[PeriodicThread]
    battery_monitor: BatteryMonitor | None = None

    def shutdown(self) -> None:
        # Battery monitor first — stop polling before we kill anything
        # else, otherwise it might trigger a late pause/quit during
        # teardown.
        if self.battery_monitor is not None:
            self.battery_monitor.stop()
        # Always-on threads (webcam + Director) tracked here.
        for t in self.threads:
            t.stop()
        for t in self.threads:
            t.join(timeout=2.0)
        # Screen sensor threads (if any) are dynamically managed by Runner.
        self.runner.stop_screen_sensors()
        # If the speaker is a QwenSpeaker (or wrapped via MutableSpeaker),
        # ask it to terminate its worker cleanly.
        inner = getattr(self.speaker, "_inner", self.speaker)
        shutdown_fn = getattr(inner, "shutdown", None)
        if callable(shutdown_fn):
            shutdown_fn()
        else:
            self.speaker.stop()
        # Stop the VLM worker thread (drops the loaded model).
        self.vlm.shutdown()
        self.log.close(ended_at=time.time())
        if self.debug_log is not None:
            self.debug_log.close()


def build_and_start(
    *,
    personality: Personality,
    speaker: Speaker,
    vlm: MlxVlmClient,
    focus: str,
    relevance: Relevance,
    webcam_interval_s: float = 30.0,
    screen_interval_s: float = 12.0,
    director_interval_s: float = 20.0,
    min_gap_s: float = 60.0,
    post_speech_cooldown_s: float = 90.0,
    debug_log: DebugLog | None = None,
    db_path: Path | None = None,
    battery_check: bool = True,
) -> RunnerHandle:
    """Start the background threads; return a handle. Does NOT block.

    Caller owns the VLM client and the focus classification — so __main__
    can do both before construction and so tests can pass fakes. Webcam +
    Director always start; screen sensors only start if `relevance.screen`.
    Runner shuts everything down via `shutdown()`.
    """
    if not focus or not focus.strip():
        raise ValueError("session focus is required")
    focus = focus.strip()
    ensure_camera_authorized()

    intervener = Intervener(
        speaker=speaker, personality=personality,
        min_gap_s=min_gap_s, post_speech_cooldown_s=post_speech_cooldown_s,
    )
    session_start = time.time()
    log = SessionLog.open(
        db_path or default_db_path(),
        started_at=session_start,
        focus=focus,
        personality=personality.key,
    )

    from ..sensors.prompts import BASE_WEBCAM_PROMPT, compose_sensor_prompt
    webcam_prompt = compose_sensor_prompt(
        BASE_WEBCAM_PROMPT, relevance.webcam_context
    )
    webcam_sensor = WebcamSensor(
        vlm=vlm, system_prompt=webcam_prompt,
        default_interval_s=webcam_interval_s,
    )

    runner = Runner(
        personality=personality, speaker=speaker, vlm=vlm, log=log,
        session_start=session_start, intervener=intervener,
        focus=focus,
        relevance=relevance,
        webcam_sensor=webcam_sensor,
        webcam_interval_s=webcam_interval_s,
        screen_interval_s=screen_interval_s,
        director_interval_s=director_interval_s,
        debug_log=debug_log,
    )

    screen_status = (
        f"ON ({relevance.reason})" if relevance.screen
        else f"OFF ({relevance.reason})"
    )
    print(
        f"Uncle Fu — webcam@{webcam_interval_s}s screens@{screen_interval_s}s "
        f"director@{director_interval_s}s\n"
        f"Personality: {personality.display_name} ({personality.voice})\n"
        f"Focus: {focus}\n"
        f"VLM: {vlm.model_id}\n"
        f"Screen sensors: {screen_status}\n"
        f"DB: {log.db_path}  (session #{log.session_id})\n"
    )

    # Always-on threads: webcam + Director. Tracked by the Handle so
    # shutdown() can stop them. Screen sensors live on the Runner and
    # spin up/down as focus changes.
    threads: list[PeriodicThread] = []
    threads.append(PeriodicThread(
        name="webcam", interval_s=webcam_interval_s,
        fn=lambda: runner._run_webcam(webcam_sensor),
    ))
    threads.append(PeriodicThread(
        name="director", interval_s=director_interval_s,
        fn=runner._run_director,
    ))
    for t in threads:
        t.start()
    if relevance.screen:
        runner.start_screen_sensors()

    # Battery monitor: pause when on battery below 30%, quit below 15%.
    # Disabled with battery_check=False (CLI --no-battery-check).
    battery_monitor: BatteryMonitor | None = None
    if battery_check:
        battery_monitor = BatteryMonitor(
            on_warn=runner.on_battery_warn,
            on_pause=runner.on_battery_pause,
            on_resume=runner.on_battery_resume,
            on_quit=runner.on_battery_quit,
        )
        battery_monitor.start()

    return RunnerHandle(runner=runner, log=log, speaker=speaker, vlm=vlm,
                        debug_log=debug_log, threads=threads,
                        battery_monitor=battery_monitor)


def run(
    *,
    personality: Personality,
    speaker: Speaker,
    vlm: MlxVlmClient,
    focus: str,
    relevance: Relevance,
    webcam_interval_s: float = 30.0,
    screen_interval_s: float = 12.0,
    director_interval_s: float = 20.0,
    min_gap_s: float = 60.0,
    post_speech_cooldown_s: float = 90.0,
    debug_log: DebugLog | None = None,
    db_path: Path | None = None,
    battery_check: bool = True,
) -> None:
    """Headless / --cli entrypoint. Blocks pumping NSRunLoop until Ctrl-C."""
    handle = build_and_start(
        personality=personality, speaker=speaker, vlm=vlm,
        focus=focus, relevance=relevance,
        battery_check=battery_check,
        webcam_interval_s=webcam_interval_s,
        screen_interval_s=screen_interval_s,
        director_interval_s=director_interval_s,
        min_gap_s=min_gap_s, post_speech_cooldown_s=post_speech_cooldown_s,
        debug_log=debug_log, db_path=db_path,
    )
    print("Press Ctrl-C to stop.\n")

    # AVFoundation needs the main thread's NSRunLoop alive (see docs/learnings.md).
    try:
        while True:
            NSRunLoop.currentRunLoop().runUntilDate_(
                NSDate.dateWithTimeIntervalSinceNow_(0.25)
            )
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        handle.shutdown()
