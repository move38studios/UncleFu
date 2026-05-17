"""Run a function on a fixed interval in its own thread, until stopped.

Catches exceptions inside the function (so a transient VLM error doesn't kill
the thread) and yields them via an `on_error` callback so the runner can
record them. Sleep uses an Event for instant shutdown.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class PeriodicThread(threading.Thread):
    def __init__(
        self,
        name: str,
        interval_s: float,
        fn: Callable[[], None],
        *,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.interval_s = float(interval_s)
        self.fn = fn
        self.on_error = on_error
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.time()
            try:
                self.fn()
            except Exception as e:
                # Threads never die mid-session. Either hand the exception to
                # on_error or print + continue. KeyboardInterrupt is a
                # BaseException, not Exception, so it still propagates.
                if self.on_error is not None:
                    try:
                        self.on_error(e)
                    except Exception:
                        pass
                else:
                    import sys
                    import traceback
                    print(
                        f"[{self.name}] uncaught {type(e).__name__}: {e}",
                        file=sys.stderr, flush=True,
                    )
                    traceback.print_exc(file=sys.stderr)
            elapsed = time.time() - t0
            wait_s = max(0.0, self.interval_s - elapsed)
            if self._stop_event.wait(timeout=wait_s):
                return

    def stop(self) -> None:
        self._stop_event.set()

    # NOTE: do NOT name the Event `self._stop` — `threading.Thread` already
    # has a private `_stop()` method that `join()` calls internally when the
    # thread is already dead (to clean up tstate locks). Shadowing it makes
    # `join()` crash with `TypeError: 'Event' object is not callable` —
    # which we hit during menu-bar Quit. The Event lives at `_stop_event`.
