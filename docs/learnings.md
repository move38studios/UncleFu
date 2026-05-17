# Learnings

Things that bit us, with the fix that worked. Append-only; don't relitigate.

---

## Webcam capture (macOS / pyobjc)

### The default camera might be black

`AVCaptureDevice.defaultDeviceWithMediaType_(AVMediaTypeVideo)` returns whatever macOS considers default. Enumerate explicitly via `AVCaptureDeviceDiscoverySession` and prefer `AVCaptureDeviceTypeBuiltInWideAngleCamera`. We saw the MBP enumerate both a wide-angle camera and a `DeskViewCamera`; either could end up first.

### AVCapturePhotoOutput default codec is HEIC, not JPEG

First webcam smoke test gave a 21 KB "valid" JPEG that was actually all black. Cause: macOS's default photo codec is HEIC, and round-tripping through Pillow's HEIC decoder produced garbage. Fix:

```python
fmt = {AVF.AVVideoCodecKey: AVF.AVVideoCodecTypeJPEG}
settings = AVF.AVCapturePhotoSettings.photoSettingsWithFormat_(fmt)
```

Then `photo.fileDataRepresentation()` returns real JPEG bytes — write to disk directly, no Pillow.

### Sensor needs 1–2 s to settle

Snapping a photo immediately after `session.startRunning()` returns underexposed frames. We pump the run loop for 2 s after `session.isRunning()` is true. Don't shorten this without testing — overall cycle time is dominated by VLM inference anyway.

### Permission attribution is per-parent-process

macOS attributes camera permission to the binary that owns the request, but the *prompt* targets the parent process (your terminal). Practical consequence: Claude Code's own Bash tool runs in a different parent than the user's terminal — even after the user grants permission, our Bash invocations still fail. We hand the script to the user to run from their terminal during dev.

### `NSKVONotifying_AVCapturePhotoOutput not linked` warning is harmless

Stderr noise from pyobjc's KVO bridging. Ignore.

### pyobjc 12: `dispatch_queue_create` is not in `Foundation`

If we ever switch to `AVCaptureVideoDataOutput` (sample buffer delegate, requires a `dispatch_queue_t`), we'll have to find it elsewhere — `Foundation` doesn't export it in pyobjc 12.1. `AVCapturePhotoOutput` sidesteps this entirely, which is why we use it.

---

## Screen capture

`mss` is fine. `mss.mss()` is deprecated in favor of `mss.MSS()` — cosmetic, but quiet the warning when we graduate the smoke test.

---

## VLM (Gemma 4 E4B via in-process mlx-vlm)

### `rumps.App.icon` setter hardcodes 20×20

rumps' `_nsimage_from_file` (line 127 in rumps.py at v0.4.0) calls
`image.setSize_((20, 20) if dimensions is None else dimensions)` every
time, but `App.icon = path` never passes a `dimensions` arg. So every
menu-bar icon ends up at 20pt logical = 40px @ retina, period. For our
88×88 character sprites that destroyed expression detail to the point
we couldn't see the icon change between states — looked like a static
icon. Took live testing to catch.

Fix: in `_apply_icon`, construct the NSImage ourselves, call
`setSize_((22, 22))` (22pt = 44px @ retina = clean 2:1 from 88), and
poke it directly via `_nsapp.setStatusBarIcon()` rather than going
through `App.icon`. Keep `_icon` / `_icon_nsimage` private attrs in
sync so any other rumps code reading them still works.

If we ever upgrade rumps and they expose a proper size parameter on
the App icon setter, we can drop the manual path.

### `rumps.notification` doesn't work outside an .app bundle

`rumps.notification(title=..., subtitle=..., message=...)` crashes with

```
RuntimeError: Failed to setup the notification center. This issue
occurs when the "Info.plist" file cannot be found or is missing
"CFBundleIdentifier".
```

when we run via `uv run python -m unclefu`. The notification center
needs a bundle identifier to register against; a uv-launched Python
process has none. rumps' suggested workaround (`PlistBuddy` writing
`Info.plist` next to the venv binary) is brittle — venv rebuilds wipe
it, and it confuses other tools.

**Fix when running unpackaged:** use `osascript -e 'display notification
… with title … subtitle …'`. AppleScript doesn't care about bundle
identity. Helper in `__main__._macos_notify`. Best-effort, never raises.

Inside the packaged `.app` bundle (Info.plist carries `CFBundleIdentifier`),
`rumps.notification` works. `osascript` is fine to keep either way and
removes the bundle dependency entirely.

### Don't name attributes `_stop` on a `threading.Thread` subclass

`threading.Thread` has a private `_stop()` method that `join()` calls
internally when the thread is already dead — it's part of the tstate
lock cleanup path. If you subclass `Thread` and add `self._stop =
threading.Event()`, you shadow that method, and the next `join()` on
a finished thread blows up with:

```
TypeError: 'Event' object is not callable
  File ".../threading.py", line 1171, in _wait_for_tstate_lock
    self._stop()
```

We hit this in `PeriodicThread` — symptom was the menu-bar Quit button
silently doing nothing (the exception killed the rumps callback). Tests
didn't catch it because the smoke-test shutdown path doesn't go through
`Thread.join()` after the thread has already returned.

Fix: name the Event anything else (`_stop_event`, `_stopping`, …).

### Pre-warm `transformers` on the main thread or workers race the import lock

When both `MlxVlmClient` (mlx-vlm) and `QwenSpeaker` (mlx-audio) spawn
their worker threads at startup, they both eventually try to
`from transformers import AutoTokenizer`. Python's import lock is
per-module: whichever thread starts importing `transformers` first
runs `transformers/__init__.py` from the top; the other thread sees a
partially-initialised `transformers` namespace where `AutoTokenizer`
isn't defined yet, and dies with:

```
ImportError: Missing dependency while loading qwen3_tts: cannot import
name 'AutoTokenizer' from 'transformers' (.../transformers/__init__.py)
```

In our case the loser was always QwenSpeaker — TTS would silently fail
to load and Director cycles would log `outcome=spoke` while no audio
ever played. Tricky because the speaker.say() interface is fire-and-
forget; the only signal was the stderr line at startup that's easy to
miss in `--debug` output.

**Fix:** force the `transformers` init on the main thread *before* any
worker spawns:

```python
# at the top of __main__.py
from transformers import AutoTokenizer as _AutoTokenizer  # noqa: F401
```

Costs ~1-2 s of synchronous startup time on the main thread (which
would otherwise have been paid by whichever worker won the race), so
no net wall-clock loss. Both workers now find the fully-constructed
module and proceed normally.

Generalisation: any time you have multiple worker threads that
eagerly-or-lazily import the same heavy third-party module
(`transformers`, `torch`, `tensorflow`, anything that does meaningful
work in `__init__.py`), pre-warm on the main thread. Python's per-module
import lock does NOT protect you from "partial module" reads across
threads.

### Multi-image generation is supported but we don't use it

`mlx-vlm`'s `generate(..., image=[paths_or_PIL])` supports multiple images in one call (with matching `num_images=N` on `apply_chat_template`). Each sensor currently sends one image at a time — if we ever want to bundle webcam + screen into one call to save overhead, the API is ready.

### Inference is the bottleneck

End-to-end on a quiet Mac with mlx-vlm + Gemma 4 E4B-4bit: ~2.3 s warm per single-image call. Webcam capture is ~4 s on top of that, so one webcam cycle is ~6–7 s. **Default loop interval is 20 s** — still has headroom but we could pull it down if needed.

### Image tokens aren't reported

mlx-vlm's `generate()` returns just a string. No `usage` object,
no `prompt_tokens` / `completion_tokens`. `ChatResult` keeps those
fields as `None` for log-shape compat. Don't use them for anything.

### Prompt sensitivity — same as before

Gemma 4 E4B still occasionally misclassifies things a stronger model
would catch (e.g., "code_editor" instead of "Cursor — main.py"
when looking at Cursor). Same prompt-engineering job as before;
nothing about the backend swap changed Gemma's behavior. Strip
leading/trailing ` ```json … ``` ` markdown fences before
`json.loads` (we do this in `parse_json_response`) — Gemma
sometimes wraps output in fences even when told not to.

### MLX streams are thread-local (same as mlx-audio)

`mlx-vlm 0.5.0` ports the same thread-local generation stream pattern
from `mlx-lm`. The model can be loaded on one thread, but `generate()`
must be called from the same thread that loaded it (or every thread
needs its own stream context, which adds bookkeeping).

Our pattern: `MlxVlmClient` owns one worker thread that does both
the `load()` and every `generate()`. Sensors and Director on other
threads submit `_Request` objects via a `queue.Queue`, get responses
back on a per-request response queue. Naturally serializes VLM calls
(which is what we want — only one mx.gpu generation at a time anyway).

### Hallucination escalation via history feedback

When we feed the model its own past narratives, it tends to trust them over what it sees in the current frame. Once it hallucinates "YouTube tutorial" in cycle 1, by cycle 3 it's "YouTube video tutorial about X" — escalating specificity, all invented.

Fix: explicit prompt rule that the current image overrides recent context. "Never carry forward an app name (e.g. 'YouTube', 'VS Code') that isn't visible in the current screen images." Helps a lot.

### rumps.Window before App.run() needs explicit NSApplication activation

If you show a `rumps.Window` (NSAlert under the hood) BEFORE `rumps.App.run()`
has been called — e.g. an at-launch focus modal — the alert appears on
screen but **keyboard focus stays in the launching terminal** (Ghostty,
iTerm, …). The user can click the text field, see it highlight, and type
into nothing.

Cause: a Python script launched from a terminal inherits
`NSApplicationActivationPolicyProhibited` by default. The modal renders
but its parent app isn't a "real" frontmost app, so focus doesn't shift.
`rumps.App.run()` does an `activateIgnoringOtherApps_(True)` call internally
(see rumps.py line 1187), which is why interactions through the menu bar
work fine — but that only happens once `run()` is reached.

Fix: before showing any modal at launch, do it ourselves.

```python
from AppKit import NSApplication
nsapp = NSApplication.sharedApplication()
nsapp.setActivationPolicy_(1)              # Accessory — menu-bar app, no Dock icon
nsapp.activateIgnoringOtherApps_(True)     # steal focus from the terminal
# now rumps.Window.run() takes keyboard focus correctly
```

Activation policy values: `0` Regular (Dock icon), `1` Accessory (no Dock,
can have windows), `2` Prohibited (no UI). For Uncle Fu we want
Accessory — it's a menu-bar app.

### Qwen3-TTS 0.6B has unreliable EOS — use the 1.7B variant

We initially shipped on `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit`
because it was the smallest CustomVoice quantization. It produced visible
truncation on Uncle_Fu English lines — e.g. *"YouTube. You said you were
working on the compan"* and then dead silence. Diagnostics (afplay
spawn/exit breadcrumbs) showed the WAV file itself was that short — 6.00 s
of speech, no kill — so the bug was in synthesis, not playback.

Root cause: `_generate_with_instruct` in `mlx_audio/tts/models/qwen3_tts/qwen3_tts.py`
(line 2370 in v0.4.3) hard-caps generation at:

```python
effective_max_tokens = min(max_tokens, max(75, target_token_count * 6))
```

With Uncle_Fu's actual rate of ~9 codec tokens per text token doing
English (he's a Chinese-native preset speaker), the `× 6` factor
undershoots. The comment on that line is honest: *"Cap max_tokens
based on target text length to prevent runaway generation when EOS
logit doesn't become dominant (seen especially with 0.6B model)."*

The `min(max_tokens, …)` clause means raising the `max_tokens` kwarg
can't override the cap — `max_tokens` is a *secondary* upper bound,
not an override. We tried monkey-patching `tokenizer.encode` to inflate
the perceived text length and remove the cap that way. Result: the
model spoke the full sentence then degenerated into babble —
*"code code coden coden…"* — because the 0.6B EOS logit genuinely
doesn't fire reliably. Confirmed upstream
([QwenLM/Qwen3-TTS#118](https://github.com/QwenLM/Qwen3-TTS/issues/118),
acknowledged but unfixed).

The fix that actually works: **swap to
`mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit`**. The 1.7B has
materially lower cross-lingual WER, so Uncle_Fu's codec/text token
ratio falls back inside the `× 6` margin and the existing cap stops
biting. EOS reliability is also much better on 1.7B (the 0.5%
runaway rate from issue #118 still applies, but on 1-20 word lines
we're rarely in the failure regime).

We also raised `repetition_penalty` from the default 1.05 to 1.3 —
this is the same belt-and-suspenders mlx-audio uses internally for
ICL (voice-clone) calls (see `Qwen3TTS.generate` line ~1160, where
ICL bumps the penalty to 1.5). It further discourages tail-babble
without distorting prosody.

Costs of the swap:
- Disk: ~500 MB → ~2 GB.
- RAM peak: ~5 GB → ~4 GB (1.7B is more memory-efficient post
  Blaizzy's PR #435 work — slight win).
- Synth latency: ~2-3 s → ~5-7 s per line on M-series. First line
  feels slower; subsequent ones are still well under playback time.

If we ever need to drop back to 0.6B (e.g. cheaper distribution),
the only known mitigation is text-splitting on sentence/clause
boundaries with one `generate_audio` call per chunk and WAV
concatenation — short chunks trigger EOS more reliably. Not worth
it for the disk savings.

### TTS lifecycle diagnostics

`QwenSpeaker` emits stderr breadcrumbs for every afplay spawn and every
*real* kill (interrupting a still-running process). Each spawn also
records the expected WAV duration; a daemon watcher thread per afplay
logs the actual elapsed time + exit code when it finishes:

```
[QwenSpeaker 765895.926] spawned afplay pid=19210 file=out.wav expected=3.68s text='Step outside. Sun is free.'
[QwenSpeaker 765900.382] afplay pid=19210 exited code=0 elapsed=4.46s expected=3.68s
```

`elapsed` runs a fraction longer than `expected` under normal play
(afplay spawn/teardown is ~0.5–0.8 s). Reading the breadcrumbs to
diagnose a perceived cut-off:

- `code != 0` → afplay was killed; the matching `KILL afplay … reason=…`
  line tells you which path did it (`say()`, `stop()`, or
  `worker:before-spawn`).
- `code == 0` and `elapsed ≪ expected` → audio file itself is shorter
  than expected; look at the synth path (`text=…` shows what we asked
  Qwen for, vs. the actual WAV duration).
- `code == 0` and `elapsed ≈ expected` → audio played fully; if the user
  perceives a cut-off it's a subjective issue (Uncle Fu lines really are
  short).

This is how we ruled out a real cut-off bug: every afplay exited with
code 0 and `elapsed ≈ expected`. The Uncle Fu lines are just terse.

### `rumps.App.icon` and `.title` coexist — set one, blank the other

The menu bar can show an image (`app.icon = "/path.png"`), text/emoji
(`app.title = "🧙‍♂️"`), or both side-by-side. For the PNG-or-emoji fallback
in our character system we want exactly one at a time, so the apply-icon
path sets `app.icon = None` and `app.title = emoji` for the emoji branch,
and `app.icon = path` + `app.title = ""` for the sprite branch. Forgetting
to blank the other field gives you an icon *and* a stray emoji in the bar.

### `rumps.Timer` must be held as an attribute or it stops firing

`rumps.Timer(callback, interval).start()` returns a Timer object whose lifetime controls the underlying NSTimer's reference back to the Python callback. If you don't store the returned object, Python eventually GCs it and the menu bar stops refreshing — silently, no error.

Fix: `self._timer = rumps.Timer(self._refresh, 4.0); self._timer.start()`. Same pattern for any rumps Timer.

### MLX streams are thread-local

`RuntimeError: There is no Stream(gpu, 0) in current thread.` shows up if you call `mx.eval` (or any MLX op that triggers eval, like `generate_audio`) from a worker thread that hasn't initialised a GPU stream. The model can be *loaded* on one thread and *used* on another — tensors aren't bound to streams — but each thread needs its own stream context for evaluation.

Fix: wrap the call in `with mx.stream(mx.gpu):`. This sets up the stream for the current thread for the duration of the block.

This bit us because synthesis happens on the Director thread (called via the Intervener), not the main thread. The mlx-audio CLI works fine because it's all on the main thread.

### Qwen3-TTS via mlx-audio is the TTS path on Apple Silicon

- Package: `mlx-audio` on PyPI (Blaizzy/mlx-audio on GitHub). MIT license.
- Models on `mlx-community/Qwen3-TTS-12Hz-*` — both 0.6B and 1.7B, in 4/5/6/8/bf16 quantisations. Base (voice clone), CustomVoice (9 preset speakers), VoiceDesign (NL voice description — not on mlx-community yet, only the upstream HF transformers version).
- 1.7B-CustomVoice-4bit is what we ship: ~2 GB on disk, ~4 GB peak RAM during synth, ~5–7 s per one-sentence line on M-series. (The 0.6B-4bit is smaller / faster but truncates lines mid-word due to an EOS bug — see the dedicated entry above.)
- Load the model once via `mlx_audio.tts.utils.load_model(path)` after `huggingface_hub.snapshot_download(repo_id=...)`, then call `mlx_audio.tts.generate.generate_audio(text=..., model=..., voice=..., save=True, output_path=..., file_prefix=...)`. Reusing the loaded model across calls is essential — re-loading per call would burn ~5 s every speech.
- Play the resulting WAV with `afplay` as a non-blocking subprocess; the same `terminate()` pattern as our old `say` wrapper.
- The 9 CustomVoice presets: Ryan, Aiden (English-native); Vivian, Serena, Uncle_Fu, Dylan, Eric (Chinese-native with various dialects); Ono_Anna (Japanese-native); Sohee (Korean-native). Only Ryan, Aiden are clean English. Uncle_Fu speaks English with a Chinese accent that lands well as a "seasoned mentor".

### Webcam capture on a worker thread needs the main thread to pump NSRunLoop

`AVCaptureSession` and `AVCapturePhotoOutput` can be *called* from any thread, but the capture pipeline needs the **main thread's NSRunLoop** to be alive to actually function. If the main thread is just `time.sleep`-ing, the camera pipeline stalls and the photo callback never fires (we time out waiting for a frame even though everything looks correctly wired). Symptom: webcam sensor cycles all log `RuntimeError: Timed out waiting for photo callback`, while screen sensors are fine.

Fix: in `runtime/runner.py`, the main thread pumps the run loop in small slices:

```python
from Foundation import NSDate, NSRunLoop
while True:
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.25))
```

Short slices (0.25 s) keep `KeyboardInterrupt` responsive. Same pattern any future Cocoa-based UI (rumps, NSStatusItem) will already need.

### "Voice" prompts → poetry. WHAT-TO-SAY prompts → useful lines.

A personality fragment that describes **how the character sounds** ("cryptic, weird, strange word choices") makes the model dutifully produce on-voice but useless lines like *"The wise words are buffering"* or *"A chorus of echoes"*. Personality fragments should instead describe **what the character does** — "roasts the user because they care, names the specific thing on screen, tells them what to do about it." Reinforce in the Director system prompt with a WHAT-TO-SAY section carrying concrete GOOD/BAD example lines; abstract "be specific" rules don't land. Forbid specific bad words explicitly ("no 'siren song', no 'chorus', no 'wisdom buffering'") — the model treats those as positive constraints.

### Throttle tuning

Min-gap 60 s + post-speech-cooldown 90 s is the default. Both are CLI-tunable via `--min-gap` / `--cooldown`. Higher values (90/180 s) make the model speak roughly once every 3 minutes — calmer for long sessions, but a 5-minute first run can feel broken because you hear one line and never another.

### Speak-or-shut-up bias is hard to set

Initial "DO NOT speak every cycle. Silence is fine." prompt made the model so conservative it stayed silent even when text on screen literally said "SAY SOMETHING". Be careful with absolute negative phrasing in prompts. The host already throttles, so the prompt should err slightly toward speaking when uncertain — let the throttle be the safety net, not the prompt.

---

## Project setup

- Python 3.12 via `uv`. Avoid 3.14 — wheel availability for some deps (pyobjc, pillow) is patchy at the time of writing.
- Don't install `pyobjc` (the meta-package, ~25 frameworks). Pull only the framework packages we actually use.
