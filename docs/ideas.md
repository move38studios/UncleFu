# Parking lot

Design ideas surfaced but not built. Not committed-to. Revisit when the
matching pain shows up.

---

## Commitment-aware perception

The Director knows the user's commitment but sensors don't. Two unresolved
ideas about whether to fix that:

### A. Pass the commitment into sensor prompts

E.g. focus="tuning my piano" → webcam sensor prompt becomes
"Is this person tuning a piano?" instead of generic "describe what you see".

- Open question: how to keep the wellness signals (posture collapse,
  alertness) we'd otherwise lose by narrowing the prompt.
- Open question: how to handle the screen sensor when the commitment is
  not screen-bound.
- Architectural concern: today sensors are dumb describers; pushing
  judgment into them couples layers we've kept separate.

User pushed back on the "leave it as the Director's job" framing, so this
needs more thought, not deletion.

### B. Pre-classify which sensors are relevant for a given commitment

Run Gemma once at session start: "Given commitment X, which sensors
matter?" Disable irrelevant ones for the session.

- Open question: hidden state — user said "watch me", now we sometimes
  aren't.
- Open question: false negatives ("just reading a book" might still need
  the webcam for wellness; "studying" might or might not need the screen).
- Lower priority than A.

### C. One-shot focus expansion at session start

Run Gemma once at session start: "User said X. In one sentence each,
what would on-task look like in (a) the webcam, (b) the screen?" Cache
the answer, splice into the Director prompt every cycle.

Compromise between A and "do nothing". Doesn't change sensor architecture,
doesn't disable anything, gives the Director a calibrated yardstick. May
become moot if A turns out to be the right architectural call.

---

## Other ideas

(Add as they come up.)
