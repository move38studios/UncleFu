"""Iteratively test the focus classifier prompt against a battery of
realistic focus strings. Runs every case through the real Gemma model
and reports per-case verdicts + overall accuracy.

Run:
    uv run python tests/smoke/test_focus_classifier.py

Each case asserts:
- expected `valid` (true for real focus statements, false for gibberish)
- expected `screen` (only checked when valid=true)
"""

from __future__ import annotations

import sys
import time

from unclefu.director.focus_classifier import Relevance, classify_focus
from unclefu.vlm.client import MlxVlmClient


# (focus, expected_valid, expected_screen, severity)
#   expected_screen is only checked when expected_valid=True
#   severity: "must"  → if we get this wrong the classifier is broken
#             "soft"  → ambiguous but documented default
CASES: list[tuple[str, bool, bool, str]] = [
    # ── valid + screen=true (must) ───────────────────────────────────────
    ("writing the auth migration",            True, True,  "must"),
    ("debugging the payment flow",            True, True,  "must"),
    ("coding the React component",            True, True,  "must"),
    ("studying for the LSAT online",          True, True,  "must"),
    ("drafting the team update in Notion",    True, True,  "must"),
    ("designing the landing page in Figma",   True, True,  "must"),
    ("answering pull request comments",       True, True,  "must"),
    ("doing my taxes in TurboTax online",     True, True,  "must"),
    ("watching the new Marvel movie",         True, True,  "must"),
    ("writing chapter 4 of my novel",         True, True,  "must"),
    # ── valid + screen=false (must) ──────────────────────────────────────
    ("tuning my piano",                       True, False, "must"),
    ("cooking dinner",                        True, False, "must"),
    ("meditating",                            True, False, "must"),
    ("stretching and back exercises",         True, False, "must"),
    ("reading 'War and Peace' (paperback)",   True, False, "must"),
    ("going for a run",                       True, False, "must"),
    ("drawing in my sketchbook",              True, False, "must"),
    ("playing chess with my grandfather",     True, False, "must"),
    ("writing in my paper journal",           True, False, "must"),
    ("practicing scales on the guitar",       True, False, "must"),
    # ── ambiguous valid with documented default (soft) ───────────────────
    ("studying for the LSAT",                 True, True,  "soft"),
    ("reading a book",                        True, True,  "soft"),
    ("writing in my journal",                 True, False, "soft"),
    ("thinking about the proposal",           True, True,  "soft"),
    ("practicing French",                     True, True,  "soft"),
    ("calling my mom",                        True, False, "soft"),
    # ── stress: adversarial unseen patterns (must) ───────────────────────
    ("answering emails",                      True, True,  "must"),
    ("playing Minecraft",                     True, True,  "must"),
    ("playing soccer in the park",            True, False, "must"),
    ("video call with my therapist",          True, True,  "must"),
    ("doing yoga following a YouTube video",  True, True,  "must"),
    ("doing 50 pushups",                      True, False, "must"),
    ("fixing the dishwasher",                 True, False, "must"),
    ("having lunch with my wife",             True, False, "must"),
    ("watching TV",                           True, True,  "must"),
    ("playing the piano",                     True, False, "must"),
    ("doing my taxes by hand",                True, False, "must"),
    ("writing a poem",                        True, True,  "soft"),
    ("calling Jonathan on Zoom",              True, True,  "must"),
    ("editing my podcast in Logic",           True, True,  "must"),
    ("knitting a scarf",                      True, False, "must"),
    ("rehearsing my talk in front of a mirror", True, False, "must"),
    ("doing dishes",                          True, False, "must"),
    ("scrolling through Instagram",           True, True,  "must"),
    # ── INVALID: gibberish, test inputs (must) ───────────────────────────
    ("asdfasdf",                              False, True, "must"),
    ("dsf;alksdjfa;lsdkfjasd;flkj",           False, True, "must"),
    ("x",                                     False, True, "must"),
    ("test",                                  False, True, "must"),
    ("asdf",                                  False, True, "must"),
    ("...",                                   False, True, "must"),
    (";;;;;;;;",                              False, True, "must"),
    # ── valid but vague (soft) — we should NOT reject ────────────────────
    ("work",                                  True, True,  "soft"),
    ("stuff",                                 True, True,  "soft"),
    ("vibing",                                True, True,  "soft"),
    ("thinking",                              True, True,  "soft"),
]


def main() -> int:
    print("Loading VLM (this is slow on first launch)…")
    vlm = MlxVlmClient()
    if not vlm.wait_ready(timeout_s=600):
        print("FAIL: VLM did not become ready", file=sys.stderr)
        return 2
    print("  ready.\n")

    results: list[tuple[str, bool, bool, bool, bool, str, str, str, str, float]] = []
    # focus, exp_valid, got_valid, exp_screen, got_screen, webcam_ctx, screen_ctx, reason, severity, ms
    for focus, exp_valid, exp_screen, severity in CASES:
        t0 = time.perf_counter()
        try:
            rel: Relevance = classify_focus(vlm, focus)
        except Exception as e:
            print(f"  EXC {focus!r}: {type(e).__name__}: {e}")
            results.append((focus, exp_valid, not exp_valid, exp_screen,
                            not exp_screen, "", "", f"EXC {e}", severity, 0.0))
            continue
        dt = (time.perf_counter() - t0) * 1000
        results.append((
            focus, exp_valid, rel.valid, exp_screen, rel.screen,
            rel.webcam_context, rel.screen_context, rel.reason, severity, dt,
        ))

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    must_pass = must_fail = soft_pass = soft_fail = 0
    failed_must: list[str] = []
    for focus, exp_v, got_v, exp_s, got_s, webcam_ctx, screen_ctx, reason, sev, ms in results:
        v_ok = exp_v == got_v
        # Only check screen when valid is expected — when we expect invalid,
        # we don't care what screen says (caller re-prompts anyway).
        s_ok = (got_s == exp_s) if exp_v else True
        ok = v_ok and s_ok
        mark = "✓" if ok else "✗"
        if sev == "must":
            if ok: must_pass += 1
            else:
                must_fail += 1
                failed_must.append(
                    f"{focus!r}  exp v={exp_v} s={exp_s}  got v={got_v} s={got_s}  ({reason[:40]})"
                )
        else:
            if ok: soft_pass += 1
            else:  soft_fail += 1
        print(f"{mark} {focus}")
        print(f"    valid={got_v}  screen={got_s}  ({ms:.0f}ms)")
        if got_v:
            print(f"    webcam ctx: {webcam_ctx}")
            if got_s:
                print(f"    screen ctx: {screen_ctx}")
        else:
            print(f"    reject reason: {reason}")

    total_must = must_pass + must_fail
    total_soft = soft_pass + soft_fail
    print(f"{'='*100}")
    print(f"MUST:  {must_pass}/{total_must}   "
          f"SOFT (default sanity): {soft_pass}/{total_soft}")
    if failed_must:
        print(f"\nFailed MUST cases:")
        for f in failed_must:
            print(f"  - {f}")

    vlm.shutdown()
    return 0 if must_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
