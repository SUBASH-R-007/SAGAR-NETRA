# Clutter sweep — what happens when the seabed stops being clean

Generated 2026-08-28T12:05:23+00:00 by `scripts/eval_clutter.py` in 471 s.

**SYNTHETIC.** Physics-simulated scenes only; no real sonar data. Treat as
a *relative* comparison between methods on identical pixels.

## Why this table exists

`docs/baseline_comparison.md` finds a tuned classical CAD baseline slightly
*ahead* of SAGAR-NETRA on the standard held-out scenes. That comparison is
confounded, and this table is the evidence.

In the scene simulator `rock_cluster` — the only natural clutter class — has
reflectivity **2.0-3.0**, the lowest of any class, while most man-made
targets sit at **4.0-8.0**. Brightness therefore *is* the man-made/natural
label for most of the catalogue, and a detector that thresholds on
brightness is handed the answer by the data generator. Real sonar offers no
such gap: a boulder and a steel drum can return comparable amplitude, which
is precisely why the problem needs shape, shadow geometry and learning.

So clutter is swept under two conditions, identical in every other respect —
same scenes, same debris, same seeds, rocks in the same positions:

- **native** — rocks keep catalogue reflectivity 2.0-3.0 — the simulator's brightness gap is intact and a threshold can exploit it.
- **matched** — each rock borrows a real target's reflectivity — brightness carries no information about whether an object is debris.

Levels are nested; every rock is a false positive by construction (truth
boxes are man-made only). Recall should stay near-flat because the debris
field never changes; what moves is precision.

## Protocol

- 12 scenes, 55 man-made truth boxes, 0.1529 km², seed base 12000.
- Classical baseline tuned on a separate split (seed base 11000): `k_sigma=0.25`, `score>=34.8`, shadow gate on (its stronger form).
- SAGAR-NETRA at its shipped 50% floor, tuned against nothing.
- TP: IoU >= 0.3, **class match not required** (localization only).
- Decoy rocks are placed clear of man-made targets so no truth box is corrupted.

## NATIVE — rocks keep catalogue reflectivity 2.0-3.0 — the simulator's brightness gap is intact and a threshold can exploit it

| extra rocks | classical P | classical R | classical F1 | SAGAR P | SAGAR R | SAGAR F1 |
|---|---|---|---|---|---|---|
| +0 | 0.882 | 0.818 | 0.849 | 0.739 | 0.618 | 0.673 |
| +6 | 0.523 | 0.836 | 0.643 | 0.304 | 0.636 | 0.412 |
| +12 | 0.358 | 0.800 | 0.494 | 0.206 | 0.600 | 0.307 |
| +24 | 0.253 | 0.818 | 0.386 | 0.121 | 0.545 | 0.199 |

Precision change from +0 to +24 rocks: **classical -0.630**, **SAGAR-NETRA -0.618**.

## MATCHED — each rock borrows a real target's reflectivity — brightness carries no information about whether an object is debris

| extra rocks | classical P | classical R | classical F1 | SAGAR P | SAGAR R | SAGAR F1 |
|---|---|---|---|---|---|---|
| +0 | 0.882 | 0.818 | 0.849 | 0.739 | 0.618 | 0.673 |
| +6 | 0.377 | 0.836 | 0.520 | 0.265 | 0.655 | 0.377 |
| +12 | 0.243 | 0.800 | 0.373 | 0.166 | 0.618 | 0.262 |
| +24 | 0.148 | 0.800 | 0.250 | 0.107 | 0.673 | 0.185 |

Precision change from +0 to +24 rocks: **classical -0.734**, **SAGAR-NETRA -0.632**.

## Reading it

- Compare the two `matched` slopes, not the headline numbers. The question
  is which method degrades more slowly when brightness stops being a
  shortcut, because that is the only condition resembling real seabed.
- `rock_cluster` is a trained hard negative for the classifier and a feature
  in the Stage-2 verifier's cue vector; the classical detector has no way to
  express "bright, shadowed, and not debris".
- If the two methods degrade alike under `matched`, the honest conclusion is
  that this simulator cannot separate them, and the claim needs real data —
  not a louder version of this table.
