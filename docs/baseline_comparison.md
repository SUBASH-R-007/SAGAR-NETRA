# SAGAR-NETRA vs. classical CAD — SYNTHETIC held-out scenes

Generated 2026-08-28T13:09:05+00:00 by `scripts/eval_baseline.py` in 1771 s.

**Read this before quoting any number.** Every scene is rendered by the
SAGAR-NETRA physics scene simulator — no real sonar data is involved.
Synthetic targets are easier than real debris in real clutter, so treat
these as an upper bound and, above all, as a *relative* comparison between
methods measured on identical pixels.

## What the baseline is

Rows (1) and (2) are `tridentnet/baseline.py` — a faithful reimplementation
of the threshold-and-blob computer-aided-detection scheme that side-scan
survey software used before learned detectors: per-range-column robust
background, threshold at `median + k*sigma`, morphological opening,
connected components, then area and aspect filters. Row (2) adds the one
physical cue the classical method can cheaply exploit — a required dark
region down-range of the highlight. It reimplements the *approach*, not any
particular commercial product, and it is not a product comparison.

## Protocol

- **Split A (tuning)**: 8 scenes, 33 truth boxes, seed base 11000.
- **Split B (evaluation)**: 16 scenes, 66 truth boxes, 0.2054 km², seed base 12000.
- Both splits are disjoint from detector training (seed 0) and from the
  calibration set (seed base 9000). **Every reported number is measured on
  split B; every hyperparameter is chosen on split A.**
- TP: IoU >= 0.3 with a man-made truth box, **class match not
  required**, one detection per truth (greedy, score-ranked). Hits on
  natural targets (rock clusters) or background count as FP.
- **Localization-only scoring** is a concession to the baseline: a blob
  detector emits no class, so scoring class would penalise it for a task it
  does not attempt. SAGAR-NETRA earns no credit here for classifying.
- The baseline gets **both** its knobs tuned on split A — `k_sigma` over
  [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0] and its score cut swept — against SAGAR-NETRA's single
  confidence floor. Row (4) is the **shipped** 50% floor, tuned against
  nothing at all.
- The baseline detects on **gain-corrected, pre-CLAHE** imagery, where it is
  strongest: targets peak at 8-30 sigma above background there versus
  1.7-3.2 sigma after contrast equalization.
- Stage-2 verifier checkpoint: held-out AUC 0.955.

## Results (all measured on split B)

| method | P | R | F1 | PR-AUC | FP/km² | TP | FP | classifies? | operating point |
|---|---|---|---|---|---|---|---|---|---|
| (1) classical threshold + blob | 0.904 | 0.712 | 0.797 | 0.917 | 24.34 | 47 | 5 | no | k_sigma=0.25, score>=39.7 (tuned on split A) |
| (2) classical + shadow gate | 0.841 | 0.803 | 0.822 | 0.828 | 48.69 | 53 | 10 | no | k_sigma=0.05, score>=33.4 (tuned on split A) |
| (3) SAGAR-NETRA detector only (no physics) | 0.171 | 0.652 | 0.270 | 0.569 | 1017.53 | 43 | 209 | yes | shipped floor, conf>=50% |
| (4) SAGAR-NETRA full stack | 0.759 | 0.667 | 0.710 | 0.738 | 68.16 | 44 | 14 | yes | shipped floor, conf>=50% |
| (5) SAGAR-NETRA full stack, threshold tuned | 0.643 | 0.682 | 0.662 | 0.738 | 121.71 | 45 | 25 | yes | conf>=28.7% (tuned on split A) |

> **Tuning note.** `blob_shadow` selected k_sigma=0.05 at a grid endpoint, but the F1-vs-k curve is flat there (endpoint 0.914 vs best interior 0.914, within 0.02) — a plateau, not a truncated optimum, so widening the grid would only move the argmax around inside the noise.
>
> Selection instability is itself informative: a threshold detector has
> to be re-tuned per survey, and on a flat objective that tuning is
> noise-sensitive. The full split-B sweep below lets you see the shape
> of the curve rather than take the selected point on trust.

## The confound — read this before drawing a conclusion

**This benchmark structurally favours a brightness threshold, and the table
above should not be read as a general statement about the two approaches.**

In the scene simulator `rock_cluster` — the only natural clutter class — has
reflectivity **2.0-3.0**, the lowest of any class, while most man-made
targets sit at **4.0-8.0**. Brightness is therefore very nearly the
man-made/natural label, and a detector that thresholds on brightness is
handed the answer by the data generator. Real sonar offers no such gap: a
boulder and a steel drum can return comparable amplitude, which is the whole
reason the problem needs shape, shadow geometry and learning.

`docs/clutter_sweep.md` quantifies this by re-running with decoy clutter
whose brightness is drawn from the *real targets'* distribution, removing the
shortcut while changing nothing else.

Two things in this table are **not** affected by the confound, because both
rows score the same raw detections from the same detector:

- Rows (3) -> (4): the physics and verifier stages take false alarms from
  1018 to 68 per km² (15x) at comparable recall.
**Rows (1) and (2) are not a shadow-gate ablation.** Each variant is tuned
independently, so they usually land on different `k_sigma` and differ in two
things at once. Ablated properly — same k, only the gate changing — the
shadow requirement *raises precision where detection is hard* (+0.13 at
k=1, +0.23 at k=3) but costs recall, and at the permissive thresholds this
baseline actually prefers it is net-negative on F1 (0.909 -> 0.848 at
k=0.25). The cue is real and conditional, not a free win.

## Reading it

- Against the **stronger** classical baseline (2), the deployed system at
  its shipped threshold moves F1 from 0.822 to 0.710 and
  false alarms per km² from 49 to 68.
- Do **not** read rows (1) vs (2) as a shadow-gate ablation; they are tuned
  separately and usually differ in `k_sigma` too. See the note above for the
  matched-k measurement, where the gate helps precision only at strict
  thresholds.
- **PR-AUC is the threshold-free comparison.** It ranks detections without
  reference to any cut point, so it is the column least sensitive to how
  generously either family was tuned.
- The `classifies?` column is the part no threshold sweep can close. The
  baseline localizes; it cannot name a class, estimate height from shadow,
  score severity, or populate a report. Rows (1) and (2) are an upper bound
  on what the classical approach delivers operationally.

## Split-B threshold sweep (published, not used for selection)

Selection happened on split A. This is what the same sweep looks like on
the evaluation scenes, so the gap between the tuned point and the best
achievable point is visible rather than hidden.

| k_sigma | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 1 | 1.5 | 2 | 3 | 5 | 8 | 12 | 16 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blob best F1 | 0.891 | 0.884 | 0.887 | 0.849 | 0.765 | 0.720 | 0.594 | 0.508 | 0.365 | 0.303 | 0.028 | 0.000 | 0.000 | 0.000 | 0.000 |
| blob FP/km² | 24 | 29 | 39 | 68 | 24 | 68 | 151 | 107 | 93 | 88 | 365 | 5 | 0 | 0 | 0 |
| blob+shadow best F1 | 0.833 | 0.833 | 0.846 | 0.831 | 0.771 | 0.743 | 0.649 | 0.523 | 0.400 | 0.293 | 0.053 | 0.000 | 0.000 | 0.000 | 0.000 |
| blob+shadow FP/km² | 19 | 19 | 24 | 49 | 5 | 24 | 44 | 63 | 10 | 19 | 34 | 5 | 0 | 0 | 0 |

Selected on split A: `blob` k_sigma=0.25 score>=39.7, `blob_shadow` k_sigma=0.05 score>=33.4.
