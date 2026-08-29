# Detection-stack ablation — SYNTHETIC held-out scenes

Generated 2026-08-29T21:33:34+00:00 by `scripts/eval_detector.py` in 14 s.

**Read this before quoting any number.** Every scene below is rendered by
the SAGAR-NETRA physics scene simulator — no real sonar data is involved.
The scenes are *held out* (seed base 12000, disjoint from detector training and calibration seeds), so no
model in the stack has seen their speckle, seabed texture or targets; but
synthetic targets are inevitably easier than real debris in real clutter,
so treat these numbers as an upper bound and, above all, as a *relative*
comparison between pipeline stages measured on identical raw detections.

## Protocol

- 8 scenes, 34 man-made truth boxes, 0.1023 km² surveyed (per `geoscribe.build.survey_stats`).
- Deployed detector stack ran ONCE per scene; the four configurations
  re-score the SAME raw detections (raw score >= 0.25).
- TP: IoU >= 0.3 with a man-made truth box, same class required;
  one detection per truth (greedy, confidence-ranked). Hits on natural
  targets (rock clusters) or background count as FP.
- Point metrics at calibrated confidence >= 50%; PR-AUC is
  average precision swept over all confidence thresholds.
- Stage-2 verifier checkpoint: held-out AUC 0.955, accuracy 0.957 (own scene-level held-out split).

## Results

| configuration | P@50 | R@50 | F1@50 | PR-AUC | FP/km² | TP | FP | dets |
|---|---|---|---|---|---|---|---|---|
| (a) raw detector (calibrated score) | 0.270 | 0.588 | 0.370 | 0.514 | 527.86 | 20 | 54 | 93 |
| (b) + physics gate (Stage-1) | 0.538 | 0.618 | 0.575 | 0.472 | 175.95 | 21 | 18 | 93 |
| (c) + ML verifier (Stage-2) | 0.528 | 0.559 | 0.543 | 0.425 | 166.18 | 19 | 17 | 93 |
| (d) + temporal persistence (deployed) | 0.528 | 0.559 | 0.543 | 0.425 | 166.18 | 19 | 17 | 93 |

Reading the ladder: (b) applies the Stage-1 highlight/shadow multipliers,
(c) adds the learned Stage-2 cue-vector multiplier, (d) adds the
thin-detection persistence gate — the full deployed configuration.
Confidence multipliers demote rather than delete, so `dets` is constant
by construction; what moves is how many false alarms stay above the
operator's confidence floor. When rows (c) and (d) coincide, no scored
detection in this set was thinner than the persistence minimum — the
ensemble consensus already suppresses 1-2-ping impulsive returns, and
the temporal gate is the deployed backstop for single-model operation.
