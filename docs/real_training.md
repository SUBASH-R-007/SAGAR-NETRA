# Real-data training - acceptance gates

Generated 2026-08-29T21:31:11+00:00 by `scripts/eval_gates_real.py` in 56 s.

Candidate `detector_real.pt` vs deployed baseline `detector.pt`,
measured through one harness in one run - no numbers quoted from memory.

## Gate 1 - per-source val mAP50

| val set | boxes | baseline | candidate |
|---|---|---|---|
| synth_xl | synthetic truth | 0.701 | 0.808 |
| klsg_yolo | weak (measured) | 0.007 | 0.637 |
| uatd_yolo | **real, annotated** | 0.002 | 0.819 |

Synthetic regression: -0.107 against a tolerance of 0.1 - **PASS**.

The KLSG row is against weak boxes and is reported for completeness, not
compared against anything: its labels are measured approximations.

## Gate 2 - KLSG class-reach (top-1 in {wreck, aircraft})

| model | hits | share |
|---|---|---|
| baseline | 10 / 88 | 11.4% |
| candidate | 87 / 88 | 98.9% |

## Gate 3 - demo survey smoke check

- baseline: 9 contacts, classes {'ghost_net': 4, 'mine_like': 2, 'wreck': 1, 'container': 1, 'cylinder_drum': 1}
- candidate: 12 contacts, classes {'ghost_net': 3, 'wreck': 2, 'mine_like': 2, 'aircraft': 1, 'container': 1, 'cylinder_drum': 1, 'pipeline': 1, 'tire': 1}

## Verdict

- Gate 1 (synthetic within tolerance): PASS
- Gate 2 (class-reach improved): PASS (11.4% -> 98.9%)
- Gate 3 is judgement, not arithmetic - read the class spread above.

Swapping the deployed weights is a human decision on these numbers;
this script never copies a checkpoint anywhere.
