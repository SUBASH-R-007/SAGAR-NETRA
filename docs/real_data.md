# Real sonar — what the pipeline actually does on it

Generated 2026-08-29T06:35:08+00:00 by `scripts/eval_real_data.py` over 447 images in 181 s.

## The corpus

KLSG (`SeabedObjects-Ship-and-Airplane-dataset`) — **385 real shipwreck and 62
real aircraft side-scan images**, contributed by L-3 Klein Associates, EdgeTech,
Lcocean, Hydro-tech Marine and Tritech. Released by the authors for academic use;
cite the KLSG paper, and do not ship it in a commercial build.

These are target-centred chips and mosaics, not raw survey logs. They carry no
navigation, no recorded range and no altitude, so geometry is **declared**
(15 m altitude, 75 m range) and every
metre-valued quantity derived from it — height, shadow length, position — is
arbitrary. This report therefore measures *detection behaviour* and *conditioning*,
and deliberately quotes no height in metres.

## What works: L1 conditioning

All 447 images parsed and ran the full signal chain — bottom tracking, slant
correction, despeckle, CLAHE, tiling — with no format-specific handling and no
crashes. Gain normalization is skipped automatically because these images are
already display-normalized (`meta['gain_normalized']`), which is the same guard
that fixed the synthetic image-upload path.

![Real sonar before and after conditioning](images/real_data.png)

*Left: real KLSG shipwreck imagery as supplied. Right: the same frame after
the L1 chain. The highlight-and-shadow structure the physics gate keys on is
clearly present in real data — a bright hull return with a long dark shadow
extending down-range.*

## What does not work: detection

This is the honest part.

| measure | value |
|---|---|
| images processed | 447 (385 wreck, 62 aircraft) |
| raw detections | 3602 (8.1 per image) |
| from Brain A (supervised detector) | 515 (14.3%) |
| from Brain C (open-set autoencoder) | 3087 (85.7%) |
| surviving the shipped 50% floor | 19 |
| mean top detector score per image | 0.670
| wreck-image runs that ever predict wreck/aircraft | 53 / 385 |

### Predicted class distribution

| class | detections |
|---|---|
| `unknown_anomaly` | 3087 |
| `pipeline` | 178 |
| `ghost_net` | 123 |
| `aircraft` | 62 |
| `wreck` | 50 |
| `tire` | 31 |
| `human_body` | 25 |
| `container` | 25 |
| `rock_cluster` | 9 |
| `cylinder_drum` | 6 |
| `mine_like` | 6 |

### Reading it

**The supervised detector does not transfer.** Trained on 172 synthetic tiles, it
has never seen a real hull, real seabed texture or a real acoustic shadow. On
unmistakable shipwreck imagery it produces low-confidence guesses spread across
classes, and reaches for `wreck` or `aircraft` in only a minority of images. No
amount of downstream physics can recover a label the detector never proposed.

**The open-set brain floods.** Brain C is an autoencoder trained to reconstruct
*synthetic* clean seabed. Real seabed — sand ripples, rock fields, biological
scatter, survey artefacts — reconstructs badly everywhere, so almost everything
reads as anomalous. This is the third appearance of one failure class: the
anomaly brain reacting to imagery normalized or textured differently from its
calibration set. It flooded in live-stream mode, it flooded on uploaded images,
and it floods here.

**The physics gate holds the line.** The shipped confidence floor still admits
only 19 of 3602 raw detections, so the stage that was measured
as a 15x false-alarm reduction on synthetic data is doing visible work here too —
it is the only reason this output is not unusable.

## What this changes

The claim that survives is narrower and more defensible than the one before it:

> The **signal chain** runs on real sonar from five different manufacturers with no
> per-format handling. The **detection models** are trained on synthetic data and do
> not yet transfer — measured, not assumed.

The fix is not a better gate; it is real training data, and this corpus is the
start of it. KLSG carries folder-level class labels but no bounding boxes, so the
two candidate paths are (a) domain-adapt Brain C's autoencoder on real seabed,
which needs no labels at all, and (b) weakly supervised fine-tuning of Brain A on
target-centred chips.

## Path (a) was attempted and did not work

`scripts/train_anomaly.py --klsg` mixes real seabed -- the border bands of the 81
KLSG chips large enough to have a margin clear of their centred target -- into the
autoencoder's training set. The retrained checkpoint was measured against the
shipped one on both domains before any decision to adopt it:

| checkpoint | real (one wreck image) | synthetic scene | demo survey |
|---|---|---|---|
| shipped `anomaly.pt` | 539 anomalies | 15 | 17 raw -> 14 contacts, 1 open-set |
| retrained, own threshold | 307 (-43%) | 45 (3x worse) | 19 raw -> 16 contacts, 3 open-set |
| retrained, old threshold | 157 (-71%) | 1 | 16 raw -> 13 contacts, **0 open-set** |

**Neither operating point is an improvement.** At its own calibrated threshold the
flood halves but synthetic false anomalies triple, and the demo survey gains two
contacts that are not there. At the shipped threshold the flood drops by 71% but
open-set detection stops entirely -- zero `unknown_anomaly` contacts -- which
removes the capability Brain C exists to provide.

One small convolutional autoencoder with a single global threshold cannot model
two domains this different, and 324 border bands from 81 usable chips is thin. The
shipped weights were therefore **left unchanged**; the retrained checkpoint is kept
out of the deployed path. The likelier real fix is matching the two domains'
statistics in preprocessing rather than asking one autoencoder to span both, or
training on real data with real labels -- which is path (b), and needs boxes this
corpus does not have.
