# Real sonar — what the pipeline actually does on it

Generated 2026-08-29T21:34:46+00:00 by `scripts/eval_real_data.py` over 447 images in 67 s.

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
| raw detections | 3728 (8.3 per image) |
| from Brain A (supervised detector) | 636 (17.1%) |
| from Brain C (open-set autoencoder) | 3092 (82.9%) |
| surviving the shipped 50% floor | 22 |
| mean top detector score per image | 0.742
| wreck-image runs that ever predict wreck/aircraft | 359 / 385 |

### Predicted class distribution

| class | detections |
|---|---|
| `unknown_anomaly` | 3092 |
| `wreck` | 610 |
| `aircraft` | 22 |
| `container` | 4 |

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
only 22 of 3728 raw detections, so the stage that was measured
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

## Why Brain C is loud on real seabed, and what fixed it

The first diagnosis was wrong and worth recording. The assumption was that the
autoencoder reconstructs real seabed *badly* -- unfamiliar texture, high error,
everything looks anomalous -- so it was retrained with real seabed mixed in
(`scripts/train_anomaly.py --klsg`, still available). That produced no operating
point worth shipping: at its own threshold the count halved but synthetic false
anomalies tripled, and at the shipped threshold open-set detection stopped dead.

Measuring the error distributions showed why it could not have worked. Against the
shipped checkpoint, real imagery reconstructs **better** than synthetic:

| corpus | median error | fraction above threshold |
|---|---|---|
| synthetic clean seabed | 0.0688 | 1.49% |
| real KLSG, whole chips | 0.0405 | 0.61% |
| real KLSG, seabed borders | 0.0399 | 0.67% |

Error magnitude was never the problem, so retraining addressed something that was
not broken. What differs is **spatial structure**. Synthetic speckle is
incoherent, so its above-threshold pixels are scattered singletons that the
`min_blob_px` filter discards. Real seabed texture -- sand ripples, rock fields,
wreck framing -- is spatially coherent, so the same fraction of pixels forms
connected blobs that survive. Measured per tile: a synthetic **maximum of 12**
blobs against a real **median of 12 and a maximum of 62**.

Blob peak-to-threshold ratios overlap too much to separate on (real median 1.45
vs synthetic 1.33), so a strength gate would delete true positives before it
deleted texture. A per-tile *candidate budget* keeping the highest-scoring blobs
was implemented and measured next -- and it is not the fix either:

| budget | candidates | surviving the 50% floor |
|---|---|---|
| off | 946 | 6 |
| 16 | 635 | **0** |
| 32 | 801 | 3 |
| 48 | 902 | 5 |

*(70 real KLSG images.)* Capping at 16 removes **every** detection that would have
survived the confidence floor. The ranking is by raw reconstruction peak, while
survival downstream is decided by highlight/shadow physics -- a texture blob can
peak higher than a real target the gate would later promote, so ranking on one and
selecting on the other is close to anti-correlated.

`max_blobs_per_tile` therefore ships **disabled**, and every number in this report
is measured with it off. It remains available as a deliberate recall-for-compute
trade on fixed edge hardware, with the cost table above attached to it so nobody
enables it expecting a free win.

Two hypotheses tested, two rejected. What is left standing is the measurement
itself: Brain C is loud on real seabed because real seabed is genuinely busy, and
the physics stack -- not the anomaly brain -- is what makes the output usable.

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

## Why Brain C is loud on real seabed, and what fixed it

The first diagnosis was wrong and worth recording. The assumption was that the
autoencoder reconstructs real seabed *badly* -- unfamiliar texture, high error,
everything looks anomalous -- so it was retrained with real seabed mixed in
(`scripts/train_anomaly.py --klsg`, still available). That produced no operating
point worth shipping: at its own threshold the count halved but synthetic false
anomalies tripled, and at the shipped threshold open-set detection stopped dead.

Measuring the error distributions showed why it could not have worked. Against the
shipped checkpoint, real imagery reconstructs **better** than synthetic:

| corpus | median error | fraction above threshold |
|---|---|---|
| synthetic clean seabed | 0.0688 | 1.49% |
| real KLSG, whole chips | 0.0405 | 0.61% |
| real KLSG, seabed borders | 0.0399 | 0.67% |

Error magnitude was never the problem, so retraining addressed something that was
not broken. What differs is **spatial structure**. Synthetic speckle is
incoherent, so its above-threshold pixels are scattered singletons that the
`min_blob_px` filter discards. Real seabed texture -- sand ripples, rock fields,
wreck framing -- is spatially coherent, so the same fraction of pixels forms
connected blobs that survive. Measured per tile: a synthetic **maximum of 12**
blobs against a real **median of 12 and a maximum of 62**.

Blob peak-to-threshold ratios overlap too much to separate on (real median 1.45
vs synthetic 1.33), so a strength gate would delete true positives before it
deleted texture. What is safe to bound is *cost*: `max_blobs_per_tile` keeps the
highest-scoring blobs per tile and drops the rest, since a real target peaks well
above the threshold while texture barely crosses it.

The shipped budget of 16 sits above the synthetic maximum on purpose:

| | candidates |
|---|---|
| synthetic held-out scenes (6 seeds) | **bit-identical, box for box** |
| real ship-113 | 539 -> 311 (**-42%**) |

This is a cost bound, not a transfer fix. Brain C is still answering honestly --
a boulder field genuinely is unlike flat sediment -- and the detector still does
not recognise real wrecks. What changed is that an image full of rocks can no
longer spend unbounded downstream time, and it changed nothing about the scenes
every published table was measured on.

## What this changes

The claim that survives is narrower and more defensible than the one before it:

> The **signal chain** runs on real sonar from five different manufacturers with no
> per-format handling. The **detection models** are trained on synthetic data and do
> not yet transfer -- measured, not assumed.

The fix for that is real training data. KLSG carries folder-level class labels but
no bounding boxes, so the remaining path is weakly supervised fine-tuning on its
target-centred chips; the pseudo-boxes would be approximate, and any mAP from them
must not be quoted beside the synthetic numbers.
