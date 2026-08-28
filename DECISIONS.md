# DECISIONS.md — engineering assumptions & rationale

A running log of assumptions made while building SAGAR-NETRA. Newest entries at the bottom of each milestone section.

## M1 — Skeleton

1. **Repo root = `sagar-netra/`.** The GitHub repo is already named `SAGAR-NETRA`, so the layout from the
   spec is created at the repo root rather than in a nested `sagar-netra/` subdirectory.
2. **Bundled sample data is synthetic, spec-compliant XTF.** Real SSS survey logs are large and
   license-encumbered, and the build must work offline. `scripts/make_sample_xtf.py` renders a
   physically consistent scene (Rayleigh speckle seabed, sand ripples, seeded targets with
   highlight + down-range shadow pairs whose shadow length obeys L = H·R/A, TVG-like range banding,
   water-column gap sized by altitude) and writes it as a real XTF file (per the Triton XTF spec,
   via `pyxtf` ctypes structures). The parser is therefore exercised against genuine XTF bytes,
   not a mock. Noted per the master prompt's instruction to substitute synthetic stand-ins.
3. **`PingArray` layout.** Intensity is `float32`, shape `(n_pings, n_samples)` per side, raw
   (un-normalized) backscatter amplitude. Navigation is a NumPy structured array (`NAV_DTYPE`)
   with one record per ping — vectorizable, and it keeps per-ping slant range/altitude so later
   corrections never assume constancy along the track.
4. **GDAL avoided as a hard dependency.** `rasterio` (which bundles GDAL in its wheels) is an
   optional extra (`pip install .[geo]`); the GeoTIFF adapter degrades with a clear error if it
   is missing. Core pipeline needs only NumPy/SciPy/Pillow.
5. **JSF adapter is a from-spec implementation.** No EdgeTech sample file could be bundled, so
   `jsf.py` implements the published JSF message framing (0x1601 marker, 16-byte header) and the
   message-type-80 sonar record, and is validated by a round-trip test against our own
   spec-following writer. Field offsets follow the EdgeTech JSF rev. 1.20 description.
6. **Windows-first dev environment** (this machine), Linux via docker-compose. All paths handled
   with `pathlib`; no shell-outs in library code.
7. **Dependency pin: `albumentations==1.4.15` + `albucore==0.0.16`.** Newer releases depend on
   `stringzilla`, which ships no Windows wheel and hits an MSVC 2019 internal compiler error when
   built from source. The pinned pair installs cleanly everywhere we target.
8. **Bundled sample is committed AND regenerable.** `data/samples/survey_alpha.xtf` (5.4 MB) is
   checked in for instant demos; `scripts/make_sample_xtf.py` regenerates it byte-identically
   from the seed, so it can be deleted from history at any time without loss.
9. **Port/starboard convention.** In waterfall renderings, port is mirrored so its far range is at
   the image's left edge and both nadirs meet at the image centerline — matching common survey
   software (SonarWiz/Chesapeake) display convention. Column→(side, sample) bookkeeping lives in
   one place (`sonar_core/waterfall.py`) so pixel→ping/sample mapping stays exact.

## M2 — Preprocessing

1. **EGN excludes a nadir guard band from statistics.** The bright first-bottom-return transient
   sweeps across samples as altitude wobbles and is not seabed reflectivity; without an 8-sample
   guard it inflated near-nadir gain by up to 2.2x (found by adversarial review, verified
   numerically). Normalization still *applies* from the first return onward.
2. **Nadir blend columns are honestly NaN.** Ground columns whose source slant sample precedes the
   first bottom return would interpolate blanked water-column fill into "valid-looking" pixels;
   they are masked to NaN like the far-range swath edge.
3. **CLAHE via OpenCV 16-bit** (not scikit-image): faster, and cv2 ships anyway with the ML stack.
4. **Detection runs per side on ground-range imagery** (not on the combined waterfall): shadows are
   side-local and both sides are stored nadir-first, so shadow direction is uniform (+columns).

## M3 — Detector

1. **The offline training set is fully synthetic** (physics scene renderer through the real M2
   chain). This is honest closed-world training: the bundled demo detects targets rendered by the
   same physics — the value demonstrated is the *pipeline*, not open-world generalization. The
   eight public datasets (download scripts with licenses) plus the copy-paste factory are the
   documented path to real-world weights; conversion utilities land with them.
2. **Augmentation physics policy:** mirroring across columns, rotation, shear and perspective are
   forbidden (they would put acoustic shadows up-range of highlights — impossible geometry);
   ISOTROPIC scaling and window translation are allowed (equivalent to different survey
   resolution / crop), along-track flips are valid (a reversed survey line). Pinned explicitly in
   `scripts/train_detector.py` so Ultralytics default changes can't silently violate the rule.
3. **workers=0 on Windows** for every Ultralytics call (spawn-semantics deadlock).

## M4 — Physics

1. **Shadow statistics use the central 60% of box rows.** Objects taper along-track, so box-end
   rows carry little shadow and diluted the dark-fraction below threshold (found when the height
   test failed at 0.14 m vs truth 2.0 m; central-rows fix recovers height within tolerance).
2. **Physics demotes, never deletes.** Implausible or cue-less detections keep flowing to the
   operator with multiplied-down calibrated confidence and an explicit violation reason.

## M5 — Geo & Reports

1. **SQLite is the default store; PostGIS optional** (compose profile). Offline-first demo on a
   laptop must not require a database server; the repository interface is swappable.
2. **No shapely:** point-in-polygon by ray casting + point-to-edge distances on a local azimuthal
   equidistant projection (pyproj only) — exact enough below a few hundred km, zero extra deps.
3. **Severity is explainable by construction:** every contact carries its per-term breakdown
   (hazard/size/height/depth/proximity + nearest layer and distance).

## M6 — Dashboard

1. **No deck.gl.** The spec named deck.gl for the heatmap; `leaflet.heat` (8 kB) delivers the same
   severity heatmap without a ~1 MB WebGL dependency that fights Leaflet for the canvas. Noted as
   a deliberate deviation.
2. **Client-side filtering** over the survey's contact fetch (limit 500) instead of re-querying per
   filter change: instant, consistent across Map/Waterfall/Contacts tabs; server-side filters
   remain available on the endpoint for large deployments.
3. **Offline map story:** the backend proxies and disk-caches OSM tiles (`/tiles/{z}/{x}/{y}.png`);
   once an area has been viewed online it renders offline forever, and with no cache the proxy
   serves a neutral sea-grid tile so the map stays usable.

## M7 — Intelligence

1. **Anomaly brain masks a nadir guard (32 columns).** The slant-to-ground stretch zone next to
   nadir magnifies a few samples into smooth streaks whose reconstruction error is systematically
   high on normal seabed (17 false blobs on a clean scene without the guard, 0-6 with it). The
   supervised detector still covers that ~1.3 m strip.
2. **`unknown_anomaly` is an ensemble-level reportable label**, not a detector training class.
3. **Brain B (segmenter) is deferred**: the ensemble treats it as optional (masks refine boxes when
   weights exist). Detector + anomaly + physics already close the demo loop; a U-Net trained on
   the synthetic factory's free masks is the documented follow-up.
4. **Error smoothing happens BEFORE swath masking** in the anomaly map, or smoothing bleeds error
   back into the masked zone (caught by test).

## Blueprint audit round (post-M8)

1. **Brain B is a net/rope specialist U-Net**, not SegFormer/SAM-LoRA: filamentous classes
   (`ghost_net`, `pipeline`) are the ones whose bounding box lies about extent; the U-Net trains
   offline in minutes on the scene renderer's free masks. SAM-LoRA is the documented upgrade once
   real imagery (AI4Shipwrecks) is downloaded.
2. **Deep ensemble over MC-dropout** for L3 uncertainty: YOLOv8n contains no dropout layers to
   sample at inference, so ensemble disagreement across three seed-trained members
   (`configs/detector.yaml: ensemble_weights`) is the honest source of epistemic uncertainty.
   Fusion divides summed matched scores by the member count — a lone find is demoted, consensus
   keeps its mean score.
3. **OpenMax not implemented**: it requires penultimate-layer activation surgery inside the
   Ultralytics head. Open-set detection duty is carried by Brain C (reconstruction error) plus
   cross-brain consensus; contacts the detector cannot name surface as `unknown_anomaly`.
4. **Citizen-sonar parsers are spec-implementations validated by round-trip** (the jsf.py
   precedent): Lowrance offsets per opensounder/sonarlight, Humminbird per PING-Mapper (its
   cm-scaling and R=6378388 mercator chosen where references disagree). Humminbird recordings
   (.DAT + .SON directory) upload as a .zip; the API extracts and locates the .DAT.
5. **Blueprint's diffusion/CycleGAN/S3Simulator synthesis is a growth path, not shipped**: those
   need GPU training time and real style-target imagery; the working subset is the physics
   renderer + shadow-consistent copy-paste + physically-safe augmentation.
6. **Mission profiles re-rank, never re-detect**: a mission YAML only overrides the severity
   hazard table and the detector confidence floor, so imagery and physics evidence stay
   comparable across missions. Mission names are validated against the profile listing (an HTTP
   form value must never resolve a path).
7. **The review round found and fixed**: copilot substring hijacks ("lane" in "planes", "high" in
   "highest"), the LLM path bypassing Python-side dimension filters, missing .sl2/.sl3/.zip
   upload suffixes (frontend + backend), mission path traversal, SL2 writer epoch overflow, and
   segmenter YAML keys frozen by checkpoint config precedence.

## Technical-approach round (from TECHNICAL APPROACH.pdf + SIH26057 strategy doc)

1. **Stage-2 verifier is explainable by construction**: a gradient-boosted classifier over 13
   hand-crafted physics features (shadow-edge linearity, contour regularity, texture-entropy
   delta, cross-ping persistence, ...) trained WITHOUT the detector — labels come from rendered
   truth geometry, negatives from rocks/ripple-band/background boxes, split by scene. With no
   checkpoint present, confidences are bit-identical to the Stage-1 pipeline (golden-tested).
2. **Cross-ping temporal gate demotes, never deletes** (`min_persistence_pings`), and its reason
   string states the measured extent vs the threshold.
3. **Streaming dedup keeps the STRONGER observation, cross-window pairs only** (adversarial
   review caught the first-seen rule storing boundary-clipped fragments and merging distinct
   same-window neighbours 8 m apart; both are regression-tested now). Replacements keep the
   operator-visible id plus any review/recovery already recorded.
4. **INT8 honesty**: dynamic-quantized ONNX is 3.1 MB (vs 11.6 fp32, sub-10 MB target met) but
   SLOWER on this x86 CPU (no VNNI int8 conv path) — its speed claim belongs to Jetson TensorRT,
   and the benchmark table says so instead of hiding it.
5. **Ablation on synthetic held-out scenes is a relative ladder, not a real-data claim**
   (`outputs/metrics/ablation.md` preamble): FP/km² 469 → 147 (physics gate) → 98 (ML verifier)
   at held recall; rows (c)/(d) coincide because ensemble consensus already kills 1–2-ping
   returns — the temporal gate is the single-model backstop.
6. **The PDF's stack deltas we did not adopt**: TypeScript/Tailwind/Recharts (working JSX+CSS
   dashboard stays; charts are hand-rolled where needed), DeepLabV3+ (U-Net specialist), YOLOv8-s
   (v8n per the PDF's own Jetson-class advice). Functional parity, not framework churn.

## Live-stream consistency round

1. **Streaming windows must be equalised like the whole survey.** CLAHE divides an image into a
   fixed `tile_grid` of cells and equalises each independently, so the grid's *row* count decides
   how many pings share one transfer curve. A 200-ping window under the stock `(8, 8)` grid gets
   25 pings per cell against a 600-ping survey's 75 — a 3x more aggressive equaliser on identical
   seabed, which lifts Rayleigh speckle into structure the anomaly autoencoder never saw in
   training. Because Brain C's threshold is calibrated once at training time, it then fired
   everywhere: measured **0.50 anomalies/tile** over the whole survey against **19.8/tile** in
   windows, i.e. a 600-ping survey with 3 seeded targets streamed as **179 contacts** (175 of them
   spurious `unknown_anomaly`) versus 12 in batch.
2. **The fix scales the CLAHE row count by the window's share of the survey**
   (`api.realtime._window_preprocess_config`), restoring comparable pings-per-cell. Streamed
   output fell to 27 contacts with the real detections identical to batch (ghost_net, container,
   cylinder_drum, tire — one each). Column count is untouched: swath width does not vary window
   to window. Batch is unaffected — a window covering the whole survey resolves to the stock grid.
3. **Ruled out by measurement, not assumption:** forcing the window to use the full survey's
   percentile-stretch bounds *increased* the flood (23.3/tile), so the global stretch was not the
   cause. Only the cell geometry mattered.
4. **The residual gap is coverage, not normalisation.** Streaming re-processes overlapping pings,
   so it examines 24 tiles where batch examines 12; the remaining anomaly excess scales with that
   and is dominated by low-confidence (~11%) contacts that sort to the bottom of the severity
   queue.
5. **Latent issue worth knowing:** the same scale dependence means two *batch* surveys of very
   different lengths are also equalised slightly differently. Training scenes span 700–1100 pings
   so the deployed models see a narrow band of this, but expressing the CLAHE grid in pixels
   rather than cell counts is the principled long-term fix.

## Image-upload round

1. **A nav-less format needs declared geometry, not a guess.** A survey log records altitude,
   range and position per ping; an image records none of it, so uploading a PNG failed at
   slant-range correction ("needs finite positive altitude for every ping"). The operator now
   states what the sonar was set to at upload time (`altitude_m`, `range_m` required; position,
   heading and tow depth optional) and the full chain runs: slant correction, height from shadow
   and WGS-84 geotagging. `range_m <= altitude_m` is rejected — a swath only exists beyond the
   first bottom return.
2. **The synthesised track is labelled as a fiction.** With a declared start position the parser
   lays a straight constant-heading line so contacts geotag correctly relative to it, and records
   `meta["nav_source"] = "declared-line"`. It positions detections correctly for a benchmark image
   or a demo; it is not recorded navigation and the metadata says so.
3. **A display image must not be gain-normalised twice.** A waterfall written for display has
   already had its range falloff flattened and its contrast stretched. Running EGN over it again
   invents range structure that was never in the water, and the open-set anomaly brain — calibrated
   on singly normalised imagery — reads that structure as debris. Measured on the bundled
   waterfall: **92 spurious anomalies with EGN on, 0 with it off**; total contacts fell from 105 to
   13, all named classes, with heights matching the log path (wreck 2.99 m, container 2.36 m).
   The image parser therefore declares `gain_normalized=True` by default and `preprocess` skips
   EGN for such sources — an explicit `egn.enabled` from the caller always wins, and an export of
   raw uncorrected amplitudes can pass `gain_normalized=False`.
4. **Same failure class as the streaming flood.** Both were the anomaly brain reacting to imagery
   normalised differently from its calibration set, not to debris. Worth remembering whenever a
   new input path is added: match the normalisation the brains were calibrated on, or recalibrate.

## Baseline-comparison round

1. **The ablation ladder had no floor that wasn't us.** Every published number compared
   SAGAR-NETRA against SAGAR-NETRA — how much each stage adds. That cannot answer the first
   question a reviewer asks, which is whether any of it beats what survey teams already run.
   `tridentnet/baseline.py` adds a faithful classical CAD baseline (per-range-column robust
   background, threshold at `median + k*sigma`, morphological opening, connected components,
   area/aspect filters, optional shadow gate) and `scripts/eval_baseline.py` scores it against
   the deployed stack through the same matcher, the same metric arithmetic and the same scenes.
2. **The baseline detects on gain-corrected, pre-CLAHE imagery.** The first draft ran it on the
   enhanced image the learned brains see, where it looked terrible. Measured cause: true targets
   peak at **8–30 sigma** above the robust per-column background on `ground_raw` but only
   **1.7–3.2 sigma** after CLAHE, because local contrast equalization compresses exactly the
   global separation a fixed threshold depends on. Scoring it there would have been a strawman,
   so `ClassicalConfig.use_raw_imagery` defaults to True.
3. **Hyperparameters are selected on a separate split, for both families.** An earlier draft
   swept the baseline's two knobs directly on the evaluation set and reported the best — and the
   baseline duly "won". That is test-set fitting: with a few dozen truth boxes it manufactures a
   winner out of noise. Selection now happens on seed base 11000 (disjoint from training seed 0,
   calibration 9000 and evaluation 12000) and is applied unchanged to the evaluation scenes.
4. **The simulator encodes the class label in brightness, which confounds the whole comparison.**
   `rock_cluster` — the only natural clutter class — has reflectivity **2.0–3.0**, the lowest of
   any class, while most man-made targets sit at **4.0–8.0**. A detector that thresholds on
   brightness is handed the man-made/natural answer by the data generator. Real sonar has no such
   gap; a boulder and a steel drum can return comparable amplitude, which is the entire reason
   the problem needs shape, shadow geometry and learning. Any brightness-threshold-versus-learned
   comparison on this simulator is therefore structurally biased toward the threshold.
5. **So clutter is swept under two conditions.** `scripts/eval_clutter.py` holds the debris field
   fixed and layers nested decoy rock clusters, under `native` (catalogue reflectivity, gap
   intact) and `matched` (each decoy borrows a real target's reflectivity, gap removed). Every
   rock is a false positive by construction, so the sweep measures one thing: how fast precision
   falls as target-shaped natural objects accumulate, with and without the shortcut.
6. **The two conditions must differ in brightness and nothing else.** Branching on an RNG draw
   (`rng.uniform` for native, `rng.integers` for matched) consumed different amounts of the bit
   stream and silently moved every subsequent rock — two unrelated experiments wearing the same
   label, and the failure produces perfectly plausible tables. Both draws now happen in both
   modes, in a fixed order; `tests/test_clutter.py::test_modes_place_identical_rocks` pins it.
7. **The threshold sweep is vectorized because the honest grid is wide.** Widening `k_sigma` down
   to 0.25 (so the selected value is interior, not clamped at an endpoint) hands the sweep tens of
   thousands of blobs, and the obvious re-score-at-every-threshold loop is O(n^2) — one run hung
   in it. Now O(n log n) via a cumulative-TP scan, verified identical to the loop on 300
   randomized label streams.
8. **Two suspected handicaps on the baseline were checked and cleared by measurement, not
   assertion.** The aspect-ratio filter is *inert*: identical F1, precision, recall and FP/km²
   from `max_aspect=6` through effectively infinite. The area cap was not inert but was close —
   the largest truth box in the held-out set is an aircraft at ~19.5k px against a 20k cap, one
   bad seed from silently dropping a real target and charging the miss to the baseline — so it
   was raised to 100k.
9. **The confound is real and measured, but it does not rescue the result.** The clutter sweep
   (12 scenes, 55 man-made truths, nested decoys) shows the classical baseline is far more
   dependent on the brightness shortcut than the learned stack is. Precision lost between +0 and
   +24 decoy rocks: classical **-0.630 native vs -0.734 matched**, SAGAR-NETRA **-0.618 native vs
   -0.632 matched**. Removing the shortcut costs the classical detector an extra **0.104** of
   precision and costs SAGAR-NETRA **0.014** — roughly a sevenfold difference in sensitivity,
   which is what you would expect from a method that reads shape and shadow rather than
   amplitude. But the classical baseline still holds higher absolute precision at every clutter
   level in both conditions, so the confound explains part of its lead and does not erase it.
10. **What this licenses us to claim, and what it does not.** On this synthetic benchmark a tuned
    classical CAD baseline matches or beats the deployed stack at localization, and no amount of
    reframing changes that. "SAGAR-NETRA outperforms classical sonar software" is therefore **not
    a supported claim** and must not be presented as one. Three things remain fully supported and
    are unaffected by the confound, because they compare rows scored from identical detections:
    the physics and verifier stages cut false alarms roughly **15x** (1018 -> 68 per km²) at
    comparable recall, both rows scoring identical detections; and the baseline cannot classify,
    estimate height, score severity or produce a report at any threshold. The likeliest
    reason the learned stack does not pull ahead is that a clean simulated seabed of
    high-contrast targets is the regime a tuned threshold is best at, compounded by a small
    CPU-trained detector (mAP50 0.656, precision 0.552). Settling it needs real survey data, not
    a louder synthetic table — which is exactly what the dataset and active-learning path is for.
11. **The published rows (1) and (2) are not a shadow-gate ablation, and an early draft wrongly
    read them as one.** Each classical variant is tuned independently, so they land on different
    `k_sigma` and differ in two things at once. Ablated properly — same k, only the gate moving —
    the shadow requirement raises precision **where detection is hard** (+0.128 at k=1, +0.229 at
    k=3) but costs recall, and at the permissive thresholds the baseline actually prefers it is
    net-negative on F1 (0.909 -> 0.848 at k=0.25, 0.899 -> 0.812 at k=0.05). The shadow cue is
    real and conditional. Claiming it as independent corroboration of Stage-1 would have been
    unsupported, and the claim was removed from the README and the generated table.
