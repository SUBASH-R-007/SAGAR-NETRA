# SAGAR-NETRA Â· à¤¸à¤¾à¤—à¤° à¤¨à¥‡à¤¤à¥à¤°

**AI-Powered Automated Underwater Marine Debris & Anomaly Detection from Side-Scan Sonar**

Smart India Hackathon 2026 Â· Problem Statement **26057** Â· Ministry of Earth Sciences (MoES) / National Institute of Ocean Technology (NIOT) Â· Category: Software Â· Theme: Disaster Management

> Raw sonar survey logs go in one end. Calibrated, geotagged, **physics-verified**, priority-ranked
> debris reports come out the other â€” on a laptop, with no internet, in about ten seconds.

| | |
|---|---|
| **Status** | Complete end-to-end prototype, all 8 milestones + 2 hardening rounds |
| **Tests** | **290 passing**, 0 failures (52 s), `ruff` clean across 8 packages |
| **Code** | 15,813 lines Python (9,854 library) Â· 3,395 lines frontend Â· 112 Python modules |
| **Cloud dependency** | **None.** Zero network calls at inference |
| **Input formats** | XTF Â· EdgeTech JSF Â· Lowrance SL2/SL3 Â· Humminbird DAT/SON Â· GeoTIFF Â· PNG/JPG |
| **Output formats** | JSON (+ JSON Schema) Â· CSV Â· GeoJSON Â· KML Â· PDF |

---

## 1. The problem

Between **500,000 and 1,000,000 tonnes** of ghost fishing gear enter the ocean every year. It
persists for **600â€“800 years**, makes up roughly **46%** of the Great Pacific Garbage Patch by
mass, and kills an estimated **650,000 marine animals annually**. A Kerala study measured
**167.5 kg of gear lost per vessel per year**; the Fishery Survey of India has mapped **14
ghost-gear hotspots** off the east coast alone, and India's **11,098 km** coastline is surveyed
by NIOT vessels and the OMe-6000 AUV that already generate side-scan sonar data.

**The bottleneck is not the sonar â€” it is the human review.** WWF's Baltic campaign surveyed
5,820 ha over 45 sea-days and produced 549 suspect contacts, each one hand-picked by an expert
scrolling waterfall imagery. Sonar search was already 12Ã— faster and 17Ã— cheaper than divers;
AI triage removes the last manual step.

**Market gap:** SonarWiz, Triton Perspective, Klein SonarPro and EdgeTech Discover ship **no AI
auto-detection**. SeeByte's ATR is naval, at defence pricing. GhostNetZero.ai is closed, cloud-only,
net-only and Baltic-trained â€” and is now expanding toward the Indian Ocean. There is no indigenous,
open, edge-deployable Indian system. This is that system.

---

## 2. What SAGAR-NETRA does

![Preprocessing pipeline](docs/images/pipeline.png)

*Real output from the bundled survey: raw slant-range imagery (black water column, TVG banding)
â†’ ground-range corrected and gain-flattened â†’ despeckled + CLAHE, the image the detector sees.*

The approach is four steps, and the second-to-last one is what makes it defensible:

| | **1 Â· CONDITION** | **2 Â· DETECT** | **3 Â· VERIFY** | **4 Â· ACT** |
|---|---|---|---|---|
| | raw log â†’ clean imagery | triple-brain AI | physics + calibration | decision-ready output |
| | Bottom tracking Â· slant-range correction âˆš(RÂ²âˆ’AÂ²) Â· empirical gain normalization Â· shadow-preserving despeckle Â· CLAHE Â· SAHI tiling | **YOLOv8 deep ensemble** (rigid debris) âˆ¥ **U-Net masks** (ghost nets, ropes) âˆ¥ **Autoencoder** (open-set unknowns) | Highlightâ€“shadow pairing Â· height from shadow **H = LÂ·A/R** Â· class plausibility gates Â· 13-feature ML verifier Â· cross-ping persistence Â· temperature-scaled 0â€“100% | WGS-84 geotag Â· entanglement severity index Â· clustered recovery routes Â· 5 report formats Â· live console |

**The core insight.** A side-scan sonar image is an *acoustic reflectance map*, not a photograph.
A real object proud of the seabed produces a **bright highlight paired with a dark shadow
extending down-range** â€” and the shadow is often more diagnostic than the object, because it
encodes silhouette and height. Every detection in SAGAR-NETRA must satisfy that geometry before
it reaches an operator. A "2-metre-tall bottle" is demoted with an explicit written reason,
not silently dropped.

---

## 3. Architecture

```mermaid
flowchart TD
    A["SONAR SURVEY LOG<br/>XTF Â· JSF Â· SL2/SL3 Â· SON Â· GeoTIFF Â· PNG"] --> B

    subgraph L1["L1 Â· SonicPrep â€” signal conditioning"]
        B["PingArray<br/>intensities + per-ping navigation"] --> C["bottom tracking<br/>water-column removal"]
        C --> D["slantâ†’ground correction<br/>empirical gain normalization"]
        D --> E["Lee despeckle Â· CLAHE<br/>SAHI overlap tiling"]
    end

    subgraph L2["L2 Â· TridentNet â€” triple brain"]
        E --> F["Brain A<br/>YOLOv8 deep ensemble<br/>rigid debris boxes"]
        E --> G["Brain B<br/>U-Net segmentation<br/>net / rope masks"]
        E --> H["Brain C<br/>conv-autoencoder<br/>open-set anomalies"]
        F --> I["ensemble merge<br/>corroboration + provenance"]
        G --> I
        H --> I
    end

    subgraph L3["L3 Â· PhysiCheck â€” verification"]
        I --> J["highlight + shadow present?<br/>H = LÂ·A/R plausible for class?"]
        J --> K["13-feature ML verifier<br/>cross-ping persistence"]
        K --> L["temperature-scaled confidence<br/>Evidence Card per contact"]
    end

    subgraph L4["L4 Â· GeoScribe â€” reporting"]
        L --> M["WGS-84 geotag + layback<br/>dimensions LÃ—WÃ—H"]
        M --> N["Entanglement Severity Index<br/>vs shipping lanes / turtle zones / MPAs"]
        N --> O["JSON Â· CSV Â· GeoJSON Â· KML Â· PDF"]
    end

    subgraph L5["L5 Â· DRISHTI Console"]
        O --> P["FastAPI + WebSocket + SQLite"]
        P --> Q["map Â· waterfall overlay Â· evidence cards<br/>review queue Â· change detection Â· copilot<br/>recovery routes Â· 4 mission profiles"]
    end

    F -.-> R["EDGE PATH<br/>ONNX Â· INT8 Â· Jetson / Pi<br/>offline, no network"]
```

Five layers, each independently testable, each with its own config file and test suite.

---

## 4. Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev,ml,api,geo]"
python scripts/make_sample_xtf.py     # deterministic bundled sample survey
python scripts/demo.py --serve        # full pipeline, then the console at :8000
```

Or containerised: `docker compose up --build` (add `--profile postgis` for a shore station).

### The 90-second judge demo

```bash
python scripts/demo.py --serve
```

narrates the whole flow on the bundled survey and prints the contact table:

```
Done in 13.0 s: 1200 pings -> 18 tiles -> 17 raw detections -> 14 verified contacts

ID                   class           conf%   sev   H(m)  position
SN-20260101-0002     ghost_net        75.8  83.7    1.3  13.05016, 80.35064
SN-20260101-0003     ghost_net        66.1  82.8    0.9  13.05027, 80.35195
SN-20260101-0005     ghost_net        57.3  80.4    0.7  13.04986, 80.35073
SN-20260101-0001     wreck           100.0  79.9    3.0  13.04975, 80.35035
SN-20260101-0006     container        46.6  77.3    2.4  13.04981, 80.35148
SN-20260101-0004     mine_like        63.5  75.6    0.9  13.04970, 80.35095
...
SN-20260101-0014     unknown_anomaly  10.6  62.7      -  13.05040, 80.35008
```

Note the ranking: severity puts **ghost nets at the top even when their confidence is lower than
the wreck's** â€” because entanglement hazard, not detector certainty, is what a cleanup crew
prioritises. The last row is an open-set find with no class and no measurable shadow, surfaced by
the autoencoder rather than the detector.

â€¦writes all five report formats (the KML opens directly in Google Earth), and opens the console
with the results loaded. Upload another survey from the dashboard â€” in **LIVE STREAM** mode you
watch detections arrive one by one over the WebSocket, as they would on a towed survey.

### Training from scratch (all CPU, all offline)

```bash
python scripts/train_detector.py                     # Brain A  ~15 min
python scripts/train_detector.py --seed 1 --dest weights/detector_seed1.pt --name detector_seed1
python scripts/train_segmenter.py --epochs 60        # Brain B  ~9 min
python scripts/train_anomaly.py                      # Brain C  ~10 min
python scripts/train_verifier.py --scenes 10         # Stage-2 verifier  ~2 min
python scripts/fit_calibration.py --apply            # temperature scaling
python scripts/eval_detector.py                      # the ablation table
```

Every script has a `--smoke` profile that finishes in under a minute for CI.

---

## 5. Layer by layer â€” what is actually implemented

### L1 Â· SonicPrep â€” ingestion and signal conditioning

| Module | What it does |
|---|---|
| `parsers/base.py` | `PingArray` â€” the format-agnostic contract: `(n_pings, n_samples)` float32 per side, sample 0 at nadir, plus a 10-field `NAV_DTYPE` record per ping (time, lat, lon, heading, altitude, sensor depth, sound velocity, slant range, speed, layback) |
| `parsers/xtf.py` | Triton XTF via `pyxtf`; nav from per-ping headers, port/starboard resolved from `ChanInfo` |
| `parsers/jsf.py` | EdgeTech JSF message-type-80, implemented from the spec (offsets per MB-System's `mbsys_jstar.h`) |
| `parsers/lowrance.py` | **Citizen sonar**: Lowrance `.sl2`/`.sl3`, both frame layouts, spherical-mercator inverse, composite sidescan split at nadir |
| `parsers/humminbird.py` | **Citizen sonar**: Humminbird `.DAT` + `.SON`/`.IDX`, tag-walked records per PING-Mapper |
| `parsers/geotiff.py` | Georeferenced mosaics (optional `rasterio`); marks `ground_range` so slant stages are skipped |
| `parsers/image.py` | Plain PNG/JPG waterfall crops for public benchmark datasets |
| `preprocess/bottom_track.py` | First-bottom-return detection with sustained-run thresholding, outlier rejection against a median-smoothed reference, header-altitude fallback |
| `preprocess/slant_range.py` | Ground range = âˆš(RÂ²âˆ’AÂ²), per-ping altitude aware, **invertible to 6Ã—10â»Â¹Â² px**; nadir blend columns honestly NaN-masked |
| `preprocess/egn.py` | Empirical gain normalization; per-range medians with an 8-sample nadir guard band (without it, near-nadir gain inflated up to **2.2Ã—**) â€” **19.8Ã— flatness improvement** |
| `preprocess/despeckle.py` | Lee MMSE filter (NaN-safe normalised convolution) + adaptive median; **shadow edges move 0â€“0.4 px** |
| `preprocess/clahe.py` | 16-bit OpenCV CLAHE, NaN-swath safe |
| `preprocess/tiler.py` | SAHI-style overlapping tiles, shifted-last-tile edge policy, lossless coverage, exact tileâ†’ping/column bookkeeping |
| `preprocess/pipeline.py` | Config-driven orchestrator with progress callbacks and per-stage timings |
| `waterfall.py` | Waterfall rendering with exact column â†” (side, sample) mapping |

**Synthetic data factory** (`synth/`): a physics-consistent scene renderer â€” Rayleigh multiplicative
speckle, grazing-angle falloff, residual-TVG banding, sand-ripple fields, a water-column gap sized
by altitude, a bright first bottom return, and **ray-traced shadows whose length obeys the same
geometry PhysiCheck inverts**. Plus `shadow_render` (shadow compositing), `copy_paste`
(radiometry-matched chip pasting), `augment` (sonar-legal augmentations only), and `artifacts`
(the five artifacts the problem statement names: **heave banding, pitch stretch, roll shear, ping
dropout, resolution jitter**).

### L2 Â· TridentNet â€” the triple brain

The **12-class frozen label set** â€” 9 reportable object classes plus 3 *hard negatives* that exist
purely to absorb false positives and are never shown to an operator:

```
0 ghost_net   3 pipeline        6 container    9  rock_cluster â†hard negative
1 wreck       4 cylinder_drum   7 human_body   10 sand_ripple  â†hard negative
2 aircraft    5 tire            8 mine_like    11 reef         â†hard negative
```

plus `unknown_anomaly`, an ensemble-level open-set label (not a training class).

| Brain | Model | Role | Trained result |
|---|---|---|---|
| **A** | YOLOv8n Ã— 3 seeds, deep ensemble | Rigid debris â€” boxes | mAP50 **0.656** / 0.651 / 0.587 (P 0.552, R 0.878) |
| **B** | U-Net, ~1.9 M params, pure torch | Ghost nets & ropes â€” pixel masks | val Dice **0.924** |
| **C** | Convolutional autoencoder | Open-set: "this does not belong here" | threshold auto-calibrated at train time |

**Why three brains.** A ghost net is a sprawling irregular tangle â€” a bounding box around it is
mostly water, so it gets segmentation. A drum is a compact ellipse â€” boxes are perfect, so
segmentation would be wasted compute. And an object class nobody trained on still has to surface,
so an autoencoder trained *only on clean seabed* flags reconstruction-error blobs as
`unknown_anomaly`. Most teams pick one architecture and force everything through it.

**Deep-ensemble uncertainty** (`deep_ensemble.py`): three independently seeded detectors vote;
fused score = `sum(matched member scores) / n_models`, so unanimous agreement keeps the mean while
a 1-of-3 lone find is cut to a third. This replaces MC-dropout, which YOLOv8n cannot provide â€”
it has no dropout layers to sample.

**Ensemble merge** (`ensemble.py`): a Brain-C blob overlapping a Brain-A box at IoU â‰¥ 0.30
*corroborates* it (+0.05 score, provenance `AC`); Brain-B masks refine boxes to their true pixel
extent (provenance `AB`/`ABC`); standalone C blobs survive as open-set finds. Every contact
records which brains fired.

### L3 Â· PhysiCheck â€” verification and calibrated confidence

![Evidence cards](docs/images/evidence.png)

*Two real Evidence Cards. Yellow = the detection box, red = the measured acoustic shadow, caption =
the cues that fired. The wreck's 19.24 m shadow at 8 m altitude yields a 2.99 m height; the
container's 10.16 m shadow yields 2.36 m against a seeded truth of 2.4 m.*

- **`shadow.py`** â€” locates the shadow down-range of each box (dark-run detection over the central
  60% of box rows, because objects taper along-track), measures its ground-range length, and
  inverts the geometry to a height. Cues: highlight present, shadow present, ratios, length, height.
- **`calibrate.py`** â€” temperature scaling fitted by NLL on held-out scenes, expected calibration
  error, reliability diagrams, and a class-conditional plausibility gate driven by
  `configs/physics.yaml` (per-class height bands: tire â‰¤ 1.2 m, ghost net â‰¤ 2.5 m, wreck â‰¤ 15 mâ€¦).
- **`features.py` + `verifier.py`** â€” the **Stage-2 ML verifier**: a gradient-boosted classifier
  over **13 explicit physics features** â€” not a black box:

  | | | |
  |---|---|---|
  | `highlight_ratio` | `shadow_ratio` | `shadow_len_m` |
  | `height_m` | `has_height` | **`shadow_linearity`** â€” RÂ² of a line through the shadow's trailing edge; pipes and hulls cast straight shadows, rocks cast ragged ones |
  | **`contour_regularity`** â€” man-made objects trend toward regular geometry | **`texture_entropy_delta`** â€” debris breaks the background texture statistics; sand ripples do not | **`ping_persistence`** â€” real objects persist across scan lines |
  | `aspect_ratio` | `area_px` Â· `range_frac` | `score_raw` |

  Trained **without the detector** â€” labels come from rendered truth geometry, negatives from rock
  clusters, ripple-band boxes and clear seabed, split by scene. Held-out **AUC 0.955, accuracy 0.957**.
- **`verify.py`** â€” applies the gate, the verifier multiplier and the cross-ping persistence gate.
  With no verifier checkpoint present, output is **bit-identical** to the Stage-1 pipeline (golden-tested).
- **`evidence.py`** â€” renders the Evidence Card PNG + a JSON cue list per contact (with optional
  gradient-free EigenCAM overlay).
- **`crossview.py`** â€” contacts seen in two overlapping surveys corroborate each other; lone
  contacts are demoted and flagged `resurvey_recommended`, mirroring naval re-acquire doctrine.

![Calibration](docs/images/calibration.png)

*Reliability diagrams before and after temperature scaling: **ECE 0.204 â†’ 0.146** at T = 2.54.
A displayed "70%" now means roughly 70% of such detections are real.*

### L4 Â· GeoScribe â€” geotagging, severity and reports

- **Geotagging** â€” the towfish is placed at the ping's nav fix (pushed astern by layback), the
  target offset perpendicular to heading at its ground range, solved on the WGS-84 ellipsoid with
  pyproj. Verified against seeded scene geometry to within **6%**.
- **Dimensions** â€” length Ã— width from the footprint (mask-refined for nets), **height from the
  shadow**. Measured container: 6.0 Ã— 2.5 Ã— 2.4 m against a seeded 6.1 Ã— 2.4 Ã— 2.4 m.
- **Position accuracy** â€” every contact carries an honest `position_accuracy_m` =
  `2 Ã— ground_res + layback_term + nav_uncertainty`, summed linearly because each term is a bias,
  not independent noise: a recovery diver wants the conservative number.
- **Entanglement Severity Index (0â€“100)** â€” a weighted, saturating blend, fully explainable
  because every contact carries its own breakdown:

  | Term | Weight | Detail |
  |---|---|---|
  | Class hazard | 0.40 | ghost_net 1.0, human_body 1.0, mine_like 0.95, container 0.80, drum 0.75, wreck/aircraft 0.60, pipeline 0.50, tire 0.30, unknown 0.65 |
  | Footprint area | 0.15 | saturating, 40 mÂ² scale |
  | Height | 0.10 | saturating, 2 m scale |
  | Depth band | 0.15 | shallow objects entangle gear and strike hulls |
  | Layer proximity | 0.20 | geodesic distance to shipping lanes, turtle-nesting zones, MPAs (demo layers for the Chennai coast ship with the repo) |

- **Priority & recommended action** â€” HIGH â‰¥ 75, MEDIUM â‰¥ 50, else LOW, with an operator-facing
  action per class: *"Entanglement hazard â€” flag for ROV recovery"*, *"Notify SAR authority
  immediately"*, *"Do NOT approach â€” notify naval EOD"*.
- **Reports** â€” `contacts.json` (with a published JSON Schema), `contacts.csv`, `contacts.geojson`,
  severity-styled `contacts.kml`, and a branded `report.pdf` with per-contact thumbnails. Plus a
  survey summary block: contacts, high-confidence count, **area surveyed (kmÂ²)** and **debris
  density per kmÂ²**.
- **Recovery routing** â€” geodesic union-find clustering into recovery zones, then nearest-neighbour
  + 2-opt ordering within and across clusters.

### L5 Â· DRISHTI Console

A React 18 + Vite + Leaflet single-page console served by FastAPI, skinned as a **Government of
India / Ministry of Earth Sciences portal** (bilingual header with the Ashoka Chakra, tricolour
ribbon, working A- / A / A+ accessibility controls, skip-to-content link, navy nav and footer).
Design system committed at [`web/DESIGN.md`](web/DESIGN.md).

| Tab | What it does |
|---|---|
| **Map** | Contacts on satellite imagery, coloured by severity, with sensitive-zone GeoJSON overlays, a severity heatmap toggle, and click-through Evidence Cards with Confirm/Reject |
| **Waterfall** | The processed sonar image with detection boxes registered pixel-exactly over it, zoom, raw/enhanced toggle, click a box â†’ contact detail |
| **Contacts** | Dense ledger: thumbnail, class, confidence, severity swatch, LÃ—WÃ—H, depth, physics badges, review state, **recovery state**, and one-click download of all five report formats |
| **Diff** | Change detection between two surveys â€” what is NEW since the last pass (the post-cyclone port-clearance feature) |
| **Copilot** | Natural language over the contact store: *"contacts longer than 5 m between 20 and 40 m depth"*, *"how many ghost nets?"*, *"contacts near the turtle nesting zone"*, plus auto-drafted survey summaries |

Plus a persistent **ingest rail**: drag-and-drop upload, **BATCH or LIVE STREAM** mode, live
WebSocket progress, and a per-detection feed while streaming.

---

## 6. Measured results

Everything below was measured on this machine and is regenerable. Nothing is projected.

### Ablation â€” does each stage earn its place?

From [`docs/ablation.md`](docs/ablation.md), 8 held-out synthetic scenes
(seed base 12000, disjoint from training and calibration), 34 man-made truth boxes, 0.1023 kmÂ²,
one detection pass re-scored four ways:

| Configuration | Precision | Recall | F1 | PR-AUC | **FP / kmÂ²** |
|---|---|---|---|---|---|
| (a) raw detector | 0.294 | 0.588 | 0.392 | 0.486 | **469.2** |
| (b) + physics gate | 0.583 | 0.618 | 0.600 | 0.594 | **146.6** |
| (c) + ML verifier | 0.667 | 0.588 | 0.625 | 0.627 | **97.8** |
| (d) + temporal persistence *(deployed)* | 0.667 | 0.588 | 0.625 | 0.627 | **97.8** |

**False alarms per kmÂ² drop 4.8Ã— while recall holds.** Rows (c) and (d) coincide because the
ensemble consensus already suppresses 1â€“2-ping impulsive returns; the temporal gate is the
backstop for single-model operation.

> **Read honestly:** these are *synthetic* held-out scenes. Synthetic targets are easier than real
> debris in real clutter, so treat the absolute values as an upper bound and the **relative ladder**
> as the result. The path to real-data numbers is `scripts/download_datasets.py` plus the
> active-learning flywheel.

### Throughput ([`edge/benchmark.md`](edge/benchmark.md), CPU only)

| Stage | ms / tile | Throughput |
|---|---|---|
| L1 preprocessing | â€” | **2,717 pings/s** (800-ping survey in 0.29 s) |
| Detector â€” PyTorch CPU | 55.7 | 17.96 tiles/s |
| Detector â€” **ONNX Runtime CPU** | **26.2** | **38.14 tiles/s** |
| Detector â€” ONNX INT8 CPU | 223.6 | 4.47 tiles/s *(see note)* |
| Anomaly autoencoder | 40.7 | 24.57 tiles/s |

A side-scan sonar produces 1â€“10 ping lines per second. The pipeline runs far ahead of the sensor.

> **INT8 note, stated honestly:** dynamic quantisation shrinks the model from **11.6 MB â†’ 3.1 MB**
> (the sub-10 MB edge target is met) but runs *slower* on this x86 laptop, which has no VNNI int8
> convolution path. The INT8 speed claim belongs to Jetson TensorRT, on hardware we do not have.
> The benchmark says so rather than hiding it.

### Models and verification

| Item | Result |
|---|---|
| Brain A detector (30 epochs, imgsz 512) | mAP50 **0.656**, mAP50-95 0.559, precision 0.552, recall 0.878 |
| Deep-ensemble members | mAP50 0.656 / 0.651 / 0.587 |
| Brain B segmenter (60 epochs) | val Dice **0.924**; mask alignment verified to **0.02 px** |
| Stage-2 verifier | held-out AUC **0.955**, accuracy **0.957** |
| Confidence calibration | ECE **0.204 â†’ 0.146**, T = 2.54 |
| ONNX export parity | mAP50 delta **0.0000** vs PyTorch |
| Height from shadow | seeded 2.0 m recovered within 25%; 2.4 m container measured at 2.36 m |
| Geotag accuracy | within **6%** of seeded across-track offset |
| Slantâ†”ground round-trip | **6Ã—10â»Â¹Â² px** |
| End-to-end demo (full triple-brain + verifier) | 1200 pings â†’ 18 tiles â†’ 17 detections â†’ **14 contacts + 5 reports in 13.0 s** |

![Detections](docs/images/detections.png)

*Detector output on the bundled survey â€” every seeded target localised, each with its
highlightâ€“shadow signature intact.*

---

## 7. The twelve innovations

Mapped to the design blueprint; full per-feature audit in [`COMPLIANCE.md`](COMPLIANCE.md)
(**45 implemented Â· 5 partial Â· 3 documented non-goals** across 53 audited features).

| # | Innovation | Status |
|---|---|---|
| N-01 | **Open-set triple brain** â€” detector + net segmentation + unknown-anomaly autoencoder | All three trained and fused |
| N-02 | **PhysiCheck engine** â€” highlightâ€“shadow pairing and H = LÂ·A/R as a false-positive firewall | Implemented, ablation-measured |
| N-03 | **Calibrated Evidence Cards** â€” ensemble + temperature scaling + per-detection cue list | Implemented |
| N-04 | **Synthetic Data Factory** â€” physics-consistent generation with sim-to-real path | Physics renderer + copy-paste + artifact augs; diffusion/CycleGAN documented as growth path |
| N-05 | **Entanglement Severity Index** â€” class Ã— size Ã— height Ã— depth Ã— habitat proximity | Implemented with explainable breakdown |
| N-06 | **Retrieval Mission Planner** â€” density clustering + optimised vessel routing | Implemented |
| N-07 | **Temporal change detection** â€” "new anomaly since last pass" | Implemented (`/api/diff`) |
| N-08 | **Active-learning flywheel** â€” analyst verdicts become training labels | Implemented (`scripts/export_review_labels.py`) |
| N-09 | **Edge-first, offline-first** â€” ONNX/INT8, zero cloud | Implemented + measured |
| N-10 | **Citizen sonar** â€” Lowrance & Humminbird recreational sonar ingestion | Implemented, dashboard-uploadable |
| N-11 | **SONAR-GPT copilot** â€” natural-language querying + survey summaries | Implemented, works fully offline |
| N-12 | **Disaster-mode profiles** â€” one pipeline, four missions | Implemented |

### Four missions, one pipeline

A mission profile re-weights the severity hazard table and the detector confidence floor â€” it
**re-ranks, it never re-detects**, so imagery and physics evidence stay comparable across missions.

| Mission | Confidence floor | Emphasis |
|---|---|---|
| `ghost_net_cleanup` | 0.20 | Ghost gear and entanglement hazards |
| `port_clearance` | 0.25 | Containers, wrecks, pipelines â€” post-cyclone harbour clearance |
| `sar` | **0.10** (recall first) | Human body â€” search and rescue |
| `aircraft_search` | 0.15 | Aircraft, wreckage â€” black-box search |

---

## 8. Data strategy

**Offline training is fully synthetic and honest about it.** The scene renderer produces
physically correct imagery through the *same* preprocessing chain used at inference, so training
chips are exactly what the detector sees in production. Ghost nets get 3Ã— sampling weight as the
rarest and hardest class.

**Augmentation obeys sonar physics.** Mirroring across columns, rotation, shear and perspective are
*forbidden* â€” they would place acoustic shadows up-range of their highlights, geometry no sonar can
produce. Isotropic scaling, translation and along-track flips are valid. The rule is pinned
explicitly in the training script so an upstream default change cannot silently violate it.

**The real-data path is scripted, licensed and documented.** `python scripts/download_datasets.py --list`
prints the table: Marine Debris FLS Â· UATD Â· SCTD/SCTD2 Â· Seabed Objects-KLSG Â· AI4Shipwrecks Â·
SWDD Â· Marine PULSE Â· UXO â€” each with its license and role. Combined with the active-learning
flywheel, that is how these weights become field weights.

**The bundled sample** (`data/samples/survey_alpha.xtf`, 5.1 MB, committed and byte-reproducible
from its seed) is a real XTF file containing 1200 pings and **8 seeded targets** with published
ground truth â€” 6 man-made plus 2 rock clusters as natural distractors, spanning 12.5â€“33 m ground
range at 8 m altitude.

---

## 9. API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Model versions (SHA-1 per checkpoint), load state, memory, last-survey throughput |
| `POST` | `/api/upload` | Upload a survey â€” `mode=batch\|stream`, optional `mission=` |
| `GET` | `/api/jobs` Â· `/api/jobs/{id}` | Job list / one job snapshot |
| `WS` | `/api/jobs/{id}/progress` | Live progress; in stream mode, per-detection events |
| `GET` | `/api/surveys` | Processed surveys |
| `GET` | `/api/contacts` | Filter by `survey`, `cls`, `min_conf`, `min_sev`, `review`, `limit` |
| `GET` | `/api/contacts/{id}` | One contact, full record |
| `POST` | `/api/contacts/{id}/review` | Confirm / reject (feeds the retraining flywheel) |
| `POST` | `/api/contacts/{id}/recovery` | `flagged â†’ assigned â†’ retrieved` |
| `GET` | `/api/contacts/{id}/evidence` Â· `/thumb` | Evidence Card PNG Â· thumbnail |
| `GET` | `/api/reviews/export` Â· `/api/recovery/log` | Append-only audit trails |
| `GET` | `/api/report/{json\|csv\|geojson\|kml\|pdf}?survey=` | Download a report |
| `GET` | `/api/waterfall/{survey}` Â· `/meta` | Processed imagery + overlay geometry |
| `GET` | `/api/diff?survey_a=&survey_b=&radius_m=` | Change detection |
| `GET` | `/api/crossview?survey_a=&survey_b=` | Cross-survey confirmation, resurvey flags |
| `GET` | `/api/route?cluster_eps_m=` | Recovery tour over confirmed contacts |
| `POST` | `/api/copilot` | Natural-language question â†’ answer + SQL + rows |
| `GET` | `/api/missions` Â· `/api/layers` | Mission profiles Â· sensitive-zone GeoJSON |
| `GET` | `/tiles/{z}/{x}/{y}.png` | Basemap proxy with a per-source disk cache (online once, offline forever) |

---

## 10. Repository layout

| Path | Layer | Contents |
|---|---|---|
| `sonar_core/` | L1 | 6 format parsers â†’ `PingArray`, 7-stage preprocessing pipeline, synthetic scene factory |
| `tridentnet/` | L2 | Detector + deep ensemble, U-Net segmenter, anomaly autoencoder, brain merger, dataset builder |
| `physicheck/` | L3 | Shadow geometry, calibration, 13-feature verifier, Evidence Cards, cross-view |
| `geoscribe/` | L4 | Contact model, geotagging, severity index, 5 report writers, clustering + routing |
| `api/` | L5 | FastAPI app, SQLite store, batch + streaming processing, diff, copilot |
| `web/` | L5 | React/Vite/Leaflet console + committed design system |
| `edge/` | â€” | ONNX + INT8 export with parity checks, benchmarks, TensorRT runbook |
| `configs/` | â€” | 6 YAML files + 4 mission profiles â€” every tunable, commented, with units |
| `scripts/` | â€” | 13 CLIs: training, calibration, evaluation, export, demo, dataset download |
| `tests/` | â€” | 36 files, **290 tests** |
| `docs/` | â€” | README figures (regenerate with `scripts/make_docs_images.py`) |

**Documentation index**

| File | Contents |
|---|---|
| [`COMPLIANCE.md`](COMPLIANCE.md) | Feature-by-feature audit against the design blueprint, with evidence |
| [`DECISIONS.md`](DECISIONS.md) | **41 engineering decisions** across 9 rounds â€” every assumption, with its reason |
| [`docs/ablation.md`](docs/ablation.md) | The ablation ladder and its protocol |
| [`edge/benchmark.md`](edge/benchmark.md) | Machine-stamped throughput measurements |
| [`edge/trt_int8.md`](edge/trt_int8.md) | Jetson INT8 conversion runbook + calibration-set generator |
| [`web/DESIGN.md`](web/DESIGN.md) | The console's design system |

---

## 11. Engineering quality

- **290 tests, zero failures**, running in 52 seconds â€” unit tests per module plus end-to-end
  acceptance tests per milestone, including a full raw-XTF â†’ reports run.
- **`ruff` clean** across all 8 packages (E, F, W, I, N, UP, B).
- **Every stage config-driven** â€” no magic numbers in code; defaults live in one place and the YAML
  mirrors them.
- **Adversarially reviewed.** Each build round ended with review agents that probed the new code
  empirically and reported only substantiated defects. Over **20 real bugs** were found and fixed
  with regression tests, including: EGN inflating near-nadir gain 2.2Ã—, streaming dedup storing
  boundary-clipped fragments instead of full re-detections, a copilot query hijack (`"lane"`
  matching inside `"planes"`), a mission-name path traversal, and an SL2 timestamp overflow.
- **Graceful degradation everywhere**: no GPU â†’ CPU; no trained weights â†’ documented fallback with
  a warning; no LLM â†’ rule-based copilot; no internet â†’ cached tiles, then a neutral sea-grid tile.

---

## 12. Honest limitations

Stated plainly, because a prototype that hides its edges is not trustworthy:

1. **Model metrics are on synthetic held-out data.** The blueprint target of â‰¥0.90 mAP@50 refers to
   published results on real KLSG/SCTD benchmarks after full training campaigns. Our 0.656 is
   honest closed-world synthetic performance. The real-data path is scripted, not yet run.
2. **INT8 is a size win here, not a speed win** â€” the speed claim needs Jetson TensorRT hardware.
3. **Jetson and Hailo paths are documented runbooks**, not executed measurements â€” we do not have
   the devices.
4. **The State Emblem is not used** (it is legally restricted). The header renders an Ashoka Chakra,
   with a marked slot for an official asset if the submission is entitled to one.
5. **Sensitive-zone layers are illustrative demo geometry** for the Chennai coast, clearly labelled
   as such â€” not official maritime boundaries.
6. **Not an official Government of India website** â€” the console states this in its footer.
7. **Deliberate non-goals**, each with a written rationale in `DECISIONS.md`: OpenMax fusion
   (Brain C + consensus covers open-set), MC-dropout (YOLOv8n has no dropout layers), and
   diffusion/CycleGAN synthesis (needs GPUs and real style targets).

---

## 13. Credits and licensing

Code: MIT (see [`LICENSE`](LICENSE)). Ultralytics YOLOv8 is AGPL-3.0 â€” relevant if the trained
detector is redistributed.

Public datasets referenced by `scripts/download_datasets.py` carry their own licenses (CC BY-NC-SA,
CC-BY, BSD-3) and are attributed in the table it prints. Basemap imagery: Esri World Imagery
(Maxar, Earthstar Geographics, GIS community), proxied and cached locally.

Built for **Smart India Hackathon 2026**, Problem Statement **26057**, Ministry of Earth Sciences /
National Institute of Ocean Technology.

---

<div align="center">

**à¤¸à¤¾à¤—à¤° à¤¨à¥‡à¤¤à¥à¤°** â€” *the eye of the ocean*

</div>
