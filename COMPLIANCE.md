# Blueprint Compliance Matrix

Audit of the implementation against `SAGAR-NETRA_Blueprint.docx` (August 2026),
feature by feature. Statuses: ✅ implemented & tested · 🟡 partial (working
subset, rest documented) · 📋 documented non-goal (requires hardware/data we
don't have, or research beyond a hackathon prototype — honest rationale given).

## L1 — SonicPrep

| Blueprint feature | Status | Where | Evidence |
|---|---|---|---|
| XTF parser (pyxtf) | ✅ | `sonar_core/parsers/xtf.py` | round-trip tests vs spec-compliant writer |
| EdgeTech JSF parser | ✅ | `sonar_core/parsers/jsf.py` | offsets per MB-System reference; round-trip tests |
| Humminbird .SON/.DAT (PING-Mapper approach) | ✅ | `sonar_core/parsers/humminbird.py` | DAT/SON/IDX per PING-Mapper docs; byte-level round-trip tests |
| Lowrance .sl2/.sl3 (sonarlight approach) | ✅ | `sonar_core/parsers/lowrance.py` | SL2+SL3 offset tables per opensounder; port-reversal verified at byte level |
| GeoTIFF mosaics + plain images | ✅ | `parsers/geotiff.py`, `parsers/image.py` | tests + mosaic passthrough path |
| Bottom tracking + water-column removal | ✅ | `preprocess/bottom_track.py` | recovers heave through wobble; header fallback |
| Slant-range correction √(Rs²−A²) | ✅ | `preprocess/slant_range.py` | invertible to 6e-12 px; nadir blend NaN-masked |
| EGN | ✅ | `preprocess/egn.py` | flatness 19.8× improvement; nadir guard band |
| Shadow-preserving despeckle (Lee/adaptive median) | ✅ | `preprocess/despeckle.py` | shadow edge moved 0–0.4 px in tests |
| CLAHE | ✅ | `preprocess/clahe.py` | 16-bit OpenCV, NaN-swath safe |
| SAHI overlap tiling | ✅ | `preprocess/tiler.py` | lossless coverage; shifted-last-tile policy |

## L2 — TridentNet

| Feature | Status | Where | Evidence |
|---|---|---|---|
| Brain A: nano-YOLO, 9 classes + 3 hard negatives | ✅ | `tridentnet/detector.py`, `classes.py` | mAP50 0.653 (synthetic val); 5.6 tiles/s CPU |
| Brain B: net/rope segmentation → masks + dims | ✅ | `tridentnet/segmenter.py`, `segdata.py` | pure-torch U-Net; mask alignment verified to 0.02 px; mask-refined boxes feed dims |
| Brain B backbone choice | 🟡 | U-Net (blueprint also lists SegFormer/SAM-LoRA) | U-Net chosen: offline-trainable in minutes on the synthetic factory; SAM-LoRA documented as the real-data upgrade path |
| Brain C: conv-AE anomaly brain | ✅ | `tridentnet/anomaly.py` | train-time-calibrated threshold; nadir stretch guard; seeded-unknown tests |
| OpenMax open-set fusion | 📋 | — | open-set duty is covered by Brain C + cross-brain consensus; OpenMax needs classifier-logit surgery inside Ultralytics heads — documented in DECISIONS.md |
| Ensemble merge of A+B+C with provenance | ✅ | `tridentnet/ensemble.py` | corroboration lifts, brains recorded per contact |

## L3 — PhysiCheck

| Feature | Status | Where | Evidence |
|---|---|---|---|
| Highlight + down-range shadow verification | ✅ | `physicheck/shadow.py` | rendered-scene tests |
| Height from shadow H = L·A/R + class plausibility | ✅ | `shadow.py`, `calibrate.py`, `configs/physics.yaml` | seeded 2.0 m recovered within 25%; "2-m bottle" demoted with reason |
| Deep-ensemble uncertainty | ✅ | `tridentnet/deep_ensemble.py` | 3 seed models trained (mAP50 .653/.651/.587); consensus fusion tested |
| MC-dropout | 📋 | — | YOLOv8n has no dropout layers to sample; ensemble disagreement is the honest epistemic-uncertainty source (DECISIONS.md) |
| Temperature scaling → meaningful 0–100% | ✅ | `physicheck/calibrate.py` | fitted T=2.54 on held-out scenes; ECE 0.204→0.146; reliability diagrams saved |
| Evidence Card: Grad-CAM + cue list | ✅ | `physicheck/evidence.py` | EigenCAM (gradient-free) best-effort + JSON cues per detection |
| Multi-view/overlap cross-confirmation + re-survey flag | ✅ | `physicheck/crossview.py`, `GET /api/crossview` | boost on agreement, demote + `resurvey_recommended` on lone contacts |

## L4 — GeoScribe

| Feature | Status | Where | Evidence |
|---|---|---|---|
| Ping-header nav + layback → WGS-84 | ✅ | `geoscribe/geotag.py` | position matches seeded across-track offset within 6% |
| JSON + published schema, CSV, GeoJSON, KML, PDF | ✅ | `geoscribe/report.py` | all validated in tests; KML severity-styled with footprints |
| Dims L×W (mask when available) × H (shadow) | ✅ | `build.py` + Brain-B refined boxes | container 6.0×2.5×2.4 m vs seeded 6.1×2.4×2.4 |
| Entanglement Severity Index (class/size/depth/lanes/turtle/MPA) | ✅ | `geoscribe/severity.py` + `data/layers/` | explainable per-term breakdown on every contact |

## L5 — DRISHTI Console

| Feature | Status | Where | Evidence |
|---|---|---|---|
| Drag-drop upload + live WebSocket progress | ✅ | `api/main.py`, `web/` | browser-verified end to end |
| Map + synchronized waterfall with overlays | ✅ | `web/src/components/` | port-mirror overlay math review-verified |
| Density heatmap | 🟡 | leaflet.heat (blueprint: deck.gl) | same capability, 8 kB instead of ~1 MB; DECISIONS.md |
| Retrieval mission planner (clustering + routing) | ✅ | `geoscribe/cluster.py`, `route.py`, `/api/route?cluster_eps_m=` | union-find geodesic clustering + NN+2-opt per cluster |
| Temporal change detection | ✅ | `api/diff.py`, Diff tab | geodesic matching, new-vs-matched |
| Confirm/reject → active-learning queue | ✅ | review endpoints + `scripts/export_review_labels.py` | verdicts export as YOLO chips (rejected → hard negatives) |
| SONAR-GPT copilot (NL queries incl. dims/depth, summaries) | ✅ | `api/copilot.py` | rule grammar + optional local LLM; "net-like contacts longer than 5 m…" works offline |
| PostGIS store | 🟡 | SQLite default, PostGIS compose profile | offline-first demo needs zero servers; repository interface swappable |
| Disaster-mode profiles (4 missions) | ✅ | `configs/missions/`, `/api/missions`, upload `mission` field | SAR re-ranks human_body above debris in tests |

## Edge deployment

| Feature | Status | Where | Evidence |
|---|---|---|---|
| PyTorch → ONNX | ✅ | `edge/export_onnx.py` | mAP50 parity delta 0.0000 |
| TensorRT INT8 (Jetson) | 🟡 | `edge/trt_int8.md` runbook + calibration-set generator | needs the physical Jetson to execute |
| Hailo .hef (Pi 5) | 📋 | — | needs the Hailo accelerator + DFC toolchain |
| Benchmarks | ✅ | `edge/benchmark.py` → `benchmark.md` | measured: 43 tiles/s ONNX CPU, 2233 pings/s preprocess |

## Innovation ledger

| # | Innovation | Status |
|---|---|---|
| N-01 | Open-set triple brain | ✅ A+B+C all trained and fused |
| N-02 | PhysiCheck engine | ✅ |
| N-03 | Calibrated Evidence Cards | ✅ ensemble + temperature + CAM |
| N-04 | Synthetic Data Factory | 🟡 physics renderer + shadow-consistent copy-paste + safe augments implemented; diffusion/CycleGAN/S3Simulator are the documented growth path (need GPUs + real style-target data) |
| N-05 | Entanglement Severity Index | ✅ |
| N-06 | Retrieval Mission Planner | ✅ clustering + routing |
| N-07 | Temporal change detection | ✅ |
| N-08 | Active-learning flywheel | ✅ review trail → YOLO label export → `train_detector.py --data` |
| N-09 | Edge-first, offline-first | ✅ (ONNX measured; TRT/Hailo runbooks) |
| N-10 | Citizen sonar | ✅ Lowrance + Humminbird parsers, dashboard upload (.sl2/.sl3/.zip) |
| N-11 | SONAR-GPT copilot | ✅ queries + survey summaries |
| N-12 | Disaster-mode profiles | ✅ ghost-net / port-clearance / SAR / aircraft-search |

## Blueprint targets vs measured reality

| Target (blueprint §6) | Measured here | Note |
|---|---|---|
| ≥0.90 mAP@50 on held-out SSS benchmark | 0.653 on *synthetic* val | The 0.90–0.94 published numbers are on real KLSG/SCTD after full training campaigns; our number is honest closed-world synthetic. Real-dataset path: `scripts/download_datasets.py` + flywheel |
| ≥60% FP reduction from PhysiCheck | demote-not-delete design; seeded FP demonstrably demoted | measured FP-reduction % needs a real labeled benchmark |
| <25 ms/tile on Orin Nano | 23 ms/tile ONNX on this *laptop CPU* | Orin INT8 will be faster; needs the device |
| Staged FLS→SSS transfer, three-seed ensemble | three-seed ensemble ✅; staged transfer 📋 (needs dataset downloads) | |

Tech-stack deltas from §7, all deliberate and logged in DECISIONS.md: leaflet.heat for
deck.gl, hand-rolled CSS for Tailwind, SQLite-first for PostGIS, pyproj-only geometry for
Shapely/GeoPandas, Ultralytics `runs/` for MLflow, PyWavelets unused (wavelet speckle
decoupling is WPG-DetNet-frontier research, not required by our despeckle chain).
