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
