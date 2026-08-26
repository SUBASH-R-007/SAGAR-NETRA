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
