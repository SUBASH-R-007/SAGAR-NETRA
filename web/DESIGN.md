# DRISHTI Console — design system

Identity: **"Chart room, daylight."** The console borrows its visual language from
paper nautical charts — the documents hydrographers actually trust: light ground,
ink-precise hairlines, condensed uppercase labels, tabular figures, and the one
famous chart accent, **chart magenta** (the color of buoys, restricted areas and
notes on every NOAA/Admiralty chart). Dark sonar imagery sits framed inside light
chrome, the way a photo editor frames photographs. One deliberate theme — light —
committed everywhere; no dark mode, no theme toggle.

## Tokens (the only colors and fonts allowed)

```css
:root {
  /* ground */
  --paper:      #F3F5F2;  /* page ground — cool chart paper, green-grey bias */
  --panel:      #FDFDFC;  /* raised panels, table ground */
  --well:       #E8EEF0;  /* imagery/map wells — pale water tint */
  --hairline:   #D4DBD6;  /* 1px rules everywhere */
  --hairline-2: #B9C4BD;  /* emphasized rules (header keylines, table heads) */

  /* ink */
  --ink:        #1C2A35;  /* chart ink — blue-black */
  --ink-dim:    #5C6E7A;  /* secondary text */
  --ink-faint:  #8C9BA5;  /* captions, disabled */

  /* identity accent — use SPARINGLY: active-tab underline, primary buttons,
     focus rings, links, the survey track. Never for status. */
  --magenta:      #C0107E;
  --magenta-ink:  #8F0A5D;  /* hover/pressed */
  --magenta-wash: #F9E7F2;  /* selected-row / active chip wash */

  /* severity semantics — independent of the accent, square swatches only */
  --sev-critical: #BC3116;
  --sev-high:     #C67102;
  --sev-moderate: #A08C00;
  --sev-low:      #3E7D52;

  /* state semantics */
  --ok:     #2E6E44;
  --error:  #B3261E;
}
```

## Typography (bundled via @fontsource — the console is offline-first, NO
Google Fonts links)

| Role | Face | Usage |
|---|---|---|
| Display / wordmark / tab labels | **Archivo** 600–700 | `SAGAR-NETRA` wordmark, section titles. Uppercase eyebrows: 11px, 600, letter-spacing 0.08em |
| Body / UI | **Public Sans** 400/500/600 | everything readable; 13.5–14px base |
| Data | **IBM Plex Mono** 400/500 | IDs, coordinates, numbers, file formats; ALWAYS `font-variant-numeric: tabular-nums` |

Numbers right-align in tables. Coordinates always mono. No italics anywhere.

## Form rules — the anti-generic kill list

- **Radius: 2px maximum.** No rounded-lg, no pill shapes. Chips/tags are
  rectangular, 1px hairline border, 2px radius, 10.5px uppercase Archivo.
- **Hairlines instead of shadows.** Panels separate by 1px `--hairline` borders
  on `--panel`. The ONLY shadow in the app: popovers/menus
  (`0 4px 16px rgba(28,42,53,.14)`).
- **No emoji in the UI.** No decorative icons. Where a mark is needed, use
  drawn geometry (a 8×8 square severity swatch, a 6px triangle disclosure).
- **No gradients, no glow, no backdrop blur.**
- **Color discipline:** magenta is identity, severity ramp is status; they never
  swap jobs. Body chrome is paper/ink only.
- **Buttons:** rectangular. Primary = magenta fill, white text. Secondary =
  1px ink border on panel. Tertiary/inline = magenta text link. 32px tall,
  13px Public Sans 600.
- **Focus:** 2px `--magenta` outline, 1px offset, on every interactive element.
- **Density:** tables at 34–36px rows, 8px grid throughout, generous ONLY around
  the header block and section titles.

## Structure

- **Header = chart title block.** Like the title panel of a survey chart:
  wordmark line (`SAGAR-NETRA` Archivo 700 + thin vertical rule + `DRISHTI
  SURVEY CONSOLE` eyebrow), beneath it a metadata line in mono
  (`DATUM WGS-84 · SURVEY survey_alpha.xtf · 14 CONTACTS · MoES / NIOT PS 26057`),
  closed by a **3px magenta keyline** above a 1px hairline. This keyline is the
  page's single flourish.
- **Tabs:** plain uppercase Archivo text, ink-dim → ink on hover; active = ink +
  2px magenta underline flush with the header hairline. No boxes.
- **Filter bar:** one hairline strip; selects and inputs are 1px-bordered
  rectangles on panel; labels are mono 10.5px uppercase `--ink-faint`.
- **Upload rail → "INGEST LEDGER":** dashed 1px `--hairline-2` rectangular drop
  target with mono caption of accepted formats; below, jobs as a ledger — each
  row: mono name, right-aligned status tag, 2px progress rule in magenta.
  BATCH/LIVE STREAM = a two-cell segmented control (rectangular, shared
  hairline, active cell = magenta-wash + magenta text).
- **Tables:** header row uppercase 10.5px Archivo 600 `--ink-dim` on `--paper`,
  double rule beneath (`--hairline-2`); body rows hairline-separated; hover =
  `--well`; selected = `--magenta-wash`. Severity cell = square swatch + mono
  number. Physics cues = rectangular tags `HL` `SH` (ink on panel; violation tag
  in `--sev-critical` outline).
- **Map & waterfall:** sit in `--well` wells with a 1px `--hairline-2` frame.
  Leaflet controls restyled to match (white panel, hairline, ink glyphs, 2px
  radius). Legend = white panel, hairline, uppercase eyebrows, square swatches.
- **Toasts:** bottom-left, rectangular panel, 3px left rule in `--ok`/`--error`.
- **Empty states:** centered, eyebrow + one sentence, ink-dim; no illustration.

## Voice

Labels name what operators recognize: "Survey", "Contacts", "Evidence",
"Recovery", "Confirm", "Reject". No jargon-as-decoration, no exclamation marks,
no "..." except genuine progress. Buttons say what happens.
