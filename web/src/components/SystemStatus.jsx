import { useCallback, useEffect, useState } from 'react'
import { fetchHealth } from '../api'
import EmptyState from './EmptyState'

// ── model registry ─────────────────────────────────────────────────────────
// One row per trained artefact the stack loads. `loadedKey` indexes
// health.models_loaded, `versionKey` indexes health.versions — both come
// straight from GET /api/health (api/realtime.py :: model_inventory), which
// reports file existence and a content hash only, never a live model load.
// The physics-gate config has no models_loaded entry: file_version() returns
// null exactly when the file is absent, so its presence is read off the
// fingerprint itself rather than invented.
const REGISTRY = [
  {
    name: 'Brain A — detector',
    role: 'Supervised YOLO ensemble over preprocessed tiles',
    artefact: 'weights/detector.pt',
    loadedKey: 'detector',
    versionKey: 'detector',
  },
  {
    name: 'Brain B — segmenter',
    role: 'U-Net masks; footprints for filamentous targets',
    artefact: 'weights/segmenter.pt',
    loadedKey: 'segmenter',
    versionKey: 'segmenter',
  },
  {
    name: 'Brain C — anomaly autoencoder',
    role: 'Reconstruction error; open-set finds with no labels',
    artefact: 'weights/anomaly.pt',
    loadedKey: 'anomaly',
    versionKey: 'anomaly',
  },
  {
    name: 'Stage-2 physics verifier',
    role: 'Learned gate over highlight / shadow features',
    artefact: 'weights/verifier.pkl',
    loadedKey: 'verifier',
    versionKey: 'verifier_model',
  },
  {
    name: 'Physics gate tunables',
    role: 'Thresholds the verifier and shadow analysis read',
    artefact: 'configs/physics.yaml',
    loadedKey: null, // presence is derived from the fingerprint (see above)
    versionKey: 'verifier_config',
  },
]

// ── processing chain ───────────────────────────────────────────────────────
// Layer names and stages mirror the packages on disk: sonar_core/preprocess
// (bottom_track, slant_range, egn, despeckle, clahe, tiler), tridentnet,
// physicheck, geoscribe, and this console.
const CHAIN = [
  {
    id: 'L1',
    name: 'SonicPrep',
    w: 232,
    stages: [
      'Bottom track',
      'Slant-range correction',
      'Gain normalization',
      'Despeckle',
      'CLAHE',
      'Tiling',
    ],
  },
  {
    id: 'L2',
    name: 'TridentNet',
    w: 240,
    stages: [
      'Brain A — YOLO ensemble',
      'Brain B — U-Net masks',
      'Brain C — anomaly autoencoder',
    ],
  },
  {
    id: 'L3',
    name: 'PhysiCheck',
    w: 230,
    stages: [
      'Highlight + shadow',
      'Height  H = L·A / R',
      'Stage-2 verifier',
      'Confidence calibration',
    ],
  },
  {
    id: 'L4',
    name: 'GeoScribe',
    w: 190,
    stages: ['WGS-84 geotag', 'Severity index', 'Reports (5 formats)'],
  },
  {
    id: 'L5',
    name: 'DRISHTI',
    w: 156,
    stages: ['Map + waterfall', 'Review + recovery', 'Change detection'],
  },
]

const CHAIN_GAP = 28
const CHAIN_TOP = 28
const CHAIN_BOX_H = 170
const CHAIN_W = CHAIN.reduce((sum, l) => sum + l.w, 0) + CHAIN_GAP * (CHAIN.length - 1)
const CHAIN_H = CHAIN_TOP + CHAIN_BOX_H + 22
const CHAIN_MID = CHAIN_TOP + CHAIN_BOX_H / 2 + 0.5

// x offsets resolved once, so the SVG geometry can never drift from the data
const CHAIN_BOXES = (() => {
  let x = 0
  return CHAIN.map((l) => {
    const box = { ...l, x }
    x += l.w + CHAIN_GAP
    return box
  })
})()

// The SVG carries the whole chain as its accessible name — a screen reader
// gets the same five layers and stage lists a sighted operator reads.
const CHAIN_LABEL =
  'Processing chain, five layers in order. ' +
  CHAIN.map((l) => `${l.id} ${l.name}: ${l.stages.join(', ')}`).join('. Feeds ') +
  '.'

// ── ingest capability ──────────────────────────────────────────────────────
// Extensions match api/main.py :: UPLOAD_SUFFIXES exactly; each row has a
// parser module under sonar_core/parsers/.
const FORMATS = [
  {
    format: 'XTF',
    ext: '.xtf',
    what: 'eXtended Triton Format — the side-scan survey standard',
  },
  { format: 'EdgeTech JSF', ext: '.jsf', what: 'EdgeTech native side-scan / sub-bottom' },
  { format: 'Lowrance SL2 / SL3', ext: '.sl2 .sl3', what: 'Recreational and citizen sonar' },
  {
    format: 'Humminbird DAT + SON',
    ext: '.zip',
    what: 'Citizen sonar — upload the recording folder as a .zip',
  },
  { format: 'GeoTIFF mosaic', ext: '.tif .tiff', what: 'Georeferenced sonar mosaic' },
  { format: 'PNG / JPG waterfall', ext: '.png .jpg .jpeg', what: 'Waterfall image without navigation' },
]

// ── small presentational pieces ────────────────────────────────────────────

function Section({ title, children }) {
  return (
    <section className="sys-section">
      <h2 className="rail-title sys-title">{title}</h2>
      {children}
    </section>
  )
}

function FactStrip({ items }) {
  return (
    <dl className="sys-facts">
      {items.map((it) => (
        <div className="sys-fact" key={it.label}>
          <dt className="sys-fact-label mono">{it.label}</dt>
          <dd className="sys-fact-value">{it.value}</dd>
        </div>
      ))}
    </dl>
  )
}

// A fingerprint is one of three real things: a sha1-8 hash, the literal
// "pretrained-fallback" the API substitutes when detector weights are absent,
// or null. Each renders as itself — nothing is filled in.
function Fingerprint({ value }) {
  if (value === 'pretrained-fallback') {
    return <span className="sys-fp-note">pretrained-fallback</span>
  }
  if (!value) {
    return <span className="mono sys-dash">—</span>
  }
  return <span className="mono">{value}</span>
}

function ChainDiagram() {
  return (
    <div className="sys-chain-scroll">
      <svg
        className="sys-chain"
        viewBox={`0 0 ${CHAIN_W} ${CHAIN_H}`}
        width={CHAIN_W}
        height={CHAIN_H}
        role="img"
        aria-label={CHAIN_LABEL}
      >
        {CHAIN_BOXES.map((l, i) => (
          <g key={l.id}>
            <rect
              className="sys-chain-box"
              x={l.x + 0.5}
              y={CHAIN_TOP + 0.5}
              width={l.w - 1}
              height={CHAIN_BOX_H - 1}
              shapeRendering="crispEdges"
            />
            <text className="sys-chain-id" x={l.x + 12} y={CHAIN_TOP + 22}>
              {l.id}
            </text>
            <text className="sys-chain-name" x={l.x + 36} y={CHAIN_TOP + 22}>
              {l.name.toUpperCase()}
            </text>
            <line
              className="sys-chain-rule"
              x1={l.x + 12}
              y1={CHAIN_TOP + 34.5}
              x2={l.x + l.w - 12}
              y2={CHAIN_TOP + 34.5}
              shapeRendering="crispEdges"
            />
            {l.stages.map((s, j) => (
              <g key={s}>
                <rect
                  className="sys-chain-dot"
                  x={l.x + 13}
                  y={CHAIN_TOP + 48 + j * 20}
                  width={4}
                  height={4}
                  shapeRendering="crispEdges"
                />
                <text className="sys-chain-stage" x={l.x + 25} y={CHAIN_TOP + 54 + j * 20}>
                  {s}
                </text>
              </g>
            ))}
            {i < CHAIN_BOXES.length - 1 && (
              <g className="sys-chain-arrow">
                <line
                  x1={l.x + l.w + 6}
                  y1={CHAIN_MID}
                  x2={l.x + l.w + CHAIN_GAP - 9}
                  y2={CHAIN_MID}
                  shapeRendering="crispEdges"
                />
                <polygon
                  points={
                    `${l.x + l.w + CHAIN_GAP - 9},${CHAIN_MID - 4} ` +
                    `${l.x + l.w + CHAIN_GAP - 9},${CHAIN_MID + 4} ` +
                    `${l.x + l.w + CHAIN_GAP - 1},${CHAIN_MID}`
                  }
                />
              </g>
            )}
          </g>
        ))}
      </svg>
    </div>
  )
}

// ── view ───────────────────────────────────────────────────────────────────

export default function SystemStatus({ pushToast }) {
  const [health, setHealth] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [readAt, setReadAt] = useState(null)

  const load = useCallback(
    async (alive = () => true) => {
      setLoading(true)
      try {
        const data = await fetchHealth()
        if (!alive()) return
        setHealth(data)
        setError(null)
        setReadAt(new Date())
      } catch (err) {
        if (!alive()) return
        setError(err.message)
        pushToast(`Health check failed: ${err.message}`, 'error')
      } finally {
        if (alive()) setLoading(false)
      }
    },
    [pushToast],
  )

  useEffect(() => {
    let mounted = true
    load(() => mounted)
    return () => {
      mounted = false
    }
  }, [load])

  const versions = (health && health.versions) || {}
  const loaded = (health && health.models_loaded) || {}
  const lastSurvey = health ? health.last_survey : null
  const tilesPerS = health ? health.tiles_per_s_last : null

  // service strip — every cell omitted or dashed when the field is missing
  const serviceItems = []
  if (health) {
    serviceItems.push({
      label: 'Name',
      value: health.service ? (
        <span className="mono">{health.service}</span>
      ) : (
        <span className="mono sys-dash">—</span>
      ),
    })
    serviceItems.push({
      label: 'Status',
      value: (
        <span className={`tag ${health.status === 'ok' ? 'tag-done' : 'tag-error'}`}>
          {String(health.status || 'unknown')}
        </span>
      ),
    })
    serviceItems.push({
      label: 'Resident memory',
      value:
        health.memory_mb == null ? (
          <span className="mono sys-dash">—</span>
        ) : (
          <span className="mono">{Number(health.memory_mb).toFixed(1)} MB</span>
        ),
    })
    serviceItems.push({
      label: 'Pipeline',
      value: <Fingerprint value={versions.pipeline} />,
    })
  }

  // last run — each fact appears only when the API actually reported it
  const runItems = []
  if (lastSurvey && lastSurvey.name) {
    runItems.push({ label: 'Survey', value: <span className="mono">{lastSurvey.name}</span> })
  }
  if (lastSurvey && lastSurvey.n_contacts != null) {
    runItems.push({
      label: 'Contacts',
      value: <span className="mono">{lastSurvey.n_contacts}</span>,
    })
  }
  if (lastSurvey && lastSurvey.seconds != null) {
    runItems.push({
      label: 'Wall time',
      value: <span className="mono">{Number(lastSurvey.seconds).toFixed(2)} s</span>,
    })
  }
  if (tilesPerS != null) {
    runItems.push({
      label: 'Throughput',
      value: <span className="mono">{Number(tilesPerS).toFixed(2)} tiles/s</span>,
    })
  }

  return (
    <div className="sys">
      <div className="sys-toolbar">
        <span className="ctl-label">Edge node telemetry</span>
        <button type="button" className="btn small" onClick={() => load()} disabled={loading}>
          Refresh
        </button>
        <span className="filter-count mono" aria-live="polite">
          {loading
            ? 'Reading /api/health…'
            : readAt
              ? `Read at ${readAt.toLocaleTimeString()}`
              : 'Not read'}
        </span>
      </div>

      {!health ? (
        loading ? (
          <EmptyState
            title="Reading node telemetry"
            hint="Querying /api/health for model fingerprints, resident memory and last-run throughput."
          />
        ) : (
          <EmptyState
            title="Telemetry unavailable"
            hint={`The health endpoint did not answer: ${error || 'unknown error'}. Use Refresh to try again.`}
          />
        )
      ) : (
        <div className="sys-body">
          {error && (
            <p className="sys-stale">
              Showing the last successful read — the most recent refresh failed: {error}
            </p>
          )}

          {/* 1 · service */}
          <Section title="Service">
            <FactStrip items={serviceItems} />
          </Section>

          {/* 2 · model registry */}
          <Section title="Model registry">
            <div className="sys-table-wrap">
              <table className="contacts-table sys-table">
                <thead>
                  <tr>
                    <th>Brain</th>
                    <th>Role</th>
                    <th>Artefact</th>
                    <th>State</th>
                    <th>Fingerprint</th>
                  </tr>
                </thead>
                <tbody>
                  {REGISTRY.map((row) => {
                    const fp = versions[row.versionKey]
                    const on = row.loadedKey ? !!loaded[row.loadedKey] : fp != null
                    const known = row.loadedKey
                      ? loaded[row.loadedKey] !== undefined
                      : fp !== undefined
                    const label = row.loadedKey
                      ? on
                        ? 'LOADED'
                        : 'ABSENT'
                      : on
                        ? 'PRESENT'
                        : 'ABSENT'
                    return (
                      <tr key={row.artefact}>
                        <td>{row.name}</td>
                        <td className="muted">{row.role}</td>
                        <td className="mono sys-artefact">{row.artefact}</td>
                        <td>
                          {known ? (
                            // absent is a fact, not a fault: the plain neutral
                            // tag, never the error ramp
                            <span className={on ? 'tag tag-done' : 'tag'}>{label}</span>
                          ) : (
                            <span className="mono sys-dash">—</span>
                          )}
                        </td>
                        <td>
                          <Fingerprint value={fp} />
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
            <p className="sys-caption">
              A fingerprint is a sha1-8 content hash of the artefact file itself, not a version
              number somebody typed. Retraining changes the bytes, so it changes the hash — the
              exact weights behind any detection stay identifiable, and a silent model swap is
              impossible to hide. ABSENT means no file on disk: the detector then falls back to
              pretrained weights, and Brains B and C are simply skipped by the ensemble.
            </p>
          </Section>

          {/* 3 · last run */}
          <Section title="Last run">
            {runItems.length > 0 ? (
              <FactStrip items={runItems} />
            ) : (
              <p className="sys-caption">
                No survey has been processed on this node yet. Ingest one from the rail on the
                right and its contact count, wall time and tile throughput appear here.
              </p>
            )}
            {lastSurvey && lastSurvey.seconds == null && (
              <p className="sys-caption">
                Wall time is reported only for runs completed by the server process now running —
                this survey was processed earlier, so its timing is not claimed here.
              </p>
            )}
            {tilesPerS != null && (
              <p className="sys-caption">
                Throughput is the detector tile rate this node measured on its own last run —
                the tiles that run produced divided by its wall-clock seconds, timed by the
                server process that executed it. It is the record of one run on this hardware,
                not a benchmark and not a claim about any other machine or sensor.
              </p>
            )}
          </Section>

          {/* 4 · processing chain */}
          <Section title="Processing chain">
            <ChainDiagram />
            <p className="sys-caption">
              Every stage above runs on this machine. L3 gates L2&apos;s candidates against sonar
              physics before L4 is allowed to write a contact, which is why a detection in this
              console carries measured highlight and shadow evidence rather than a bare
              confidence score.
            </p>
          </Section>

          {/* 5 · ingest capability */}
          <Section title="Ingest capability">
            <div className="sys-table-wrap">
              <table className="contacts-table sys-table">
                <thead>
                  <tr>
                    <th>Format</th>
                    <th>Extension</th>
                    <th>What it is</th>
                  </tr>
                </thead>
                <tbody>
                  {FORMATS.map((f) => (
                    <tr key={f.format}>
                      <td>{f.format}</td>
                      <td className="mono">{f.ext}</td>
                      <td className="muted">{f.what}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="sys-caption">
              Citizen-sonar support matters operationally: Lowrance and Humminbird units are
              already fitted to fishing boats, so cooperatives can contribute usable surveys with
              hardware they own, without a survey-grade sonar or a research vessel.
            </p>
          </Section>

          {/* 6 · offline posture */}
          <Section title="Offline posture">
            <div className="sys-posture">
              <p>
                Inference makes zero network calls. Preprocessing, the three brains, physics
                verification, geotagging and report writing all execute in this process; no
                imagery, position or contact leaves the machine.
              </p>
              <p>
                Basemap tiles are proxied by this server and cached to disk per source. A tile
                fetched once is served from the cache thereafter, and when a tile has never been
                seen and there is no link, the map draws a neutral sea grid instead of failing.
              </p>
              <p>
                Models and fonts are bundled locally — weights sit beside the code, typefaces are
                compiled into the bundle. There is no CDN, no external font link and no API key
                anywhere in the deployment.
              </p>
            </div>
          </Section>
        </div>
      )}
    </div>
  )
}
