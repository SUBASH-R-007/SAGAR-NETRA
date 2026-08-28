import { useEffect, useMemo, useState } from 'react'
import { fetchSummary } from '../api'
import { SEVERITY_BANDS, fmtMeters, prettyName, sevColor } from '../utils'
import EmptyState from './EmptyState'
import SeverityChip from './SeverityChip'

// COMMAND OVERVIEW — the landing view. It answers two questions with real
// numbers only: what did this survey find, and what does the operator do next.
//
// Two sources feed it, and nothing else:
//   * `contacts` — the full unfiltered contact list for the selected survey
//     (every count in sections 2-5 is derived from it);
//   * GET /api/summary — the survey-level block geoscribe wrote into
//     contacts.json (coverage, density, sonar config, provenance). It 404s
//     until a report exists, which is a normal state, not an error: the
//     contact-derived tiles still stand and the rest shows an honest dash.

const QUEUE_SIZE = 5

// geoscribe.report.HIGH_CONFIDENCE_PCT — the same floor the pipeline uses for
// the summary block, so the fallback count below can never disagree with the
// reported one.
const HIGH_CONFIDENCE_PCT = 70

// SEVERITY_BANDS is ordered critical-first (utils.js); index 0 is the >= 75 band.
const CRITICAL_BAND = SEVERITY_BANDS[0]

// Indian digit grouping — this is a Government of India portal. Values are
// printed as the API returned them (max 4 dp covers area_surveyed_sqkm).
function fmtNum(v) {
  if (v == null || v === '') return null
  const n = Number(v)
  return Number.isFinite(n) ? n.toLocaleString('en-IN', { maximumFractionDigits: 4 }) : null
}

// 'critical  ≥75' -> ['critical', '≥75'] (the label carries both halves).
function splitBand(label) {
  const parts = String(label).split(/\s{2,}/)
  return { name: parts[0], range: parts[1] || '' }
}

function Tile({ label, value, unit, unitLabel, context, tone, target, onTab }) {
  const spoken = value == null ? 'not available' : `${value}${unitLabel ? ` ${unitLabel}` : ''}`
  return (
    <button
      type="button"
      className="ov-tile"
      onClick={() => onTab(target)}
      aria-label={`${label}: ${spoken}. Opens the ${target} view.`}
    >
      <span className="ov-tile-value mono" style={tone ? { color: tone } : undefined}>
        {value == null ? '—' : value}
        {value != null && unit ? <span className="ov-tile-unit"> {unit}</span> : null}
      </span>
      <span className="ov-tile-label">{label}</span>
      <span className="ov-tile-context">{context}</span>
    </button>
  )
}

export default function Overview({ contacts, survey, surveys, onTab, pushToast }) {
  const [summary, setSummary] = useState(null)
  // idle (no survey) | loading | ready | absent (404, no report yet) | error
  const [state, setState] = useState('idle')

  useEffect(() => {
    if (!survey) {
      setSummary(null)
      setState('idle')
      return undefined
    }
    let alive = true
    setSummary(null)
    setState('loading')
    fetchSummary(survey)
      .then((doc) => {
        if (!alive) return
        setSummary(doc)
        setState('ready')
      })
      .catch((err) => {
        if (!alive) return
        setSummary(null)
        // 404 = this survey has no generated report yet. Expected, so quiet.
        if (String(err.message).startsWith('404')) {
          setState('absent')
        } else {
          setState('error')
          pushToast(`Could not load the survey summary: ${err.message}`, 'error')
        }
      })
    return () => {
      alive = false
    }
  }, [survey, pushToast])

  // Bucket by the band color sevColor() returns, so the thresholds live in
  // exactly one place (utils.js) and the bar can never drift from the ramp.
  const bands = useMemo(() => {
    const counts = new Map()
    for (const c of contacts) {
      const key = sevColor(c.severity)
      counts.set(key, (counts.get(key) || 0) + 1)
    }
    return SEVERITY_BANDS.map((b) => ({
      ...b,
      ...splitBand(b.label),
      count: counts.get(b.color) || 0,
    }))
  }, [contacts])

  const classRows = useMemo(() => {
    const counts = new Map()
    for (const c of contacts) counts.set(c.cls, (counts.get(c.cls) || 0) + 1)
    return [...counts.entries()]
      .map(([cls, count]) => ({ cls, count }))
      .sort((a, b) => b.count - a.count || a.cls.localeCompare(b.cls))
  }, [contacts])

  const queue = useMemo(
    () => [...contacts].sort((a, b) => b.severity - a.severity).slice(0, QUEUE_SIZE),
    [contacts],
  )

  const physics = useMemo(() => {
    let both = 0
    let height = 0
    let violation = 0
    let confSum = 0
    for (const c of contacts) {
      const p = c.physics || {}
      if (p.highlight && p.shadow) both += 1
      if (c.dims && c.dims.height_m != null) height += 1
      if (p.physics_violation) violation += 1
      confSum += Number(c.confidence) || 0
    }
    return {
      both,
      height,
      violation,
      meanConf: contacts.length ? confSum / contacts.length : null,
    }
  }, [contacts])

  const derivedHighConf = useMemo(
    () => contacts.filter((c) => c.confidence >= HIGH_CONFIDENCE_PCT).length,
    [contacts],
  )

  if (!survey) {
    return (
      <EmptyState
        title="No survey selected"
        hint="Drop a sonar file into the ingest rail on the right — the command overview for that survey appears here once the pipeline finishes."
      />
    )
  }

  if (state === 'loading' && contacts.length === 0) {
    return <EmptyState title="Loading survey" hint={`Reading the roll-up for ${survey}.`} />
  }

  // The surveys row carries the ingested ping count, so the pings tile still
  // has a real figure when a survey has no generated report yet.
  const surveyRow = (surveys || []).find((s) => s.name === survey)
  const cfg = summary?.sonar_config || null
  const pings = cfg?.n_pings ?? surveyRow?.n_pings ?? null
  const total = contacts.length
  const critical = bands[0].count
  const maxClass = classRows.length ? classRows[0].count : 0

  // Stacked-bar geometry: one 1000-unit viewBox row, bands in ramp order. The
  // painted rect and its overlay hit button are driven by the same percentage,
  // so the click targets line up with the paint exactly.
  let acc = 0
  const segments = bands
    .filter((b) => b.count > 0)
    .map((b) => {
      const pct = (b.count / total) * 100
      const seg = { ...b, pct, x: acc * 10 }
      acc += pct
      return seg
    })

  // Why a summary-fed figure is missing, in the tile's own context line.
  const summaryNote =
    state === 'loading'
      ? 'reading survey report'
      : state === 'absent'
        ? 'no survey report for this run'
        : 'survey report unavailable'

  const area = fmtNum(summary?.area_surveyed_sqkm)
  const density = fmtNum(summary?.debris_density_per_sqkm)
  const pingText = fmtNum(pings)

  const sonarContext = () => {
    if (cfg?.range_m != null) {
      return cfg.altitude_m != null
        ? `${fmtMeters(cfg.range_m, 0)} range · ${fmtMeters(cfg.altitude_m, 0)} altitude`
        : `${fmtMeters(cfg.range_m, 0)} range`
    }
    if (pings != null) return 'ping records ingested'
    return summaryNote
  }

  // Provenance: only fields actually present, joined into one mono line.
  const prov = []
  prov.push(`SURVEY ${summary?.survey || survey}`)
  if (cfg?.range_m != null) prov.push(`RANGE ${fmtMeters(cfg.range_m)}`)
  if (cfg?.altitude_m != null) prov.push(`ALTITUDE ${fmtMeters(cfg.altitude_m)}`)
  if (pingText != null) prov.push(`PINGS ${pingText}`)
  if (summary?.pipeline_version) prov.push(`PIPELINE v${summary.pipeline_version}`)
  if (summary?.generated_at) prov.push(`GENERATED ${summary.generated_at}`)

  return (
    <div className="overview">
      {/* 1 · KPI tile row */}
      <section className="ov-section">
        <h2 className="ov-eyebrow">Survey roll-up</h2>
        <div className="ov-tiles">
          <Tile
            label="Total contacts"
            value={fmtNum(total)}
            context={`${classRows.length} ${classRows.length === 1 ? 'class' : 'classes'} detected`}
            target="Contacts"
            onTab={onTab}
          />
          <Tile
            label="Critical · sev ≥ 75"
            value={fmtNum(critical)}
            tone={CRITICAL_BAND.color}
            context={total ? `${Math.round((critical / total) * 100)}% of contacts` : 'no contacts'}
            target="Recovery"
            onTab={onTab}
          />
          <Tile
            label="High confidence"
            value={fmtNum(summary?.high_confidence ?? derivedHighConf)}
            context={`at or above ${HIGH_CONFIDENCE_PCT}% calibrated`}
            target="Contacts"
            onTab={onTab}
          />
          <Tile
            label="Area surveyed"
            value={area}
            unit="km²"
            unitLabel="square kilometres"
            context={area == null ? summaryNote : 'swath coverage from ping navigation'}
            target="Map"
            onTab={onTab}
          />
          <Tile
            label="Debris density"
            value={density}
            unit="/km²"
            unitLabel="per square kilometre"
            context={density == null ? summaryNote : 'contacts per square kilometre'}
            target="Map"
            onTab={onTab}
          />
          <Tile
            label="Pings surveyed"
            value={pingText}
            context={sonarContext()}
            target="Waterfall"
            onTab={onTab}
          />
        </div>
      </section>

      {total === 0 ? (
        <EmptyState
          title="No contacts in this survey"
          hint="The pipeline completed without retaining a contact for this survey. The run that produced it is listed in the ingest ledger on the right."
        />
      ) : (
        <>
          <div className="ov-split">
            {/* 2 · severity distribution — one SVG, four segments */}
            <section className="ov-section">
              <h2 className="ov-eyebrow">Severity distribution</h2>
              <div className="ov-panel">
                <div className="ov-bar-wrap">
                  <svg
                    className="ov-bar"
                    viewBox="0 0 1000 28"
                    preserveAspectRatio="none"
                    aria-hidden="true"
                    focusable="false"
                  >
                    {segments.map((s) => (
                      <rect
                        key={s.color}
                        x={s.x}
                        y="0"
                        width={s.pct * 10}
                        height="28"
                        fill={s.color}
                      />
                    ))}
                  </svg>
                  <div className="ov-bar-hits">
                    {segments.map((s) => (
                      <button
                        key={s.color}
                        type="button"
                        className="ov-bar-hit"
                        style={{ width: `${s.pct}%` }}
                        title={`${s.name} ${s.range} — ${s.count} contacts`}
                        aria-label={`${s.name} severity ${s.range}: ${s.count} contacts. Opens the Contacts view.`}
                        onClick={() => onTab('Contacts')}
                      />
                    ))}
                  </div>
                </div>
                <div className="ov-bands">
                  {bands.map((b) => (
                    <div key={b.color} className="ov-band">
                      <span className="ov-band-sq" style={{ background: b.color }} aria-hidden="true" />
                      <span className="ov-band-count mono">{b.count}</span>
                      <span className="ov-band-name">{b.name}</span>
                      <span className="ov-band-range mono">{b.range}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            {/* 3 · class breakdown */}
            <section className="ov-section">
              <h2 className="ov-eyebrow">Class breakdown</h2>
              <div className="ov-panel ov-classes">
                {classRows.map((r) => (
                  <div key={r.cls} className="ov-class-row">
                    <span className="ov-class-name" title={prettyName(r.cls)}>
                      {prettyName(r.cls)}
                    </span>
                    <span className="ov-class-track">
                      <span
                        className="ov-class-fill"
                        style={{ width: `${(r.count / maxClass) * 100}%` }}
                      />
                    </span>
                    <span className="ov-class-count mono">{r.count}</span>
                  </div>
                ))}
              </div>
            </section>
          </div>

          {/* 4 · priority action queue */}
          <section className="ov-section">
            <h2 className="ov-eyebrow">Priority action queue</h2>
            <p className="ov-sub">
              A decision tool, not a bounding-box generator: the{' '}
              {Math.min(QUEUE_SIZE, total)}{' '}
              {Math.min(QUEUE_SIZE, total) === 1 ? 'contact' : 'contacts'} to act on first, each
              carrying the instruction the pipeline recommends.
            </p>
            <div className="ov-queue">
              <table className="contacts-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Class</th>
                    <th className="num">Sev</th>
                    <th>Priority</th>
                    <th className="ov-action-col">Recommended action</th>
                  </tr>
                </thead>
                <tbody>
                  {/* The ID cell's button is each row's keyboard-reachable
                      control. A click handler on the <tr> itself would be
                      mouse-only, so the rows carry none. */}
                  {queue.map((c) => (
                    <tr key={c.id} className="row">
                      <td className="mono id-cell">
                        <button
                          type="button"
                          className="cell-btn"
                          aria-label={`Open ${c.id} in the Contacts view`}
                          onClick={() => onTab('Contacts')}
                        >
                          {c.id}
                        </button>
                      </td>
                      <td>{prettyName(c.cls)}</td>
                      <td className="num">
                        <SeverityChip value={c.severity} />
                      </td>
                      <td>
                        {c.priority ? (
                          <span className={`tag ov-prio-${String(c.priority).toLowerCase()}`}>
                            {c.priority}
                          </span>
                        ) : (
                          <span className="mono">—</span>
                        )}
                      </td>
                      <td className="ov-action-col">{c.recommended_action || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* 5 · physics evidence summary */}
          <section className="ov-section">
            <h2 className="ov-eyebrow">Physics evidence</h2>
            <p className="ov-sub">
              Computed across all {total} {total === 1 ? 'contact' : 'contacts'} in this survey.
            </p>
            <div className="ov-phys">
              <div className="ov-phys-item">
                <span className="ov-phys-value mono">{physics.both}</span>
                <span className="stat-label">highlight + shadow</span>
              </div>
              <div className="ov-phys-item">
                <span className="ov-phys-value mono">{physics.height}</span>
                <span className="stat-label">height measured</span>
              </div>
              <div className="ov-phys-item">
                <span
                  className="ov-phys-value mono"
                  style={physics.violation ? { color: CRITICAL_BAND.color } : undefined}
                >
                  {physics.violation}
                </span>
                <span className="stat-label">physics violation</span>
              </div>
              <div className="ov-phys-item">
                <span className="ov-phys-value mono">
                  {physics.meanConf == null ? '—' : `${physics.meanConf.toFixed(1)}%`}
                </span>
                <span className="stat-label">mean calibrated confidence</span>
              </div>
            </div>
          </section>
        </>
      )}

      {/* 6 · survey provenance */}
      <p className="ov-prov mono">{prov.join(' · ')}</p>
    </div>
  )
}
