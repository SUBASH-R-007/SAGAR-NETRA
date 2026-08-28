import { useEffect, useState } from 'react'
import { CircleMarker, MapContainer, TileLayer, Tooltip } from 'react-leaflet'
import { fetchCrossview, fetchDiff } from '../api'
import EmptyState from './EmptyState'
import MapResize from './MapResize'
import SeverityChip from './SeverityChip'

const CENTER = [13.05, 80.35]

// Percentages arrive as floats; a missing one shows an honest dash, never 0.
const pct = (v) => (v == null ? '—' : `${Math.round(Number(v))}%`)
const sep = (v) => (v == null ? '—' : `${Number(v).toFixed(1)} m`)
// physicheck.crossview matches an open-set anomaly against any class, so the
// two sides of a confirmed pair can legitimately carry different labels.
const pairClass = (a, b) => (a?.cls === b?.cls ? a?.cls || '—' : `${a?.cls} / ${b?.cls}`)
const confTitle = (c) => `raw ${pct(c?.confidence)} → adjusted ${pct(c?.adjusted_confidence)}`

// Compare two surveys of the same area: which contacts are NEW in survey B?
export default function DiffView({ surveys, pushToast }) {
  const [surveyA, setSurveyA] = useState('')
  const [surveyB, setSurveyB] = useState('')
  const [radius, setRadius] = useState(25)
  // Cross-view runs tighter than change detection: both passes are already
  // georeferenced, so 15 m is the API's own default residual-position budget.
  const [xRadius, setXRadius] = useState(15)
  const [result, setResult] = useState(null)
  const [diffError, setDiffError] = useState(null)
  const [cross, setCross] = useState(null)
  const [crossError, setCrossError] = useState(null)
  const [loading, setLoading] = useState(false)

  // Prefill A = oldest-ish first entry, B = second when the list arrives.
  useEffect(() => {
    if (surveys.length && !surveyA) {
      setSurveyA(surveys[0].name)
      setSurveyB((surveys[1] || surveys[0]).name)
    }
  }, [surveys, surveyA])

  const run = async () => {
    if (!surveyA || !surveyB) {
      pushToast('Pick both surveys first', 'error')
      return
    }
    setLoading(true)
    setResult(null)
    setDiffError(null)
    setCross(null)
    setCrossError(null)
    try {
      // One click, both analyses — issued together so neither waits on the
      // other, and a failure on one side still renders the other.
      const [diffRes, crossRes] = await Promise.allSettled([
        fetchDiff(surveyA, surveyB, radius),
        fetchCrossview(surveyA, surveyB, xRadius),
      ])
      if (diffRes.status === 'fulfilled') {
        setResult(diffRes.value)
      } else {
        setDiffError(diffRes.reason.message)
        pushToast(`Diff failed: ${diffRes.reason.message}`, 'error')
      }
      if (crossRes.status === 'fulfilled') {
        setCross(crossRes.value)
      } else {
        setCrossError(crossRes.reason.message)
        pushToast(`Cross-view failed: ${crossRes.reason.message}`, 'error')
      }
    } finally {
      setLoading(false)
    }
  }

  if (surveys.length === 0) {
    return (
      <EmptyState
        title="Nothing to compare yet"
        hint="Upload at least two surveys of the same area to run change detection and cross-view confirmation."
      />
    )
  }

  const newContacts = result?.new_contacts || []
  const matched = result?.matched || []
  const first = newContacts[0] || matched[0]?.b
  const center = first ? [first.lat, first.lon] : CENTER

  // Single-pass contacts from both sides, tagged with the pass they came from.
  const singles = [
    ...(cross?.a_only || []).map((c) => ({ c, side: 'A', survey: cross.survey_a })),
    ...(cross?.b_only || []).map((c) => ({ c, side: 'B', survey: cross.survey_b })),
  ]

  return (
    <div className="diff">
      <div className="diff-toolbar">
        <label className="ctl">
          <span className="ctl-label">Baseline (A)</span>
          <select value={surveyA} onChange={(e) => setSurveyA(e.target.value)}>
            {surveys.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
        </label>
        <label className="ctl">
          <span className="ctl-label">Revisit (B)</span>
          <select value={surveyB} onChange={(e) => setSurveyB(e.target.value)}>
            {surveys.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
        </label>
        <label className="ctl" title="drives change detection: what is NEW in B">
          <span className="ctl-label">Change radius (m)</span>
          <input
            className="num-input"
            type="number"
            min="1"
            max="500"
            value={radius}
            onChange={(e) => setRadius(Number(e.target.value) || 25)}
          />
        </label>
        <label className="ctl" title="drives cross-view: what both passes corroborate">
          <span className="ctl-label">Cross-view radius (m)</span>
          <input
            className="num-input"
            type="number"
            min="1"
            max="500"
            value={xRadius}
            onChange={(e) => setXRadius(Number(e.target.value) || 15)}
          />
        </label>
        <button type="button" className="btn primary" onClick={run} disabled={loading}>
          {loading ? 'Comparing…' : 'Compare'}
        </button>
      </div>

      {!result && !diffError && !cross && !crossError && !loading && (
        <EmptyState
          title="Run a comparison"
          hint={
            'Both analyses run on one click: change detection flags contacts in B with no A ' +
            'contact inside the change radius; cross-view corroborates contacts that both ' +
            'passes saw inside the cross-view radius.'
          }
        />
      )}

      {/* A toast is gone in six seconds; a half-empty page must still say why
          that half is missing, because the other analysis renders beside it. */}
      {diffError && !result && (
        <div className="diff-fail">
          <h3 className="diff-col-title">Change Detection</h3>
          <p className="xv-err">Change detection could not be computed — {diffError}</p>
        </div>
      )}

      {result && (
        <div className="diff-result">
          <div className="diff-stats">
            <div className="stat">
              <span className="stat-value mono">{result.n_a ?? '—'}</span>
              <span className="stat-label">contacts in A</span>
            </div>
            <div className="stat">
              <span className="stat-value mono">{result.n_b ?? '—'}</span>
              <span className="stat-label">contacts in B</span>
            </div>
            <div className="stat new">
              <span className="stat-value mono">{newContacts.length}</span>
              <span className="stat-label">NEW in B</span>
            </div>
            <div className="stat">
              <span className="stat-value mono">{matched.length}</span>
              <span className="stat-label">matched pairs</span>
            </div>
          </div>

          <div className="diff-map">
            <MapContainer
              key={`${surveyA}|${surveyB}|${newContacts.length}|${matched.length}`}
              center={center}
              zoom={13}
              className="map"
              preferCanvas
            >
              <MapResize />
              <TileLayer
                url="/tiles/{z}/{x}/{y}.png"
                attribution="Esri, Maxar, Earthstar Geographics, and the GIS community"
                maxNativeZoom={17}
                maxZoom={19}
              />
              {/* matched = ink-faint, NEW = sev-critical (DESIGN.md tokens) */}
              {matched.map((m, i) =>
                m.b ? (
                  <CircleMarker
                    key={`m-${m.b.id || i}`}
                    center={[m.b.lat, m.b.lon]}
                    radius={6}
                    pathOptions={{ color: '#8A94A0', fillColor: '#8A94A0', fillOpacity: 0.4, weight: 1.5 }}
                  >
                    <Tooltip>{`matched: ${m.b.id} · ${m.a?.id} (${Number(m.distance_m).toFixed(1)} m)`}</Tooltip>
                  </CircleMarker>
                ) : null,
              )}
              {newContacts.map((c) => (
                <CircleMarker
                  key={`n-${c.id}`}
                  center={[c.lat, c.lon]}
                  radius={8}
                  pathOptions={{ color: '#BC3116', fillColor: '#BC3116', fillOpacity: 0.65, weight: 2 }}
                >
                  <Tooltip>{`NEW: ${c.id} · ${c.cls} · sev ${Math.round(c.severity)}`}</Tooltip>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>

          <div className="diff-tables">
            <div className="diff-col">
              <h3 className="diff-col-title">New Contacts ({newContacts.length})</h3>
              {newContacts.length === 0 ? (
                <p className="muted">No new contacts — nothing appeared between the surveys.</p>
              ) : (
                <div className="table-scroll short">
                  <table className="contacts-table">
                    <thead>
                      <tr>
                        <th>
                          <span className="th-quiet">Change flag</span>
                        </th>
                        <th>ID</th>
                        <th>Class</th>
                        <th className="num">Conf</th>
                        <th className="num">Sev</th>
                        <th>Position</th>
                      </tr>
                    </thead>
                    <tbody>
                      {newContacts.map((c) => (
                        <tr key={c.id}>
                          <td>
                            <span className="tag new-tag">NEW</span>
                          </td>
                          <td className="mono">{c.id}</td>
                          <td>{c.cls}</td>
                          <td className="num mono">{Math.round(c.confidence)}%</td>
                          <td className="num">
                            <SeverityChip value={c.severity} />
                          </td>
                          <td className="mono">
                            {c.lat.toFixed(5)}, {c.lon.toFixed(5)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div className="diff-col">
              <h3 className="diff-col-title">Matched Pairs ({matched.length})</h3>
              {matched.length === 0 ? (
                <p className="muted">No matches within {radius} m.</p>
              ) : (
                <div className="table-scroll short">
                  <table className="contacts-table">
                    <thead>
                      <tr>
                        <th>B contact</th>
                        <th>A contact</th>
                        <th className="num">Distance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {matched.map((m, i) => (
                        <tr key={m.b?.id || i}>
                          <td className="mono">{m.b?.id ?? '—'}</td>
                          <td className="mono">{m.a?.id ?? '—'}</td>
                          <td className="num mono">{Number(m.distance_m).toFixed(1)} m</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {crossError && !cross && (
        <div className="xv">
          <h3 className="diff-col-title">Cross-View Confirmation</h3>
          <p className="xv-err">Cross-view could not be computed — {crossError}</p>
        </div>
      )}

      {cross && (
        <div className="xv">
          <div className="xv-head">
            <h3 className="diff-col-title">Cross-View Confirmation</h3>
            <p className="xv-doctrine">
              Two passes over the same seabed image the same real object twice. A contact
              matched on both passes at the same position is corroborated by independent
              acoustics, so its confidence is raised. A contact only one pass saw is demoted —
              never deleted — and flagged for a re-look, because absence on the other pass is
              a data-collection gap until a second sweep proves otherwise.
            </p>
            <p className="xv-note mono">
              CORROBORATION RADIUS {sep(cross.radius_m)} · A {cross.survey_a} · B{' '}
              {cross.survey_b}
            </p>
          </div>

          <div className="diff-stats">
            <div className="stat corroborated">
              <span className="stat-value mono">{cross.n_confirmed ?? '—'}</span>
              <span className="stat-label">confirmed pairs</span>
            </div>
            <div className="stat single">
              <span className="stat-value mono">{cross.n_a_only ?? '—'}</span>
              <span className="stat-label">A only</span>
            </div>
            <div className="stat single">
              <span className="stat-value mono">{cross.n_b_only ?? '—'}</span>
              <span className="stat-label">B only</span>
            </div>
          </div>

          <div className="xv-tables">
            <div className="xv-block">
              <h3 className="diff-col-title">Confirmed Pairs ({cross.confirmed?.length ?? 0})</h3>
              {(cross.confirmed || []).length === 0 ? (
                <p className="muted">
                  No contact was seen from both passes within {sep(cross.radius_m)}.
                </p>
              ) : (
                <div className="table-scroll short">
                  <table className="contacts-table">
                    <thead>
                      <tr>
                        <th>A contact</th>
                        <th>B contact</th>
                        <th>Class</th>
                        <th className="num">Separation</th>
                        <th className="num">Adj conf (A)</th>
                        <th className="num">Adj conf (B)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {cross.confirmed.map((p, i) => (
                        <tr key={`${p.a?.id}|${p.b?.id}|${i}`}>
                          <td className="mono">{p.a?.id ?? '—'}</td>
                          <td className="mono">{p.b?.id ?? '—'}</td>
                          <td>{pairClass(p.a, p.b)}</td>
                          <td className="num mono">{sep(p.distance_m)}</td>
                          <td className="num mono" title={confTitle(p.a)}>
                            {pct(p.a?.adjusted_confidence)}
                          </td>
                          <td className="num mono" title={confTitle(p.b)}>
                            {pct(p.b?.adjusted_confidence)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <p className="xv-note">
                Separation is the geodesic distance between the two fixes. Adjusted confidence
                is the corroboration-raised figure returned by the API — hover a cell for the
                raw detector value. A pair may carry two class labels: an open-set anomaly
                corroborates a classed contact, since both passes agree something non-seabed
                is there.
              </p>
            </div>

            <div className="xv-block">
              <h3 className="diff-col-title">Unconfirmed — Single Pass ({singles.length})</h3>
              {singles.length === 0 ? (
                <p className="muted">Every contact in both surveys was corroborated.</p>
              ) : (
                <div className="table-scroll short">
                  <table className="contacts-table">
                    <thead>
                      <tr>
                        <th>
                          <span className="th-quiet">Re-survey flag</span>
                        </th>
                        <th>Pass</th>
                        <th>ID</th>
                        <th>Class</th>
                        <th className="num">Conf</th>
                        <th className="num">Adj conf</th>
                        <th className="num">Sev</th>
                        <th>Position</th>
                      </tr>
                    </thead>
                    <tbody>
                      {singles.map(({ c, side, survey }) => (
                        <tr key={`${side}|${c.id}`}>
                          <td>
                            {c.resurvey_recommended ? (
                              <span className="tag resurvey-tag">RESURVEY</span>
                            ) : null}
                          </td>
                          <td>
                            <span className="tag side-tag" title={survey}>
                              {side}
                            </span>
                          </td>
                          <td className="mono">{c.id}</td>
                          <td>{c.cls}</td>
                          <td className="num mono">{pct(c.confidence)}</td>
                          <td className="num mono" title={confTitle(c)}>
                            {pct(c.adjusted_confidence)}
                          </td>
                          <td className="num">
                            <SeverityChip value={c.severity} />
                          </td>
                          <td className="mono">
                            {c.lat.toFixed(5)}, {c.lon.toFixed(5)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <p className="xv-note">
                RESURVEY marks a contact only one pass saw. Its confidence is demoted, it stays
                in the queue, and the re-look either corroborates it or clears it.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
