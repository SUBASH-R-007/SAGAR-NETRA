import { useEffect, useState } from 'react'
import { CircleMarker, MapContainer, TileLayer, Tooltip } from 'react-leaflet'
import { fetchDiff } from '../api'
import EmptyState from './EmptyState'
import SeverityChip from './SeverityChip'

const CENTER = [13.05, 80.35]

// Compare two surveys of the same area: which contacts are NEW in survey B?
export default function DiffView({ surveys, pushToast }) {
  const [surveyA, setSurveyA] = useState('')
  const [surveyB, setSurveyB] = useState('')
  const [radius, setRadius] = useState(25)
  const [result, setResult] = useState(null)
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
    try {
      setResult(await fetchDiff(surveyA, surveyB, radius))
    } catch (err) {
      pushToast(`Diff failed: ${err.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  if (surveys.length === 0) {
    return (
      <EmptyState
        title="Nothing to compare yet"
        hint="Upload at least two surveys of the same area to run change detection."
      />
    )
  }

  const newContacts = result?.new_contacts || []
  const matched = result?.matched || []
  const first = newContacts[0] || matched[0]?.b
  const center = first ? [first.lat, first.lon] : CENTER

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
        <label className="ctl">
          <span className="ctl-label">Match radius (m)</span>
          <input
            className="num-input"
            type="number"
            min="1"
            max="500"
            value={radius}
            onChange={(e) => setRadius(Number(e.target.value) || 25)}
          />
        </label>
        <button type="button" className="btn primary" onClick={run} disabled={loading}>
          {loading ? 'Comparing…' : 'Compare'}
        </button>
      </div>

      {!result && !loading && (
        <EmptyState
          title="Run a comparison"
          hint="Contacts in B with no A contact within the radius are flagged NEW — likely fresh debris."
        />
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
              <TileLayer url="/tiles/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" />
              {matched.map((m, i) =>
                m.b ? (
                  <CircleMarker
                    key={`m-${m.b.id || i}`}
                    center={[m.b.lat, m.b.lon]}
                    radius={6}
                    pathOptions={{ color: '#8296ab', fillColor: '#8296ab', fillOpacity: 0.4, weight: 1.5 }}
                  >
                    <Tooltip>{`matched: ${m.b.id} ↔ ${m.a?.id} (${Number(m.distance_m).toFixed(1)} m)`}</Tooltip>
                  </CircleMarker>
                ) : null,
              )}
              {newContacts.map((c) => (
                <CircleMarker
                  key={`n-${c.id}`}
                  center={[c.lat, c.lon]}
                  radius={8}
                  pathOptions={{ color: '#ff3b30', fillColor: '#ff3b30', fillOpacity: 0.65, weight: 2 }}
                >
                  <Tooltip>{`NEW: ${c.id} · ${c.cls} · sev ${Math.round(c.severity)}`}</Tooltip>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>

          <div className="diff-tables">
            <div className="diff-col">
              <h3 className="mono">NEW CONTACTS ({newContacts.length})</h3>
              {newContacts.length === 0 ? (
                <p className="muted">No new contacts — nothing appeared between the surveys.</p>
              ) : (
                <div className="table-scroll short">
                  <table className="contacts-table">
                    <thead>
                      <tr>
                        <th />
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
                            <span className="new-tag mono">NEW</span>
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
              <h3 className="mono">MATCHED PAIRS ({matched.length})</h3>
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
    </div>
  )
}
