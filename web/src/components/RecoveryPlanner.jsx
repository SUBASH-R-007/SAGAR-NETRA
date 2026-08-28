import { useEffect, useMemo, useState } from 'react'
import {
  CircleMarker,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  Tooltip,
  useMap,
} from 'react-leaflet'
import { fetchRoute } from '../api'
import { SEVERITY_BANDS, sevColor } from '../utils'
import EmptyState from './EmptyState'
import MapResize from './MapResize'
import SeverityChip from './SeverityChip'

// RECOVERY MISSION PLANNER — closes the loop from sonar pixel to cleanup
// sortie. Everything rendered here comes from GET /api/route
// (geoscribe/route.py: nearest-neighbour + 2-opt over geodesic distances;
// geoscribe/cluster.py: geodesic single-linkage retrieval zones). No value is
// derived beyond summing the leg distances the API returns.

// DESIGN.md v2: navy is structure — the planned track is structure, not status.
const TRACK_NAVY = '#153874'
const START_INK = '#1B2733'
const DEFAULT_EPS = 150

// < 1000 m reads in metres (the leg a coxswain steers); beyond that, km.
function fmtDist(v) {
  const m = Number(v)
  if (v == null || !Number.isFinite(m)) return '—'
  return m < 1000 ? `${m.toFixed(1)} m` : `${(m / 1000).toFixed(2)} km`
}

const fmtCoord = (v) => (Number.isFinite(Number(v)) ? Number(v).toFixed(5) : '—')

// A waypoint with no severity gets the neutral ink-faint mark rather than a
// band colour it has not earned.
const markerColor = (sev) => (Number.isFinite(Number(sev)) ? sevColor(sev) : '#8A94A0')

// Frame the whole tour. Leaflet sizes itself once at init, so fit on two
// animation frames (same reasoning as MapResize) with the size re-measured
// first — otherwise the fit is computed against a 0x0 container.
function FitTour({ points }) {
  const map = useMap()
  useEffect(() => {
    if (!points.length) return undefined
    let raf2 = 0
    const fit = () => {
      map.invalidateSize({ animate: false })
      map.fitBounds(points, { padding: [32, 32], maxZoom: 16, animate: false })
    }
    const raf1 = requestAnimationFrame(() => {
      fit()
      raf2 = requestAnimationFrame(fit)
    })
    return () => {
      cancelAnimationFrame(raf1)
      cancelAnimationFrame(raf2)
    }
  }, [map, points])
  return null
}

export default function RecoveryPlanner({ survey, contacts, pushToast }) {
  const [scope, setScope] = useState('confirmed')
  const [clusterOn, setClusterOn] = useState(false)
  const [epsText, setEpsText] = useState(String(DEFAULT_EPS))
  const [startLat, setStartLat] = useState('')
  const [startLon, setStartLon] = useState('')
  const [result, setResult] = useState(null)
  const [planned, setPlanned] = useState(null)
  const [planError, setPlanError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [planSeq, setPlanSeq] = useState(0)

  // A plan belongs to the survey it was planned over — never let one linger.
  useEffect(() => {
    setResult(null)
    setPlanned(null)
    setPlanError(null)
  }, [survey])

  const confirmedCount = useMemo(
    () => contacts.filter((c) => c.review === 'confirmed').length,
    [contacts],
  )

  const firstContact = contacts[0]

  const fillStart = () => {
    if (!firstContact || !Number.isFinite(firstContact.lat) || !Number.isFinite(firstContact.lon)) {
      pushToast('No contact position available to seed the start from', 'error')
      return
    }
    setStartLat(firstContact.lat.toFixed(5))
    setStartLon(firstContact.lon.toFixed(5))
  }

  const plan = async () => {
    const latText = startLat.trim()
    const lonText = startLon.trim()
    const wantStart = latText !== '' || lonText !== ''
    const lat = Number(latText)
    const lon = Number(lonText)
    if (wantStart) {
      if (latText === '' || lonText === '' || !Number.isFinite(lat) || !Number.isFinite(lon)) {
        pushToast('Enter both a start latitude and longitude, or leave both blank', 'error')
        return
      }
      if (Math.abs(lat) > 90 || Math.abs(lon) > 180) {
        pushToast('Start position out of range (lat ±90, lon ±180)', 'error')
        return
      }
    }
    const eps = Number(epsText)
    if (clusterOn && (!Number.isFinite(eps) || eps <= 0)) {
      pushToast('Cluster radius must be a positive number of metres', 'error')
      return
    }

    setLoading(true)
    setResult(null)
    setPlanError(null)
    try {
      const res = await fetchRoute({
        survey,
        review: scope,
        clusterEpsM: clusterOn ? eps : undefined,
        startLat: wantStart ? lat : '',
        startLon: wantStart ? lon : '',
      })
      setResult(res)
      setPlanned({ scope, eps: clusterOn ? eps : null })
      setPlanSeq((n) => n + 1)
    } catch (err) {
      setPlanned(null)
      setPlanError(err.message)
      pushToast(`Route planning failed: ${err.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  const startPos = useMemo(() => {
    const s = result?.start
    if (!s || !Number.isFinite(Number(s.lat)) || !Number.isFinite(Number(s.lon))) return null
    return [Number(s.lat), Number(s.lon)]
  }, [result])

  // legs_m[k] prices the k-th hop of the tour the API walked: with a vessel
  // start the first hop is start -> waypoint 1, so waypoint i takes legs_m[i];
  // without one, waypoint 1 has no inbound leg and waypoint i takes
  // legs_m[i-1]. Anything the API did not send degrades to a dash.
  const rows = useMemo(() => {
    const wps = result?.waypoints || []
    const legs = Array.isArray(result?.legs_m) ? result.legs_m : []
    const offset = startPos ? 0 : 1
    let cum = 0
    let cumOk = true
    return wps.map((wp, i) => {
      const legIdx = i - offset
      let leg = null
      if (legIdx >= 0) {
        const v = Number(legs[legIdx])
        if (legIdx < legs.length && Number.isFinite(v)) {
          leg = v
          cum += v
        } else {
          cumOk = false
        }
      }
      return { ...wp, leg, cum: cumOk ? cum : null }
    })
  }, [result, startPos])

  const clusters = Array.isArray(result?.clusters) ? result.clusters : []
  const clustered = clusters.length > 0

  const longest = useMemo(() => {
    const legs = (result?.legs_m || []).map(Number).filter((v) => Number.isFinite(v))
    if (legs.length === 0) return null
    const max = Math.max(...legs)
    const idx = legs.indexOf(max)
    // the leg arrives at this waypoint (see the offset note above)
    const arrival = rows[startPos ? idx : idx + 1]
    return { max, arrival }
  }, [result, rows, startPos])

  const tourPoints = useMemo(
    () =>
      rows
        .filter((r) => Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon)))
        .map((r) => [Number(r.lat), Number(r.lon)]),
    [rows],
  )

  const linePoints = useMemo(
    () => (startPos ? [startPos, ...tourPoints] : tourPoints),
    [startPos, tourPoints],
  )

  if (!survey) {
    return (
      <EmptyState
        title="No survey selected"
        hint="Upload or select a survey from the rail on the right — the recovery tour is planned over its contacts."
      />
    )
  }

  return (
    <div className="rp">
      <div className="rp-toolbar">
        <label className="ctl">
          <span className="ctl-label">Scope</span>
          <select
            value={scope}
            onChange={(e) => setScope(e.target.value)}
            aria-label="Which contacts to route"
          >
            <option value="confirmed">Confirmed only</option>
            <option value="">All contacts</option>
          </select>
        </label>

        <label className="ctl check">
          <input
            type="checkbox"
            checked={clusterOn}
            onChange={(e) => setClusterOn(e.target.checked)}
          />
          <span className="ctl-label">Retrieval zones</span>
        </label>

        <label className="ctl">
          <span className="ctl-label">Cluster radius (m)</span>
          <input
            className="num-input"
            type="number"
            min="1"
            step="10"
            value={epsText}
            disabled={!clusterOn}
            onChange={(e) => setEpsText(e.target.value)}
          />
        </label>

        <span className="rp-group">
          <span className="ctl-label">Vessel start</span>
          <input
            className="num-input rp-coord"
            type="number"
            step="any"
            inputMode="decimal"
            placeholder="lat"
            aria-label="Vessel start latitude"
            value={startLat}
            onChange={(e) => setStartLat(e.target.value)}
          />
          <input
            className="num-input rp-coord"
            type="number"
            step="any"
            inputMode="decimal"
            placeholder="lon"
            aria-label="Vessel start longitude"
            value={startLon}
            onChange={(e) => setStartLon(e.target.value)}
          />
          <button
            type="button"
            className="btn small"
            onClick={fillStart}
            disabled={!firstContact}
            title="Fill the start from the first contact in this survey (the highest-severity one)"
          >
            Use survey start
          </button>
        </span>

        <button type="button" className="btn primary" onClick={plan} disabled={loading}>
          {loading ? 'Planning…' : 'Plan route'}
        </button>
      </div>

      <p className="rp-caption">
        Confirmed only is the operational default: a vessel sorties on verified targets, not on
        unreviewed detections.{' '}
        <span className="mono">
          {confirmedCount} of {contacts.length} contacts confirmed in {survey}.
        </span>{' '}
        Leave the start position blank to begin the tour at the first target; give a position to
        anchor the tour at the vessel.
      </p>

      {/* persistent live region — present before the message so it announces */}
      <p className="rp-status mono" role="status">
        {loading
          ? `Planning route over ${scope === 'confirmed' ? 'confirmed' : 'all'} contacts…`
          : ''}
      </p>

      {/* A failed plan must not fall back to "not planned yet" — the operator
          pressed the button and is owed the reason, on the page. */}
      {planError && !loading && (
        <EmptyState
          title="Route planning failed"
          hint={`The planner did not answer: ${planError}. The contacts are unchanged — press Plan route to try again.`}
        />
      )}

      {!result && !planError && !loading && (
        <EmptyState
          title="No route planned yet"
          hint="Press Plan route. DRISHTI orders the targets into a short visiting tour — nearest-neighbour construction with a 2-opt improvement pass over geodesic (WGS-84) distances."
        />
      )}

      {result && rows.length === 0 && !loading && (
        <EmptyState
          title="No waypoints to sortie on"
          hint={
            planned?.scope === 'confirmed'
              ? 'This survey has no confirmed contacts yet. Confirm the targets you intend to recover on the Contacts tab, then plan again — or switch the scope to all contacts.'
              : 'This survey has no contacts to route.'
          }
        />
      )}

      {result && rows.length > 0 && (
        <div className="rp-result">
          <div className="rp-stats">
            <div className="stat">
              <span className="stat-value mono">{rows.length}</span>
              <span className="stat-label">waypoints</span>
            </div>
            <div className="stat">
              <span
                className="stat-value mono"
                title={
                  startPos
                    ? 'includes the run-in from the vessel start position'
                    : 'sum of the legs between waypoints'
                }
              >
                {fmtDist(result.total_m)}
              </span>
              <span className="stat-label">tour length</span>
            </div>
            {clustered && (
              <div className="stat">
                <span className="stat-value mono">{clusters.length}</span>
                <span className="stat-label">retrieval zones</span>
              </div>
            )}
            <div className="stat">
              <span
                className="stat-value mono"
                title={longest?.arrival ? `arrives at ${longest.arrival.id}` : undefined}
              >
                {longest ? fmtDist(longest.max) : '—'}
              </span>
              <span className="stat-label">longest leg</span>
            </div>
          </div>

          {linePoints.length > 0 && (
            <div className="rp-map">
              <MapContainer
                key={`plan-${planSeq}-${rows.length}`}
                center={linePoints[0]}
                zoom={14}
                className="map"
                preferCanvas
              >
                <MapResize />
                <FitTour points={linePoints} />
                {/* Esri World Imagery via the backend's offline-caching proxy. */}
                <TileLayer
                  url="/tiles/{z}/{x}/{y}.png"
                  attribution="Esri, Maxar, Earthstar Geographics, and the GIS community"
                  maxNativeZoom={17}
                  maxZoom={19}
                />

                {linePoints.length > 1 && (
                  <Polyline
                    positions={linePoints}
                    pathOptions={{ color: TRACK_NAVY, weight: 2, opacity: 0.9 }}
                  />
                )}

                {startPos && (
                  <CircleMarker
                    center={startPos}
                    radius={7}
                    pathOptions={{
                      color: START_INK,
                      weight: 2,
                      fillColor: '#FFFFFF',
                      fillOpacity: 1,
                    }}
                  >
                    <Tooltip permanent direction="right" offset={[8, 0]} className="rp-tip">
                      START
                    </Tooltip>
                  </CircleMarker>
                )}

                {rows.map((r) =>
                  Number.isFinite(Number(r.lat)) && Number.isFinite(Number(r.lon)) ? (
                    <CircleMarker
                      key={`${r.seq}-${r.id}`}
                      center={[Number(r.lat), Number(r.lon)]}
                      radius={6}
                      pathOptions={{
                        color: markerColor(r.severity),
                        weight: 2,
                        fillColor: markerColor(r.severity),
                        fillOpacity: 0.6,
                      }}
                    >
                      <Tooltip permanent direction="right" offset={[8, 0]} className="rp-tip">
                        {r.seq}
                      </Tooltip>
                      <Popup className="contact-popup">
                        <div className="rp-pop mono">
                          <div>
                            {r.seq} · {r.id}
                          </div>
                          <div className="muted">{r.cls}</div>
                          <div className="muted">
                            {fmtCoord(r.lat)}, {fmtCoord(r.lon)}
                          </div>
                          {Number.isFinite(Number(r.severity)) && (
                            <SeverityChip value={r.severity} />
                          )}
                        </div>
                      </Popup>
                    </CircleMarker>
                  ) : null,
                )}
              </MapContainer>

              <div className="map-legend">
                <div className="legend-title">Route</div>
                <div className="legend-row">
                  <span className="legend-swatch rp-legend-track" aria-hidden="true" />
                  <span className="legend-band mono">planned track</span>
                </div>
                {startPos && (
                  <div className="legend-row">
                    <span className="rp-legend-start" aria-hidden="true" />
                    <span className="legend-band mono">vessel start</span>
                  </div>
                )}
                <div className="legend-title">Severity</div>
                {SEVERITY_BANDS.map((b) => (
                  <div key={b.label} className="legend-row">
                    <span className="legend-sq" style={{ background: b.color }} aria-hidden="true" />
                    <span className="legend-band mono">{b.label}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {clustered && (
            <div className="rp-zones">
              <h3 className="diff-col-title">Retrieval zones ({clusters.length})</h3>
              <p className="rp-caption">
                Contacts linked within {planned?.eps ?? '—'} m of one another form one zone
                (geodesic single-linkage). The tour visits the zones in the order below and works
                every contact in a zone before moving on. The anchor is the zone&apos;s
                severity-weighted centroid, so the hook splashes nearest the worst debris.
              </p>
              <div className="table-scroll short">
                <table className="contacts-table">
                  <thead>
                    <tr>
                      <th>Zone</th>
                      <th className="num">Contacts</th>
                      <th className="num">Total sev</th>
                      <th>Anchor (lat, lon)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {clusters.map((z, i) => (
                      <tr key={z.seq ?? i}>
                        <td>
                          <span className="tag">Z{z.seq ?? i + 1}</span>
                        </td>
                        <td className="num mono">
                          {Number.isFinite(Number(z.n))
                            ? z.n
                            : (z.contact_ids || []).length || '—'}
                        </td>
                        <td className="num mono">
                          {Number.isFinite(Number(z.total_severity))
                            ? Number(z.total_severity).toFixed(0)
                            : '—'}
                        </td>
                        <td className="mono">
                          {fmtCoord(z.centroid_lat)}, {fmtCoord(z.centroid_lon)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="rp-waypoints">
            <h3 className="diff-col-title">Waypoint list ({rows.length})</h3>
            <div className="table-scroll">
              <table className="contacts-table">
                <thead>
                  <tr>
                    <th className="num">Seq</th>
                    <th>ID</th>
                    <th>Class</th>
                    <th className="num">Sev</th>
                    <th className="num">Leg</th>
                    <th className="num">Cumulative</th>
                    <th>Position (lat, lon)</th>
                    {clustered && <th>Zone</th>}
                  </tr>
                </thead>
                <tbody>
                  {startPos && (
                    <tr className="rp-start-row">
                      <td className="num mono">—</td>
                      <td className="mono">START</td>
                      <td className="muted">vessel position</td>
                      <td className="num">—</td>
                      <td className="num mono">—</td>
                      <td className="num mono">{fmtDist(0)}</td>
                      <td className="mono">
                        {fmtCoord(startPos[0])}, {fmtCoord(startPos[1])}
                      </td>
                      {clustered && <td>—</td>}
                    </tr>
                  )}
                  {rows.map((r) => (
                    <tr key={`${r.seq}-${r.id}`} className="row">
                      <td className="num mono">{r.seq}</td>
                      <td className="mono">{r.id}</td>
                      <td>{r.cls}</td>
                      <td className="num">
                        {Number.isFinite(Number(r.severity)) ? (
                          <SeverityChip value={r.severity} />
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="num mono">{r.leg == null ? '—' : fmtDist(r.leg)}</td>
                      <td className="num mono">{r.cum == null ? '—' : fmtDist(r.cum)}</td>
                      <td className="mono">
                        {fmtCoord(r.lat)}, {fmtCoord(r.lon)}
                      </td>
                      {clustered && (
                        <td>
                          {r.cluster == null ? '—' : <span className="tag">Z{r.cluster}</span>}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
