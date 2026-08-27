import { useEffect, useMemo, useRef, useState } from 'react'
import L from 'leaflet'
import 'leaflet.heat'
import { CircleMarker, GeoJSON, MapContainer, Popup, TileLayer, useMap } from 'react-leaflet'
import { fetchLayers } from '../api'
import { SEVERITY_BANDS, prettyName, sevColor } from '../utils'
import ContactPopover from './ContactPopover'
import EmptyState from './EmptyState'
import MapResize from './MapResize'

const CENTER = [13.05, 80.35]
// Zone outlines rotate through the ink/state/navy family (DESIGN.md v2
// tokens) — dashed hairwork like restricted areas on a paper chart.
const LAYER_COLORS = ['#1B2733', '#55616E', '#2E6E44', '#153874']

// leaflet.heat attaches L.heatLayer to the Leaflet global; drive it manually.
function HeatLayer({ points, enabled }) {
  const map = useMap()
  const layerRef = useRef(null)
  useEffect(() => {
    if (!enabled || points.length === 0) return undefined
    const layer = L.heatLayer(points, { radius: 28, blur: 20, maxZoom: 17, max: 1.0 })
    layer.addTo(map)
    layerRef.current = layer
    return () => {
      map.removeLayer(layer)
      layerRef.current = null
    }
  }, [map, points, enabled])
  return null
}

export default function MapView({ contacts, onReview, pushToast, hasSurvey }) {
  const [layers, setLayers] = useState({})
  const [visible, setVisible] = useState({})
  const [heat, setHeat] = useState(false)

  useEffect(() => {
    let alive = true
    fetchLayers()
      .then((docs) => {
        if (!alive) return
        setLayers(docs)
        setVisible(Object.fromEntries(Object.keys(docs).map((k) => [k, true])))
      })
      .catch(() => {
        /* sensitive layers are optional — the map still works without them */
      })
    return () => {
      alive = false
    }
  }, [])

  const heatPoints = useMemo(
    () => contacts.map((c) => [c.lat, c.lon, Math.max(c.severity, 5) / 100]),
    [contacts],
  )

  return (
    <div className="map-wrap">
      <MapContainer center={CENTER} zoom={13} className="map" preferCanvas>
        <MapResize />
        {/* Esri World Imagery via the backend's offline-caching proxy. */}
        <TileLayer
          url="/tiles/{z}/{x}/{y}.png"
          attribution="Esri, Maxar, Earthstar Geographics, and the GIS community"
          maxNativeZoom={17}
          maxZoom={19}
        />

        {Object.entries(layers).map(
          ([name, fc], i) =>
            visible[name] && (
              <GeoJSON
                key={name}
                data={fc}
                style={() => ({
                  color: LAYER_COLORS[i % LAYER_COLORS.length],
                  weight: 2,
                  dashArray: '6 4',
                  fillColor: LAYER_COLORS[i % LAYER_COLORS.length],
                  fillOpacity: 0.06,
                })}
                onEachFeature={(feature, layer) => {
                  const extra = feature?.properties?.name
                  const label = extra ? `${prettyName(name)} · ${extra}` : prettyName(name)
                  layer.bindTooltip(label, { sticky: true })
                }}
              />
            ),
        )}

        {contacts.map((c) => (
          <CircleMarker
            key={c.id}
            center={[c.lat, c.lon]}
            radius={5 + (c.confidence / 100) * 6}
            pathOptions={{
              color: sevColor(c.severity),
              weight: 2,
              fillColor: sevColor(c.severity),
              fillOpacity: 0.55,
            }}
          >
            <Popup maxWidth={300} className="contact-popup">
              <ContactPopover contact={c} onReview={onReview} pushToast={pushToast} />
            </Popup>
          </CircleMarker>
        ))}

        <HeatLayer points={heatPoints} enabled={heat} />
      </MapContainer>

      <div className="map-legend">
        <div className="legend-title">Overlays</div>
        <label className="legend-row">
          <input type="checkbox" checked={heat} onChange={(e) => setHeat(e.target.checked)} />
          <span className="legend-ramp" aria-hidden="true">
            {SEVERITY_BANDS.map((b) => (
              <span key={b.color} className="legend-cell" style={{ background: b.color }} />
            ))}
          </span>
          severity heatmap
        </label>
        {Object.keys(layers).map((name, i) => (
          <label key={name} className="legend-row">
            <input
              type="checkbox"
              checked={Boolean(visible[name])}
              onChange={(e) => setVisible((v) => ({ ...v, [name]: e.target.checked }))}
            />
            <span
              className="legend-swatch"
              style={{ borderColor: LAYER_COLORS[i % LAYER_COLORS.length] }}
            />
            {prettyName(name)}
          </label>
        ))}
        <div className="legend-title">Severity</div>
        {SEVERITY_BANDS.map((b) => (
          <div key={b.label} className="legend-row">
            <span className="legend-sq" style={{ background: b.color }} />
            <span className="legend-band mono">{b.label}</span>
          </div>
        ))}
      </div>

      {!hasSurvey && (
        <div className="map-empty">
          <EmptyState
            title="No survey loaded yet"
            hint="Drop an .xtf / .jsf / GeoTIFF file into the upload rail to run the detection pipeline."
          />
        </div>
      )}
      {hasSurvey && contacts.length === 0 && (
        <div className="map-empty small">No contacts match the current filters.</div>
      )}
    </div>
  )
}
