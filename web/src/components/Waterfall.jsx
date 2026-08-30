import { useEffect, useMemo, useState } from 'react'
import { fetchWaterfallMeta, waterfallUrl } from '../api'
import { sevColor } from '../utils'
import ContactPopover from './ContactPopover'
import EmptyState from './EmptyState'

// Enhanced waterfall image with an SVG overlay of contact boxes.
// Pixel mapping (image is port mirrored | starboard):
//   port:      x = n_port_cols - 1 - col   (so col1 maps to the left edge)
//   starboard: x = n_port_cols + col
//   y = ping row
export default function Waterfall({
  survey, contacts, onReview, pushToast, canReview, permissions,
}) {
  const [meta, setMeta] = useState(null)
  const [metaErr, setMetaErr] = useState(null)
  const [zoom, setZoom] = useState(1)
  const [raw, setRaw] = useState(false)
  const [selectedId, setSelectedId] = useState(null)

  useEffect(() => {
    setMeta(null)
    setMetaErr(null)
    setSelectedId(null)
    if (!survey) return undefined
    let alive = true
    fetchWaterfallMeta(survey)
      .then((m) => {
        if (alive) setMeta(m)
      })
      .catch((err) => {
        if (alive) setMetaErr(err.message)
      })
    return () => {
      alive = false
    }
  }, [survey])

  const boxes = useMemo(() => {
    if (!meta) return []
    return contacts
      .filter((c) => c.pixel)
      .map((c) => {
        const px = c.pixel
        let x0
        let x1
        if (px.side === 'port') {
          x0 = meta.n_port_cols - 1 - px.col1
          x1 = meta.n_port_cols - 1 - px.col0
        } else {
          x0 = meta.n_port_cols + px.col0
          x1 = meta.n_port_cols + px.col1
        }
        return {
          c,
          x: Math.min(x0, x1),
          y: px.ping0,
          w: Math.max(Math.abs(x1 - x0), 3),
          h: Math.max(px.ping1 - px.ping0, 3),
        }
      })
  }, [meta, contacts])

  if (!survey) {
    return (
      <EmptyState
        title="No survey selected"
        hint="Upload a survey from the rail on the right, then pick it in the toolbar above."
      />
    )
  }

  const selected = contacts.find((c) => c.id === selectedId) || null
  const W = meta ? meta.n_port_cols + meta.n_stbd_cols : 0
  const H = meta ? meta.n_pings : 0

  return (
    <div className="wf">
      <div className="wf-toolbar">
        <label className="ctl slider">
          <span className="ctl-label">Zoom</span>
          <input
            type="range"
            min="0.25"
            max="2"
            step="0.05"
            value={zoom}
            onChange={(e) => setZoom(Number(e.target.value))}
          />
          <span className="mono ctl-value">{zoom.toFixed(2)}×</span>
        </label>
        <label className="ctl check">
          <input type="checkbox" checked={raw} onChange={(e) => setRaw(e.target.checked)} />
          raw (unenhanced)
        </label>
        {meta && (
          <span className="wf-meta mono">
            {H} pings · {W} cols
            {meta.ground_res != null ? ` · ${Number(meta.ground_res).toFixed(3)} m/px` : ''}
            {' · '}
            {boxes.length} boxes
          </span>
        )}
        {metaErr && <span className="wf-err">waterfall metadata unavailable ({metaErr})</span>}
      </div>

      <div className="wf-body">
        <div className="wf-scroll">
          {meta ? (
            // Outer stage owns the scrollable footprint (scaled size); the inner
            // container is transformed, so image and SVG overlay stay registered.
            <div className="wf-stage" style={{ width: W * zoom, height: H * zoom }}>
              <div
                className="wf-inner"
                style={{ width: W, height: H, transform: `scale(${zoom})` }}
              >
                <img
                  className="wf-img"
                  src={waterfallUrl(survey, raw)}
                  width={W}
                  height={H}
                  alt={`waterfall for ${survey}`}
                  draggable={false}
                  onError={() => pushToast('Waterfall image failed to load', 'error')}
                />
                <svg className="wf-overlay" width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
                  {boxes.map(({ c, x, y, w, h }) => (
                    <rect
                      key={c.id}
                      x={x}
                      y={y}
                      width={w}
                      height={h}
                      fill={selectedId === c.id ? `${sevColor(c.severity)}33` : 'transparent'}
                      stroke={sevColor(c.severity)}
                      strokeWidth={Math.max(2, 2 / zoom)}
                      style={{ cursor: 'pointer', pointerEvents: 'all' }}
                      role="button"
                      tabIndex={0}
                      aria-label={`${c.id} ${c.cls}, severity ${Math.round(c.severity)}`}
                      onClick={() => setSelectedId(c.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          setSelectedId(c.id)
                        }
                      }}
                    >
                      <title>{`${c.id} · ${c.cls} · sev ${Math.round(c.severity)}`}</title>
                    </rect>
                  ))}
                </svg>
              </div>
            </div>
          ) : !metaErr ? (
            <div className="wf-loading">Loading waterfall…</div>
          ) : (
            <img
              className="wf-img-fallback"
              src={waterfallUrl(survey, raw)}
              alt={`waterfall for ${survey}`}
              onError={(e) => {
                e.currentTarget.style.display = 'none'
              }}
            />
          )}
        </div>

        {selected && (
          <aside className="wf-panel">
            <div className="wf-panel-head">
              <span className="mono">{selected.id}</span>
              <button
                type="button"
                className="btn ghost small"
                onClick={() => setSelectedId(null)}
              >
                Close
              </button>
            </div>
            <ContactPopover
              contact={selected}
              onReview={onReview}
              pushToast={pushToast}
              canReview={canReview}
              permissions={permissions}
              showEvidence
            />
          </aside>
        )}
      </div>
    </div>
  )
}
