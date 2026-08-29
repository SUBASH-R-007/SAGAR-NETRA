import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchGeometry, fetchShadow, fetchSimClasses, simulateScene } from '../api'

// ── Physics Lab ────────────────────────────────────────────────────────────
// Three panels a visitor can drive. Every number shown is computed by the
// backend — sonar_core.geometry and physicheck.shadow, the same functions that
// process an uploaded survey. Nothing here re-derives the physics in
// JavaScript, because a browser copy would let the picture and the product
// drift apart and the picture would always look right.
//
// The drawings are plain SVG in the portal palette: navy is structure, saffron
// is the thing under test, hairlines instead of shadows, radius <= 2px.

const NAVY = '#153874'
const SAFFRON = '#E07C00'
const HAIRLINE = '#D6DBE3'
const INK_DIM = '#55616E'
const SHADOW_FILL = '#1B2733'

function useDebounced(value, ms = 120) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return v
}

function Slider({ label, unit, value, min, max, step, onChange, hint }) {
  return (
    <label className="lab-slider">
      <span className="lab-slider-head">
        <span className="lab-slider-label">{label}</span>
        <span className="lab-slider-value mono">
          {Number(value).toFixed(step < 1 ? 2 : 0)}
          <span className="lab-unit">{unit}</span>
        </span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <span className="lab-slider-hint">{hint}</span>}
    </label>
  )
}

function Readout({ items }) {
  return (
    <dl className="lab-readout">
      {items.map((it) => (
        <div key={it.k} className={it.emphasis ? 'lab-stat emphasis' : 'lab-stat'}>
          <dt>{it.k}</dt>
          <dd className="mono">
            {it.v}
            {it.unit ? <span className="lab-unit">{it.unit}</span> : null}
          </dd>
        </div>
      ))}
    </dl>
  )
}

// ── 1 · shadow simulator ───────────────────────────────────────────────────
// The core novelty made checkable: forward-model a shadow from a known height,
// then invert it with the deployed estimator and show that the height comes
// back. The ray diagram is the argument; the round-trip error is the proof.

function ShadowPanel({ pushToast }) {
  const [altitude, setAltitude] = useState(10)
  const [height, setHeight] = useState(2)
  const [ground, setGround] = useState(20)
  const [res, setRes] = useState(null)

  const q = useDebounced({ altitude, height, ground })

  useEffect(() => {
    let alive = true
    fetchShadow(q.altitude, q.height, q.ground)
      .then((r) => alive && setRes(r))
      .catch((e) => pushToast(`Shadow model failed: ${e.message}`, 'error'))
    return () => {
      alive = false
    }
  }, [q, pushToast])

  const W = 620
  const H = 272
  // Bottom padding carries two stacked rows: the shadow-length callout sits
  // directly under the seabed, the axis labels below it. Sharing one row put
  // them on top of each other whenever the shadow reached the right edge.
  const PAD = { l: 44, r: 20, t: 18, b: 50 }
  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b

  const view = useMemo(() => {
    if (!res) return null
    const xMax = Math.max(res.shadow_end_m * 1.15, res.ground_range_m * 1.4, 10)
    const yMax = Math.max(res.altitude_m * 1.15, 1)
    const sx = (x) => PAD.l + (x / xMax) * plotW
    const sy = (y) => PAD.t + plotH - (y / yMax) * plotH
    return { xMax, yMax, sx, sy }
  }, [res, plotW, plotH])

  return (
    <section className="lab-panel">
      <header className="lab-panel-head">
        <span className="eyebrow">Simulation 01</span>
        <h3>Height from shadow</h3>
        <p className="lab-lede">
          A side-scan sonar cannot measure height directly — it only records how much
          sound came back. But an object blocks the beam, and the length of the dark
          gap behind it encodes how tall it is. Drag the sliders: the shadow is
          forward-modelled, then inverted by the same estimator that runs on real
          contacts. If the recovered height matches, the inversion is sound.
        </p>
      </header>

      <div className="lab-split">
        <div className="lab-controls">
          <Slider
            label="Towfish altitude"
            unit="m"
            value={altitude}
            min={3}
            max={30}
            step={0.5}
            onChange={(v) => {
              setAltitude(v)
              if (height > 0.9 * v) setHeight(Number((0.9 * v).toFixed(1)))
            }}
            hint="height above the seabed"
          />
          <Slider
            label="Object height"
            unit="m"
            value={height}
            min={0}
            max={Math.max(0.9 * altitude, 0.5)}
            step={0.1}
            onChange={setHeight}
            hint="what we are trying to recover"
          />
          <Slider
            label="Ground range"
            unit="m"
            value={ground}
            min={2}
            max={80}
            step={1}
            onChange={setGround}
            hint="across-track distance from nadir"
          />

          {res && (
            <Readout
              items={[
                { k: 'Shadow length', v: res.shadow_length_m.toFixed(2), unit: 'm' },
                {
                  k: 'Shadow gain',
                  v: res.shadow_gain == null ? '—' : `${res.shadow_gain.toFixed(2)}×`,
                  emphasis: true,
                },
                { k: 'Grazing angle', v: res.grazing_angle_deg.toFixed(1), unit: '°' },
                {
                  k: 'Recovered height',
                  v: res.recovered_height_m.toFixed(3),
                  unit: 'm',
                  emphasis: true,
                },
                {
                  k: 'Round-trip error',
                  v: res.round_trip_error_m.toExponential(1),
                  unit: 'm',
                },
              ]}
            />
          )}

          {res && res.shadow_gain > 1 && (
            <p className="lab-insight">
              The shadow is <strong>{res.shadow_gain.toFixed(1)}× longer</strong> than the
              object is tall. That lever is why we measure the shadow instead of the
              object: a {res.height_m.toFixed(1)} m target is a few bright pixels, but its{' '}
              {res.shadow_length_m.toFixed(1)} m shadow spans{' '}
              {Math.round(res.shadow_length_m / 0.075)} range samples.
            </p>
          )}
          {res && res.height_clamped && (
            <p className="lab-warn">
              Clamped to {res.max_height_m.toFixed(1)} m. An object as tall as the towfish
              casts an infinite shadow — the geometry has no solution.
            </p>
          )}
        </div>

        <div className="lab-figure">
          {view && res && (
            <svg viewBox={`0 0 ${W} ${H}`} className="lab-svg" role="img"
                 aria-label="Cross-section of the sonar beam, object and its acoustic shadow">
              {/* water column + seabed */}
              <rect x={PAD.l} y={PAD.t} width={plotW} height={plotH} fill="#F7F9FB" />
              <line x1={PAD.l} y1={view.sy(0)} x2={PAD.l + plotW} y2={view.sy(0)}
                    stroke={NAVY} strokeWidth="1.5" />

              {/* shadow region — the measured quantity */}
              <rect
                x={view.sx(res.shadow_start_m)}
                y={view.sy(0)}
                width={Math.max(view.sx(res.shadow_end_m) - view.sx(res.shadow_start_m), 0)}
                height={10}
                fill={SHADOW_FILL}
              />

              {/* beam edge grazing the object top, continuing to where it lands */}
              <line
                x1={view.sx(0)} y1={view.sy(res.altitude_m)}
                x2={view.sx(res.shadow_end_m)} y2={view.sy(0)}
                stroke={SAFFRON} strokeWidth="1.5" strokeDasharray="4 3"
              />
              {/* unobstructed ray to the near edge of the object */}
              <line
                x1={view.sx(0)} y1={view.sy(res.altitude_m)}
                x2={view.sx(res.ground_range_m)} y2={view.sy(0)}
                stroke={HAIRLINE} strokeWidth="1"
              />

              {/* the object */}
              <rect
                x={view.sx(res.ground_range_m) - 4}
                y={view.sy(res.height_m)}
                width={8}
                height={Math.max(view.sy(0) - view.sy(res.height_m), 1)}
                fill={NAVY}
              />

              {/* towfish */}
              <circle cx={view.sx(0)} cy={view.sy(res.altitude_m)} r={5} fill={NAVY} />
              <line x1={view.sx(0)} y1={view.sy(res.altitude_m)} x2={view.sx(0)}
                    y2={view.sy(0)} stroke={HAIRLINE} strokeWidth="1" strokeDasharray="2 3" />

              {/* labels */}
              <text x={view.sx(0) + 9} y={view.sy(res.altitude_m) - 6} className="lab-svg-label">
                towfish · A={res.altitude_m.toFixed(1)} m
              </text>
              <text x={view.sx(res.ground_range_m) + 7} y={view.sy(res.height_m) - 5}
                    className="lab-svg-label">
                H={res.height_m.toFixed(1)} m
              </text>
              <text
                x={(view.sx(res.shadow_start_m) + view.sx(res.shadow_end_m)) / 2}
                y={view.sy(0) + 24}
                textAnchor="middle"
                className="lab-svg-label strong"
              >
                shadow {res.shadow_length_m.toFixed(2)} m
              </text>
              <text x={PAD.l} y={H - 10} className="lab-svg-axis">
                nadir
              </text>
              <text x={PAD.l + plotW} y={H - 10} textAnchor="end" className="lab-svg-axis">
                ground range →
              </text>
            </svg>
          )}
          <p className="lab-formula mono">
            H = A · (x_end − x_far) / x_end
            <span className="lab-formula-src">physicheck/shadow.py</span>
          </p>
        </div>
      </div>
    </section>
  )
}

// ── 2 · resolution explorer ────────────────────────────────────────────────
// Two limits that behave differently. Across-track is set by the pulse and is
// flat across the swath; along-track is set by the beam and grows linearly.
// That divergence is why a far-range length is a softer number.

function ResolutionPanel({ pushToast }) {
  const [range, setRange] = useState(50)
  const [altitude, setAltitude] = useState(8)
  const [beam, setBeam] = useState(0.5)
  const [pulseUs, setPulseUs] = useState(100)
  const [geo, setGeo] = useState(null)

  const q = useDebounced({ range, altitude, beam, pulseUs })

  useEffect(() => {
    let alive = true
    fetchGeometry({
      altitude: q.altitude, range: q.range, beam: q.beam, pulseUs: q.pulseUs,
    })
      .then((r) => alive && setGeo(r))
      .catch((e) => pushToast(`Geometry failed: ${e.message}`, 'error'))
    return () => {
      alive = false
    }
  }, [q, pushToast])

  const W = 620
  const H = 240
  const PAD = { l: 52, r: 20, t: 16, b: 36 }
  const plotW = W - PAD.l - PAD.r
  const plotH = H - PAD.t - PAD.b

  const chart = useMemo(() => {
    if (!geo || !geo.curve.length) return null
    const xs = geo.curve.map((c) => c.ground_range_m)
    const yMax = Math.max(...geo.curve.map((c) => c.along_track_m), 0.2) * 1.15
    const xMax = Math.max(...xs, 1)
    const sx = (x) => PAD.l + (x / xMax) * plotW
    const sy = (y) => PAD.t + plotH - (y / yMax) * plotH
    const along = geo.curve.map((c) => `${sx(c.ground_range_m)},${sy(c.along_track_m)}`)
    const across = geo.curve.map((c) => `${sx(c.ground_range_m)},${sy(c.across_track_m)}`)
    return { sx, sy, xMax, yMax, along: along.join(' '), across: across.join(' ') }
  }, [geo, plotW, plotH])

  return (
    <section className="lab-panel">
      <header className="lab-panel-head">
        <span className="eyebrow">Simulation 02</span>
        <h3>What the sonar can resolve</h3>
        <p className="lab-lede">
          Two limits, behaving differently. Across-track resolution is{' '}
          <span className="mono">c·τ/2</span> — set by the pulse alone, so it is the same
          at the swath edge as at nadir. Along-track is <span className="mono">θ·R</span> —
          set by the beam, so it degrades linearly with range. That divergence is why a
          length measured far out is a much softer number than the same length near the
          track.
        </p>
      </header>

      <div className="lab-split">
        <div className="lab-controls">
          <Slider label="Slant range" unit="m" value={range} min={20} max={120} step={5}
                  onChange={setRange} hint="sonar range setting" />
          <Slider label="Altitude" unit="m" value={altitude} min={3} max={30} step={0.5}
                  onChange={setAltitude} />
          <Slider label="Along-track beam" unit="°" value={beam} min={0.2} max={2} step={0.1}
                  onChange={setBeam} hint="narrower beam, sharper along-track" />
          <Slider label="Pulse length" unit="µs" value={pulseUs} min={20} max={400} step={10}
                  onChange={setPulseUs} hint="shorter pulse, sharper across-track" />

          {geo && (
            <Readout
              items={[
                {
                  k: 'Across-track (all ranges)',
                  v: (geo.across_track_resolution_m * 100).toFixed(1),
                  unit: 'cm',
                  emphasis: true,
                },
                {
                  k: 'Along-track at far edge',
                  v: geo.along_track_resolution_far_m.toFixed(2),
                  unit: 'm',
                  emphasis: true,
                },
                { k: 'Usable swath', v: geo.max_ground_range_m.toFixed(1), unit: 'm' },
                {
                  k: '2nd bottom return',
                  v: geo.multipath_ground_range_m.toFixed(1),
                  unit: 'm',
                },
                {
                  k: 'Sound-speed error (1%)',
                  v: geo.sound_speed_error_far_m.toFixed(2),
                  unit: 'm',
                },
              ]}
            />
          )}
          {geo && (
            <p className="lab-insight">
              At the far edge the beam smears an object over{' '}
              <strong>{geo.along_track_resolution_far_m.toFixed(2)} m</strong> along-track
              but only {(geo.across_track_resolution_m * 100).toFixed(1)} cm across —
              a {(geo.along_track_resolution_far_m / geo.across_track_resolution_m).toFixed(0)}
              × anisotropy. Contacts are wider than they are long for a reason.
            </p>
          )}
        </div>

        <div className="lab-figure">
          {chart && geo && (
            <svg viewBox={`0 0 ${W} ${H}`} className="lab-svg" role="img"
                 aria-label="Resolution against ground range">
              <rect x={PAD.l} y={PAD.t} width={plotW} height={plotH} fill="#F7F9FB" />
              {[0, 0.25, 0.5, 0.75, 1].map((f) => (
                <g key={f}>
                  <line x1={PAD.l} y1={PAD.t + plotH * f} x2={PAD.l + plotW}
                        y2={PAD.t + plotH * f} stroke={HAIRLINE} strokeWidth="1" />
                  <text x={PAD.l - 8} y={PAD.t + plotH * f + 4} textAnchor="end"
                        className="lab-svg-axis">
                    {(chart.yMax * (1 - f)).toFixed(2)}
                  </text>
                </g>
              ))}
              <polyline points={chart.along} fill="none" stroke={SAFFRON} strokeWidth="2" />
              <polyline points={chart.across} fill="none" stroke={NAVY} strokeWidth="2"
                        strokeDasharray="5 3" />
              <text x={PAD.l + plotW - 6} y={chart.sy(geo.along_track_resolution_far_m) - 8}
                    textAnchor="end" className="lab-svg-label" fill={SAFFRON}>
                along-track θ·R
              </text>
              <text x={PAD.l + plotW - 6} y={chart.sy(geo.across_track_resolution_m) - 8}
                    textAnchor="end" className="lab-svg-label" fill={NAVY}>
                across-track c·τ/2
              </text>
              <text x={PAD.l} y={H - 8} className="lab-svg-axis">0 m</text>
              <text x={PAD.l + plotW} y={H - 8} textAnchor="end" className="lab-svg-axis">
                {chart.xMax.toFixed(0)} m ground range
              </text>
              <text x={10} y={PAD.t + 10} className="lab-svg-axis">metres</text>
            </svg>
          )}
        </div>
      </div>
    </section>
  )
}

// ── 3 · scene builder ──────────────────────────────────────────────────────
// Place objects on a seabed, render it through the real L1 chain, then measure
// each one back from its shadow. The measurement is allowed to disagree with
// the truth, and when it does the visitor sees the error column.

const DEFAULT_TARGETS = [
  { cls: 'cylinder_drum', ground_range_m: 16, height_m: 1.0, side: 'starboard' },
  { cls: 'ghost_net', ground_range_m: 28, height_m: 1.6, side: 'starboard' },
  { cls: 'rock_cluster', ground_range_m: 22, height_m: 1.1, side: 'port' },
]

function ScenePanel({ pushToast }) {
  const [classes, setClasses] = useState([])
  const [targets, setTargets] = useState(DEFAULT_TARGETS)
  const [altitude, setAltitude] = useState(9)
  const [range, setRange] = useState(50)
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null)
  const ranOnce = useRef(false)

  useEffect(() => {
    fetchSimClasses()
      .then(setClasses)
      .catch(() => setClasses([]))
  }, [])

  const run = useCallback(async () => {
    if (!targets.length) {
      pushToast('Place at least one object on the seabed first.', 'error')
      return
    }
    setBusy(true)
    try {
      const body = {
        targets, altitude_m: altitude, slant_range_m: range, n_pings: 420, seed: 26057,
      }
      setResult(await simulateScene(body))
    } catch (e) {
      pushToast(`Simulation failed: ${e.message}`, 'error')
    } finally {
      setBusy(false)
    }
  }, [targets, altitude, range, pushToast])

  useEffect(() => {
    if (!ranOnce.current && classes.length) {
      ranOnce.current = true
      run()
    }
  }, [classes, run])

  const update = (i, patch) =>
    setTargets((ts) => ts.map((t, j) => (j === i ? { ...t, ...patch } : t)))

  const errors = (result?.targets || [])
    .map((t) => t.height_error_m)
    .filter((e) => e != null)
  const meanAbsErr = errors.length
    ? errors.reduce((a, b) => a + Math.abs(b), 0) / errors.length
    : null

  return (
    <section className="lab-panel">
      <header className="lab-panel-head">
        <span className="eyebrow">Simulation 03</span>
        <h3>Build a seabed, then measure it back</h3>
        <p className="lab-lede">
          Place objects, render them through the full L1 chain — bottom tracking, gain
          normalization, slant correction, despeckle, CLAHE — and then recover each
          object's height from its shadow alone. The truth column is what the renderer
          was told to draw; the measured column is what the physics got back from the
          image. They are allowed to disagree.
        </p>
      </header>

      <div className="lab-scene-controls">
        <Slider label="Altitude" unit="m" value={altitude} min={4} max={20} step={0.5}
                onChange={setAltitude} />
        <Slider label="Slant range" unit="m" value={range} min={25} max={90} step={5}
                onChange={setRange} />
        <button type="button" className="btn primary" onClick={run} disabled={busy}>
          {busy ? 'Rendering…' : 'Run simulation'}
        </button>
      </div>

      <div className="lab-target-editor">
        {targets.map((t, i) => (
          <div className="lab-target-row" key={i}>
            <select value={t.cls} onChange={(e) => update(i, { cls: e.target.value })}>
              {classes.map((c) => (
                <option key={c.cls} value={c.cls}>
                  {c.cls.replace(/_/g, ' ')}
                  {c.natural ? ' (natural)' : ''}
                </option>
              ))}
            </select>
            <label className="lab-mini">
              range
              <input type="number" min="2" max="90" step="1" value={t.ground_range_m}
                     onChange={(e) => update(i, { ground_range_m: Number(e.target.value) })} />
              m
            </label>
            <label className="lab-mini">
              height
              <input type="number" min="0.1" max="8" step="0.1" value={t.height_m}
                     onChange={(e) => update(i, { height_m: Number(e.target.value) })} />
              m
            </label>
            <select value={t.side} onChange={(e) => update(i, { side: e.target.value })}>
              <option value="port">port</option>
              <option value="starboard">starboard</option>
            </select>
            <button type="button" className="btn tiny"
                    onClick={() => setTargets((ts) => ts.filter((_, j) => j !== i))}>
              remove
            </button>
          </div>
        ))}
        {targets.length < 8 && (
          <button
            type="button"
            className="btn tiny add"
            onClick={() =>
              setTargets((ts) => [
                ...ts,
                { cls: 'tire', ground_range_m: 20, height_m: 0.4, side: 'starboard' },
              ])
            }
          >
            + add object
          </button>
        )}
      </div>

      {result && (
        <div className="lab-scene-out">
          <figure className="lab-waterfall">
            <img src={`data:image/png;base64,${result.waterfall_png_b64}`}
                 alt="Simulated sonar waterfall" />
            <figcaption className="mono">
              {result.n_pings} pings · {result.ground_res} m/px · port | starboard,
              nadir at centre
            </figcaption>
          </figure>

          <table className="lab-table">
            <thead>
              <tr>
                <th>object</th>
                <th>range</th>
                <th>truth H</th>
                <th>measured H</th>
                <th>error</th>
                <th>shadow</th>
                <th>cues</th>
              </tr>
            </thead>
            <tbody>
              {result.targets.map((t, i) => (
                <tr key={i} className={t.natural ? 'natural' : ''}>
                  <td>
                    {t.cls.replace(/_/g, ' ')}
                    {t.natural && <span className="lab-tag">natural</span>}
                    {t.multipath_suspect && <span className="lab-tag warn">multipath band</span>}
                  </td>
                  <td className="mono">{t.ground_range_m.toFixed(1)} m</td>
                  <td className="mono">{t.truth_height_m.toFixed(2)} m</td>
                  <td className="mono">
                    {t.measured_height_m == null ? '—' : `${t.measured_height_m.toFixed(2)} m`}
                  </td>
                  <td className="mono">
                    {t.height_error_m == null ? '—' : (
                      <span className={Math.abs(t.height_error_m) > 0.5 ? 'err-hi' : 'err-ok'}>
                        {t.height_error_m > 0 ? '+' : ''}
                        {t.height_error_m.toFixed(2)} m
                      </span>
                    )}
                  </td>
                  <td className="mono">{t.shadow_len_m.toFixed(2)} m</td>
                  <td className="mono">
                    {t.has_highlight ? 'HL' : '--'} {t.has_shadow ? 'SH' : '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {meanAbsErr != null && (
            <p className="lab-insight">
              Mean absolute height error <strong>{meanAbsErr.toFixed(2)} m</strong> across{' '}
              {errors.length} object{errors.length === 1 ? '' : 's'}, recovered from shadow
              geometry alone — no model, no training, just the ray triangle. Objects placed
              beyond the usable swath ({result.max_ground_range_m.toFixed(1)} m) are pulled
              back in, because a sonar cannot image what its geometry does not reach.
            </p>
          )}
        </div>
      )}
    </section>
  )
}

export default function PhysicsLab({ pushToast }) {
  return (
    <div className="lab">
      <div className="lab-intro">
        <span className="eyebrow">Physics Lab</span>
        <p>
          Interactive models of the acoustics behind every contact. Each panel calls the
          deployed backend — <span className="mono">sonar_core.geometry</span> and{' '}
          <span className="mono">physicheck.shadow</span> — rather than re-deriving the
          formulas in the browser, so what you see here is what runs on a real survey.
        </p>
      </div>
      <ShadowPanel pushToast={pushToast} />
      <ResolutionPanel pushToast={pushToast} />
      <ScenePanel pushToast={pushToast} />
    </div>
  )
}
