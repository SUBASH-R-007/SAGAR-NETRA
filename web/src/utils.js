// Shared formatting + severity-band helpers.

export function sevColor(sev) {
  if (sev >= 75) return '#ff3b30'
  if (sev >= 50) return '#ff9500'
  if (sev >= 25) return '#ffcc00'
  return '#34c759'
}

export const SEVERITY_BANDS = [
  { label: 'critical  ≥75', color: '#ff3b30' },
  { label: 'high  50–74', color: '#ff9500' },
  { label: 'moderate  25–49', color: '#ffcc00' },
  { label: 'low  <25', color: '#34c759' },
]

export function fmtDims(d) {
  if (!d) return '—'
  const n = (v) => (v == null ? '?' : Number(v).toFixed(1))
  const base = `${n(d.length_m)} × ${n(d.width_m)}`
  return d.height_m != null ? `${base} × ${Number(d.height_m).toFixed(1)} m` : `${base} m`
}

export function fmtMeters(v, digits = 1) {
  return v == null ? '—' : `${Number(v).toFixed(digits)} m`
}

export function prettyName(name) {
  return String(name).replace(/_/g, ' ')
}
