// Shared formatting + severity-band helpers.

// Hex values mirror the --sev-* tokens in styles.css (DESIGN.md ramp).
export function sevColor(sev) {
  if (sev >= 75) return '#BC3116'
  if (sev >= 50) return '#C67102'
  if (sev >= 25) return '#A08C00'
  return '#3E7D52'
}

export const SEVERITY_BANDS = [
  { label: 'critical  ≥75', color: '#BC3116' },
  { label: 'high  50–74', color: '#C67102' },
  { label: 'moderate  25–49', color: '#A08C00' },
  { label: 'low  <25', color: '#3E7D52' },
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
