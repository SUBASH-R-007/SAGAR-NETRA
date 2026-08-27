import { sevColor } from '../utils'

// Severity reads as a square swatch (band color) plus a tabular mono number.
export default function SeverityChip({ value }) {
  const c = sevColor(value)
  return (
    <span className="sev mono" title={`severity ${Math.round(value)}/100`}>
      <span className="sev-swatch" style={{ background: c }} aria-hidden="true" />
      {Math.round(value)}
    </span>
  )
}
