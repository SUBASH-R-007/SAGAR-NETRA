import { sevColor } from '../utils'

export default function SeverityChip({ value }) {
  const c = sevColor(value)
  return (
    <span
      className="sev-chip mono"
      style={{ background: `${c}22`, color: c, borderColor: `${c}88` }}
      title={`severity ${Math.round(value)}/100`}
    >
      {Math.round(value)}
    </span>
  )
}
