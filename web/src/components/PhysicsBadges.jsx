// Acoustic-cue badges from PhysiCheck: highlight, shadow, and (when the
// return geometry is impossible) a red VIOLATION badge with the reason.
export default function PhysicsBadges({ physics }) {
  const p = physics || {}
  const ratio = (v) => (v == null ? '' : ` (ratio ${Number(v).toFixed(2)})`)
  return (
    <div className="badges">
      <span
        className={p.highlight ? 'badge on' : 'badge'}
        title={`acoustic highlight ${p.highlight ? 'present' : 'absent'}${ratio(p.highlight_ratio)}`}
      >
        HL
      </span>
      <span
        className={p.shadow ? 'badge on' : 'badge'}
        title={`acoustic shadow ${p.shadow ? 'present' : 'absent'}${ratio(p.shadow_ratio)}`}
      >
        SH
      </span>
      {p.physics_violation && (
        <span className="badge violation" title={p.violation_reason || 'physics violation'}>
          VIOLATION
        </span>
      )}
    </div>
  )
}
