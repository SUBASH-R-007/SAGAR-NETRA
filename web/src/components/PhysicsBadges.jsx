// Acoustic-cue tags from PhysiCheck: highlight, shadow, and (when the return
// geometry is impossible) a VIOLATION tag outlined in sev-critical.
//
// Two things this deliberately does NOT do any more. It no longer carries the
// present/absent state in text colour alone — a red-green-blind operator, or
// anyone looking at a projector, saw two identical grey tags. And it no longer
// hides the explanation in a `title`, which is unreachable by keyboard, by
// touch, and by anyone driving the demo with a clicker. The glyph now states
// the answer and the aria-label carries a plain sentence.
export default function PhysicsBadges({ physics }) {
  const p = physics || {}
  const ratio = (v) => (v == null ? '' : `, measured ratio ${Number(v).toFixed(2)}`)
  return (
    <div className="badges">
      <span
        className={p.highlight ? 'tag cue on' : 'tag cue'}
        title={`Acoustic highlight ${p.highlight ? 'present' : 'absent'}${ratio(p.highlight_ratio)}. A highlight is sound bounced back harder than the surrounding seabed — a hard object.`}
        aria-label={`Acoustic highlight ${p.highlight ? 'present' : 'absent'}${ratio(p.highlight_ratio)}`}
      >
        HL {p.highlight ? '\u2713' : '\u2014'}
      </span>
      <span
        className={p.shadow ? 'tag cue on' : 'tag cue'}
        title={`Acoustic shadow ${p.shadow ? 'present' : 'absent'}${ratio(p.shadow_ratio)}. A shadow is the sound-less patch an object casts behind itself — it is what gives the object a height.`}
        aria-label={`Acoustic shadow ${p.shadow ? 'present' : 'absent'}${ratio(p.shadow_ratio)}`}
      >
        SH {p.shadow ? '\u2713' : '\u2014'}
      </span>
      {p.physics_violation && (
        <span
          className="tag cue violation"
          title={p.violation_reason || 'physics violation'}
          aria-label={`Physics violation: ${p.violation_reason || 'the return geometry is not consistent with this class'}`}
        >
          VIOLATION
        </span>
      )}
    </div>
  )
}
