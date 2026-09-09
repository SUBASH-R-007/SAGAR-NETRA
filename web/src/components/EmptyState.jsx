// A wall the user has hit. The `action` slot exists because telling someone
// where to go in prose and then making them find it in a nine-item nav of
// unfamiliar labels is the loop that defeats a first-time user under demo
// pressure — every empty state in the console could describe a destination but
// none could offer it.
export default function EmptyState({ title, hint, action }) {
  return (
    <div className="empty">
      <div className="empty-eyebrow">{title}</div>
      {hint && <div className="empty-hint">{hint}</div>}
      {action && <div className="empty-action">{action}</div>}
    </div>
  )
}
