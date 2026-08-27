export default function EmptyState({ title, hint }) {
  return (
    <div className="empty">
      <div className="empty-eyebrow">{title}</div>
      {hint && <div className="empty-hint">{hint}</div>}
    </div>
  )
}
