export default function EmptyState({ title, hint }) {
  return (
    <div className="empty">
      <div className="empty-icon" aria-hidden="true">
        ◌
      </div>
      <div className="empty-title">{title}</div>
      {hint && <div className="empty-hint">{hint}</div>}
    </div>
  )
}
