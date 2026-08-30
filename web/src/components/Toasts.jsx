// Notification region.
//
// Both live regions are mounted unconditionally, even with nothing to say. A
// region that appears in the same commit as its first message is routinely
// dropped by NVDA, JAWS and VoiceOver — it has to already exist for the screen
// reader to be watching it. Returning `null` when empty, which is what this
// component used to do, meant every error the console raised was announced to
// sighted users only.
//
// Errors get their own assertive region and do not self-dismiss: a survey that
// failed to ingest is still not ingested six seconds later.
export default function Toasts({ toasts, onDismiss }) {
  const errors = toasts.filter((t) => t.kind === 'error')
  const rest = toasts.filter((t) => t.kind !== 'error')
  return (
    <div className="toast-stack">
      <div role="status" aria-live="polite" className="toast-region">
        {rest.map((t) => (
          <div key={t.id} className={`toast toast-${t.kind}`}>
            {t.text}
          </div>
        ))}
      </div>
      <div role="alert" aria-live="assertive" className="toast-region">
        {errors.map((t) => (
          <div key={t.id} className={`toast toast-${t.kind}`}>
            <span>{t.text}</span>
            {onDismiss && (
              <button
                type="button"
                className="toast-dismiss"
                aria-label="Dismiss this error"
                onClick={() => onDismiss(t.id)}
              >
                &times;
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
