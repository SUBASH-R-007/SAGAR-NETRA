import { useEffect, useRef, useState } from 'react'
import { askCopilot } from '../api'

const EXAMPLES = [
  'How many ghost nets?',
  'Top 5 most severe contacts',
  'Contacts near the turtle nesting zone',
  'How many confirmed contacts?',
]

function RowsTable({ rows }) {
  if (!Array.isArray(rows) || rows.length === 0) return null
  const shown = rows.slice(0, 12)
  const objectRows = typeof shown[0] === 'object' && shown[0] !== null && !Array.isArray(shown[0])
  const headers = objectRows ? Object.keys(shown[0]) : null
  return (
    <div className="table-scroll copilot-rows">
      <table className="contacts-table">
        {headers && (
          <thead>
            <tr>
              {headers.map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
        )}
        <tbody>
          {shown.map((row, i) => (
            <tr key={i}>
              {(objectRows ? headers.map((h) => row[h]) : [].concat(row)).map((cell, j) => (
                <td key={j} className="mono">
                  {cell == null ? '—' : String(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > shown.length && (
        <p className="muted small-note">…and {rows.length - shown.length} more rows</p>
      )}
    </div>
  )
}

export default function Copilot({ pushToast }) {
  const [log, setLog] = useState([])
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef(null)

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    endRef.current?.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'end' })
  }, [log, busy])

  const ask = async (question) => {
    const text = (question ?? q).trim()
    if (!text || busy) return
    setQ('')
    setLog((l) => [...l, { role: 'user', text }])
    setBusy(true)
    try {
      const res = await askCopilot(text)
      setLog((l) => [
        ...l,
        { role: 'bot', text: res.answer, sql: res.sql, rows: res.rows, mode: res.mode },
      ])
    } catch (err) {
      setLog((l) => [...l, { role: 'bot', text: `Copilot error: ${err.message}`, error: true }])
      pushToast(`Copilot failed: ${err.message}`, 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="copilot">
      <div className="copilot-log">
        {log.length === 0 && (
          <div className="copilot-hello">
            <div className="empty-eyebrow">Survey Copilot</div>
            <p>
              Ask the survey copilot about detected contacts — it answers from the contact
              database and shows its SQL.
            </p>
          </div>
        )}
        {log.map((entry, i) => (
          <div key={i} className={`msg msg-${entry.role}${entry.error ? ' msg-error' : ''}`}>
            <div className="msg-role">{entry.role === 'user' ? 'YOU' : 'COPILOT'}</div>
            <div className="msg-text">{entry.text}</div>
            {entry.sql && (
              <pre className="msg-sql">
                <code>{entry.sql}</code>
              </pre>
            )}
            <RowsTable rows={entry.rows} />
          </div>
        ))}
        {busy && (
          <div className="msg msg-bot">
            <div className="msg-role">COPILOT</div>
            <div className="msg-text muted">thinking…</div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="copilot-chips">
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" className="chip" onClick={() => ask(ex)} disabled={busy}>
            {ex}
          </button>
        ))}
      </div>

      <form
        className="copilot-input"
        onSubmit={(e) => {
          e.preventDefault()
          ask()
        }}
      >
        <input
          type="text"
          value={q}
          placeholder="Ask about contacts, severity, zones…"
          onChange={(e) => setQ(e.target.value)}
        />
        <button type="submit" className="btn primary" disabled={busy || !q.trim()}>
          Ask
        </button>
      </form>
    </div>
  )
}
