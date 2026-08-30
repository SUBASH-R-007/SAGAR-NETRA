import { useState } from 'react'
import { evidenceUrl, postReview, thumbUrl } from '../api'
import { fmtDims, fmtMeters } from '../utils'
import PhysicsBadges from './PhysicsBadges'
import SeverityChip from './SeverityChip'

// Shared contact card: used inside map popups (thumbnail) and in the
// waterfall side panel (full evidence card).
export default function ContactPopover({
  contact,
  onReview,
  pushToast,
  showEvidence = false,
  canReview = true,
} ) {
  const [busy, setBusy] = useState(false)
  const [notes, setNotes] = useState('')
  const c = contact

  const setStatus = async (status) => {
    setBusy(true)
    try {
      const updated = await postReview(c.id, status, notes.trim() || null)
      onReview(updated)
      setNotes('')
      pushToast(`${c.id} marked ${status}`, 'ok')
    } catch (err) {
      pushToast(`Review failed: ${err.message}`, 'error')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="contact-pop">
      <img
        className="pop-img"
        src={showEvidence ? evidenceUrl(c.id) : thumbUrl(c.id)}
        alt={`${c.id} sonar evidence`}
        onError={(e) => {
          e.currentTarget.style.display = 'none'
        }}
      />
      <div className="pop-head">
        <span className="pop-id mono">{c.id}</span>
        <SeverityChip value={c.severity} />
      </div>
      <div className="pop-grid">
        <span>Class</span>
        <b>{c.cls}</b>
        <span>Confidence</span>
        <b className="mono">{Math.round(c.confidence)}%</b>
        <span>Dims</span>
        <b className="mono">{fmtDims(c.dims)}</b>
        <span>Depth</span>
        <b className="mono">{fmtMeters(c.depth_m)}</b>
        <span>Along-track res</span>
        <b className="mono" title="Beam footprint at this range - the resolution floor under length_m">
          {c.dims && c.dims.along_track_resolution_m != null
            ? `±${c.dims.along_track_resolution_m.toFixed(2)} m`
            : '—'}
        </b>
        <span>Position ±</span>
        <b className="mono">
          {c.position_accuracy_m != null ? `${c.position_accuracy_m.toFixed(1)} m` : '—'}
        </b>
        <span>Brains</span>
        <b>{c.brains && c.brains.length ? c.brains.join(' · ') : '—'}</b>
        <span>Review</span>
        <b className={`rv rv-${c.review}`}>{c.review}</b>
      </div>
      <PhysicsBadges physics={c.physics} />
      {canReview && (
      <input
        className="pop-notes"
        type="text"
        maxLength={200}
        placeholder="Review note (optional) - kept in the audit trail"
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        aria-label="Review note"
      />
      )}
      <div className="pop-actions">
        {/* Hiding these is a courtesy: the API refuses the call regardless. */}
        {canReview && (
          <>
        <button
          type="button"
          className="btn ok"
          disabled={busy || c.review === 'confirmed'}
          onClick={() => setStatus('confirmed')}
        >
          Confirm
        </button>
        <button
          type="button"
          className="btn danger"
          disabled={busy || c.review === 'rejected'}
          onClick={() => setStatus('rejected')}
        >
          Reject
        </button>
          </>
        )}
        <a className="btn link" href={evidenceUrl(c.id)} target="_blank" rel="noreferrer">
          Evidence
        </a>
      </div>
    </div>
  )
}
