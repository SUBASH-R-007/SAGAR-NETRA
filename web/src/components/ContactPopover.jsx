import { useState } from 'react'
import { evidenceUrl, postReview, thumbUrl } from '../api'
import { fmtDims, fmtMeters } from '../utils'
import PhysicsBadges from './PhysicsBadges'
import SeverityChip from './SeverityChip'

// Shared contact card: used inside map popups (thumbnail) and in the
// waterfall side panel (full evidence card).
export default function ContactPopover({ contact, onReview, pushToast, showEvidence = false }) {
  const [busy, setBusy] = useState(false)
  const c = contact

  const setStatus = async (status) => {
    setBusy(true)
    try {
      const updated = await postReview(c.id, status)
      onReview(updated)
      pushToast(`${c.id} → ${status}`, 'ok')
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
        <b>{Math.round(c.confidence)}%</b>
        <span>Dims</span>
        <b>{fmtDims(c.dims)}</b>
        <span>Depth</span>
        <b>{fmtMeters(c.depth_m)}</b>
        <span>Brains</span>
        <b>{c.brains && c.brains.length ? c.brains.join(' · ') : '—'}</b>
        <span>Review</span>
        <b className={`rv rv-${c.review}`}>{c.review}</b>
      </div>
      <PhysicsBadges physics={c.physics} />
      <div className="pop-actions">
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
        <a className="btn link" href={evidenceUrl(c.id)} target="_blank" rel="noreferrer">
          Evidence ↗
        </a>
      </div>
    </div>
  )
}
