import { Fragment, useMemo, useState } from 'react'
import { evidenceUrl, postRecovery, postReview, reportUrl, thumbUrl } from '../api'
import { fmtDims, fmtMeters } from '../utils'
import EmptyState from './EmptyState'
import PhysicsBadges from './PhysicsBadges'
import SeverityChip from './SeverityChip'

// HIGH / MEDIUM / LOW is an ordered band (geoscribe.report.priority_for), so it
// sorts on rank, not alphabetically.
const PRIORITY_RANK = { HIGH: 3, MEDIUM: 2, LOW: 1 }

const SORTS = {
  severity: (c) => c.severity,
  confidence: (c) => c.confidence,
  class: (c) => c.cls,
  id: (c) => c.id,
  priority: (c) => PRIORITY_RANK[c.priority] ?? 0,
  action: (c) => c.recommended_action || '',
}
// Header keys that read best low-to-high on first click.
const ASCENDING_FIRST = new Set(['class', 'id', 'action'])
const REPORT_FMTS = ['json', 'csv', 'geojson', 'kml', 'pdf']
const BREAKDOWN_KEYS = ['hazard', 'size', 'height', 'depth', 'proximity']
// Recovery workflow ring: flagged -> assigned -> retrieved (-> flagged to undo).
const NEXT_RECOVERY = { flagged: 'assigned', assigned: 'retrieved', retrieved: 'flagged' }

function BreakdownBars({ breakdown }) {
  const b = breakdown || {}
  return (
    <div className="breakdown">
      {BREAKDOWN_KEYS.map((k) => {
        const v = Number(b[k] || 0)
        return (
          <div key={k} className="bk-row">
            <span className="bk-label mono">{k}</span>
            <div className="bk-bar">
              <div className="bk-fill" style={{ width: `${Math.min(v, 100)}%` }} />
            </div>
            <span className="bk-value mono">{v.toFixed(0)}</span>
          </div>
        )
      })}
      <div className="bk-layer">
        Nearest sensitive layer:{' '}
        <b>
          {b.nearest_layer
            ? `${b.nearest_layer} — ${fmtMeters(b.nearest_layer_distance_m, 0)}`
            : 'none within range'}
        </b>
      </div>
    </div>
  )
}

// Triage priority as a band-coloured tag — HIGH reads on the critical token,
// MEDIUM on high, LOW on low (geoscribe.report.priority_for).
function PriorityTag({ value }) {
  if (!value) return <span className="muted">—</span>
  return (
    <span
      className={`tag pri-${String(value).toLowerCase()}`}
      title={`triage priority ${value}`}
    >
      {value}
    </span>
  )
}

// Position error budget from geoscribe.build.position_accuracy. The stored 0.0
// means "no estimate on this record", so it shows a dash, never "±0 m".
function PositionAccuracy({ value }) {
  return (
    <div className="pos-acc">
      <div className="pos-acc-value">
        Position accuracy <b>{value ? `±${Number(value).toFixed(1)} m` : '—'}</b>
      </div>
      <div className="pos-acc-note">
        {value
          ? `Ground resolution, layback and nav-fix terms are summed linearly rather than in
             quadrature — each is a bias, not independent noise — so this is a conservative
             search radius a diver can plan against.`
          : 'Not recorded: this contact was stored before the accuracy estimate existed.'}
      </div>
    </div>
  )
}

export default function ContactsTable({
  contacts,
  survey,
  onReview,
  pushToast,
  canReview = true,
  canRecover = true,
}) {
  const [sortKey, setSortKey] = useState('severity')
  const [dir, setDir] = useState(-1)
  const [expandedId, setExpandedId] = useState(null)
  const [busyId, setBusyId] = useState(null)

  const sorted = useMemo(() => {
    const get = SORTS[sortKey]
    return [...contacts].sort((a, b) => {
      const va = get(a)
      const vb = get(b)
      if (va < vb) return -1 * dir
      if (va > vb) return 1 * dir
      return 0
    })
  }, [contacts, sortKey, dir])

  const clickSort = (key) => {
    if (key === sortKey) {
      setDir((d) => -d)
    } else {
      setSortKey(key)
      setDir(ASCENDING_FIRST.has(key) ? 1 : -1)
    }
  }

  // Drawn sort indicator — a CSS triangle, no glyphs.
  const sortMark = (key) =>
    key === sortKey ? (
      <span className={dir === 1 ? 'sort-tri asc' : 'sort-tri desc'} aria-hidden="true" />
    ) : null

  const ariaSort = (key) =>
    key === sortKey ? (dir === 1 ? 'ascending' : 'descending') : undefined

  const review = async (c, status) => {
    setBusyId(c.id)
    try {
      onReview(await postReview(c.id, status))
      pushToast(`${c.id} marked ${status}`, 'ok')
    } catch (err) {
      pushToast(`Review failed: ${err.message}`, 'error')
    } finally {
      setBusyId(null)
    }
  }

  const cycleRecovery = async (c) => {
    const status = NEXT_RECOVERY[c.recovery] || 'assigned'
    setBusyId(c.id)
    try {
      onReview(await postRecovery(c.id, status))
      pushToast(`${c.id} recovery set to ${status}`, 'ok')
    } catch (err) {
      pushToast(`Recovery update failed: ${err.message}`, 'error')
    } finally {
      setBusyId(null)
    }
  }

  if (!survey) {
    return (
      <EmptyState
        title="No survey selected"
        hint="Upload a survey from the rail on the right — its contacts will list here."
      />
    )
  }

  return (
    <div className="contacts-wrap">
      <div className="contacts-toolbar">
        <span className="ctl-label">Download report</span>
        {REPORT_FMTS.map((fmt) => (
          <button
            key={fmt}
            type="button"
            className="btn small"
            onClick={() => window.open(reportUrl(fmt, survey), '_blank')}
          >
            {fmt.toUpperCase()}
          </button>
        ))}
        <span className="filter-count mono">{sorted.length} contacts</span>
      </div>

      {sorted.length === 0 ? (
        <EmptyState
          title="No contacts to show"
          hint="This survey has no contacts matching the current class / confidence filters."
        />
      ) : (
        <div className="table-scroll">
          <table className="contacts-table">
            <thead>
              <tr>
                <th className="sortable" aria-sort={ariaSort('id')}>
                  <button type="button" className="cell-btn" onClick={() => clickSort('id')}>
                    ID{sortMark('id')}
                  </button>
                </th>
                <th>Thumb</th>
                <th className="sortable" aria-sort={ariaSort('class')}>
                  <button type="button" className="cell-btn" onClick={() => clickSort('class')}>
                    Class{sortMark('class')}
                  </button>
                </th>
                <th className="sortable num" aria-sort={ariaSort('confidence')}>
                  <button type="button" className="cell-btn" onClick={() => clickSort('confidence')}>
                    Conf{sortMark('confidence')}
                  </button>
                </th>
                <th className="sortable num" aria-sort={ariaSort('severity')}>
                  <button type="button" className="cell-btn" onClick={() => clickSort('severity')}>
                    Sev{sortMark('severity')}
                  </button>
                </th>
                <th className="sortable" aria-sort={ariaSort('priority')}>
                  <button type="button" className="cell-btn" onClick={() => clickSort('priority')}>
                    Priority{sortMark('priority')}
                  </button>
                </th>
                <th className="sortable" aria-sort={ariaSort('action')}>
                  <button type="button" className="cell-btn" onClick={() => clickSort('action')}>
                    Recommended action{sortMark('action')}
                  </button>
                </th>
                <th>Dims L×W×H</th>
                <th className="num">Depth</th>
                <th>Physics</th>
                <th>Review</th>
                <th>Recovery</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((c) => (
                <Fragment key={c.id}>
                  <tr className={expandedId === c.id ? 'row expanded' : 'row'}>
                    <td className="mono id-cell">
                      <button
                        type="button"
                        className="cell-btn"
                        aria-expanded={expandedId === c.id}
                        title="toggle severity breakdown"
                        onClick={() => setExpandedId((cur) => (cur === c.id ? null : c.id))}
                      >
                        <span
                          className={expandedId === c.id ? 'tri open' : 'tri'}
                          aria-hidden="true"
                        />
                        {c.id}
                      </button>
                    </td>
                    <td>
                      <img
                        className="row-thumb"
                        src={thumbUrl(c.id)}
                        alt=""
                        loading="lazy"
                        onError={(e) => {
                          e.currentTarget.style.visibility = 'hidden'
                        }}
                      />
                    </td>
                    <td>{c.cls}</td>
                    <td className="num mono">{Math.round(c.confidence)}%</td>
                    <td className="num">
                      <SeverityChip value={c.severity} />
                    </td>
                    <td>
                      <PriorityTag value={c.priority} />
                    </td>
                    <td>
                      {c.recommended_action ? (
                        <span className="rec-action" title={c.recommended_action}>
                          {c.recommended_action}
                        </span>
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                    <td className="mono">{fmtDims(c.dims)}</td>
                    <td className="num mono">{fmtMeters(c.depth_m)}</td>
                    <td>
                      <PhysicsBadges physics={c.physics} />
                    </td>
                    <td>
                      <span className={`rv rv-${c.review}`}>{c.review}</span>
                    </td>
                    <td className="recovery-cell">
                      <div className="row-actions">
                        <span className={`rc rc-${c.recovery || 'flagged'}`}>
                          {c.recovery || 'flagged'}
                        </span>
                        {canRecover && (
                        <button
                          type="button"
                          className="btn small"
                          disabled={busyId === c.id}
                          title={`advance to ${NEXT_RECOVERY[c.recovery] || 'assigned'}`}
                          onClick={() => cycleRecovery(c)}
                        >
                          Advance
                        </button>
                        )}
                      </div>
                    </td>
                    <td className="actions-cell">
                      <div className="row-actions">
                        {canReview && (
                        <button
                          type="button"
                          className="btn ok small"
                          disabled={busyId === c.id || c.review === 'confirmed'}
                          onClick={() => review(c, 'confirmed')}
                        >
                          Confirm
                        </button>
                        )}
                        {canReview && (
                        <button
                          type="button"
                          className="btn danger small"
                          disabled={busyId === c.id || c.review === 'rejected'}
                          onClick={() => review(c, 'rejected')}
                        >
                          Reject
                        </button>
                        )}
                        <a
                          className="btn link small"
                          href={evidenceUrl(c.id)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          Evidence
                        </a>
                      </div>
                    </td>
                  </tr>
                  {expandedId === c.id && (
                    <tr className="expand-row">
                      <td colSpan={13}>
                        <BreakdownBars breakdown={c.severity_breakdown} />
                        <PositionAccuracy value={c.position_accuracy_m} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
