import { Fragment, useMemo, useState } from 'react'
import { evidenceUrl, postRecovery, postReview, reportUrl, thumbUrl } from '../api'
import { fmtDims, fmtMeters } from '../utils'
import EmptyState from './EmptyState'
import PhysicsBadges from './PhysicsBadges'
import SeverityChip from './SeverityChip'

const SORTS = {
  severity: (c) => c.severity,
  confidence: (c) => c.confidence,
  class: (c) => c.cls,
  id: (c) => c.id,
}
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

export default function ContactsTable({ contacts, survey, onReview, pushToast }) {
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
      setDir(key === 'class' || key === 'id' ? 1 : -1)
    }
  }

  const arrow = (key) => (key === sortKey ? (dir === 1 ? ' ▲' : ' ▼') : '')

  const review = async (c, status) => {
    setBusyId(c.id)
    try {
      onReview(await postReview(c.id, status))
      pushToast(`${c.id} → ${status}`, 'ok')
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
      pushToast(`${c.id} recovery → ${status}`, 'ok')
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
                <th className="sortable" onClick={() => clickSort('id')}>
                  ID{arrow('id')}
                </th>
                <th>Thumb</th>
                <th className="sortable" onClick={() => clickSort('class')}>
                  Class{arrow('class')}
                </th>
                <th className="sortable num" onClick={() => clickSort('confidence')}>
                  Conf{arrow('confidence')}
                </th>
                <th className="sortable num" onClick={() => clickSort('severity')}>
                  Sev{arrow('severity')}
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
                    <td
                      className="mono id-cell"
                      onClick={() => setExpandedId((cur) => (cur === c.id ? null : c.id))}
                      title="toggle severity breakdown"
                    >
                      <span className="chev">{expandedId === c.id ? '▾' : '▸'}</span> {c.id}
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
                    <td className="mono">{fmtDims(c.dims)}</td>
                    <td className="num mono">{fmtMeters(c.depth_m)}</td>
                    <td>
                      <PhysicsBadges physics={c.physics} />
                    </td>
                    <td>
                      <span className={`rv rv-${c.review}`}>{c.review}</span>
                    </td>
                    <td className="recovery-cell">
                      <span className={`rc rc-${c.recovery || 'flagged'}`}>
                        {c.recovery || 'flagged'}
                      </span>
                      <button
                        type="button"
                        className="btn small"
                        disabled={busyId === c.id}
                        title={`advance to ${NEXT_RECOVERY[c.recovery] || 'assigned'}`}
                        onClick={() => cycleRecovery(c)}
                      >
                        ⟳
                      </button>
                    </td>
                    <td className="actions-cell">
                      <button
                        type="button"
                        className="btn ok small"
                        disabled={busyId === c.id || c.review === 'confirmed'}
                        onClick={() => review(c, 'confirmed')}
                      >
                        Confirm
                      </button>
                      <button
                        type="button"
                        className="btn danger small"
                        disabled={busyId === c.id || c.review === 'rejected'}
                        onClick={() => review(c, 'rejected')}
                      >
                        Reject
                      </button>
                      <a
                        className="btn link small"
                        href={evidenceUrl(c.id)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Evidence
                      </a>
                    </td>
                  </tr>
                  {expandedId === c.id && (
                    <tr className="expand-row">
                      <td colSpan={11}>
                        <BreakdownBars breakdown={c.severity_breakdown} />
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
