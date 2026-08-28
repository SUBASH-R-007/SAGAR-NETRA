import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchJob, fetchJobs, fetchMissions, jobSocketUrl, uploadFile } from '../api'
import { prettyName } from '../utils'

// .sl2/.sl3 = Lowrance citizen sonar; .zip = Humminbird recording archive
// (.DAT + .SON directory). Keep in sync with api/main.py UPLOAD_SUFFIXES.
const ACCEPT = '.xtf,.jsf,.tif,.tiff,.png,.jpg,.jpeg,.sl2,.sl3,.zip'

// Right-hand rail: drag-drop upload, live WebSocket progress, survey refresh.
// Falls back to 1 s polling of GET /api/jobs/{id} if the socket fails.
export default function UploadRail({ onJobDone, pushToast }) {
  const [jobs, setJobs] = useState([])
  const [drag, setDrag] = useState(false)
  const [mode, setMode] = useState('batch') // 'batch' | 'stream' (live replay)
  const [missions, setMissions] = useState([]) // [{name, description}]
  const [missionState, setMissionState] = useState('loading') // loading|ready|error
  const [mission, setMission] = useState('') // '' = standard survey, no profile
  const fileRef = useRef(null)
  const finishedRef = useRef(new Set())

  // Disaster-mode profiles from configs/missions/*.yaml. An empty list is a
  // legitimate answer (a deployment can ship none), so it is not an error.
  useEffect(() => {
    let alive = true
    fetchMissions()
      .then((list) => {
        if (!alive) return
        setMissions(Array.isArray(list) ? list : [])
        setMissionState('ready')
      })
      .catch((err) => {
        if (!alive) return
        setMissionState('error')
        pushToast(`Mission profiles unavailable: ${err.message}`, 'error')
      })
    return () => {
      alive = false
    }
  }, [pushToast])

  useEffect(() => {
    fetchJobs()
      .then((list) => {
        // Jobs that were already finished before this page loaded should not
        // re-fire the "survey processed" side effects.
        list.forEach((j) => {
          if (j.status === 'done' || j.status === 'error') finishedRef.current.add(j.id)
        })
        setJobs([...list].reverse())
      })
      .catch(() => {
        /* no backend yet — the rail still renders */
      })
  }, [])

  const upsert = useCallback((snap) => {
    setJobs((js) => {
      const i = js.findIndex((j) => j.id === snap.id)
      if (i === -1) return [snap, ...js]
      const next = [...js]
      next[i] = { ...next[i], ...snap }
      return next
    })
  }, [])

  const finish = useCallback(
    (snap) => {
      if (finishedRef.current.has(snap.id)) return
      finishedRef.current.add(snap.id)
      if (snap.status === 'done') {
        pushToast(`${snap.survey || 'Survey'} processed — ${snap.n_contacts ?? 0} contacts`, 'ok')
        onJobDone(snap.survey)
      } else {
        pushToast(`Processing failed: ${snap.error || snap.message || 'unknown error'}`, 'error')
      }
    },
    [onJobDone, pushToast],
  )

  const track = useCallback(
    (jobId) => {
      let settled = false

      const poll = async () => {
        if (finishedRef.current.has(jobId)) return
        try {
          const snap = await fetchJob(jobId)
          upsert(snap)
          if (snap.status === 'done' || snap.status === 'error') {
            finish(snap)
            return
          }
        } catch {
          /* transient — keep polling */
        }
        setTimeout(poll, 1000)
      }

      let ws
      try {
        ws = new WebSocket(jobSocketUrl(jobId))
      } catch {
        poll()
        return
      }
      ws.onmessage = (ev) => {
        try {
          const snap = JSON.parse(ev.data)
          if (!snap.id) return
          upsert(snap)
          if (snap.status === 'done' || snap.status === 'error') {
            settled = true
            finish(snap)
          }
        } catch {
          /* ignore malformed frame */
        }
      }
      ws.onerror = () => {
        if (!settled) {
          settled = true
          try {
            ws.close()
          } catch {
            /* already closed */
          }
          poll()
        }
      }
      ws.onclose = () => {
        if (!settled) {
          settled = true
          poll()
        }
      }
    },
    [upsert, finish],
  )

  const handleFiles = useCallback(
    async (files) => {
      const file = files && files[0]
      if (!file) return
      // The API rejects mission + stream with 422. The select is disabled in
      // stream mode, so a stale selection is dropped rather than sent.
      const profile = mode === 'stream' ? '' : mission
      try {
        const { job_id: jobId } = await uploadFile(file, mode, profile)
        pushToast(
          `Uploaded ${file.name} — ${mode === 'stream' ? 'live stream' : 'processing'} started` +
            (profile ? ` under the ${prettyName(profile)} profile` : ''),
          'ok',
        )
        upsert({
          id: jobId, status: 'queued', stage: 'queued', fraction: 0, message: file.name, mode,
          mission: profile || null,
        })
        track(jobId)
      } catch (err) {
        pushToast(`Upload failed: ${err.message}`, 'error')
      }
    },
    [mission, mode, pushToast, track, upsert],
  )

  const pct = (j) => Math.round((j.fraction || 0) * 100)

  // A profile only reaches the API in batch mode; the select is locked
  // otherwise, and every lock state says why in the caption below it.
  const missionLocked =
    mode === 'stream' || missionState !== 'ready' || missions.length === 0
  const selected = missions.find((m) => m.name === mission)
  let missionCaption
  if (mode === 'stream') {
    missionCaption = 'Mission profiles apply to batch processing only — a live stream is'
      + ' replayed ping by ping and cannot be re-ranked mid-run.'
  } else if (missionState === 'loading') {
    missionCaption = 'Loading mission profiles…'
  } else if (missionState === 'error') {
    missionCaption = 'Mission profiles could not be read — uploads run the standard survey.'
  } else if (missions.length === 0) {
    missionCaption = 'No mission profiles installed on this deployment.'
  } else if (selected) {
    missionCaption = selected.description
  } else {
    missionCaption = 'Default hazard weighting — no mission re-ranking applied.'
  }

  return (
    <aside className="rail">
      <div className="rail-section">
        <h2 className="rail-title">Survey Ingest</h2>
        <div
          className={drag ? 'dropzone drag' : 'dropzone'}
          onDragOver={(e) => {
            e.preventDefault()
            setDrag(true)
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDrag(false)
            handleFiles(e.dataTransfer.files)
          }}
          onClick={() => fileRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') fileRef.current?.click()
          }}
        >
          <div className="dz-text">
            Drop a sonar file here
            <br />
            <span className="muted">or click to browse</span>
          </div>
          <div className="dz-formats">.xtf · .jsf · .tif · .png</div>
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPT}
            hidden
            onChange={(e) => {
              handleFiles(e.target.files)
              e.target.value = ''
            }}
          />
        </div>
        <div className="rail-field">
          <label className="ctl-label" htmlFor="mission-profile">
            Mission profile
          </label>
          <select
            id="mission-profile"
            value={mission}
            disabled={missionLocked}
            aria-describedby="mission-profile-note"
            onChange={(e) => setMission(e.target.value)}
          >
            <option value="" title="default hazard weighting, no mission re-ranking">
              Standard survey
            </option>
            {/* Uppercased so the SAR acronym reads correctly and the control
                matches the console's label idiom — no name table, just the
                API's own profile id with its underscores opened out. */}
            {missions.map((m) => (
              <option key={m.name} value={m.name} title={m.description || undefined}>
                {prettyName(m.name).toUpperCase()}
              </option>
            ))}
          </select>
          <p
            id="mission-profile-note"
            className={missionState === 'error' ? 'rail-caption warn' : 'rail-caption'}
          >
            {missionCaption}
          </p>
        </div>

        <div className="seg" role="radiogroup" aria-label="processing mode">
          {['batch', 'stream'].map((m) => (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={mode === m}
              className={mode === m ? 'seg-cell active' : 'seg-cell'}
              onClick={() => setMode(m)}
              title={m === 'stream' ? 'replay as a live towed stream' : 'one-shot processing'}
            >
              {m === 'stream' ? 'LIVE STREAM' : 'BATCH'}
            </button>
          ))}
        </div>
      </div>

      <div className="rail-section grow">
        <h2 className="rail-title">Ingest Ledger</h2>
        {jobs.length === 0 && (
          <p className="muted rail-hint">No jobs yet. Uploaded surveys are parsed, enhanced,
            detected on, and scored here in real time.</p>
        )}
        <div className="job-list">
          {jobs.map((j) => (
            <div key={j.id} className={j.status === 'error' ? 'job job-error' : 'job'}>
              <div className="job-top">
                <span className="job-name mono" title={j.id}>
                  {j.survey || j.name || j.filename || j.message || j.id}
                </span>
                {j.mode === 'stream' && <span className="tag tag-live">live</span>}
                <span className={`tag tag-${j.status}`}>{j.status}</span>
              </div>
              {(j.status === 'running' || j.status === 'queued') && (
                <>
                  <div className="bar">
                    <div className="bar-fill" style={{ width: `${pct(j)}%` }} />
                  </div>
                  <div className="job-sub">
                    <span>{j.stage || '…'}</span>
                    <span className="mono">{pct(j)}%</span>
                  </div>
                </>
              )}
              {j.status === 'done' && (
                <div className="job-sub ok-text">
                  <span>{j.n_contacts ?? 0} contacts</span>
                  <span className="mono">100%</span>
                </div>
              )}
              {j.status === 'error' && (
                <div className="job-sub err-text">{j.error || j.message || 'failed'}</div>
              )}
              {j.mission && (
                <div className="job-sub job-mission">
                  <span className="ctl-label">Mission</span>
                  <span className="mono">{prettyName(j.mission).toUpperCase()}</span>
                </div>
              )}
              {j.mode === 'stream' && (j.recent_events || []).some((e) => e.type === 'contact') && (
                <div className="live-feed">
                  {(j.recent_events || [])
                    .filter((e) => e.type === 'contact')
                    .slice(-5)
                    .map((e) => (
                      <div key={e.contact.id} className="live-row mono">
                        <span className="live-id">{e.contact.id}</span>
                        <span className="live-cls">{e.contact.cls}</span>
                        <span className="live-conf">{Math.round(e.contact.confidence)}%</span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </aside>
  )
}
