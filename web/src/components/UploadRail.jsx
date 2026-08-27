import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchJob, fetchJobs, jobSocketUrl, uploadFile } from '../api'

// .sl2/.sl3 = Lowrance citizen sonar; .zip = Humminbird recording archive
// (.DAT + .SON directory). Keep in sync with api/main.py UPLOAD_SUFFIXES.
const ACCEPT = '.xtf,.jsf,.tif,.tiff,.png,.jpg,.jpeg,.sl2,.sl3,.zip'

// Right-hand rail: drag-drop upload → live WebSocket progress → survey refresh.
// Falls back to 1 s polling of GET /api/jobs/{id} if the socket fails.
export default function UploadRail({ onJobDone, pushToast }) {
  const [jobs, setJobs] = useState([])
  const [drag, setDrag] = useState(false)
  const fileRef = useRef(null)
  const finishedRef = useRef(new Set())

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
      try {
        const { job_id: jobId } = await uploadFile(file)
        pushToast(`Uploaded ${file.name} — processing started`, 'ok')
        upsert({ id: jobId, status: 'queued', stage: 'queued', fraction: 0, message: file.name })
        track(jobId)
      } catch (err) {
        pushToast(`Upload failed: ${err.message}`, 'error')
      }
    },
    [pushToast, track, upsert],
  )

  const pct = (j) => Math.round((j.fraction || 0) * 100)

  return (
    <aside className="rail">
      <div className="rail-section">
        <h2 className="rail-title mono">SURVEY INGEST</h2>
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
          <div className="dz-icon" aria-hidden="true">
            ⇪
          </div>
          <div className="dz-text">
            Drop a sonar file here
            <br />
            <span className="muted">or click to browse</span>
          </div>
          <div className="dz-formats mono">.xtf · .jsf · .tif · .png</div>
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
      </div>

      <div className="rail-section grow">
        <h2 className="rail-title mono">JOBS</h2>
        {jobs.length === 0 && (
          <p className="muted rail-hint">No jobs yet. Uploaded surveys are parsed, enhanced,
            detected on, and scored here in real time.</p>
        )}
        <div className="job-list">
          {jobs.map((j) => (
            <div key={j.id} className={`job job-${j.status}`}>
              <div className="job-top">
                <span className="job-name mono" title={j.id}>
                  {j.survey || j.name || j.filename || j.message || j.id}
                </span>
                <span className={`pill pill-${j.status}`}>{j.status}</span>
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
            </div>
          ))}
        </div>
      </div>
    </aside>
  )
}
