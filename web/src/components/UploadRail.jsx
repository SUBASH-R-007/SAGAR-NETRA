import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchJob, fetchJobs, fetchMissions, jobSocketUrl, uploadFile } from '../api'
import { prettyName } from '../utils'

// .sl2/.sl3 = Lowrance citizen sonar; .zip = Humminbird recording archive
// (.DAT + .SON directory). Keep in sync with api/main.py UPLOAD_SUFFIXES.
const ACCEPT = '.xtf,.jsf,.tif,.tiff,.png,.jpg,.jpeg,.sl2,.sl3,.zip'

// Formats that record no navigation of their own. A survey log carries the
// sonar's altitude, range and position per ping; a picture of the seabed
// carries none of it, so the operator states the geometry the sonar was set
// to. Keep in sync with api/main.py GEOMETRY_REQUIRED_SUFFIXES.
const GEOMETRY_REQUIRED = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
const needsGeometry = (name) =>
  GEOMETRY_REQUIRED.some((ext) => (name || '').toLowerCase().endsWith(ext))

// Right-hand rail: drag-drop upload, live WebSocket progress, survey refresh.
// Falls back to 1 s polling of GET /api/jobs/{id} if the socket fails.
export default function UploadRail({ onJobDone, pushToast }) {
  const [jobs, setJobs] = useState([])
  const [drag, setDrag] = useState(false)
  const [mode, setMode] = useState('batch') // 'batch' | 'stream' (live replay)
  const [missions, setMissions] = useState([]) // [{name, description}]
  const [missionState, setMissionState] = useState('loading') // loading|ready|error
  const [mission, setMission] = useState('') // '' = standard survey, no profile
  // An image is held here until its geometry is supplied; logs never stage.
  const [pending, setPending] = useState(null)
  const [geom, setGeom] = useState({
    altitude_m: '', range_m: '', lat: '', lon: '', heading_deg: '90', sensor_depth_m: '',
  })
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

  const submit = useCallback(
    async (file, geometry) => {
      // The API rejects mission + stream with 422. The select is disabled in
      // stream mode, so a stale selection is dropped rather than sent.
      const profile = mode === 'stream' ? '' : mission
      try {
        const { job_id: jobId } = await uploadFile(file, mode, profile, geometry)
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
        setPending(null)
      } catch (err) {
        pushToast(`Upload failed: ${err.message}`, 'error')
      }
    },
    [mission, mode, pushToast, track, upsert],
  )

  const handleFiles = useCallback(
    (files) => {
      const file = files && files[0]
      if (!file) return
      // A picture of the seabed cannot be processed until someone says what
      // the sonar was set to, so hold it and ask rather than fail downstream.
      if (needsGeometry(file.name)) {
        setPending(file)
        return
      }
      submit(file, null)
    },
    [submit],
  )

  const submitPending = useCallback(
    (e) => {
      e.preventDefault()
      if (!pending) return
      const altitude = Number(geom.altitude_m)
      const range = Number(geom.range_m)
      if (!(altitude > 0) || !(range > 0)) {
        pushToast('Altitude and range are required for an image', 'error')
        return
      }
      if (range <= altitude) {
        pushToast('Range must exceed altitude — the swath starts past nadir', 'error')
        return
      }
      const geometry = { altitude_m: altitude, range_m: range }
      if (geom.lat !== '' && geom.lon !== '') {
        geometry.lat = Number(geom.lat)
        geometry.lon = Number(geom.lon)
        if (geom.heading_deg !== '') geometry.heading_deg = Number(geom.heading_deg)
      }
      if (geom.sensor_depth_m !== '') geometry.sensor_depth_m = Number(geom.sensor_depth_m)
      submit(pending, geometry)
    },
    [geom, pending, pushToast, submit],
  )

  const setG = (k) => (e) => setGeom((g) => ({ ...g, [k]: e.target.value }))

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
        {pending && (
          <form className="geom-form" onSubmit={submitPending}>
            <h3 className="geom-title">Sonar geometry</h3>
            <p className="rail-caption">
              <span className="mono">{pending.name}</span> is an image: it records no
              navigation. State what the sonar was set to and the pipeline can correct
              slant range, measure height from shadow, and geotag every contact.
            </p>
            <div className="geom-grid">
              <label className="geom-cell">
                <span className="ctl-label">Altitude (m) *</span>
                <input className="num-input" type="number" step="0.1" min="0.1" required
                  value={geom.altitude_m} onChange={setG('altitude_m')}
                  placeholder="8" aria-label="Towfish altitude above seabed in metres" />
              </label>
              <label className="geom-cell">
                <span className="ctl-label">Range (m) *</span>
                <input className="num-input" type="number" step="1" min="1" required
                  value={geom.range_m} onChange={setG('range_m')}
                  placeholder="50" aria-label="Slant range setting in metres" />
              </label>
              <label className="geom-cell">
                <span className="ctl-label">Start lat</span>
                <input className="num-input" type="number" step="0.00001"
                  value={geom.lat} onChange={setG('lat')} placeholder="13.05"
                  aria-label="Survey line start latitude" />
              </label>
              <label className="geom-cell">
                <span className="ctl-label">Start lon</span>
                <input className="num-input" type="number" step="0.00001"
                  value={geom.lon} onChange={setG('lon')} placeholder="80.35"
                  aria-label="Survey line start longitude" />
              </label>
              <label className="geom-cell">
                <span className="ctl-label">Heading (deg)</span>
                <input className="num-input" type="number" step="1" min="0" max="360"
                  value={geom.heading_deg} onChange={setG('heading_deg')}
                  aria-label="Survey line heading in degrees" />
              </label>
              <label className="geom-cell">
                <span className="ctl-label">Tow depth (m)</span>
                <input className="num-input" type="number" step="0.1" min="0"
                  value={geom.sensor_depth_m} onChange={setG('sensor_depth_m')}
                  placeholder="22" aria-label="Towfish depth below surface in metres" />
              </label>
            </div>
            <p className="rail-caption">
              Altitude and range are required. Without a position the contacts are still
              detected, measured and ranked — they simply carry no map coordinate. The
              track is recorded as a declared straight line, never as recorded navigation.
            </p>
            <div className="geom-actions">
              <button type="submit" className="btn primary">Process image</button>
              <button type="button" className="btn" onClick={() => setPending(null)}>
                Cancel
              </button>
            </div>
          </form>
        )}

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
