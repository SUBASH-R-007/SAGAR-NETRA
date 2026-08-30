// Thin fetch helpers around the DRISHTI Console API.
// Every helper throws an Error with a readable message on non-2xx responses;
// callers wrap in try/catch and surface a toast.

// Sessions last 12 hours. Before this existed, expiry mid-shift showed up as a
// six-second toast reading "401: Not authenticated" and the action the operator
// had just taken simply had not happened - no re-login, no indication that the
// console was now read-only in practice. App.jsx registers a handler here so
// one expiry is handled once, centrally, instead of at twenty call sites.
let onUnauthorized = null
export function setUnauthorizedHandler(fn) {
  onUnauthorized = fn
}

// What we say to a person, per status. The raw code and the server's detail
// are still attached to the error for the console and for callers that want
// them; they are not what a user should have to read.
const HUMAN = {
  401: 'Your session has expired. Please sign in again.',
  403: 'Your role does not have permission to do that.',
  404: 'That item no longer exists - it may have been deleted.',
  409: 'Something else changed this first. Reload and try again.',
  413: 'That file is too large for this deployment to accept.',
  422: 'The server could not read that request.',
  500: 'The server hit an internal error. Check the backend log.',
  502: 'The backend is unreachable.',
  503: 'The backend is starting up or overloaded. Try again shortly.',
}

async function handle(res, opts = {}) {
  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* body was not JSON */
    }
    const err = new Error(HUMAN[res.status] || detail)
    err.status = res.status
    err.detail = detail
    if (res.status === 401 && !opts.skipAuthHandler && onUnauthorized) {
      onUnauthorized()
    }
    throw err
  }
  return res.json()
}

// The session cookie is HttpOnly — JavaScript cannot read it, which is the
// point — so every request must opt in to sending it. 'same-origin' rather
// than 'include': the console is served by the same FastAPI app it calls.
const CREDS = { credentials: 'same-origin' }

export const getJSON = (url) => fetch(url, CREDS).then(handle)

export const postJSON = (url, body) =>
  fetch(url, {
    ...CREDS,
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(handle)

// ------------------------------------------------------------- endpoints --

export const fetchSurveys = () => getJSON('/api/surveys')

export const fetchContacts = (survey, limit = 500) =>
  getJSON(`/api/contacts?survey=${encodeURIComponent(survey)}&limit=${limit}`)

export const fetchLayers = () => getJSON('/api/layers')

export const deleteSurvey = (name) =>
  fetch(`/api/surveys/${encodeURIComponent(name)}`, { ...CREDS, method: 'DELETE' }).then(handle)

export const fetchJobs = () => getJSON('/api/jobs')

export const fetchJob = (jobId) => getJSON(`/api/jobs/${encodeURIComponent(jobId)}`)

export const fetchWaterfallMeta = (survey) =>
  getJSON(`/api/waterfall/${encodeURIComponent(survey)}/meta`)

export const fetchDiff = (surveyA, surveyB, radiusM) =>
  getJSON(
    `/api/diff?survey_a=${encodeURIComponent(surveyA)}` +
      `&survey_b=${encodeURIComponent(surveyB)}&radius_m=${radiusM}`,
  )

export const postReview = (contactId, status, notes = null) =>
  postJSON(`/api/contacts/${encodeURIComponent(contactId)}/review`, { status, notes })

export const postRecovery = (contactId, status) =>
  postJSON(`/api/contacts/${encodeURIComponent(contactId)}/recovery`, { status })

export const askCopilot = (question) => postJSON('/api/copilot', { question })

// mode: 'batch' (default one-shot processing) or 'stream' (towed real-time
// replay — the job snapshot then carries a recent_events detections feed).
// mission: a profile name from GET /api/missions (configs/missions/*.yaml) that
// re-weights the severity index. Batch only — the API answers 422 for
// mission + stream — and omitted entirely when blank, so a standard survey
// sends exactly the form it always did.
export async function uploadFile(file, mode = 'batch', mission = '', geometry = null) {
  const form = new FormData()
  form.append('file', file)
  form.append('mode', mode)
  if (mission) form.append('mission', mission)
  // Declared sonar geometry — only nav-less formats (images) need it, and the
  // API ignores it for survey logs that carry their own navigation.
  if (geometry) {
    for (const [k, v] of Object.entries(geometry)) {
      if (v !== null && v !== undefined && v !== '') form.append(k, String(v))
    }
  }
  return fetch('/api/upload', { ...CREDS, method: 'POST', body: form }).then(handle)
}

// ------------------------------------------------------------------ urls --

export const jobSocketUrl = (jobId) => {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}/api/jobs/${encodeURIComponent(jobId)}/progress`
}

export const thumbUrl = (contactId) => `/api/contacts/${encodeURIComponent(contactId)}/thumb`

export const evidenceUrl = (contactId) =>
  `/api/contacts/${encodeURIComponent(contactId)}/evidence`

export const reportUrl = (fmt, survey) =>
  `/api/report/${fmt}?survey=${encodeURIComponent(survey)}`

export const waterfallUrl = (survey, raw = false) =>
  `/api/waterfall/${encodeURIComponent(survey)}${raw ? '?raw=1' : ''}`

// --------------------------------------------- overview / ops / telemetry --

export const fetchSummary = (survey) =>
  getJSON(`/api/summary?survey=${encodeURIComponent(survey)}`)

export const fetchHealth = () => getJSON('/api/health')

export const fetchMissions = () => getJSON('/api/missions')

export const fetchReviewLog = () => getJSON('/api/reviews/export')

export const fetchRecoveryLog = () => getJSON('/api/recovery/log')

export const fetchCrossview = (surveyA, surveyB, radiusM = 15) =>
  getJSON(
    `/api/crossview?survey_a=${encodeURIComponent(surveyA)}` +
      `&survey_b=${encodeURIComponent(surveyB)}&radius_m=${radiusM}`,
  )

// review defaults to 'confirmed' server-side; pass '' to plan over every
// contact. cluster_eps_m groups contacts into recovery zones first.
export function fetchRoute({ survey, review = 'confirmed', clusterEpsM, startLat, startLon }) {
  const q = new URLSearchParams()
  if (survey) q.set('survey', survey)
  if (review !== undefined && review !== null) q.set('review', review)
  if (clusterEpsM) q.set('cluster_eps_m', String(clusterEpsM))
  if (startLat !== undefined && startLat !== '') q.set('start_lat', String(startLat))
  if (startLon !== undefined && startLon !== '') q.set('start_lon', String(startLon))
  return getJSON(`/api/route?${q.toString()}`)
}

// ---- physics lab ----
// These call the deployed physics (sonar_core.geometry, physicheck.shadow),
// not a browser re-derivation, so the lab and the pipeline cannot disagree.

export const fetchGeometry = ({ altitude, range, beam, pulseUs }) => {
  const q = new URLSearchParams({ altitude_m: altitude, range_m: range })
  if (beam != null) q.set('beam_deg', beam)
  if (pulseUs != null) q.set('pulse_us', pulseUs)
  return getJSON(`/api/physics/geometry?${q.toString()}`)
}

export const fetchShadow = (altitude, height, groundRange) =>
  postJSON('/api/physics/shadow', {
    altitude_m: altitude,
    height_m: height,
    ground_range_m: groundRange,
  })

export const fetchSimClasses = () => getJSON('/api/physics/classes')

export const simulateScene = (body) => postJSON('/api/physics/simulate', body)

// ---- auth ----
// Enforcement lives on the API; these helpers only tell the console what to
// render. A user who bypasses the UI still hits the same guards.

// Both of these are allowed to 401 as part of normal operation - a mistyped
// password, and the bootstrap probe on a console with no session yet - so they
// opt out of the global expiry handler. Login.jsx and App.jsx handle their own.
export const login = (username, password) =>
  fetch('/api/auth/login', {
    ...CREDS,
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  }).then((res) => handle(res, { skipAuthHandler: true }))

export const logout = () => postJSON('/api/auth/logout', {})

export const fetchMe = () =>
  fetch('/api/auth/me', CREDS).then((res) => handle(res, { skipAuthHandler: true }))
