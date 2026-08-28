// Thin fetch helpers around the DRISHTI Console API.
// Every helper throws an Error with a readable message on non-2xx responses;
// callers wrap in try/catch and surface a toast.

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* body was not JSON */
    }
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}

export const getJSON = (url) => fetch(url).then(handle)

export const postJSON = (url, body) =>
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(handle)

// ------------------------------------------------------------- endpoints --

export const fetchSurveys = () => getJSON('/api/surveys')

export const fetchContacts = (survey, limit = 500) =>
  getJSON(`/api/contacts?survey=${encodeURIComponent(survey)}&limit=${limit}`)

export const fetchLayers = () => getJSON('/api/layers')

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
  return fetch('/api/upload', { method: 'POST', body: form }).then(handle)
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
