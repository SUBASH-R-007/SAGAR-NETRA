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

export const askCopilot = (question) => postJSON('/api/copilot', { question })

export async function uploadFile(file) {
  const form = new FormData()
  form.append('file', file)
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
