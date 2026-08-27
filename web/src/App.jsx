import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchContacts, fetchSurveys } from './api'
import Chakra from './components/Chakra'
import Copilot from './components/Copilot'
import ContactsTable from './components/ContactsTable'
import DiffView from './components/DiffView'
import FilterBar from './components/FilterBar'
import MapView from './components/MapView'
import Toasts from './components/Toasts'
import UploadRail from './components/UploadRail'
import Waterfall from './components/Waterfall'

const TABS = ['Map', 'Waterfall', 'Contacts', 'Diff', 'Copilot']
const FILTERED_TABS = new Set(['Map', 'Waterfall', 'Contacts'])

// A- / A / A+ — the standard GoI portal accessibility control. Applied to the
// root element font-size; the whole type scale is rem-based so it follows.
const FONT_STEPS = [
  { key: 'small', label: 'A-', pct: '87.5%', title: 'Decrease font size' },
  { key: 'normal', label: 'A', pct: '100%', title: 'Normal font size' },
  { key: 'large', label: 'A+', pct: '112.5%', title: 'Increase font size' },
]
const FONT_STORE_KEY = 'drishti.fontSize'

function readStoredFontStep() {
  try {
    const v = window.localStorage.getItem(FONT_STORE_KEY)
    return FONT_STEPS.some((s) => s.key === v) ? v : 'normal'
  } catch {
    return 'normal' // storage unavailable (private mode etc.) — default silently
  }
}

let toastSeq = 0

export default function App() {
  const [tab, setTab] = useState('Map')
  const [surveys, setSurveys] = useState([])
  const [survey, setSurvey] = useState('')
  const [contacts, setContacts] = useState([])
  const [cls, setCls] = useState('all')
  const [minConf, setMinConf] = useState(0)
  const [toasts, setToasts] = useState([])
  const [fontStep, setFontStep] = useState(readStoredFontStep)

  useEffect(() => {
    const step = FONT_STEPS.find((s) => s.key === fontStep) || FONT_STEPS[1]
    document.documentElement.style.fontSize = step.pct
    try {
      window.localStorage.setItem(FONT_STORE_KEY, step.key)
    } catch {
      // storage unavailable — the choice still applies for this session
    }
  }, [fontStep])

  const pushToast = useCallback((text, kind = 'info') => {
    const id = ++toastSeq
    setToasts((ts) => [...ts, { id, text, kind }])
    setTimeout(() => setToasts((ts) => ts.filter((t) => t.id !== id)), 6000)
  }, [])

  const refreshSurveys = useCallback(
    async (selectName) => {
      try {
        const rows = await fetchSurveys()
        setSurveys(rows)
        setSurvey((cur) => selectName || cur || (rows[0] ? rows[0].name : ''))
      } catch (err) {
        pushToast(`Could not reach the API — is the backend running on :8000? (${err.message})`, 'error')
      }
    },
    [pushToast],
  )

  useEffect(() => {
    refreshSurveys()
  }, [refreshSurveys])

  // Load the full contact set for the selected survey; class / confidence
  // filters are applied client-side so they are instant and shared by every tab.
  useEffect(() => {
    setCls('all')
    if (!survey) {
      setContacts([])
      return undefined
    }
    let alive = true
    fetchContacts(survey)
      .then((res) => {
        if (alive) setContacts(res.contacts || [])
      })
      .catch((err) => {
        if (alive) {
          setContacts([])
          pushToast(`Failed to load contacts: ${err.message}`, 'error')
        }
      })
    return () => {
      alive = false
    }
  }, [survey, pushToast])

  // A review action returns the updated contact — swap it in place.
  const onReview = useCallback((updated) => {
    setContacts((cs) => cs.map((c) => (c.id === updated.id ? updated : c)))
  }, [])

  const classes = useMemo(() => [...new Set(contacts.map((c) => c.cls))].sort(), [contacts])

  const filtered = useMemo(
    () => contacts.filter((c) => (cls === 'all' || c.cls === cls) && c.confidence >= minConf),
    [contacts, cls, minConf],
  )

  return (
    <div className="app">
      {/* 1 · government strip — 30px, navy-deep */}
      <div className="govt-strip">
        <span className="govt-name">
          <span lang="hi">भारत सरकार</span> | GOVERNMENT OF INDIA
        </span>
        <div className="govt-tools">
          <a className="skip-link" href="#main-content">
            Skip to main content
          </a>
          <div className="fs-group" role="group" aria-label="Font size">
            {FONT_STEPS.map((s) => (
              <button
                key={s.key}
                type="button"
                className={s.key === fontStep ? 'fs-btn active' : 'fs-btn'}
                aria-pressed={s.key === fontStep}
                aria-label={s.title}
                title={s.title}
                onClick={() => setFontStep(s.key)}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <header className="header">
        {/* 2 · header band — emblem slot, ministry designation, portal identity */}
        <div className="header-band">
          <div className="brand">
            <Chakra size={44} />
            <div className="ministry">
              <span className="ministry-hi" lang="hi">
                पृथ्वी विज्ञान मंत्रालय
              </span>
              <span className="ministry-en">Ministry of Earth Sciences</span>
            </div>
            <span className="band-rule" aria-hidden="true" />
            <div className="identity">
              <h1 className="wordmark">SAGAR-NETRA</h1>
              <span className="ident-sub">
                DRISHTI Survey Console — AI Marine Debris Detection
              </span>
            </div>
          </div>
          <span className="sih-tag mono">SMART INDIA HACKATHON 2026 · PS 26057</span>
        </div>

        {/* 3 · tricolor ribbon — 3px, three equal bands, appears exactly once */}
        <div className="tricolor" aria-hidden="true">
          <span className="t-saffron" />
          <span className="t-white" />
          <span className="t-green" />
        </div>

        {/* 4 · nav bar — solid navy, saffron active underline */}
        <nav className="tabs" aria-label="views">
          {TABS.map((t) => (
            <button
              key={t}
              type="button"
              className={t === tab ? 'tab active' : 'tab'}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          ))}
        </nav>

        {/* survey metadata line — kept visible on every tab, below the nav */}
        <p className="meta-strip mono">
          DATUM WGS-84 · SURVEY {survey || 'NONE'} · {contacts.length} CONTACTS · MoES / NIOT
          PS 26057
        </p>
      </header>

      {FILTERED_TABS.has(tab) && (
        <FilterBar
          surveys={surveys}
          survey={survey}
          onSurvey={setSurvey}
          classes={classes}
          cls={cls}
          onCls={setCls}
          minConf={minConf}
          onMinConf={setMinConf}
          shown={filtered.length}
          total={contacts.length}
        />
      )}

      <div className="body">
        <main className="content" id="main-content" tabIndex={-1}>
          {tab === 'Map' && (
            <MapView
              contacts={filtered}
              onReview={onReview}
              pushToast={pushToast}
              hasSurvey={Boolean(survey)}
            />
          )}
          {tab === 'Waterfall' && (
            <Waterfall
              survey={survey}
              contacts={filtered}
              onReview={onReview}
              pushToast={pushToast}
            />
          )}
          {tab === 'Contacts' && (
            <ContactsTable
              contacts={filtered}
              survey={survey}
              onReview={onReview}
              pushToast={pushToast}
            />
          )}
          {tab === 'Diff' && <DiffView surveys={surveys} pushToast={pushToast} />}
          {tab === 'Copilot' && <Copilot pushToast={pushToast} />}
        </main>
        <UploadRail pushToast={pushToast} onJobDone={(name) => refreshSurveys(name)} />
      </div>

      {/* 6 · footer — navy band, honest attribution */}
      <footer className="footer">
        <div className="footer-lines">
          <p>
            Prototype developed for Smart India Hackathon 2026 — Problem Statement 26057
            (Ministry of Earth Sciences / NIOT)
          </p>
          <p className="footer-note">
            This is a hackathon prototype, not an official Government of India website.
          </p>
        </div>
        <span className="footer-mode mono">Offline-first · Zero cloud dependency</span>
      </footer>

      <Toasts toasts={toasts} />
    </div>
  )
}
