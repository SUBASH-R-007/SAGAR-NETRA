import { useCallback, useEffect, useMemo, useState } from 'react'
import { fetchContacts, fetchSurveys } from './api'
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

let toastSeq = 0

export default function App() {
  const [tab, setTab] = useState('Map')
  const [surveys, setSurveys] = useState([])
  const [survey, setSurvey] = useState('')
  const [contacts, setContacts] = useState([])
  const [cls, setCls] = useState('all')
  const [minConf, setMinConf] = useState(0)
  const [toasts, setToasts] = useState([])

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
      <header className="header">
        <div className="masthead">
          <div className="mast-line">
            <h1 className="wordmark">SAGAR-NETRA</h1>
            <span className="mast-rule" aria-hidden="true" />
            <span className="mast-eyebrow">Drishti Survey Console</span>
          </div>
          <p className="mast-meta mono">
            DATUM WGS-84 · SURVEY {survey || 'NONE'} · {contacts.length} CONTACTS · MoES / NIOT
            PS 26057
          </p>
        </div>
        <div className="keyline" aria-hidden="true" />
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
        <main className="content">
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

      <Toasts toasts={toasts} />
    </div>
  )
}
