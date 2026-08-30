import { useState } from 'react'
import { login } from '../api'
import Chakra from './Chakra'

// Sign-in screen. Shown whenever /api/auth/me answers 401, which is the one
// unambiguous signal that the console has no usable session — cheaper and
// more honest than the frontend trying to infer auth state for itself.
//
// The error message is whatever the API returned. That endpoint deliberately
// gives the same text for an unknown user and a wrong password, so this
// screen must not try to be more helpful than the backend was willing to be.
export default function Login({ onSignedIn }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e) => {
    e.preventDefault()
    if (!username || !password) return
    setBusy(true)
    setError('')
    try {
      onSignedIn(await login(username, password))
    } catch (err) {
      // Strip the leading "401: " the fetch helper prefixes.
      setError(String(err.message).replace(/^\d+:\s*/, ''))
      setPassword('')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login-page">
      <div className="login-strip">
        <span lang="hi">भारत सरकार</span> | GOVERNMENT OF INDIA
      </div>

      <main className="login-card">
        <div className="login-brand">
          <Chakra size={52} />
          <div>
            <h1 className="login-wordmark">SAGAR-NETRA</h1>
            <p className="login-sub">DRISHTI Survey Console</p>
          </div>
        </div>

        <div className="login-tricolor" aria-hidden="true">
          <span className="t-saffron" />
          <span className="t-white" />
          <span className="t-green" />
        </div>

        <form className="login-form" onSubmit={submit}>
          <label className="login-field">
            <span className="ctl-label">Username</span>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              autoFocus
              required
            />
          </label>

          <label className="login-field">
            <span className="ctl-label">Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </label>

          {error && (
            <p className="login-error" role="alert">
              {error}
            </p>
          )}

          <button type="submit" className="btn primary login-submit" disabled={busy}>
            {busy ? 'Signing in…' : 'Sign in'}
          </button>
        </form>

        <p className="login-note">
          Accounts are created by an administrator
          (<span className="mono">scripts/seed_users.py</span>). Sessions expire after
          12 hours.
        </p>
      </main>

      <footer className="login-footer">
        Smart India Hackathon 2026 · PS 26057 · Ministry of Earth Sciences / NIOT
        <br />
        <span className="login-disclaimer">
          Hackathon prototype — not an official Government of India website.
        </span>
      </footer>
    </div>
  )
}
