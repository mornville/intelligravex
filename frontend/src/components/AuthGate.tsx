import type { FormEvent, ReactNode } from 'react'
import { useState } from 'react'
import { BACKEND_URL } from '../api/client'
import { getBasicAuthToken, setBasicAuthToken } from '../auth'

const DEFAULT_USER = 'admin'
const DEFAULT_PASS = ''

export default function AuthGate({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => getBasicAuthToken())
  const [username, setUsername] = useState(DEFAULT_USER)
  const [password, setPassword] = useState(DEFAULT_PASS)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setErr(null)
    setBusy(true)
    try {
      const candidate = btoa(`${username}:${password}`)
      const res = await fetch(`${BACKEND_URL}/api/options`, {
        method: 'GET',
        headers: { Authorization: `Basic ${candidate}` },
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setBasicAuthToken(username, password)
      setToken(candidate)
    } catch (e: any) {
      setErr('Invalid credentials or backend unreachable.')
    } finally {
      setBusy(false)
    }
  }

  if (!token) {
    return (
      <div className="authWrap">
        <div className="authCard">
          <div className="authTitle">Intelligravex Studio</div>
          <div className="authSubtitle">Sign in to continue</div>
          {err ? <div className="alert">{err}</div> : null}
          <form onSubmit={handleSubmit}>
            <div className="formRow">
              <label>Username</label>
              <input value={username} onChange={(e) => setUsername(e.target.value)} />
            </div>
            <div className="formRow">
              <label>Password</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
            </div>
            <button className="btn primary" type="submit" disabled={busy}>
              {busy ? 'Signing in…' : 'Sign in'}
            </button>
          </form>
          <div className="authNote">Credentials are stored locally on this device.</div>
        </div>
      </div>
    )
  }

  return <>{children}</>
}
