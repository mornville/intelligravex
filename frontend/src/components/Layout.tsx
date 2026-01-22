import type { ReactNode } from 'react'
import { useState } from 'react'
import { NavLink } from 'react-router-dom'

export default function Layout({ children }: { children: ReactNode }) {
  const [showCookies, setShowCookies] = useState(false)
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">Intelligravex VoiceBot Studio</div>
        <nav className="nav">
          <NavLink to="/bots" className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}>
            Bots
          </NavLink>
          <NavLink
            to="/conversations"
            className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}
          >
            Conversations
          </NavLink>
          <NavLink to="/keys" className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}>
            Keys
          </NavLink>
        </nav>
      </header>
      <main className="content">{children}</main>
      <footer className="footer">
        <div className="muted">ChatGPT can make mistakes. Check important info.</div>
        <button className="btn ghost" onClick={() => setShowCookies(true)}>
          Cookie Preferences
        </button>
      </footer>
      {showCookies ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow">
              <div className="cardTitle">Cookie Preferences</div>
              <button className="btn" onClick={() => setShowCookies(false)}>
                Close
              </button>
            </div>
            <div className="muted" style={{ marginBottom: 8 }}>
              Manage your cookie preferences for this app.
            </div>
            <div className="row gap" style={{ alignItems: 'center' }}>
              <label className="check">
                <input type="checkbox" checked readOnly /> Essential
              </label>
              <label className="check">
                <input type="checkbox" /> Analytics
              </label>
              <label className="check">
                <input type="checkbox" /> Marketing
              </label>
              <div className="spacer" />
              <button className="btn" onClick={() => setShowCookies(false)}>
                Save
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
