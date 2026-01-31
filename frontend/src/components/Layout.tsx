import type { ReactNode } from 'react'
import { NavLink } from 'react-router-dom'

export default function Layout({ children }: { children: ReactNode }) {
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
          <NavLink to="/developer" className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}>
            Developer
          </NavLink>
        </nav>
      </header>
      <main className="content">{children}</main>
    </div>
  )
}
