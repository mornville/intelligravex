import type { ReactNode } from 'react'
import { NavLink, useLocation } from 'react-router-dom'

export default function Layout({ children }: { children: ReactNode }) {
  const location = useLocation()
  const isLanding = location.pathname === '/'
  return (
    <div className="app">
      <header className={`topbar ${isLanding ? 'landingTopbar' : ''}`}>
        <div className="brand">GravexStudio</div>
        <nav className="nav">
          {isLanding ? (
            <>
              <a className="navLink" href="#capabilities">
                Capabilities
              </a>
              <a className="navLink" href="#workflow">
                Workflow
              </a>
              <NavLink to="/dashboard" className="navPill">
                Dashboard
              </NavLink>
            </>
          ) : (
            <>
              <NavLink to="/bots" className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}>
                Assistants
              </NavLink>
              <NavLink
                to="/conversations"
                className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}
              >
                Chats
              </NavLink>
              <NavLink to="/keys" className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}>
                Keys
              </NavLink>
              <NavLink to="/developer" className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}>
                Developer
              </NavLink>
            </>
          )}
        </nav>
      </header>
      <main className="content" key={location.pathname}>
        {children}
      </main>
    </div>
  )
}
