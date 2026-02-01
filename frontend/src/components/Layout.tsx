import type { ReactNode } from 'react'
import { NavLink, useLocation, useNavigate } from 'react-router-dom'

export default function Layout({ children }: { children: ReactNode }) {
  const location = useLocation()
  const isLanding = location.pathname === '/'
  const nav = useNavigate()
  const go = (to: string) => {
    const target = to.startsWith('/') ? to : `/${to}`
    nav(target)
    window.setTimeout(() => {
      if (window.location.pathname !== target) {
        window.location.assign(target)
      }
    }, 0)
  }
  const scrollToSection = (id: string) => {
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }
  return (
    <div className="app">
      <header className={`topbar ${isLanding ? 'landingTopbar' : ''}`}>
        <div className="brand">GravexStudio</div>
        <nav className="nav">
          {isLanding ? (
            <>
              <button className="navLink" onClick={() => scrollToSection('capabilities')}>
                Capabilities
              </button>
              <button className="navLink" onClick={() => scrollToSection('workflow')}>
                Workflow
              </button>
              <NavLink to="/dashboard" className="navPill" onClick={(e) => { e.preventDefault(); go('/dashboard') }}>
                Dashboard
              </NavLink>
            </>
          ) : (
            <>
              <NavLink
                to="/bots"
                className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}
                onClick={(e) => { e.preventDefault(); go('/bots') }}
              >
                Assistants
              </NavLink>
              <NavLink
                to="/conversations"
                className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}
                onClick={(e) => { e.preventDefault(); go('/conversations') }}
              >
                Chats
              </NavLink>
              <NavLink
                to="/keys"
                className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}
                onClick={(e) => { e.preventDefault(); go('/keys') }}
              >
                Keys
              </NavLink>
              <NavLink
                to="/developer"
                className={({ isActive }) => (isActive ? 'navLink active' : 'navLink')}
                onClick={(e) => { e.preventDefault(); go('/developer') }}
              >
                Developer
              </NavLink>
            </>
          )}
        </nav>
      </header>
      <main className="content">
        {children}
      </main>
    </div>
  )
}
