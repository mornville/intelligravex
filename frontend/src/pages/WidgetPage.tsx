import { useEffect, useMemo, useState } from 'react'
import { apiGet, getBackendUrl, setBackendUrl } from '../api/client'
import type { Bot } from '../types'
import MicTest from '../components/MicTest'
import LoadingSpinner from '../components/LoadingSpinner'
import '../widget.css'

const BOT_STORAGE_KEY = 'igx_widget_bot_id'
const BODY_CLASS = 'widget-body'

export default function WidgetPage() {
  const [bots, setBots] = useState<Bot[]>([])
  const [botId, setBotId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [backendReady, setBackendReady] = useState(false)

  useEffect(() => {
    document.body.classList.add(BODY_CLASS)
    return () => {
      document.body.classList.remove(BODY_CLASS)
    }
  }, [])

  useEffect(() => {
    let canceled = false
    async function loadBots() {
      if (!backendReady) return
      setLoading(true)
      setErr(null)
      try {
        const res = await apiGet<{ items: Bot[] }>('/api/bots')
        if (canceled) return
        const list = res.items || []
        setBots(list)
        if (!list.length) {
          setBotId(null)
          return
        }
        const saved = window.localStorage.getItem(BOT_STORAGE_KEY) || ''
        const next = list.find((b) => b.id === saved)?.id || list[0].id
        setBotId(next)
      } catch (e: any) {
        if (!canceled) setErr(String(e?.message || e))
      } finally {
        if (!canceled) setLoading(false)
      }
    }
    void loadBots()
    return () => {
      canceled = true
    }
  }, [backendReady])

  useEffect(() => {
    let active = true
    async function resolveBackend() {
      const tauri = (window as any).__TAURI__
      if (tauri?.invoke) {
        try {
          const url = await tauri.invoke('get_backend_url')
          if (typeof url === 'string' && url.trim()) {
            setBackendUrl(url)
          }
        } catch (e: any) {
          if (active) setErr(String(e?.message || e))
        }
      }
      if (active) setBackendReady(true)
    }
    void resolveBackend()
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!botId) return
    window.localStorage.setItem(BOT_STORAGE_KEY, botId)
  }, [botId])

  const dashboardUrl = useMemo(() => `${getBackendUrl()}/dashboard`, [backendReady])

  async function openDashboard() {
    const tauri = (window as any).__TAURI__
    if (tauri?.invoke) {
      try {
        await tauri.invoke('open_dashboard', { url: dashboardUrl })
        return
      } catch {
        // fall back to browser
      }
    }
    window.open(dashboardUrl, '_blank', 'noopener,noreferrer')
  }

  return (
    <div className="widgetRoot">
      <div className="widgetHeader" data-tauri-drag-region>
        <div className="widgetTitle">
          <span className="widgetTitleText">Gravex</span>
          <span className="widgetTitleTag">Widget</span>
        </div>
        <div className="widgetActions">
          {bots.length ? (
            <select
              className="widgetSelect"
              value={botId || ''}
              onChange={(e) => setBotId(e.target.value || null)}
            >
              {bots.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.name}
                </option>
              ))}
            </select>
          ) : null}
          <button className="btn ghost" onClick={() => void openDashboard()}>
            Open dashboard
          </button>
        </div>
      </div>
      {err ? <div className="alert">{err}</div> : null}
      {!backendReady ? (
        <div className="widgetLoading">
          <LoadingSpinner label="Starting backend" />
          <span className="muted">Starting backend…</span>
        </div>
      ) : loading ? (
        <div className="widgetLoading">
          <LoadingSpinner label="Loading assistants" />
          <span className="muted">Loading assistants…</span>
        </div>
      ) : botId ? (
        <MicTest
          key={botId}
          botId={botId}
          layout="whatsapp"
          hideWorkspace
          allowUploads={false}
          uploadDisabledReason="Uploads are disabled in the widget."
        />
      ) : (
        <div className="widgetEmpty">
          <div className="muted">No assistants found. Create one in the dashboard.</div>
          <button className="btn" onClick={() => void openDashboard()}>
            Open dashboard
          </button>
        </div>
      )}
    </div>
  )
}
