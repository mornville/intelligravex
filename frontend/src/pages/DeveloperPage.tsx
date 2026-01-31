import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiGet, apiPost } from '../api/client'

type DataAgentContainer = {
  id: string
  name: string
  image: string
  status: string
  created_at: string
  running_for: string
  conversation_id: string
  cpu?: string
  mem?: string
  mem_perc?: string
}

export default function DeveloperPage() {
  const [items, setItems] = useState<DataAgentContainer[]>([])
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)
  const [dockerAvailable, setDockerAvailable] = useState(true)
  const [stopping, setStopping] = useState<Record<string, boolean>>({})

  async function reload() {
    setLoading(true)
    setErr(null)
    try {
      const res = await apiGet<{ docker_available: boolean; items: DataAgentContainer[]; error?: string }>(
        '/api/data-agent/containers',
      )
      setDockerAvailable(Boolean(res.docker_available))
      setItems(res.items || [])
      if (res.error) setErr(res.error)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void reload()
    const id = window.setInterval(() => {
      void reload()
    }, 15000)
    return () => window.clearInterval(id)
  }, [])

  async function stopContainer(target: string) {
    if (!target) return
    const ok = window.confirm('Stop this container? The conversation will respawn a new one when needed.')
    if (!ok) return
    setStopping((p) => ({ ...p, [target]: true }))
    setErr(null)
    try {
      await apiPost(`/api/data-agent/containers/${encodeURIComponent(target)}/stop`, {})
      await reload()
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setStopping((p) => {
        const next = { ...p }
        delete next[target]
        return next
      })
    }
  }

  async function stopAll() {
    if (items.length === 0) return
    const ok = window.confirm(`Stop ${items.length} running containers?`)
    if (!ok) return
    setErr(null)
    for (const item of items) {
      const target = item.id || item.name
      if (!target) continue
      setStopping((p) => ({ ...p, [target]: true }))
      try {
        await apiPost(`/api/data-agent/containers/${encodeURIComponent(target)}/stop`, {})
      } catch (e: any) {
        setErr(String(e?.message || e))
      } finally {
        setStopping((p) => {
          const next = { ...p }
          delete next[target]
          return next
        })
      }
    }
    await reload()
  }

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h1>Developer settings</h1>
          <div className="muted">Operational controls for Data Agent containers.</div>
        </div>
        <div className="row gap">
          <button className="btn" onClick={() => void reload()} disabled={loading}>
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
          <button className="btn danger ghost" onClick={() => void stopAll()} disabled={loading || items.length === 0}>
            Stop all
          </button>
        </div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <section className="card">
        <div className="cardTitleRow">
          <div>
            <div className="cardTitle">Data Agent containers</div>
            <div className="muted">Running Docker containers. Updated every 15 seconds.</div>
          </div>
          {!dockerAvailable ? <span className="pill accent">Docker unavailable</span> : null}
        </div>

        {!dockerAvailable ? (
          <div className="muted" style={{ marginTop: 8 }}>
            Docker is not available. Install Docker Desktop or enable the Docker daemon.
          </div>
        ) : loading ? (
          <div className="muted">Loading…</div>
        ) : items.length === 0 ? (
          <div className="muted">No running containers.</div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Status</th>
                <th>Running</th>
                <th>CPU</th>
                <th>Memory</th>
                <th>Conversation</th>
                <th style={{ textAlign: 'right' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => {
                const target = item.id || item.name
                return (
                  <tr key={target}>
                    <td className="mono">{item.name || item.id}</td>
                    <td className="mono">{item.status || '—'}</td>
                    <td className="mono">{item.running_for || item.created_at || '—'}</td>
                    <td className="mono">{item.cpu || '—'}</td>
                    <td className="mono">{item.mem ? `${item.mem}${item.mem_perc ? ` (${item.mem_perc})` : ''}` : '—'}</td>
                    <td>
                      {item.conversation_id ? (
                        <Link className="link" to={`/conversations/${item.conversation_id}`}>
                          {item.conversation_id}
                        </Link>
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button className="btn danger ghost" onClick={() => void stopContainer(target)} disabled={!!stopping[target]}>
                        {stopping[target] ? 'Stopping…' : 'Stop'}
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
