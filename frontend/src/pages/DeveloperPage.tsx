import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiGet, apiPost } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import { StopCircleIcon } from '@heroicons/react/24/solid'

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

type LocalRuntimeStatus = {
  state?: string
  message?: string
  model_id?: string
  bytes_total?: number
  bytes_downloaded?: number
  percent?: number
  error?: string
  server_port?: number
  server_pid?: number | null
  last_update_ts?: number
}

function fmtBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = n
  let i = 0
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024
    i += 1
  }
  return `${value.toFixed(i === 0 ? 0 : 2)} ${units[i]}`
}

function fmtLocalTime(ts?: number): string {
  if (!ts || !Number.isFinite(ts)) return '-'
  try {
    return new Date(ts * 1000).toLocaleString()
  } catch {
    return '-'
  }
}

export default function DeveloperPage() {
  const [items, setItems] = useState<DataAgentContainer[]>([])
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)
  const [dockerAvailable, setDockerAvailable] = useState(true)
  const [stopping, setStopping] = useState<Record<string, boolean>>({})
  const [localStatus, setLocalStatus] = useState<LocalRuntimeStatus | null>(null)
  const [localErr, setLocalErr] = useState<string | null>(null)

  async function reload() {
    setLoading(true)
    setErr(null)
    setLocalErr(null)
    try {
      const [containersRes, localRes] = await Promise.allSettled([
        apiGet<{ docker_available: boolean; items: DataAgentContainer[]; error?: string }>('/api/data-agent/containers'),
        apiGet<LocalRuntimeStatus>('/api/local/status'),
      ])
      if (containersRes.status === 'fulfilled') {
        const res = containersRes.value
        setDockerAvailable(Boolean(res.docker_available))
        setItems(res.items || [])
        if (res.error) setErr(res.error)
      } else {
        setErr(String(containersRes.reason?.message || containersRes.reason))
      }
      if (localRes.status === 'fulfilled') {
        setLocalStatus(localRes.value || null)
      } else {
        setLocalErr(String(localRes.reason?.message || localRes.reason))
      }
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

  const progress =
    typeof localStatus?.percent === 'number' ? `${localStatus.percent}%` : ''
  const progressBytes =
    localStatus?.bytes_total && localStatus.bytes_total > 0
      ? `${fmtBytes(localStatus.bytes_downloaded || 0)} / ${fmtBytes(localStatus.bytes_total)}`
      : localStatus?.bytes_downloaded
        ? fmtBytes(localStatus.bytes_downloaded)
        : '-'
  const localStage = localStatus?.state || '-'

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h1>Developer settings</h1>
          <div className="muted">Operational controls for Isolated Workspace containers.</div>
        </div>
        <div className="row gap">
          <button className="btn" onClick={() => void reload()} disabled={loading} aria-label="Refresh containers">
            {loading ? <LoadingSpinner label="Refreshing" /> : 'Refresh'}
          </button>
          <button
            className="btn iconBtn danger"
            onClick={() => void stopAll()}
            disabled={loading || items.length === 0}
            aria-label="Stop all containers"
            title="Stop all containers"
          >
            <StopCircleIcon aria-hidden="true" />
          </button>
        </div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <section className="card">
        <div className="cardTitleRow">
          <div>
            <div className="cardTitle">Local runtime</div>
            <div className="muted">Local LLM setup status and runtime health.</div>
          </div>
          <div className="row gap">
            <span className={`pill ${localStage === 'error' ? 'accent' : ''}`}>status: {localStage}</span>
            {localStatus?.model_id ? <span className="pill">model: {localStatus.model_id}</span> : null}
          </div>
        </div>

        {localErr ? <div className="alert">{localErr}</div> : null}

        {localStatus ? (
          <div className="grid2">
            <div>
              <div className="muted">Message</div>
              <div>{localStatus.message || '-'}</div>
            </div>
            <div>
              <div className="muted">Progress</div>
              <div className="mono">{progress ? `${progress} • ${progressBytes}` : progressBytes}</div>
            </div>
            <div>
              <div className="muted">Server</div>
              <div className="mono">
                port {localStatus.server_port ?? '-'} • pid {localStatus.server_pid ?? '-'}
              </div>
            </div>
            <div>
              <div className="muted">Last update</div>
              <div className="mono">{fmtLocalTime(localStatus.last_update_ts)}</div>
            </div>
            <div>
              <div className="muted">Error</div>
              <div className="mono">{localStatus.error || '-'}</div>
            </div>
          </div>
        ) : (
          <div className="muted">{loading ? <LoadingSpinner /> : 'No local status available yet.'}</div>
        )}
      </section>

      <section className="card">
        <div className="cardTitleRow">
          <div>
            <div className="cardTitle">Isolated Workspace containers</div>
            <div className="muted">Running Docker containers. Updated every 15 seconds.</div>
          </div>
          {!dockerAvailable ? <span className="pill accent">Docker unavailable</span> : null}
        </div>

        {!dockerAvailable ? (
          <div className="muted" style={{ marginTop: 8 }}>
            Docker is not available. Install Docker Desktop or enable the Docker daemon.
          </div>
        ) : items.length === 0 ? (
          loading ? (
            <div className="muted">
              <LoadingSpinner />
            </div>
          ) : (
            <div className="muted">No running containers.</div>
          )
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
                    <td className="mono">{item.status || '-'}</td>
                    <td className="mono">{item.running_for || item.created_at || '-'}</td>
                    <td className="mono">{item.cpu || '-'}</td>
                    <td className="mono">{item.mem ? `${item.mem}${item.mem_perc ? ` (${item.mem_perc})` : ''}` : '-'}</td>
                    <td>
                      {item.conversation_id ? (
                        <Link className="link" to={`/conversations/${item.conversation_id}`}>
                          {item.conversation_id}
                        </Link>
                      ) : (
                        <span className="muted">-</span>
                      )}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn iconBtn danger"
                        onClick={() => void stopContainer(target)}
                        disabled={!!stopping[target]}
                        aria-label="Stop container"
                        title="Stop container"
                      >
                        {stopping[target] ? <LoadingSpinner label="Stopping" /> : <StopCircleIcon aria-hidden="true" />}
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
