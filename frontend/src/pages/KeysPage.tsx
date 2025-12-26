import { useEffect, useState } from 'react'
import { apiDelete, apiGet, apiPost } from '../api/client'
import type { ApiKey } from '../types'
import { fmtIso } from '../utils/format'

export default function KeysPage() {
  const [keys, setKeys] = useState<ApiKey[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [form, setForm] = useState({ name: '', secret: '' })

  async function reload() {
    setLoading(true)
    setErr(null)
    try {
      const r = await apiGet<{ items: ApiKey[] }>('/api/keys?provider=openai')
      setKeys(r.items)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void reload()
  }, [])

  async function onCreate() {
    if (!form.name.trim() || !form.secret.trim()) return
    setCreating(true)
    setErr(null)
    try {
      await apiPost<ApiKey>('/api/keys', { provider: 'openai', name: form.name.trim(), secret: form.secret.trim() })
      setForm({ name: '', secret: '' })
      await reload()
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setCreating(false)
    }
  }

  async function onDelete(k: ApiKey) {
    const ok = window.confirm(`Delete key "${k.name}"?\n\nThis is NOT reversible.`)
    if (!ok) return
    setErr(null)
    try {
      await apiDelete(`/api/keys/${k.id}`)
      await reload()
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Keys</h1>
        <div className="muted">Stored API keys (secrets never shown back).</div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <div className="grid2">
        <section className="card">
          <div className="cardTitle">Add OpenAI key</div>
          <div className="formRow">
            <label>Name</label>
            <input value={form.name} onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))} placeholder="e.g. Personal" />
          </div>
          <div className="formRow">
            <label>Secret</label>
            <input
              value={form.secret}
              onChange={(e) => setForm((p) => ({ ...p, secret: e.target.value }))}
              placeholder="sk-..."
              type="password"
              autoComplete="off"
            />
            <div className="muted">Stored encrypted at rest (requires `VOICEBOT_SECRET_KEY`).</div>
          </div>
          <div className="row">
            <button className="btn primary" onClick={onCreate} disabled={creating || !form.name.trim() || !form.secret.trim()}>
              {creating ? 'Saving…' : 'Save key'}
            </button>
          </div>
        </section>

        <section className="card">
          <div className="cardTitle">Existing keys</div>
          {loading ? (
            <div className="muted">Loading…</div>
          ) : keys.length === 0 ? (
            <div className="muted">No keys yet.</div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Hint</th>
                  <th>Created</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {keys.map((k) => (
                  <tr key={k.id}>
                    <td>{k.name}</td>
                    <td className="mono">{k.hint}</td>
                    <td>{fmtIso(k.created_at)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button className="btn danger ghost" onClick={() => void onDelete(k)}>
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>
    </div>
  )
}

