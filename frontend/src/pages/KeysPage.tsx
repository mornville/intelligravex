import { useEffect, useState } from 'react'
import { apiDelete, apiGet, apiPost } from '../api/client'
import type { ApiKey, ClientKey, GitTokenInfo } from '../types'
import { fmtIso } from '../utils/format'

export default function KeysPage() {
  const [keys, setKeys] = useState<ApiKey[]>([])
  const [clientKeys, setClientKeys] = useState<ClientKey[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [form, setForm] = useState({ name: '', secret: '' })
  const [creatingClient, setCreatingClient] = useState(false)
  const [clientForm, setClientForm] = useState({ name: '', allowed_origins: '' })
  const [newClientSecret, setNewClientSecret] = useState<string | null>(null)
  const [gitToken, setGitToken] = useState('')
  const [gitStatus, setGitStatus] = useState<GitTokenInfo | null>(null)
  const [savingGit, setSavingGit] = useState(false)

  async function reload() {
    setLoading(true)
    setErr(null)
    try {
      const [r, c] = await Promise.all([
        apiGet<{ items: ApiKey[] }>('/api/keys?provider=openai'),
        apiGet<{ items: ClientKey[] }>('/api/client-keys'),
      ])
      setKeys(r.items)
      setClientKeys(c.items)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setLoading(false)
    }
  }

  async function loadGitStatus() {
    try {
      const d = await apiGet<GitTokenInfo>('/api/user/git-token')
      setGitStatus(d)
    } catch (e: any) {
      setGitStatus(null)
    }
  }

  useEffect(() => {
    void reload()
    void loadGitStatus()
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

  async function onCreateClient() {
    if (!clientForm.name.trim()) return
    setCreatingClient(true)
    setErr(null)
    setNewClientSecret(null)
    try {
      const created: any = await apiPost<any>('/api/client-keys', {
        name: clientForm.name.trim(),
        allowed_origins: clientForm.allowed_origins.trim(),
      })
      if (created?.secret) setNewClientSecret(String(created.secret))
      setClientForm({ name: '', allowed_origins: '' })
      await reload()
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setCreatingClient(false)
    }
  }

  async function onSaveGitToken() {
    if (!gitToken.trim()) return
    setSavingGit(true)
    setErr(null)
    try {
      const d = await apiPost<GitTokenInfo>('/api/user/git-token', { provider: 'github', token: gitToken.trim() })
      setGitToken('')
      setGitStatus({ ...d, configured: true })
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setSavingGit(false)
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

  async function onDeleteClient(k: ClientKey) {
    const ok = window.confirm(`Delete client key "${k.name}"?\n\nThis is NOT reversible.`)
    if (!ok) return
    setErr(null)
    try {
      await apiDelete(`/api/client-keys/${k.id}`)
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

      <div style={{ height: 16 }} />

      <div className="grid2">
        <section className="card">
          <div className="cardTitle">GitHub token (fine‑grained)</div>
          <div className="muted" style={{ marginBottom: 8 }}>
            Prefer SSH? Set a per-bot SSH key path in Bot settings (Data Agent).
          </div>
          <div className="muted" style={{ marginBottom: 8 }}>
            Go to GitHub → Settings → Developer settings → Personal access tokens → Fine‑grained tokens.
          </div>
          <div className="muted" style={{ marginBottom: 8 }}>
            Set expiration (30–90 days), select only needed repositories, and grant read‑only Contents (and optional Pull requests).
          </div>
          <div className="formRow">
            <label>Token</label>
            <input
              value={gitToken}
              onChange={(e) => setGitToken(e.target.value)}
              placeholder="github_pat_..."
              type="password"
              autoComplete="off"
            />
            <div className="muted">Stored encrypted at rest (requires `VOICEBOT_SECRET_KEY`).</div>
          </div>
          <div className="row">
            <button className="btn primary" onClick={() => void onSaveGitToken()} disabled={savingGit || !gitToken.trim()}>
              {savingGit ? 'Saving…' : 'Save GitHub token'}
            </button>
          </div>
        </section>

        <section className="card">
          <div className="cardTitle">GitHub token status</div>
          {!gitStatus ? (
            <div className="muted">Loading…</div>
          ) : gitStatus.configured ? (
            <table className="table">
              <thead>
                <tr>
                  <th>Provider</th>
                  <th>Hint</th>
                  <th>Updated</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>{gitStatus.provider}</td>
                  <td className="mono">{gitStatus.hint || '—'}</td>
                  <td>{gitStatus.updated_at ? fmtIso(gitStatus.updated_at) : '—'}</td>
                </tr>
              </tbody>
            </table>
          ) : (
            <div className="muted">No GitHub token saved.</div>
          )}
        </section>
      </div>

      <div style={{ height: 16 }} />

      <div className="grid2">
        <section className="card">
          <div className="cardTitle">Add client key (embed)</div>
          {newClientSecret ? (
            <div className="alert">
              Copy this key now (it will not be shown again): <span className="mono">{newClientSecret}</span>
            </div>
          ) : null}
          <div className="formRow">
            <label>Name</label>
            <input
              value={clientForm.name}
              onChange={(e) => setClientForm((p) => ({ ...p, name: e.target.value }))}
              placeholder="e.g. Acme Website"
            />
          </div>
          <div className="formRow">
            <label>Allowed origins (comma-separated, optional)</label>
            <input
              value={clientForm.allowed_origins}
              onChange={(e) => setClientForm((p) => ({ ...p, allowed_origins: e.target.value }))}
              placeholder="https://acme.com, https://www.acme.com"
            />
          </div>
          <div className="row">
            <button className="btn primary" onClick={() => void onCreateClient()} disabled={creatingClient || !clientForm.name.trim()}>
              {creatingClient ? 'Generating…' : 'Generate key'}
            </button>
          </div>
        </section>

        <section className="card">
          <div className="cardTitle">Existing client keys</div>
          {loading ? (
            <div className="muted">Loading…</div>
          ) : clientKeys.length === 0 ? (
            <div className="muted">No client keys yet.</div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Hint</th>
                  <th>Allowed origins</th>
                  <th>Created</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {clientKeys.map((k) => (
                  <tr key={k.id}>
                    <td>{k.name}</td>
                    <td className="mono">{k.hint}</td>
                    <td className="mono" style={{ maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {k.allowed_origins || '(any)'}
                    </td>
                    <td>{fmtIso(k.created_at)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button className="btn danger ghost" onClick={() => void onDeleteClient(k)}>
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
