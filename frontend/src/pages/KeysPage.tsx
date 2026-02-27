import { useEffect, useState } from 'react'
import { apiDelete, apiGet, apiPost } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import { TrashIcon } from '@heroicons/react/24/solid'
import type { ApiKey, ClientKey, GitTokenInfo } from '../types'
import { fmtIso } from '../utils/format'

export default function KeysPage() {
  const [openaiKeys, setOpenaiKeys] = useState<ApiKey[]>([])
  const [chatgptKeys, setChatgptKeys] = useState<ApiKey[]>([])
  const [openrouterKeys, setOpenrouterKeys] = useState<ApiKey[]>([])
  const [clientKeys, setClientKeys] = useState<ClientKey[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [creatingOpenai, setCreatingOpenai] = useState(false)
  const [creatingOpenrouter, setCreatingOpenrouter] = useState(false)
  const [openaiForm, setOpenaiForm] = useState({ name: '', secret: '' })
  const [openrouterForm, setOpenrouterForm] = useState({ name: '', secret: '' })
  const [creatingClient, setCreatingClient] = useState(false)
  const [clientForm, setClientForm] = useState({ name: '', allowed_origins: '' })
  const [newClientSecret, setNewClientSecret] = useState<string | null>(null)
  const [gitToken, setGitToken] = useState('')
  const [gitStatus, setGitStatus] = useState<GitTokenInfo | null>(null)
  const [savingGit, setSavingGit] = useState(false)
  const [chatgptAuthState, setChatgptAuthState] = useState<string | null>(null)
  const [chatgptAuthErr, setChatgptAuthErr] = useState<string | null>(null)
  const [chatgptAuthBusy, setChatgptAuthBusy] = useState(false)
  const [chatgptAuthUrl, setChatgptAuthUrl] = useState<string | null>(null)

  async function reload() {
    setLoading(true)
    setErr(null)
    try {
      const [oai, cgpt, orr, c] = await Promise.all([
        apiGet<{ items: ApiKey[] }>('/api/keys?provider=openai'),
        apiGet<{ items: ApiKey[] }>('/api/keys?provider=chatgpt'),
        apiGet<{ items: ApiKey[] }>('/api/keys?provider=openrouter'),
        apiGet<{ items: ClientKey[] }>('/api/client-keys'),
      ])
      setOpenaiKeys(oai.items)
      setChatgptKeys(cgpt.items)
      setOpenrouterKeys(orr.items)
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

  useEffect(() => {
    if (!chatgptAuthState) return
    const state = chatgptAuthState
    let canceled = false
    async function poll() {
      try {
        const res = await apiGet<{ status: string; error?: string }>(`/api/chatgpt/oauth/status?state=${encodeURIComponent(state)}`)
        if (canceled) return
        if (res.status === 'ready') {
          setChatgptAuthState(null)
          setChatgptAuthUrl(null)
          setChatgptAuthErr(null)
          await reload()
          return
        }
        if (res.status === 'error' || res.status === 'expired') {
          setChatgptAuthState(null)
          setChatgptAuthUrl(null)
          setChatgptAuthErr(res.error || (res.status === 'expired' ? 'Sign-in expired. Please try again.' : 'Sign-in failed.'))
        }
      } catch (e: any) {
        if (!canceled) setChatgptAuthErr(String(e?.message || e))
      }
    }
    void poll()
    const t = window.setInterval(() => void poll(), 1200)
    return () => {
      canceled = true
      window.clearInterval(t)
    }
  }, [chatgptAuthState])

  async function onCreateOpenAI() {
    if (!openaiForm.name.trim() || !openaiForm.secret.trim()) return
    setCreatingOpenai(true)
    setErr(null)
    try {
      await apiPost<ApiKey>('/api/keys', {
        provider: 'openai',
        name: openaiForm.name.trim(),
        secret: openaiForm.secret.trim(),
      })
      setOpenaiForm({ name: '', secret: '' })
      await reload()
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setCreatingOpenai(false)
    }
  }

  async function startChatgptAuth() {
    setChatgptAuthErr(null)
    setChatgptAuthBusy(true)
    try {
      const res = await apiPost<{ state: string; auth_url: string }>('/api/chatgpt/oauth/start', {})
      if (!res?.state || !res?.auth_url) throw new Error('ChatGPT sign-in unavailable.')
      setChatgptAuthState(res.state)
      setChatgptAuthUrl(res.auth_url)
      const opened = window.open(res.auth_url, '_blank', 'noopener,noreferrer')
      if (!opened) setChatgptAuthErr('Popup blocked. Click "Open login" to continue.')
    } catch (e: any) {
      setChatgptAuthErr(String(e?.message || e))
    } finally {
      setChatgptAuthBusy(false)
    }
  }

  async function onCreateOpenRouter() {
    if (!openrouterForm.name.trim() || !openrouterForm.secret.trim()) return
    setCreatingOpenrouter(true)
    setErr(null)
    try {
      await apiPost<ApiKey>('/api/keys', {
        provider: 'openrouter',
        name: openrouterForm.name.trim(),
        secret: openrouterForm.secret.trim(),
      })
      setOpenrouterForm({ name: '', secret: '' })
      await reload()
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setCreatingOpenrouter(false)
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
        <div>
          <h1>Keys</h1>
          <div className="muted">Stored API keys (secrets never shown back).</div>
        </div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <div className="grid2">
        <section className="card">
          <div className="cardTitle">Add OpenAI key (global)</div>
          <div className="formRow">
            <label>Name</label>
            <input
              value={openaiForm.name}
              onChange={(e) => setOpenaiForm((p) => ({ ...p, name: e.target.value }))}
              placeholder="e.g. Personal"
            />
          </div>
          <div className="formRow">
            <label>Secret</label>
            <input
              value={openaiForm.secret}
              onChange={(e) => setOpenaiForm((p) => ({ ...p, secret: e.target.value }))}
              placeholder="sk-..."
              type="password"
              autoComplete="off"
            />
            <div className="muted">Stored encrypted at rest on this device.</div>
          </div>
          <div className="row">
            <button
              className="btn primary"
              onClick={onCreateOpenAI}
              disabled={creatingOpenai || !openaiForm.name.trim() || !openaiForm.secret.trim()}
            >
              {creatingOpenai ? 'Saving…' : 'Save key'}
            </button>
          </div>
        </section>
      </div>

      <div style={{ height: 16 }} />

      <div className="grid2">
        <section className="card">
          <div className="cardTitle">OpenAI keys</div>
          <div className="muted" style={{ marginBottom: 8 }}>
            The most recently added OpenAI key is used for all bots.
          </div>
          {loading ? (
            <div className="muted">
              <LoadingSpinner />
            </div>
          ) : openaiKeys.length === 0 ? (
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
                {openaiKeys.map((k) => (
                  <tr key={k.id}>
                    <td>{k.name}</td>
                    <td className="mono">{k.hint}</td>
                    <td>{fmtIso(k.created_at)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button className="btn iconBtn danger" onClick={() => void onDelete(k)} aria-label="Delete key" title="Delete key">
                        <TrashIcon aria-hidden="true" />
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
          <div className="cardTitle">ChatGPT OAuth (personal)</div>
          <div className="muted" style={{ marginBottom: 8 }}>
            Sign in with ChatGPT to enable personal-use Codex calls.
          </div>
          <div className="muted" style={{ marginBottom: 8 }}>
            Voice features (ASR/TTS) still require an OpenAI API key.
          </div>
          {chatgptAuthErr ? <div className="alert">{chatgptAuthErr}</div> : null}
          <div className="row gap">
            <button className="btn primary" onClick={() => void startChatgptAuth()} disabled={chatgptAuthBusy || !!chatgptAuthState}>
              {chatgptAuthBusy ? 'Starting…' : chatgptAuthState ? 'Waiting for approval…' : 'Sign in with ChatGPT'}
            </button>
            {chatgptAuthUrl ? (
              <button className="btn" onClick={() => window.open(chatgptAuthUrl, '_blank', 'noopener,noreferrer')}>
                Open login
              </button>
            ) : null}
          </div>
          {chatgptAuthState ? <div className="muted" style={{ marginTop: 8 }}>Waiting for approval…</div> : null}
        </section>
      </div>

      <div style={{ height: 16 }} />

      <div className="grid2">
        <section className="card">
          <div className="cardTitle">ChatGPT tokens</div>
          <div className="muted" style={{ marginBottom: 8 }}>
            The most recently added token is used for ChatGPT (OAuth) bots.
          </div>
          {loading ? (
            <div className="muted">
              <LoadingSpinner />
            </div>
          ) : chatgptKeys.length === 0 ? (
            <div className="muted">No tokens yet.</div>
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
                {chatgptKeys.map((k) => (
                  <tr key={k.id}>
                    <td>{k.name}</td>
                    <td className="mono">{k.hint}</td>
                    <td>{fmtIso(k.created_at)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button className="btn iconBtn danger" onClick={() => void onDelete(k)} aria-label="Delete token" title="Delete token">
                        <TrashIcon aria-hidden="true" />
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
          <div className="cardTitle">Add OpenRouter key (global)</div>
          <div className="formRow">
            <label>Name</label>
            <input
              value={openrouterForm.name}
              onChange={(e) => setOpenrouterForm((p) => ({ ...p, name: e.target.value }))}
              placeholder="e.g. Team"
            />
          </div>
          <div className="formRow">
            <label>Secret</label>
            <input
              value={openrouterForm.secret}
              onChange={(e) => setOpenrouterForm((p) => ({ ...p, secret: e.target.value }))}
              placeholder="sk-or-..."
              type="password"
              autoComplete="off"
            />
            <div className="muted">Stored encrypted at rest on this device.</div>
          </div>
          <div className="row">
            <button
              className="btn primary"
              onClick={onCreateOpenRouter}
              disabled={creatingOpenrouter || !openrouterForm.name.trim() || !openrouterForm.secret.trim()}
            >
              {creatingOpenrouter ? 'Saving…' : 'Save key'}
            </button>
          </div>
        </section>
      </div>

      <div style={{ height: 16 }} />

      <div className="grid2">
        <section className="card">
          <div className="cardTitle">OpenRouter keys</div>
          <div className="muted" style={{ marginBottom: 8 }}>
            The most recently added OpenRouter key is used for OpenRouter bots.
          </div>
          {loading ? (
            <div className="muted">
              <LoadingSpinner />
            </div>
          ) : openrouterKeys.length === 0 ? (
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
                {openrouterKeys.map((k) => (
                  <tr key={k.id}>
                    <td>{k.name}</td>
                    <td className="mono">{k.hint}</td>
                    <td>{fmtIso(k.created_at)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button className="btn iconBtn danger" onClick={() => void onDelete(k)} aria-label="Delete key" title="Delete key">
                        <TrashIcon aria-hidden="true" />
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
            Prefer SSH? Set a per-bot SSH key path in Bot settings (Isolated Workspace).
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
            <div className="muted">Stored encrypted at rest on this device.</div>
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
            <div className="muted">
              <LoadingSpinner />
            </div>
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
                  <td className="mono">{gitStatus.hint || '-'}</td>
                  <td>{gitStatus.updated_at ? fmtIso(gitStatus.updated_at) : '-'}</td>
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
            <div className="muted">
              <LoadingSpinner />
            </div>
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
                      <button
                        className="btn iconBtn danger"
                        onClick={() => void onDeleteClient(k)}
                        aria-label="Delete client key"
                        title="Delete client key"
                      >
                        <TrashIcon aria-hidden="true" />
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
