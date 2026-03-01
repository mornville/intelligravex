import { useEffect, useMemo, useState } from 'react'
import { apiGet, apiPost } from '../api/client'
import type { Bot } from '../types'
import type { DatabaseCredential, GitAuthMode, GitIntegrationsState, GitProvider } from '../utils/gitIntegrations'
import {
  buildGitIntegrationsAuthJson,
  clearGmailConfig,
  gmailConnected,
  newDatabaseCredential,
  readGitIntegrationsState,
} from '../utils/gitIntegrations'

type AppId = GitProvider | 'jira' | 'gmail' | 'database'

const APPS: Array<{ id: AppId; label: string; icon_url: string }> = [
  { id: 'github', label: 'GitHub', icon_url: 'https://cdn.simpleicons.org/github' },
  { id: 'gitlab', label: 'GitLab', icon_url: 'https://cdn.simpleicons.org/gitlab' },
  { id: 'jira', label: 'Jira', icon_url: '/brands/jira.svg' },
  { id: 'gmail', label: 'Gmail', icon_url: 'https://www.gstatic.com/images/branding/product/1x/gmail_2020q4_48dp.png' },
  { id: 'database', label: 'Database credentials', icon_url: 'https://cdn.simpleicons.org/postgresql' },
]

const UPCOMING_INTEGRATIONS: Array<{ label: string; icon_url: string }> = [
  { label: 'Slack (coming soon)', icon_url: 'https://cdn.simpleicons.org/slack' },
  { label: 'WhatsApp (coming soon)', icon_url: 'https://cdn.simpleicons.org/whatsapp' },
]

function statusChip(kind: 'done' | 'pending' | 'skip' | 'optional') {
  if (kind === 'done') return 'chip ok'
  if (kind === 'pending') return 'chip warn'
  return 'chip'
}

function authDone(mode: GitAuthMode, token: string, sshPath: string): boolean {
  if (mode === 'ssh') return Boolean(sshPath.trim())
  return Boolean(token.trim())
}

function dbCredentialReady(item: DatabaseCredential): boolean {
  return Boolean(item.engine.trim() && item.host.trim() && item.database.trim() && item.user.trim() && item.password.trim())
}

function dbCredentialTitle(item: DatabaseCredential): string {
  if (item.nickname.trim()) return item.nickname.trim()
  const base = item.host.trim() || item.database.trim() || item.engine.trim() || 'Credential'
  return base
}

type DbTestResult = {
  ok: boolean
  engine: string
  driver?: string
  code?: string
  message: string
  latency_ms?: number
}

type GmailOauthStart = {
  state: string
  auth_url: string
}

type GmailOauthStatus = {
  status: 'pending' | 'ready' | 'expired' | 'error'
  error?: string
  account_email?: string
  scope?: string
}

export default function BotGitIntegrationsPanel({
  bot,
  save,
  saving,
}: {
  bot: Bot
  save: (patch: Record<string, unknown>) => Promise<void>
  saving?: boolean
}) {
  const [state, setState] = useState<GitIntegrationsState>(() => readGitIntegrationsState(bot?.data_agent_auth_json || '{}'))
  const [activeApp, setActiveApp] = useState<AppId>('github')
  const [activeStep, setActiveStep] = useState<1 | 2 | 3 | 4>(1)
  const [activeJiraStep, setActiveJiraStep] = useState<1 | 2 | 3>(1)
  const [activeDbId, setActiveDbId] = useState<string>('')
  const [dbTestBusyId, setDbTestBusyId] = useState<string>('')
  const [dbTestResults, setDbTestResults] = useState<Record<string, DbTestResult>>({})
  const [gmailAuthBusy, setGmailAuthBusy] = useState(false)
  const [gmailAuthState, setGmailAuthState] = useState<string>('')
  const [gmailAuthUrl, setGmailAuthUrl] = useState<string>('')
  const [gmailAuthError, setGmailAuthError] = useState<string>('')

  useEffect(() => {
    const next = readGitIntegrationsState(bot?.data_agent_auth_json || '{}')
    setState(next)
    setActiveApp(next.provider)
    setActiveDbId(next.db_credentials[0]?.id || '')
    setDbTestBusyId('')
    setDbTestResults({})
    setGmailAuthBusy(false)
    setGmailAuthState('')
    setGmailAuthUrl('')
    setGmailAuthError('')
  }, [bot?.data_agent_auth_json])

  useEffect(() => {
    if (activeApp !== 'database') return
    if (!state.db_credentials.length) {
      if (activeDbId) setActiveDbId('')
      return
    }
    if (!state.db_credentials.some((item) => item.id === activeDbId)) {
      setActiveDbId(state.db_credentials[0].id)
    }
  }, [activeApp, activeDbId, state.db_credentials])

  const selectedProvider = state.provider
  const selected = state.providers[selectedProvider]
  const jira = state.jira
  const gmail = state.gmail
  const dbCredentials = state.db_credentials
  const activeDbCredential = dbCredentials.find((item) => item.id === activeDbId) || null
  const activeDbTestResult = activeDbCredential ? dbTestResults[activeDbCredential.id] : null

  const step1Done = Boolean(state.ssh_key_path.trim())
  const step2Done = authDone(selected.auth_mode, selected.token, state.ssh_key_path)
  const step3Done = Boolean(selected.repo_url.trim())
  const step4Done = Boolean(selected.repo_cache_path.trim() || selected.repo_source_path.trim())
  const jiraStep1Done = Boolean(jira.domain.trim())
  const jiraStep2Done = Boolean(jira.email.trim() && jira.api_token.trim())
  const jiraStep3Done = Boolean(jira.default_project_key.trim() || jira.default_issue_type.trim() || jira.default_jql.trim())
  const gmailDone = gmailConnected(gmail)
  const databaseDone = dbCredentials.some((item) => dbCredentialReady(item))

  const appConnection: Record<AppId, boolean> = {
    github: authDone(state.providers.github.auth_mode, state.providers.github.token, state.ssh_key_path),
    gitlab: authDone(state.providers.gitlab.auth_mode, state.providers.gitlab.token, state.ssh_key_path),
    jira: Boolean(jira.domain.trim() && jira.email.trim() && jira.api_token.trim()),
    gmail: gmailDone,
    database: databaseDone,
  }

  const selectedProviderConnected = useMemo(() => {
    const cfg = state.providers[selectedProvider]
    return authDone(cfg.auth_mode, cfg.token, state.ssh_key_path)
  }, [selectedProvider, state.providers, state.ssh_key_path])

  const selectedAppConnected =
    activeApp === 'jira'
      ? appConnection.jira
      : activeApp === 'gmail'
        ? appConnection.gmail
        : activeApp === 'database'
          ? appConnection.database
          : selectedProviderConnected

  async function saveAll(next?: GitIntegrationsState) {
    const target = next || state
    const nextJson = buildGitIntegrationsAuthJson(bot?.data_agent_auth_json || '{}', target)
    await save({ data_agent_auth_json: nextJson })
  }

  function updateProvider(provider: GitProvider) {
    setState((prev) => ({ ...prev, provider }))
    setActiveApp(provider)
  }

  function updateSelected(patch: Partial<GitIntegrationsState['providers']['github']>) {
    setState((prev) => ({
      ...prev,
      providers: {
        ...prev.providers,
        [prev.provider]: {
          ...prev.providers[prev.provider],
          ...patch,
        },
      },
    }))
  }

  function updateJira(patch: Partial<GitIntegrationsState['jira']>) {
    setState((prev) => ({
      ...prev,
      jira: {
        ...prev.jira,
        ...patch,
      },
    }))
  }

  function updateDbCredential(id: string, patch: Partial<DatabaseCredential>) {
    setState((prev) => ({
      ...prev,
      db_credentials: prev.db_credentials.map((item) => (item.id === id ? { ...item, ...patch } : item)),
    }))
    if (
      patch.engine !== undefined ||
      patch.host !== undefined ||
      patch.port !== undefined ||
      patch.database !== undefined ||
      patch.user !== undefined ||
      patch.password !== undefined ||
      patch.options !== undefined
    ) {
      setDbTestResults((prev) => {
        if (!prev[id]) return prev
        const next = { ...prev }
        delete next[id]
        return next
      })
    }
  }

  function addDbCredential() {
    const nextItem = newDatabaseCredential()
    setState((prev) => ({
      ...prev,
      db_credentials: [...prev.db_credentials, nextItem],
    }))
    setActiveDbId(nextItem.id)
  }

  function removeDbCredential(id: string) {
    const nextItems = state.db_credentials.filter((item) => item.id !== id)
    setState((prev) => ({ ...prev, db_credentials: nextItems }))
    if (activeDbId === id) setActiveDbId(nextItems[0]?.id || '')
    setDbTestResults((prev) => {
      if (!prev[id]) return prev
      const next = { ...prev }
      delete next[id]
      return next
    })
    if (dbTestBusyId === id) setDbTestBusyId('')
  }

  async function testDbCredential(item: DatabaseCredential) {
    if (!bot?.id) return
    setDbTestBusyId(item.id)
    try {
      const result = await apiPost<DbTestResult>(`/api/bots/${bot.id}/connected-apps/database/test`, {
        id: item.id,
        nickname: item.nickname,
        engine: item.engine,
        host: item.host,
        port: item.port,
        database: item.database,
        user: item.user,
        password: item.password,
        options: item.options,
        server_ca: item.server_ca,
        client_cert: item.client_cert,
        client_key: item.client_key,
      })
      setDbTestResults((prev) => ({ ...prev, [item.id]: result }))
    } catch (e: any) {
      setDbTestResults((prev) => ({
        ...prev,
        [item.id]: {
          ok: false,
          engine: item.engine || 'unknown',
          code: 'http_error',
          message: String(e?.message || e || 'Connection test failed.'),
        },
      }))
    } finally {
      setDbTestBusyId('')
    }
  }

  async function refreshFromServer() {
    if (!bot?.id) return
    try {
      const fresh = await apiGet<Bot>(`/api/bots/${bot.id}`)
      const next = readGitIntegrationsState(fresh?.data_agent_auth_json || '{}')
      setState((prev) => ({
        ...prev,
        gmail: next.gmail,
      }))
    } catch {
      // ignore refresh failures
    }
  }

  async function startGmailLogin() {
    if (!bot?.id) return
    setGmailAuthError('')
    setGmailAuthBusy(true)
    try {
      const res = await apiPost<GmailOauthStart>(`/api/bots/${bot.id}/connected-apps/gmail/oauth/start`, {})
      setGmailAuthState(res.state || '')
      setGmailAuthUrl(res.auth_url || '')
      if (res.auth_url) {
        try {
          window.open(res.auth_url, '_blank', 'noopener,noreferrer')
        } catch {
          // user can open manually
        }
      }
    } catch (e: any) {
      setGmailAuthError(String(e?.message || e || 'Failed to start Google sign-in.'))
    } finally {
      setGmailAuthBusy(false)
    }
  }

  async function disconnectGmail() {
    const next: GitIntegrationsState = {
      ...state,
      gmail: clearGmailConfig(state.gmail),
    }
    setState(next)
    await saveAll(next)
  }

  useEffect(() => {
    if (!gmailAuthState || !bot?.id) return
    const stateToken = gmailAuthState
    let canceled = false
    async function poll() {
      try {
        const res = await apiGet<GmailOauthStatus>(
          `/api/bots/${bot.id}/connected-apps/gmail/oauth/status?state=${encodeURIComponent(stateToken)}`
        )
        if (canceled) return
        if (res.status === 'ready') {
          setGmailAuthState('')
          setGmailAuthError('')
          await refreshFromServer()
          return
        }
        if (res.status === 'error' || res.status === 'expired') {
          setGmailAuthState('')
          setGmailAuthError(res.error || (res.status === 'expired' ? 'Google sign-in expired.' : 'Google sign-in failed.'))
        }
      } catch (e: any) {
        if (!canceled) {
          setGmailAuthState('')
          setGmailAuthError(String(e?.message || e || 'Google sign-in failed.'))
        }
      }
    }
    void poll()
    const t = window.setInterval(() => void poll(), 1300)
    return () => {
      canceled = true
      window.clearInterval(t)
    }
  }, [bot?.id, gmailAuthState])

  async function saveAndContinue() {
    await saveAll()
    if (activeApp === 'database') return
    if (activeApp === 'jira') {
      setActiveJiraStep((prev) => (prev < 3 ? ((prev + 1) as 1 | 2 | 3) : prev))
      return
    }
    setActiveStep((prev) => (prev < 4 ? ((prev + 1) as 1 | 2 | 3 | 4) : prev))
  }

  async function skipOptional() {
    if (activeApp === 'database') return
    if (activeApp === 'jira') {
      const next: GitIntegrationsState = {
        ...state,
        jira: {
          ...state.jira,
          default_project_key: '',
          default_issue_type: '',
          default_jql: '',
        },
      }
      setState(next)
      await saveAll(next)
      return
    }
    const next: GitIntegrationsState = {
      ...state,
      providers: {
        ...state.providers,
        [selectedProvider]: {
          ...selected,
          repo_url: '',
          repo_cache_path: '',
          repo_source_path: '',
        },
      },
    }
    setState(next)
    await saveAll(next)
  }

  return (
    <div className="gitIntRoot">
      <div className="gitIntHead">
        <div>
          <div className="cardTitle">Connected apps</div>
          <div className="muted">Per-assistant provider setup. Workspace SSH key is shared across providers.</div>
        </div>
        <span className={statusChip(selectedAppConnected ? 'done' : 'pending')}>
          {selectedAppConnected ? 'connected' : 'not connected'}
        </span>
      </div>

      <div className="gitIntLayout">
        <aside className="gitIntRail">
          {APPS.map((p) => (
            <button
              key={p.id}
              className={`gitIntRailTab ${activeApp === p.id ? 'active' : ''}`}
              onClick={() => {
                if (p.id === 'jira' || p.id === 'gmail' || p.id === 'database') {
                  setActiveApp(p.id)
                } else {
                  updateProvider(p.id)
                }
              }}
              type="button"
            >
              <img src={p.icon_url} alt="" aria-hidden="true" className="gitIntRailIconImg" loading="lazy" />
              {p.label}
              {appConnection[p.id] ? <span className="gitIntRailCheck" aria-label="Connected">✓</span> : null}
            </button>
          ))}
          {UPCOMING_INTEGRATIONS.map((item) => (
            <div key={item.label} className="gitIntRailTab disabled">
              <img src={item.icon_url} alt="" aria-hidden="true" className="gitIntRailIconImg" loading="lazy" />
              {item.label}
            </div>
          ))}
        </aside>

        <div className="gitIntMain">
          <div className="gitIntStepTabs">
            {activeApp === 'jira' ? (
              <>
                <button className={`gitIntStepTab ${activeJiraStep === 1 ? 'active' : ''}`} onClick={() => setActiveJiraStep(1)} type="button">1. Jira Domain</button>
                <button className={`gitIntStepTab ${activeJiraStep === 2 ? 'active' : ''}`} onClick={() => setActiveJiraStep(2)} type="button">2. Connect Account</button>
                <button className={`gitIntStepTab ${activeJiraStep === 3 ? 'active' : ''}`} onClick={() => setActiveJiraStep(3)} type="button">3. Defaults (Optional)</button>
              </>
            ) : activeApp === 'gmail' ? (
              <button className="gitIntStepTab active" type="button">1. Google Sign-In</button>
            ) : activeApp === 'database' ? (
              <button className="gitIntStepTab active" type="button">1. Manage Credentials</button>
            ) : (
              <>
                <button className={`gitIntStepTab ${activeStep === 1 ? 'active' : ''}`} onClick={() => setActiveStep(1)} type="button">1. Add SSH Key</button>
                <button className={`gitIntStepTab ${activeStep === 2 ? 'active' : ''}`} onClick={() => setActiveStep(2)} type="button">2. Connect Account</button>
                <button className={`gitIntStepTab ${activeStep === 3 ? 'active' : ''}`} onClick={() => setActiveStep(3)} type="button">3. Link Repo (Optional)</button>
                <button className={`gitIntStepTab ${activeStep === 4 ? 'active' : ''}`} onClick={() => setActiveStep(4)} type="button">4. Local Repo Cache (Optional)</button>
              </>
            )}
          </div>

          <div className="gitIntPanel">
            {activeApp !== 'jira' && activeApp !== 'gmail' && activeApp !== 'database' && activeStep === 1 ? (
              <>
                <div className="step">
                  <span>Workspace SSH key</span>
                  <span className={statusChip(step1Done ? 'done' : 'pending')}>{step1Done ? 'done' : 'pending'}</span>
                </div>
                <div className="quote">
                  Used by isolated workspace to run Git operations (clone/fetch/pull/push) without per-provider keys.
                </div>
                <div className="formRow">
                  <label>SSH private key path</label>
                  <input
                    value={state.ssh_key_path}
                    onChange={(e) => setState((prev) => ({ ...prev, ssh_key_path: e.target.value }))}
                    placeholder="/Users/you/.ssh/id_ed25519"
                  />
                </div>
              </>
            ) : null}

            {activeApp !== 'jira' && activeApp !== 'gmail' && activeApp !== 'database' && activeStep === 2 ? (
              <>
                <div className="step">
                  <span>OAuth / PAT access ({selectedProvider})</span>
                  <span className={statusChip(step2Done ? 'done' : 'pending')}>{step2Done ? 'done' : 'pending'}</span>
                </div>
                <div className="quote">
                  Authorizes provider API calls for listing repos, opening PRs/MRs, and metadata sync.
                </div>
                <div className="formRow">
                  <label>Auth method</label>
                  <select
                    value={selected.auth_mode}
                    onChange={(e) => updateSelected({ auth_mode: (e.target.value as GitAuthMode) || 'pat' })}
                  >
                    <option value="pat">Personal access token (PAT)</option>
                    <option value="oauth">OAuth access token</option>
                    <option value="ssh">SSH only (no API token)</option>
                  </select>
                </div>
                {selected.auth_mode !== 'ssh' ? (
                  <div className="formRow">
                    <label>{selected.auth_mode === 'oauth' ? 'OAuth access token' : 'PAT token'}</label>
                    <input
                      value={selected.token}
                      onChange={(e) => updateSelected({ token: e.target.value })}
                      placeholder={selected.auth_mode === 'oauth' ? 'oauth_xxx...' : 'ghp_... / glpat-...'}
                    />
                    <div className="muted">Stored in assistant auth JSON for this assistant only.</div>
                  </div>
                ) : null}
              </>
            ) : null}

            {activeApp !== 'jira' && activeApp !== 'gmail' && activeApp !== 'database' && activeStep === 3 ? (
              <>
                <div className="step">
                  <span>Repo link</span>
                  <span className={statusChip(step3Done ? 'done' : 'skip')}>{step3Done ? 'done' : 'skip'}</span>
                </div>
                <div className="quote">
                  Optional shortcut: preselect repo so clone/setup is one click. If skipped, integration still stays connected.
                </div>
                <div className="formRow">
                  <label>Repository URL</label>
                  <input
                    value={selected.repo_url}
                    onChange={(e) => updateSelected({ repo_url: e.target.value })}
                    placeholder={selectedProvider === 'gitlab' ? 'git@gitlab.com:group/project.git' : 'git@github.com:org/repo.git'}
                  />
                </div>
              </>
            ) : null}

            {activeApp !== 'jira' && activeApp !== 'gmail' && activeApp !== 'database' && activeStep === 4 ? (
              <>
                <div className="step">
                  <span>Local repo cache path</span>
                  <span className={statusChip(step4Done ? 'done' : 'optional')}>{step4Done ? 'done' : 'optional'}</span>
                </div>
                <div className="quote">
                  For large repos, reuse local checkout to avoid cloning repeatedly and speed up isolated workspace startup.
                </div>
                <div className="formRow">
                  <label>Repo cache path (host)</label>
                  <input
                    value={selected.repo_cache_path}
                    onChange={(e) => updateSelected({ repo_cache_path: e.target.value })}
                    placeholder="/Users/you/.igx_repo_cache/repo.git"
                  />
                </div>
                <div className="formRow">
                  <label>Repo source path (host)</label>
                  <input
                    value={selected.repo_source_path}
                    onChange={(e) => updateSelected({ repo_source_path: e.target.value })}
                    placeholder="/Users/you/dev/repo"
                  />
                </div>
              </>
            ) : null}

            {activeApp === 'jira' && activeJiraStep === 1 ? (
              <>
                <div className="step">
                  <span>Jira site domain</span>
                  <span className={statusChip(jiraStep1Done ? 'done' : 'pending')}>{jiraStep1Done ? 'done' : 'pending'}</span>
                </div>
                <div className="quote">
                  Use your Atlassian Jira Cloud URL.
                </div>
                <div className="formRow">
                  <label>Jira domain</label>
                  <input
                    value={jira.domain}
                    onChange={(e) => updateJira({ domain: e.target.value })}
                    placeholder="https://yourcompany.atlassian.net"
                  />
                </div>
              </>
            ) : null}

            {activeApp === 'jira' && activeJiraStep === 2 ? (
              <>
                <div className="step">
                  <span>Jira account access</span>
                  <span className={statusChip(jiraStep2Done ? 'done' : 'pending')}>{jiraStep2Done ? 'done' : 'pending'}</span>
                </div>
                <div className="quote">
                  Required: Atlassian account email and API token.
                </div>
                <div className="formRow">
                  <label>Atlassian email</label>
                  <input
                    type="email"
                    value={jira.email}
                    onChange={(e) => updateJira({ email: e.target.value })}
                    placeholder="you@company.com"
                  />
                </div>
                <div className="formRow">
                  <label>API token</label>
                  <input
                    type="password"
                    value={jira.api_token}
                    onChange={(e) => updateJira({ api_token: e.target.value })}
                    placeholder="ATATT..."
                  />
                  <div className="muted">
                    Generate from Atlassian account security: id.atlassian.com/manage-profile/security/api-tokens
                  </div>
                </div>
              </>
            ) : null}

            {activeApp === 'jira' && activeJiraStep === 3 ? (
              <>
                <div className="step">
                  <span>Jira defaults</span>
                  <span className={statusChip(jiraStep3Done ? 'done' : 'optional')}>{jiraStep3Done ? 'done' : 'optional'}</span>
                </div>
                <div className="quote">
                  Optional defaults for faster issue creation and search.
                </div>
                <div className="formRow">
                  <label>Default project key</label>
                  <input
                    value={jira.default_project_key}
                    onChange={(e) => updateJira({ default_project_key: e.target.value })}
                    placeholder="ENG"
                  />
                </div>
                <div className="formRow">
                  <label>Default issue type</label>
                  <input
                    value={jira.default_issue_type}
                    onChange={(e) => updateJira({ default_issue_type: e.target.value })}
                    placeholder="Task / Bug / Story"
                  />
                </div>
                <div className="formRow">
                  <label>Default JQL</label>
                  <input
                    value={jira.default_jql}
                    onChange={(e) => updateJira({ default_jql: e.target.value })}
                    placeholder="assignee = currentUser() ORDER BY updated DESC"
                  />
                </div>
              </>
            ) : null}

            {activeApp === 'gmail' ? (
              <>
                <div className="step">
                  <span>Google Sign-In</span>
                  <span className={statusChip(gmailDone ? 'done' : 'pending')}>{gmailDone ? 'connected' : 'pending'}</span>
                </div>
                <div className="quote">
                  Sign in with Google and grant only the requested Gmail scopes.
                </div>
                <div className="row" style={{ justifyContent: 'space-between' }}>
                  <div className="muted">
                    Requested scopes:
                    <br />
                    `gmail.send` and `gmail.readonly`
                  </div>
                  <button className="btn" onClick={() => void startGmailLogin()} disabled={gmailAuthBusy || Boolean(gmailAuthState)}>
                    {gmailAuthBusy ? 'Starting…' : gmailAuthState ? 'Waiting for approval…' : gmailDone ? 'Reconnect Google' : 'Sign in with Google'}
                  </button>
                </div>
                {gmailAuthUrl ? (
                  <div className="muted">
                    If popups are blocked, open login manually:{' '}
                    <a href={gmailAuthUrl} target="_blank" rel="noreferrer" className="link">
                      Open Google login
                    </a>
                  </div>
                ) : null}
                {gmailAuthError ? <div className="alert">{gmailAuthError}</div> : null}
                {gmailDone ? (
                  <div className="formRow">
                    <label>Connected account</label>
                    <input value={gmail.account_email || 'connected'} readOnly />
                    <div className="muted">Scopes: {gmail.scope || 'gmail.send gmail.readonly'}</div>
                    <button className="gitIntActionText danger" type="button" onClick={() => void disconnectGmail()}>
                      disconnect
                    </button>
                  </div>
                ) : null}
                <div className="muted">
                  Gmail OAuth setup docs: developers.google.com/workspace/gmail/api/auth/web-server
                </div>
              </>
            ) : null}

            {activeApp === 'database' ? (
              <>
                <div className="step">
                  <span>Database credentials</span>
                  <span className={statusChip(databaseDone ? 'done' : 'pending')}>
                    {databaseDone ? `${dbCredentials.length} saved` : 'pending'}
                  </span>
                </div>
                <div className="quote">
                  Add one or more database profiles. Isolated workspace tools can use these credentials for queries and ETL tasks.
                </div>

                <div className="gitIntDbShell">
                  <div className="gitIntDbList">
                    <button className="btn ghost gitIntDbAdd" type="button" onClick={addDbCredential}>
                      + Add credential
                    </button>
                    {dbCredentials.length ? (
                      dbCredentials.map((item) => (
                        <button
                          key={item.id}
                          className={`gitIntDbItem ${item.id === activeDbId ? 'active' : ''}`}
                          type="button"
                          onClick={() => setActiveDbId(item.id)}
                        >
                          <span>{dbCredentialTitle(item)}</span>
                        </button>
                      ))
                    ) : (
                      <div className="muted">No credentials yet. Add one to get started.</div>
                    )}
                  </div>

                  <div className="gitIntDbEditor">
                    {activeDbCredential ? (
                      <>
                        <div className="formRow">
                          <label>Nickname</label>
                          <input
                            value={activeDbCredential.nickname}
                            onChange={(e) => updateDbCredential(activeDbCredential.id, { nickname: e.target.value })}
                            placeholder="production analytics"
                          />
                        </div>

                        <div className="formRow formRowGrid2">
                          <div>
                            <label>Engine</label>
                            <select
                              value={activeDbCredential.engine}
                              onChange={(e) => updateDbCredential(activeDbCredential.id, { engine: e.target.value })}
                            >
                              <option value="postgresql">PostgreSQL</option>
                              <option value="mysql">MySQL</option>
                              <option value="mssql">MS SQL Server</option>
                              <option value="mongodb">MongoDB</option>
                              <option value="redis">Redis</option>
                              <option value="other">Other</option>
                            </select>
                          </div>
                          <div>
                            <label>Port</label>
                            <input
                              value={activeDbCredential.port}
                              onChange={(e) => updateDbCredential(activeDbCredential.id, { port: e.target.value })}
                              placeholder="5432"
                            />
                          </div>
                        </div>

                        <div className="formRow formRowGrid2">
                          <div>
                            <label>Host</label>
                            <input
                              value={activeDbCredential.host}
                              onChange={(e) => updateDbCredential(activeDbCredential.id, { host: e.target.value })}
                              placeholder="db.company.internal"
                            />
                          </div>
                          <div>
                            <label>Database</label>
                            <input
                              value={activeDbCredential.database}
                              onChange={(e) => updateDbCredential(activeDbCredential.id, { database: e.target.value })}
                              placeholder="warehouse"
                            />
                          </div>
                        </div>

                        <div className="formRow formRowGrid2">
                          <div>
                            <label>User</label>
                            <input
                              value={activeDbCredential.user}
                              onChange={(e) => updateDbCredential(activeDbCredential.id, { user: e.target.value })}
                              placeholder="db_reader"
                            />
                          </div>
                          <div>
                            <label>Password</label>
                            <input
                              type="password"
                              value={activeDbCredential.password}
                              onChange={(e) => updateDbCredential(activeDbCredential.id, { password: e.target.value })}
                              placeholder="••••••••"
                            />
                          </div>
                        </div>

                        <div className="formRow">
                          <label>Options</label>
                          <input
                            value={activeDbCredential.options}
                            onChange={(e) => updateDbCredential(activeDbCredential.id, { options: e.target.value })}
                            placeholder="sslmode=require&connect_timeout=10"
                          />
                        </div>

                        <div className="formRow formRowGrid2">
                          <div>
                            <label>Server CA</label>
                            <input
                              value={activeDbCredential.server_ca}
                              onChange={(e) => updateDbCredential(activeDbCredential.id, { server_ca: e.target.value })}
                              placeholder="/path/to/server-ca.pem"
                            />
                          </div>
                          <div>
                            <label>Client cert</label>
                            <input
                              value={activeDbCredential.client_cert}
                              onChange={(e) => updateDbCredential(activeDbCredential.id, { client_cert: e.target.value })}
                              placeholder="/path/to/client-cert.pem"
                            />
                          </div>
                        </div>

                        <div className="formRow formRowGrid2">
                          <div>
                            <label>Client key</label>
                            <input
                              value={activeDbCredential.client_key}
                              onChange={(e) => updateDbCredential(activeDbCredential.id, { client_key: e.target.value })}
                              placeholder="/path/to/client-key.pem"
                            />
                          </div>
                          <div className="row" style={{ justifyContent: 'flex-end', alignItems: 'flex-end' }}>
                            <button
                              className="gitIntActionText"
                              type="button"
                              onClick={() => void testDbCredential(activeDbCredential)}
                              disabled={dbTestBusyId === activeDbCredential.id}
                            >
                              {dbTestBusyId === activeDbCredential.id ? 'testing…' : 'test'}
                            </button>
                            <button
                              className="gitIntActionText danger"
                              type="button"
                              onClick={() => removeDbCredential(activeDbCredential.id)}
                            >
                              delete
                            </button>
                          </div>
                        </div>
                        {activeDbTestResult ? (
                          <div className={`gitIntDbTest ${activeDbTestResult.ok ? 'ok' : 'err'}`}>
                            <span>{activeDbTestResult.ok ? 'Connected' : 'Failed'}</span>
                            <span>{activeDbTestResult.message}</span>
                            {activeDbTestResult.latency_ms ? <span>{`${activeDbTestResult.latency_ms} ms`}</span> : null}
                          </div>
                        ) : null}
                      </>
                    ) : (
                      <div className="muted">Select a credential from the left, or create a new one.</div>
                    )}
                  </div>
                </div>
              </>
            ) : null}

            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button className="btn" onClick={() => void saveAndContinue()} disabled={Boolean(saving)}>
                {saving ? 'Saving…' : activeApp === 'database' ? 'Save credentials' : 'Continue'}
              </button>
              {activeApp !== 'database' ? (
                <button
                  className="btn ghost"
                  onClick={() => void skipOptional()}
                  disabled={Boolean(saving) || activeApp === 'gmail' || (activeApp === 'jira' && activeJiraStep !== 3)}
                >
                  Skip optional steps
                </button>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
