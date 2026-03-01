import { useEffect, useMemo, useState } from 'react'
import type { Bot } from '../types'
import type { GitAuthMode, GitIntegrationsState, GitProvider } from '../utils/gitIntegrations'
import { buildGitIntegrationsAuthJson, readGitIntegrationsState } from '../utils/gitIntegrations'

const PROVIDERS: Array<{ id: GitProvider; label: string; icon_url: string }> = [
  { id: 'github', label: 'GitHub', icon_url: 'https://cdn.simpleicons.org/github' },
  { id: 'gitlab', label: 'GitLab', icon_url: 'https://cdn.simpleicons.org/gitlab' },
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
  const [activeStep, setActiveStep] = useState<1 | 2 | 3 | 4>(1)

  useEffect(() => {
    setState(readGitIntegrationsState(bot?.data_agent_auth_json || '{}'))
  }, [bot?.data_agent_auth_json])

  const selectedProvider = state.provider
  const selected = state.providers[selectedProvider]
  const step1Done = Boolean(state.ssh_key_path.trim())
  const step2Done = authDone(selected.auth_mode, selected.token, state.ssh_key_path)
  const step3Done = Boolean(selected.repo_url.trim())
  const step4Done = Boolean(selected.repo_cache_path.trim() || selected.repo_source_path.trim())
  const providerConnection: Record<GitProvider, boolean> = {
    github: authDone(state.providers.github.auth_mode, state.providers.github.token, state.ssh_key_path),
    gitlab: authDone(state.providers.gitlab.auth_mode, state.providers.gitlab.token, state.ssh_key_path),
  }

  const selectedProviderConnected = useMemo(() => {
    const cfg = state.providers[selectedProvider]
    return authDone(cfg.auth_mode, cfg.token, state.ssh_key_path)
  }, [selectedProvider, state.providers, state.ssh_key_path])

  async function saveAll(next?: GitIntegrationsState) {
    const target = next || state
    const nextJson = buildGitIntegrationsAuthJson(bot?.data_agent_auth_json || '{}', target)
    await save({ data_agent_auth_json: nextJson })
  }

  function updateProvider(provider: GitProvider) {
    setState((prev) => ({ ...prev, provider }))
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

  async function saveAndContinue() {
    await saveAll()
    setActiveStep((prev) => (prev < 4 ? ((prev + 1) as 1 | 2 | 3 | 4) : prev))
  }

  async function skipOptional() {
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
        <span className={statusChip(selectedProviderConnected ? 'done' : 'pending')}>
          {selectedProviderConnected ? 'connected' : 'not connected'}
        </span>
      </div>

      <div className="gitIntLayout">
        <aside className="gitIntRail">
          {PROVIDERS.map((p) => (
            <button
              key={p.id}
              className={`gitIntRailTab ${selectedProvider === p.id ? 'active' : ''}`}
              onClick={() => updateProvider(p.id)}
              type="button"
            >
              <img src={p.icon_url} alt="" aria-hidden="true" className="gitIntRailIconImg" loading="lazy" />
              {p.label}
              {providerConnection[p.id] ? <span className="gitIntRailCheck" aria-label="Connected">✓</span> : null}
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
            <button className={`gitIntStepTab ${activeStep === 1 ? 'active' : ''}`} onClick={() => setActiveStep(1)} type="button">1. Add SSH Key</button>
            <button className={`gitIntStepTab ${activeStep === 2 ? 'active' : ''}`} onClick={() => setActiveStep(2)} type="button">2. Connect Account</button>
            <button className={`gitIntStepTab ${activeStep === 3 ? 'active' : ''}`} onClick={() => setActiveStep(3)} type="button">3. Link Repo (Optional)</button>
            <button className={`gitIntStepTab ${activeStep === 4 ? 'active' : ''}`} onClick={() => setActiveStep(4)} type="button">4. Local Repo Cache (Optional)</button>
          </div>

          <div className="gitIntPanel">
            {activeStep === 1 ? (
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

            {activeStep === 2 ? (
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

            {activeStep === 3 ? (
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

            {activeStep === 4 ? (
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

            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button className="btn" onClick={() => void saveAndContinue()} disabled={Boolean(saving)}>
                {saving ? 'Saving…' : 'Continue'}
              </button>
              <button className="btn ghost" onClick={() => void skipOptional()} disabled={Boolean(saving)}>
                Skip optional steps
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
