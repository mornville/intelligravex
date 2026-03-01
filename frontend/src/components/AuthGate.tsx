import type { ReactNode } from 'react'
import { useEffect, useState } from 'react'
import type { LocalModel, Options } from '../types'
import { BACKEND_URL, apiGet, apiPost } from '../api/client'
import { authHeader } from '../auth'
import { formatLocalModelToolSupport } from '../utils/localModels'
import { formatProviderLabel } from '../utils/llmProviders'

const PROVIDER_ICON_URL: Record<'chatgpt' | 'openai' | 'openrouter' | 'local', string> = {
  chatgpt: '/brands/openai.svg',
  openai: '/brands/openai.svg',
  openrouter: '/brands/openrouter.svg',
  local: '/marketing/static/igx-logo.svg',
}

export default function AuthGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<{
    openai_key_configured: boolean
    openrouter_key_configured?: boolean
    chatgpt_key_configured?: boolean
    llm_key_configured?: boolean
    docker_available: boolean
  } | null>(null)
  const [checkingStatus, setCheckingStatus] = useState(false)
  const [statusErr, setStatusErr] = useState<string | null>(null)
  const [setupActive, setSetupActive] = useState(false)
  const [setupStep, setSetupStep] = useState<'llm' | 'docker'>('llm')
  const [llmProvider, setLlmProvider] = useState<'openai' | 'openrouter' | 'local' | 'chatgpt'>('chatgpt')
  const [llmKey, setLlmKey] = useState('')
  const [setupErr, setSetupErr] = useState<string | null>(null)
  const [setupBusy, setSetupBusy] = useState(false)
  const [pullingImage, setPullingImage] = useState(false)
  const [pullMessage, setPullMessage] = useState<string | null>(null)
  const [options, setOptions] = useState<Options | null>(null)
  const [localModelId, setLocalModelId] = useState('')
  const [localCustomUrl, setLocalCustomUrl] = useState('')
  const [localCustomName, setLocalCustomName] = useState('')
  const [localStatus, setLocalStatus] = useState<any>(null)
  const [chatgptAuthState, setChatgptAuthState] = useState<string | null>(null)
  const [chatgptAuthErr, setChatgptAuthErr] = useState<string | null>(null)
  const [chatgptAuthBusy, setChatgptAuthBusy] = useState(false)
  const [chatgptAuthUrl, setChatgptAuthUrl] = useState<string | null>(null)

  const localModels: LocalModel[] = options?.local_models || []
  const selectedLocalModel = localModels.find((m) => m.id === localModelId) || null

  useEffect(() => {
    let canceled = false
    async function loadStatus() {
      setCheckingStatus(true)
      setStatusErr(null)
      try {
        const res = await fetch(`${BACKEND_URL}/api/status`, {
          method: 'GET',
          headers: { ...authHeader() },
        })
        if (res.status === 401) {
          throw new Error('Admin authentication is enabled for this server. Disable it to continue.')
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        if (canceled) return
        setStatus(data)
        const llmReady = Boolean(data.llm_key_configured ?? data.openai_key_configured)
        if (!llmReady) {
          setSetupActive(true)
          setSetupStep('llm')
        } else if (setupActive && setupStep === 'llm') {
          setSetupStep('docker')
        }
      } catch (e: any) {
        if (!canceled) {
          setStatus(null)
          setStatusErr(String(e?.message || 'Unable to reach backend.'))
        }
      } finally {
        if (!canceled) setCheckingStatus(false)
      }
    }
    void loadStatus()
    return () => {
      canceled = true
    }
  }, [setupActive, setupStep])

  useEffect(() => {
    let canceled = false
    async function loadOptions() {
      try {
        const res = await apiGet<Options>('/api/options')
        if (canceled) return
        setOptions(res)
        if (res.default_llm_provider && llmProvider === 'openai') {
          setLlmProvider(res.default_llm_provider as any)
        }
        if (!localModelId) {
          const rec = res.local_models?.find((m) => m.recommended) || res.local_models?.[0]
          if (rec) setLocalModelId(rec.id)
          else setLocalModelId('__custom__')
        }
      } catch {
        if (!canceled) setOptions(null)
      }
    }
    void loadOptions()
    return () => {
      canceled = true
    }
  }, [])

  useEffect(() => {
    setChatgptAuthErr(null)
    setChatgptAuthState(null)
    setChatgptAuthUrl(null)
  }, [llmProvider])

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
          await refreshStatus()
          if (!canceled) setSetupStep('docker')
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

  useEffect(() => {
    if (!setupActive || setupStep !== 'llm' || llmProvider !== 'chatgpt' || chatgptAuthState) return
    let canceled = false
    async function pollStatus() {
      try {
        const res = await apiGet<any>('/api/status')
        if (canceled) return
        if (res?.chatgpt_key_configured) {
          await refreshStatus()
        }
      } catch {
        // ignore
      }
    }
    void pollStatus()
    const t = window.setInterval(() => void pollStatus(), 2000)
    return () => {
      canceled = true
      window.clearInterval(t)
    }
  }, [setupActive, setupStep, llmProvider, chatgptAuthState])

  useEffect(() => {
    if (!setupActive || llmProvider !== 'local') return
    let canceled = false
    async function poll() {
      try {
        const res = await apiGet<any>('/api/local/status')
        if (!canceled) setLocalStatus(res)
        if (res?.state === 'ready') {
          await refreshStatus()
          if (!canceled) setSetupStep('docker')
        }
      } catch {
        if (!canceled) setLocalStatus(null)
      }
    }
    void poll()
    const t = window.setInterval(() => {
      void poll()
    }, 1200)
    return () => {
      canceled = true
      window.clearInterval(t)
    }
  }, [setupActive, llmProvider])

  if (checkingStatus && !status) {
    return (
      <div className="authWrap">
        <div className="authCard">
          <div className="authTitle">GravexStudio</div>
          <div className="muted">Checking setup…</div>
        </div>
      </div>
    )
  }

  if (statusErr && !setupActive) {
    return (
      <div className="authWrap">
        <div className="authCard">
          <div className="authTitle">GravexStudio</div>
          <div className="alert">{statusErr}</div>
          <div className="row gap" style={{ marginTop: 12 }}>
            <button className="btn primary" onClick={() => void refreshStatus()} disabled={checkingStatus}>
              {checkingStatus ? 'Retrying…' : 'Retry'}
            </button>
          </div>
        </div>
      </div>
    )
  }

  async function saveLlmKey() {
    setSetupErr(null)
    if (!llmKey.trim()) {
      setSetupErr('Please enter an API key or OAuth token.')
      return
    }
    setSetupBusy(true)
    try {
      await apiPost('/api/keys', { provider: llmProvider, name: 'default', secret: llmKey.trim() })
      setLlmKey('')
      const res = await fetch(`${BACKEND_URL}/api/status`, {
        method: 'GET',
        headers: { ...authHeader() },
      })
      if (res.ok) {
        const data = await res.json()
        setStatus(data)
        setSetupStep('docker')
      }
    } catch (e: any) {
      setSetupErr(e?.message || 'Failed to save API key.')
    } finally {
      setSetupBusy(false)
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
      if (!opened) {
        setChatgptAuthErr('Popup blocked. Click "Open login" to continue.')
      }
    } catch (e: any) {
      setChatgptAuthErr(String(e?.message || e))
    } finally {
      setChatgptAuthBusy(false)
    }
  }

  async function startLocalSetup() {
    setSetupErr(null)
    const usingCustom = localModelId === '__custom__'
    if (usingCustom && !localCustomUrl.trim()) {
      setSetupErr('Please enter a model download URL.')
      return
    }
    if (!usingCustom && !localModelId.trim()) {
      setSetupErr('Please choose a model.')
      return
    }
    setSetupBusy(true)
    try {
      await apiPost('/api/local/setup', {
        model_id: usingCustom ? (localCustomName.trim() || '') : localModelId.trim(),
        custom_url: usingCustom ? localCustomUrl.trim() : '',
        custom_name: usingCustom ? localCustomName.trim() : '',
      })
    } catch (e: any) {
      setSetupErr(e?.message || 'Failed to start local setup.')
    } finally {
      setSetupBusy(false)
    }
  }

  async function refreshStatus() {
    setCheckingStatus(true)
    try {
      const res = await fetch(`${BACKEND_URL}/api/status`, {
        method: 'GET',
        headers: { ...authHeader() },
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setStatus(data)
      setStatusErr(null)
      const llmReady = Boolean(data.llm_key_configured ?? data.openai_key_configured)
      if (llmReady && setupActive && setupStep === 'llm') {
        setSetupStep('docker')
      }
    } catch {
      setStatus(null)
      setStatusErr('Unable to reach backend.')
    } finally {
      setCheckingStatus(false)
    }
  }

  async function pullDataAgentImage() {
    if (!status?.docker_available) {
      setSetupErr('Docker not detected. Install Docker to download the Isolated Workspace image.')
      return
    }
    setSetupErr(null)
    setPullMessage(null)
    setPullingImage(true)
    try {
      const res: any = await apiPost('/api/data-agent/pull-image', {})
      const image = (res && typeof res.image === 'string' && res.image.trim()) ? res.image.trim() : 'Isolated Workspace image'
      setPullMessage(`${image} is ready.`)
    } catch (e: any) {
      setSetupErr(e?.message || 'Failed to download Isolated Workspace image.')
    } finally {
      setPullingImage(false)
    }
  }

  if (setupActive) {
    const stepIndex = setupStep === 'llm' ? 1 : 2
    const usingLocal = llmProvider === 'local'
    const usingChatgpt = llmProvider === 'chatgpt'
    const showCustom = localModelId === '__custom__'
    const progress = localStatus?.percent ? `${localStatus.percent}%` : ''
    const providerIcon = PROVIDER_ICON_URL[llmProvider]
    return (
      <div className="authWrap">
        <div className="authCard">
          <div className="authTitle">First‑time setup</div>
          <div className="authSubtitle">Step {stepIndex} of 2</div>
          <div className="setupSteps">
            <div className={`setupStep ${stepIndex >= 1 ? 'active' : ''}`} />
            <div className={`setupStep ${stepIndex >= 2 ? 'active' : ''}`} />
          </div>
          {setupErr ? <div className="alert">{setupErr}</div> : null}
          {setupStep === 'llm' ? (
            <>
              <div className="setupHeading">LLM setup</div>
              <div className="formRow">
                <label>Provider</label>
                <div className="setupProviderSelect">
                  <img src={providerIcon} alt="" aria-hidden="true" className="setupProviderLogoInline" loading="lazy" />
                  <select value={llmProvider} onChange={(e) => setLlmProvider(e.target.value as any)}>
                    <option value="chatgpt">{formatProviderLabel('chatgpt')}</option>
                    <option value="openai">{formatProviderLabel('openai')}</option>
                    <option value="openrouter">{formatProviderLabel('openrouter')}</option>
                    <option value="local">{formatProviderLabel('local')} model (no API key)</option>
                  </select>
                </div>
              </div>
              {!usingLocal ? (
                usingChatgpt ? (
                  <>
                    {chatgptAuthErr ? <div className="alert">{chatgptAuthErr}</div> : null}
                    <div className="muted" style={{ marginBottom: 12 }}>
                      Sign in with your ChatGPT account to enable personal-use Codex calls. We’ll open a browser window for approval.
                    </div>
                    <div className="muted" style={{ marginBottom: 12 }}>
                      Voice features (ASR/TTS) still require an OpenAI API key.
                    </div>
                    <div className="row gap">
                      <button className="btn primary" onClick={() => void startChatgptAuth()} disabled={chatgptAuthBusy || !!chatgptAuthState}>
                        <img src={PROVIDER_ICON_URL.chatgpt} alt="" aria-hidden="true" className="setupActionIcon" loading="lazy" />
                        {chatgptAuthBusy ? 'Starting…' : chatgptAuthState ? 'Waiting for approval…' : 'Sign in with ChatGPT'}
                      </button>
                      {chatgptAuthUrl ? (
                        <button className="btn" onClick={() => window.open(chatgptAuthUrl, '_blank', 'noopener,noreferrer')}>
                          Open login
                        </button>
                      ) : null}
                    </div>
                    {chatgptAuthState ? <div className="muted" style={{ marginTop: 8 }}>Waiting for approval…</div> : null}
                  </>
                ) : (
                  <>
                    <div className="formRow">
                      <label>{llmProvider === 'openai' ? 'OpenAI API key' : 'OpenRouter API key'}</label>
                      <input
                        type="password"
                        placeholder={llmProvider === 'openai' ? 'sk-...' : 'sk-or-...'}
                        value={llmKey}
                        onChange={(e) => setLlmKey(e.target.value)}
                      />
                    </div>
                    <button className="btn primary" onClick={() => void saveLlmKey()} disabled={setupBusy || !llmKey.trim()}>
                      {setupBusy ? 'Saving…' : 'Save and continue'}
                    </button>
                  </>
                )
              ) : (
                <>
                  <div className="formRow">
                    <label>Local model</label>
                    <select value={localModelId} onChange={(e) => setLocalModelId(e.target.value)}>
                      {localModels.length ? (
                        <>
                          <optgroup label="Recommended">
                            {localModels.filter((m) => m.recommended).map((m) => (
                              <option key={`rec-${m.id}`} value={m.id}>
                                {m.name}
                              </option>
                            ))}
                          </optgroup>
                          <optgroup label="All compatible">
                            {localModels.map((m) => (
                              <option key={m.id} value={m.id}>
                                {m.name}
                              </option>
                            ))}
                          </optgroup>
                        </>
                      ) : null}
                      <option value="__custom__">Custom download URL…</option>
                    </select>
                    <div className="muted" style={{ marginTop: 6 }}>
                      {formatLocalModelToolSupport(localModelId === '__custom__' ? null : selectedLocalModel)}
                    </div>
                  </div>
                  {showCustom ? (
                    <>
                      <div className="formRow">
                        <label>Model URL</label>
                        <input
                          type="text"
                          placeholder="https://huggingface.co/.../resolve/main/model.gguf"
                          value={localCustomUrl}
                          onChange={(e) => setLocalCustomUrl(e.target.value)}
                        />
                      </div>
                      <div className="formRow">
                        <label>Model name (optional)</label>
                        <input
                          type="text"
                          placeholder="My GGUF model"
                          value={localCustomName}
                          onChange={(e) => setLocalCustomName(e.target.value)}
                        />
                      </div>
                    </>
                  ) : null}
                  {localStatus?.error ? <div className="alert">{localStatus.error}</div> : null}
                  {localStatus ? (
                    <div className="muted" style={{ marginBottom: 10 }}>
                      {localStatus.message || 'Preparing local model...'} {progress}
                    </div>
                  ) : null}
                  <button className="btn primary" onClick={() => void startLocalSetup()} disabled={setupBusy}>
                    {setupBusy ? 'Starting…' : 'Start local setup'}
                  </button>
                </>
              )}
            </>
          ) : null}
          {setupStep === 'docker' ? (
            <>
              <div className="setupHeading">Isolated Workspace (optional)</div>
              <div className="muted" style={{ marginBottom: 10 }}>
                Docker is required to run the Isolated Workspace. {status?.docker_available ? 'Docker detected.' : 'Docker not detected.'}
              </div>
              <div className="muted" style={{ marginBottom: 10 }}>
                The first run will build a local container image.
              </div>
              <div className="muted" style={{ marginBottom: 12 }}>
                You can build it now to avoid a wait later.
              </div>
              {pullMessage ? <div className="alert" style={{ borderColor: 'rgba(80, 200, 160, 0.4)', background: 'rgba(80, 200, 160, 0.1)' }}>{pullMessage}</div> : null}
              {!status?.docker_available ? (
                <>
                  <div className="muted" style={{ marginBottom: 8 }}>
                    Install Docker Desktop (macOS) or Docker Engine (Linux) to enable it.
                  </div>
                  <a
                    className="setupDockerLink"
                    href="https://www.docker.com/products/docker-desktop/"
                    target="_blank"
                    rel="noreferrer"
                  >
                    Get Docker
                  </a>
                </>
              ) : null}
              <div className="row gap">
                <button className="btn" onClick={() => void pullDataAgentImage()} disabled={pullingImage || !status?.docker_available}>
                  {pullingImage ? 'Building (Docker)…' : 'Build Isolated Workspace (Docker)'}
                </button>
                <button className="btn" onClick={() => void refreshStatus()} disabled={checkingStatus}>
                  {checkingStatus ? 'Checking…' : 'Recheck'}
                </button>
                <button className="btn primary" onClick={() => setSetupActive(false)}>
                  Finish setup
                </button>
              </div>
            </>
          ) : null}
          <div className="authNote">Keys are encrypted at rest. You can change them later in Keys.</div>
        </div>
      </div>
    )
  }

  return <>{children}</>
}
