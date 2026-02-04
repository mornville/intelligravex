import type { ReactNode } from 'react'
import { useEffect, useState } from 'react'
import { apiPost, getBackendUrl } from '../api/client'
import { authHeader } from '../auth'

export default function AuthGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<{ openai_key_configured: boolean; docker_available: boolean } | null>(null)
  const [checkingStatus, setCheckingStatus] = useState(false)
  const [statusErr, setStatusErr] = useState<string | null>(null)
  const [setupActive, setSetupActive] = useState(false)
  const [setupStep, setSetupStep] = useState<'openai' | 'docker'>('openai')
  const [openaiKey, setOpenaiKey] = useState('')
  const [setupErr, setSetupErr] = useState<string | null>(null)
  const [setupBusy, setSetupBusy] = useState(false)
  const [pullingImage, setPullingImage] = useState(false)
  const [pullMessage, setPullMessage] = useState<string | null>(null)

  useEffect(() => {
    let canceled = false
    async function loadStatus() {
      setCheckingStatus(true)
      setStatusErr(null)
      try {
        const res = await fetch(`${getBackendUrl()}/api/status`, {
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
        if (!data.openai_key_configured) {
          setSetupActive(true)
          setSetupStep('openai')
        } else if (setupActive && setupStep === 'openai') {
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

  async function saveOpenAIKey() {
    setSetupErr(null)
    if (!openaiKey.trim()) {
      setSetupErr('Please enter an OpenAI API key.')
      return
    }
    setSetupBusy(true)
    try {
      await apiPost('/api/keys', { provider: 'openai', name: 'default', secret: openaiKey.trim() })
      setOpenaiKey('')
      const res = await fetch(`${getBackendUrl()}/api/status`, {
        method: 'GET',
        headers: { ...authHeader() },
      })
      if (res.ok) {
        const data = await res.json()
        setStatus(data)
        setSetupStep('docker')
      }
    } catch (e: any) {
      setSetupErr(e?.message || 'Failed to save OpenAI key.')
    } finally {
      setSetupBusy(false)
    }
  }

  async function refreshStatus() {
    setCheckingStatus(true)
    try {
      const res = await fetch(`${getBackendUrl()}/api/status`, {
        method: 'GET',
        headers: { ...authHeader() },
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setStatus(data)
      setStatusErr(null)
    } catch {
      setStatus(null)
      setStatusErr('Unable to reach backend.')
    } finally {
      setCheckingStatus(false)
    }
  }

  async function pullDataAgentImage() {
    if (!status?.docker_available) {
      setSetupErr('Docker not detected. Install Docker to download the Data Agent image.')
      return
    }
    setSetupErr(null)
    setPullMessage(null)
    setPullingImage(true)
    try {
      const res: any = await apiPost('/api/data-agent/pull-image', {})
      const image = (res && typeof res.image === 'string' && res.image.trim()) ? res.image.trim() : 'Data Agent image'
      setPullMessage(`${image} is ready.`)
    } catch (e: any) {
      setSetupErr(e?.message || 'Failed to download Data Agent image.')
    } finally {
      setPullingImage(false)
    }
  }

  if (setupActive) {
    const stepIndex = setupStep === 'openai' ? 1 : 2
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
          {setupStep === 'openai' ? (
            <>
              <div className="setupHeading">OpenAI key (required)</div>
              <div className="formRow">
                <label>OpenAI API key</label>
                <input
                  type="password"
                  placeholder="sk-..."
                  value={openaiKey}
                  onChange={(e) => setOpenaiKey(e.target.value)}
                />
              </div>
              <button className="btn primary" onClick={() => void saveOpenAIKey()} disabled={setupBusy || !openaiKey.trim()}>
                {setupBusy ? 'Saving…' : 'Save and continue'}
              </button>
            </>
          ) : null}
          {setupStep === 'docker' ? (
            <>
              <div className="setupHeading">Data Agent (optional)</div>
              <div className="muted" style={{ marginBottom: 10 }}>
                Docker is required to run the Data Agent. {status?.docker_available ? 'Docker detected.' : 'Docker not detected.'}
              </div>
              <div className="muted" style={{ marginBottom: 10 }}>
                The first run will download a prebuilt container image.
              </div>
              <div className="muted" style={{ marginBottom: 12 }}>
                You can pre-download it now to avoid a wait later.
              </div>
              {pullMessage ? <div className="alert" style={{ borderColor: 'rgba(80, 200, 160, 0.4)', background: 'rgba(80, 200, 160, 0.1)' }}>{pullMessage}</div> : null}
              {!status?.docker_available ? (
                <div className="muted" style={{ marginBottom: 12 }}>
                  Install Docker Desktop (macOS) or Docker Engine (Linux) to enable it.
                </div>
              ) : null}
              <div className="row gap">
                <button className="btn" onClick={() => void pullDataAgentImage()} disabled={pullingImage || !status?.docker_available}>
                  {pullingImage ? 'Downloading…' : 'Download Data Agent'}
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
