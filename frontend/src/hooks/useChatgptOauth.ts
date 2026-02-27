import { useCallback, useEffect, useState } from 'react'
import { apiGet, apiPost } from '../api/client'

type StatusPayload = {
  chatgpt_key_configured?: boolean
}

type OauthStartPayload = {
  state: string
  auth_url: string
}

type OauthStatusPayload = {
  status: 'pending' | 'ready' | 'expired' | 'error'
  error?: string
}

export function useChatgptOauth() {
  const [ready, setReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [authUrl, setAuthUrl] = useState<string | null>(null)
  const [authState, setAuthState] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      const res = await apiGet<StatusPayload>('/api/status')
      setReady(Boolean(res?.chatgpt_key_configured))
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const start = useCallback(async () => {
    setError(null)
    setBusy(true)
    try {
      const res = await apiPost<OauthStartPayload>('/api/chatgpt/oauth/start', {})
      setAuthState(res.state)
      setAuthUrl(res.auth_url)
      try {
        window.open(res.auth_url, '_blank', 'noopener,noreferrer')
      } catch {
        // ignore popup failures; user can click "Open login"
      }
    } catch (e: any) {
      setError(String(e?.message || e))
    } finally {
      setBusy(false)
    }
  }, [])

  useEffect(() => {
    if (!authState) return
    const state = authState
    let canceled = false
    async function poll() {
      try {
        const res = await apiGet<OauthStatusPayload>(`/api/chatgpt/oauth/status?state=${encodeURIComponent(state)}`)
        if (canceled) return
        if (res.status === 'ready') {
          setAuthState(null)
          setAuthUrl(null)
          setError(null)
          await refresh()
          return
        }
        if (res.status === 'error' || res.status === 'expired') {
          setAuthState(null)
          setAuthUrl(null)
          setError(res.error || (res.status === 'expired' ? 'Sign-in expired. Please try again.' : 'Sign-in failed.'))
        }
      } catch (e: any) {
        if (!canceled) setError(String(e?.message || e))
      }
    }
    void poll()
    const t = window.setInterval(() => void poll(), 1200)
    return () => {
      canceled = true
      window.clearInterval(t)
    }
  }, [authState, refresh])

  return {
    ready,
    busy,
    error,
    authUrl,
    authState,
    refresh,
    start,
  }
}
