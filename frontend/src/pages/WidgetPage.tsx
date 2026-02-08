import { useEffect, useRef, useState } from 'react'
import { apiGet, apiPost, BACKEND_URL } from '../api/client'
import { getBasicAuthToken } from '../auth'
import { createRecorder, type Recorder } from '../audio/recorder'
import { WavQueuePlayer } from '../audio/player'
import LoadingSpinner from '../components/LoadingSpinner'
import type { WidgetConfig } from '../types'
import { Cog6ToothIcon, MicrophoneIcon } from '@heroicons/react/24/solid'

type WidgetStatus = {
  openai_key_configured: boolean
}

type Stage = 'disconnected' | 'idle' | 'init' | 'recording' | 'asr' | 'llm' | 'tts' | 'error'
type WidgetMode = 'mic' | 'text'

function wsBase(): string {
  const u = new URL(BACKEND_URL)
  const proto = u.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${u.host}`
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `id_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`
}

function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i)
  return out
}

export default function WidgetPage() {
  const [status, setStatus] = useState<WidgetStatus | null>(null)
  const [statusErr, setStatusErr] = useState<string | null>(null)
  const [botId, setBotId] = useState<string | null>(null)
  const [botName, setBotName] = useState<string | null>(null)
  const [stage, setStage] = useState<Stage>('disconnected')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [recording, setRecording] = useState(false)
  const [widgetMode, setWidgetMode] = useState<WidgetMode>('mic')
  const [assistantText, setAssistantText] = useState('')
  const [textInput, setTextInput] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [menuOpen, setMenuOpen] = useState(false)
  const [loading, setLoading] = useState(true)
  const [modeSaving, setModeSaving] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const recorderRef = useRef<Recorder | null>(null)
  const playerRef = useRef<WavQueuePlayer | null>(null)
  const recordingRef = useRef(false)
  const activeReqIdRef = useRef<string | null>(null)
  const activeTextReqIdRef = useRef<string | null>(null)
  const interimShownRef = useRef(false)
  const widgetModeRef = useRef<WidgetMode>('mic')

  useEffect(() => {
    document.body.classList.add('widgetBody')
    return () => {
      document.body.classList.remove('widgetBody')
    }
  }, [])

  useEffect(() => {
    widgetModeRef.current = widgetMode
  }, [widgetMode])

  useEffect(() => {
    return () => {
      recordingRef.current = false
      void recorderRef.current?.close()
    }
  }, [])

  useEffect(() => {
    let active = true
    void (async () => {
      setLoading(true)
      setStatusErr(null)
      try {
        const [s, widget] = await Promise.all([
          apiGet<WidgetStatus>('/api/status'),
          apiGet<WidgetConfig>('/api/widget-config'),
        ])
        if (!active) return
        setStatus(s)
        let nextBotId = widget?.bot_id || null
        let nextBotName = widget?.bot_name || null
        const nextMode: WidgetMode = widget?.widget_mode === 'text' ? 'text' : 'mic'
        if (!nextBotId) {
          const sys = await apiGet<{ id: string; name: string }>('/api/system-bot')
          nextBotId = sys.id
          nextBotName = sys.name
        }
        setBotId(nextBotId)
        setBotName(nextBotName)
        setWidgetMode(nextMode)
      } catch (e: any) {
        if (!active) return
        setStatusErr(String(e?.message || e))
      } finally {
        if (active) setLoading(false)
      }
    })()
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!botId) return
    setStage('disconnected')
    setConversationId(null)
    setErr(null)
    const token = getBasicAuthToken()
    const authQuery = token ? `?auth=${encodeURIComponent(token)}` : ''
    const ws = new WebSocket(`${wsBase()}/ws/bots/${botId}/talk${authQuery}`)
    wsRef.current = ws
    playerRef.current?.close()
    playerRef.current = new WavQueuePlayer()
    ws.onopen = () => {
      setStage('idle')
      initConversation()
    }
    ws.onclose = () => {
      setStage('disconnected')
    }
    ws.onerror = () => {
      setStage('error')
      setErr('WebSocket error')
    }
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return
      let msg: any
      try {
        msg = JSON.parse(ev.data)
      } catch {
        return
      }
      if (msg.type === 'status') {
        setStage(msg.stage || 'idle')
        return
      }
      if (msg.type === 'conversation') {
        const id = String(msg.conversation_id || msg.id || '')
        if (id) setConversationId(id)
        return
      }
      if (msg.type === 'text_delta') {
        if (widgetModeRef.current !== 'text') return
        const delta = String(msg.delta || '')
        if (!delta) return
        const reqId = String(msg.req_id || '')
        if (activeTextReqIdRef.current && reqId && reqId !== activeTextReqIdRef.current) return
        if (!activeTextReqIdRef.current && reqId) activeTextReqIdRef.current = reqId
        setAssistantText((prev) => prev + delta)
        return
      }
      if (msg.type === 'interim') {
        if (widgetModeRef.current !== 'text') return
        const text = String(msg.text || '').trim()
        if (!text) return
        const reqId = String(msg.req_id || '')
        if (activeTextReqIdRef.current && reqId && reqId !== activeTextReqIdRef.current) return
        if (!activeTextReqIdRef.current && reqId) activeTextReqIdRef.current = reqId
        interimShownRef.current = true
        setAssistantText((prev) => (prev ? `${prev}\n${text}` : text))
        return
      }
      if (msg.type === 'done') {
        if (widgetModeRef.current !== 'text') return
        const reqId = String(msg.req_id || '')
        if (activeTextReqIdRef.current && reqId && reqId !== activeTextReqIdRef.current) return
        const text = String(msg.text || '')
        if (text.trim()) {
          const shouldReplace = interimShownRef.current
          setAssistantText((prev) => (shouldReplace || text.length >= prev.length ? text : prev))
        }
        interimShownRef.current = false
        activeTextReqIdRef.current = null
        return
      }
      if (msg.type === 'audio_wav') {
        const b64 = String(msg.wav_base64 || '')
        if (!b64) return
        const bytes = b64ToBytes(b64)
        void playerRef.current?.playWavBytes(bytes)
        return
      }
      if (msg.type === 'error') {
        setErr(String(msg.error || 'Unknown error'))
      }
    }
    return () => {
      try {
        ws.close()
      } catch {
        // ignore
      }
      wsRef.current = null
      void playerRef.current?.close()
      playerRef.current = null
    }
  }, [botId])

  function initConversation() {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    const reqId = makeId()
    activeReqIdRef.current = reqId
    const speak = widgetModeRef.current === 'mic'
    if (!speak) setAssistantText('')
    ws.send(JSON.stringify({ type: 'init', req_id: reqId, speak, test_flag: false, debug: false }))
  }

  async function startRecording() {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setErr('WebSocket not connected')
      return
    }
    if (widgetModeRef.current !== 'mic') {
      setErr('Switch to mic mode to record.')
      return
    }
    if (!conversationId) {
      setErr('Connecting...')
      initConversation()
      return
    }
    if (!status?.openai_key_configured) {
      setErr('OpenAI key not configured.')
      return
    }
    if (stage !== 'idle') return
    setErr(null)
    const reqId = makeId()
    activeReqIdRef.current = reqId
    wsRef.current.send(
      JSON.stringify({ type: 'start', req_id: reqId, conversation_id: conversationId, speak: true, test_flag: false, debug: false }),
    )
    recordingRef.current = true
    const rec = await createRecorder((pcm16) => {
      if (!recordingRef.current) return
      if (!activeReqIdRef.current) return
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) wsRef.current.send(pcm16)
    })
    recorderRef.current = rec
    await rec.start()
    setRecording(true)
  }

  async function stopRecording() {
    setRecording(false)
    recordingRef.current = false
    try {
      await recorderRef.current?.stop()
      await recorderRef.current?.close()
    } catch {
      // ignore
    } finally {
      recorderRef.current = null
    }
    const reqId = activeReqIdRef.current
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && reqId) {
      wsRef.current.send(JSON.stringify({ type: 'stop', req_id: reqId }))
    }
  }

  async function sendText() {
    const text = textInput.trim()
    if (!text) return
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setErr('WebSocket not connected')
      return
    }
    if (!status?.openai_key_configured) {
      setErr('OpenAI key not configured.')
      return
    }
    if (stage !== 'idle') return
    setErr(null)
    setTextInput('')
    setAssistantText('')
    interimShownRef.current = false
    const reqId = makeId()
    activeTextReqIdRef.current = reqId
    wsRef.current.send(
      JSON.stringify({
        type: 'chat',
        req_id: reqId,
        conversation_id: conversationId,
        speak: false,
        test_flag: false,
        debug: false,
        text,
      }),
    )
  }

  async function updateWidgetMode(nextMode: WidgetMode) {
    if (nextMode === widgetMode) return
    if (recording) {
      await stopRecording()
    }
    const prevMode = widgetMode
    setWidgetMode(nextMode)
    setModeSaving(true)
    setErr(null)
    activeTextReqIdRef.current = null
    if (nextMode === 'text') setAssistantText('')
    try {
      const res = await apiPost<WidgetConfig>('/api/widget-config', { widget_mode: nextMode })
      if (res?.bot_id) {
        setBotId(res.bot_id)
      }
      if (res?.bot_name) {
        setBotName(res.bot_name)
      }
      const mode: WidgetMode = res?.widget_mode === 'text' ? 'text' : 'mic'
      setWidgetMode(mode)
    } catch (e: any) {
      setWidgetMode(prevMode)
      setErr(String(e?.message || e))
    } finally {
      setModeSaving(false)
    }
  }

  async function openDashboard() {
    try {
      await apiPost('/api/open-dashboard', { path: '/dashboard' })
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  async function startNewConversation() {
    if (recording) {
      await stopRecording()
    }
    setErr(null)
    setAssistantText('')
    interimShownRef.current = false
    activeTextReqIdRef.current = null
    setConversationId(null)
    initConversation()
  }

  const isTextMode = widgetMode === 'text'
  const canRecord = Boolean(status?.openai_key_configured && conversationId && stage === 'idle')
  const canSendText = Boolean(status?.openai_key_configured && stage === 'idle')
  const busy = stage !== 'idle' && stage !== 'disconnected' && !recording
  const micClass = `widgetMic ${recording ? 'recording' : ''} ${busy ? 'busy' : ''}`
  const textClass = `widgetTextCard ${busy ? 'busy' : ''}`

  return (
    <div className="widgetRoot">
      <div className="widgetShell">
        {loading ? (
          <div className="widgetLoading">
            <LoadingSpinner label="Connecting" />
          </div>
        ) : (
          <>
            <div className="widgetCenter">
              {isTextMode ? (
                <div className={textClass}>
                  <div className={`widgetTextReply ${assistantText ? '' : 'empty'}`}>
                    {assistantText || 'Assistant reply will appear here.'}
                  </div>
                  <div className="widgetTextInputRow">
                    <input
                      className="widgetTextInput"
                      value={textInput}
                      onChange={(e) => setTextInput(e.target.value)}
                      placeholder={busy ? 'Thinking...' : 'Type a message'}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          e.preventDefault()
                          void sendText()
                        }
                      }}
                    />
                    <button className="widgetTextSend" onClick={() => void sendText()} disabled={!canSendText || !textInput.trim()}>
                      Send
                    </button>
                  </div>
                </div>
              ) : (
                <button className={micClass} onClick={() => void (recording ? stopRecording() : startRecording())} disabled={!canRecord && !recording}>
                  <MicrophoneIcon />
                </button>
              )}
            </div>
            <button className="widgetGear" onClick={() => setMenuOpen((v) => !v)} title="Settings">
              <Cog6ToothIcon />
            </button>
            {menuOpen ? (
              <div className="widgetMenu">
                <button className="widgetMenuItem" onClick={() => void startNewConversation()}>
                  New conversation
                </button>
                <button className="widgetMenuItem" onClick={() => void openDashboard()}>
                  Go to dashboard
                </button>
                <label className="widgetMenuToggle">
                  <span>Text mode</span>
                  <input
                    type="checkbox"
                    checked={isTextMode}
                    disabled={modeSaving}
                    onChange={(e) => {
                      const nextMode: WidgetMode = e.target.checked ? 'text' : 'mic'
                      void updateWidgetMode(nextMode)
                    }}
                  />
                  <span className="widgetSwitch" aria-hidden="true" />
                </label>
                {botName ? <div className="widgetMenuHint">Assistant: {botName}</div> : null}
              </div>
            ) : null}
            {!status?.openai_key_configured ? (
              <div className="widgetHint">Open dashboard to add your OpenAI key.</div>
            ) : null}
            {statusErr ? <div className="widgetError">{statusErr}</div> : null}
            {err ? <div className="widgetError">{err}</div> : null}
          </>
        )}
      </div>
    </div>
  )
}
