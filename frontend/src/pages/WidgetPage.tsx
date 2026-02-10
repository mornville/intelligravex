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
  openrouter_key_configured?: boolean
  llm_key_configured?: boolean
}

type WidgetMessage = {
  id: string
  role: string
  content: string | null
  created_at: string
  sender_name?: string | null
}

type Stage = 'disconnected' | 'idle' | 'init' | 'recording' | 'asr' | 'llm' | 'tts' | 'error'
type WidgetMode = 'mic' | 'text'

const CONVERSATION_STORAGE_PREFIX = 'igx_widget_conversation_'

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

function conversationStorageKey(botId: string): string {
  return `${CONVERSATION_STORAGE_PREFIX}${botId}`
}

function readStoredConversation(botId: string): string | null {
  try {
    return localStorage.getItem(conversationStorageKey(botId))
  } catch {
    return null
  }
}

function writeStoredConversation(botId: string, conversationId: string | null) {
  try {
    const key = conversationStorageKey(botId)
    if (conversationId) {
      localStorage.setItem(key, conversationId)
    } else {
      localStorage.removeItem(key)
    }
  } catch {
    // ignore storage failures
  }
}

export default function WidgetPage() {
  const [status, setStatus] = useState<WidgetStatus | null>(null)
  const [statusErr, setStatusErr] = useState<string | null>(null)
  const [botId, setBotId] = useState<string | null>(null)
  const [botName, setBotName] = useState<string | null>(null)
  const [stage, setStage] = useState<Stage>('disconnected')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [messages, setMessages] = useState<WidgetMessage[]>([])
  const [autoScroll, setAutoScroll] = useState(true)
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
  const messagesRef = useRef<WidgetMessage[]>([])
  const lastMessageAtRef = useRef<string | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const resumeAttemptRef = useRef<string | null>(null)

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
    messagesRef.current = messages
    const last = messages[messages.length - 1]
    lastMessageAtRef.current = last?.created_at || null
  }, [messages])

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
    const storedConversationId = readStoredConversation(botId)
    setConversationId(storedConversationId)
    setMessages([])
    setAssistantText('')
    setErr(null)
    const token = getBasicAuthToken()
    const authQuery = token ? `?auth=${encodeURIComponent(token)}` : ''
    const ws = new WebSocket(`${wsBase()}/ws/bots/${botId}/talk${authQuery}`)
    wsRef.current = ws
    playerRef.current?.close()
    playerRef.current = new WavQueuePlayer()
    ws.onopen = () => {
      setStage('idle')
      initConversation(storedConversationId)
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
        if (id) {
          setConversationId(id)
          if (botId) writeStoredConversation(botId, id)
        }
        resumeAttemptRef.current = null
        return
      }
      if (msg.type === 'text_delta') {
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
        const reqId = String(msg.req_id || '')
        if (activeTextReqIdRef.current && reqId && reqId !== activeTextReqIdRef.current) return
        const text = String(msg.text || '')
        if (text.trim()) {
          const shouldReplace = interimShownRef.current
          setAssistantText((prev) => (shouldReplace || text.length >= prev.length ? text : prev))
        }
        interimShownRef.current = false
        activeTextReqIdRef.current = null
        void refreshMessages()
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
        const errorText = String(msg.error || 'Unknown error')
        setErr(errorText)
        if (resumeAttemptRef.current && botId && /conversation/i.test(errorText)) {
          writeStoredConversation(botId, null)
          resumeAttemptRef.current = null
          setConversationId(null)
          setMessages([])
          initConversation(null)
        }
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

  function initConversation(preferredConversationId: string | null = null) {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    const reqId = makeId()
    activeReqIdRef.current = reqId
    const speak = widgetModeRef.current === 'mic'
    if (!speak) setAssistantText('')
    if (preferredConversationId) {
      resumeAttemptRef.current = preferredConversationId
    } else {
      resumeAttemptRef.current = null
    }
    ws.send(
      JSON.stringify({
        type: 'init',
        req_id: reqId,
        speak,
        test_flag: false,
        debug: false,
        conversation_id: preferredConversationId || undefined,
      }),
    )
  }

  async function loadMessages(opts: { reset?: boolean; since?: string | null } = {}) {
    if (!conversationId) return
    const params = new URLSearchParams()
    params.set('order', 'asc')
    params.set('limit', '200')
    if (opts.since) params.set('since', opts.since)
    try {
      const res = await apiGet<{ messages?: WidgetMessage[] }>(
        `/api/conversations/${conversationId}/messages?${params.toString()}`,
      )
      const incoming = Array.isArray(res?.messages) ? res.messages : []
      setMessages((prev) => {
        if (opts.reset) return incoming
        const seen = new Set(prev.map((m) => m.id))
        const next = [...prev]
        for (const msg of incoming) {
          if (!seen.has(msg.id)) {
            next.push(msg)
            seen.add(msg.id)
          }
        }
        return next
      })
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  async function refreshMessages() {
    await loadMessages({ since: lastMessageAtRef.current })
    setAssistantText('')
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
      initConversation(botId ? readStoredConversation(botId) : null)
      return
    }
    if (!status?.openai_key_configured) {
      setErr('OpenAI key not configured (required for voice).')
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
    const llmReady = Boolean(status?.llm_key_configured ?? status?.openai_key_configured)
    if (!llmReady) {
      setErr('LLM key not configured.')
      return
    }
    if (!conversationId) {
      setErr('Connecting...')
      initConversation(botId ? readStoredConversation(botId) : null)
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
    setMessages([])
    interimShownRef.current = false
    activeTextReqIdRef.current = null
    setConversationId(null)
    if (botId) writeStoredConversation(botId, null)
    initConversation(null)
  }

  useEffect(() => {
    if (!conversationId) {
      setMessages([])
      setAssistantText('')
      return
    }
    setAutoScroll(true)
    void loadMessages({ reset: true })
    const interval = window.setInterval(() => {
      void loadMessages({ since: lastMessageAtRef.current })
    }, 1500)
    return () => {
      window.clearInterval(interval)
    }
  }, [conversationId])

  useEffect(() => {
    if (!autoScroll) return
    const el = scrollRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [messages, assistantText, autoScroll])

  function handleScroll() {
    const el = scrollRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 12
    setAutoScroll(atBottom)
  }

  function renderMessage(msg: WidgetMessage) {
    const role = msg.role === 'assistant' ? 'assistant' : 'user'
    const content = String(msg.content || '').trim()
    if (!content) return null
    return (
      <div key={msg.id} className={`widgetMessage ${role}`}>
        {content}
      </div>
    )
  }

  const isTextMode = widgetMode === 'text'
  const llmReady = Boolean(status?.llm_key_configured ?? status?.openai_key_configured)
  const canRecord = Boolean(status?.openai_key_configured && conversationId && stage === 'idle')
  const canSendText = Boolean(llmReady && conversationId && stage === 'idle')
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
                  <div className="widgetConversation" onScroll={handleScroll} ref={scrollRef}>
                    {messages.length === 0 && !assistantText ? (
                      <div className="widgetEmpty">No messages yet.</div>
                    ) : null}
                    {messages.map((msg) => renderMessage(msg))}
                    {assistantText ? <div className="widgetMessage assistant pending">{assistantText}</div> : null}
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
                <div className={textClass}>
                  <div className="widgetConversation" onScroll={handleScroll} ref={scrollRef}>
                    {messages.length === 0 && !assistantText ? (
                      <div className="widgetEmpty">No messages yet.</div>
                    ) : null}
                    {messages.map((msg) => renderMessage(msg))}
                    {assistantText ? <div className="widgetMessage assistant pending">{assistantText}</div> : null}
                  </div>
                  <button
                    className={micClass}
                    onClick={() => void (recording ? stopRecording() : startRecording())}
                    disabled={!canRecord && !recording}
                  >
                    <MicrophoneIcon />
                  </button>
                </div>
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
            {isTextMode ? (
              !llmReady ? <div className="widgetHint">Open dashboard to add an LLM key.</div> : null
            ) : (
              !status?.openai_key_configured ? (
                <div className="widgetHint">Open dashboard to add your OpenAI key for voice.</div>
              ) : null
            )}
            {statusErr ? <div className="widgetError">{statusErr}</div> : null}
            {err ? <div className="widgetError">{err}</div> : null}
          </>
        )}
      </div>
    </div>
  )
}
