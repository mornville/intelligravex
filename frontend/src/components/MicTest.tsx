import { useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import {
  CpuChipIcon,
  MicrophoneIcon,
  PaperClipIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  UserIcon,
  WrenchScrewdriverIcon,
} from '@heroicons/react/24/solid'
import { apiGet, BACKEND_URL } from '../api/client'
import { getBasicAuthToken } from '../auth'
import { createRecorder, type Recorder } from '../audio/recorder'
import { WavQueuePlayer } from '../audio/player'
import { fmtMs } from '../utils/format'
import type { ConversationDetail, DataAgentStatus, ConversationFiles } from '../types'
import LoadingSpinner from './LoadingSpinner'

type Stage = 'disconnected' | 'idle' | 'init' | 'recording' | 'asr' | 'llm' | 'tts' | 'error'

type Timings = Partial<{
  asr: number
  llm_ttfb: number
  llm_total: number
  tts_first_audio: number
  total: number
}>

type ChatItem = {
  id: string
  role: 'user' | 'assistant' | 'tool'
  text: string
  created_at?: string
  details?: any
  timings?: Timings
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
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

function wsBase(): string {
  const u = new URL(BACKEND_URL)
  const proto = u.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${u.host}`
}

export default function MicTest({
  botId,
  initialConversationId,
  layout = 'default',
  startToken,
  onConversationIdChange,
  onStageChange,
  hideWorkspace = false,
  allowUploads = false,
  uploadDisabledReason = '',
  uploading = false,
  onUploadFiles,
}: {
  botId: string
  initialConversationId?: string
  layout?: 'default' | 'whatsapp'
  startToken?: number
  onConversationIdChange?: (id: string | null) => void
  onStageChange?: (stage: Stage) => void
  hideWorkspace?: boolean
  allowUploads?: boolean
  uploadDisabledReason?: string
  uploading?: boolean
  onUploadFiles?: (files: FileList) => void
}) {
  const [stage, setStage] = useState<Stage>('disconnected')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [speak, setSpeak] = useState(true)
  const [testFlag, setTestFlag] = useState(true)
  const [debug, setDebug] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [containerStatus, setContainerStatus] = useState<DataAgentStatus | null>(null)
  const [containerErr, setContainerErr] = useState<string | null>(null)
  const [containerLoading, setContainerLoading] = useState(false)
  const [showFilesPane, setShowFilesPane] = useState(true)
  const [files, setFiles] = useState<ConversationFiles | null>(null)
  const [filesErr, setFilesErr] = useState<string | null>(null)
  const [filesLoading, setFilesLoading] = useState(false)
  const [filesRecursive, setFilesRecursive] = useState(true)
  const [filesHidden, setFilesHidden] = useState(false)
  const [expandedPaths, setExpandedPaths] = useState<Record<string, boolean>>({})
  const [items, setItems] = useState<ChatItem[]>([])
  const [recording, setRecording] = useState(false)
  const [lastTimings, setLastTimings] = useState<Timings>({})
  const [chatText, setChatText] = useState('')
  const [uploadMenuOpen, setUploadMenuOpen] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const wsOpenPromiseRef = useRef<Promise<boolean> | null>(null)
  const wsOpenResolveRef = useRef<((ok: boolean) => void) | null>(null)
  const recorderRef = useRef<Recorder | null>(null)
  const playerRef = useRef<WavQueuePlayer | null>(null)
  const uploadRef = useRef<HTMLInputElement | null>(null)
  const uploadFolderRef = useRef<HTMLInputElement | null>(null)
  const activeReqIdRef = useRef<string | null>(null)
  const recordingRef = useRef(false)
  const draftAssistantIdRef = useRef<string | null>(null)
  const pendingUserIdRef = useRef<string | null>(null)
  const toolProgressIdRef = useRef<Record<string, string>>({})
  const timingsByReq = useRef<Record<string, Timings>>({})
  const scrollerRef = useRef<HTMLDivElement | null>(null)
  const hydratedConvIdRef = useRef<string | null>(null)
  const ignoreInitialConversationRef = useRef(false)
  const startTokenRef = useRef<number | undefined>(startToken)

  async function ensureWsOpen(timeoutMs = 1500): Promise<boolean> {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) return true
    const p = wsOpenPromiseRef.current
    if (!p) return false
    const result = await Promise.race([
      p,
      new Promise<boolean>((resolve) => setTimeout(() => resolve(false), timeoutMs)),
    ])
    return result === true && !!wsRef.current && wsRef.current.readyState === WebSocket.OPEN
  }

  function finalizeTurn() {
    setStage('idle')
    activeReqIdRef.current = null
    draftAssistantIdRef.current = null
    toolProgressIdRef.current = {}
    if (pendingUserIdRef.current) {
      const pendingId = pendingUserIdRef.current
      setItems((prev) => prev.filter((it) => it.id !== pendingId || it.text.trim()))
      pendingUserIdRef.current = null
    }
  }

  async function hydrateConversation(cid: string) {
    if (!cid) return
    if (hydratedConvIdRef.current === cid) return
    try {
      const d = await apiGet<ConversationDetail>(`/api/conversations/${cid}`)
      hydratedConvIdRef.current = cid
      draftAssistantIdRef.current = null
      const sorted = [...d.messages].sort((a, b) => {
        const at = new Date(a.created_at).getTime()
        const bt = new Date(b.created_at).getTime()
        if (at !== bt) return at - bt
        const aRole = a.role
        const bRole = b.role
        if (aRole !== bRole) {
          const roleOrder = (r: string) => (r === 'user' ? 0 : r === 'assistant' ? 1 : r === 'tool' ? 2 : 3)
          return roleOrder(aRole) - roleOrder(bRole)
        }
        if (aRole === 'tool' && bRole === 'tool') {
          const aReq = (a.tool as any)?.req_id ?? ''
          const bReq = (b.tool as any)?.req_id ?? ''
          const aName = a.tool_name ?? ''
          const bName = b.tool_name ?? ''
          if (aReq && bReq && aReq === bReq) {
            const kindOrder = (k: string | null) => (k === 'call' ? 0 : k === 'result' ? 1 : 2)
            if (a.tool_kind !== b.tool_kind) return kindOrder(a.tool_kind) - kindOrder(b.tool_kind)
          } else if (aName && bName && aName === bName && a.tool_kind !== b.tool_kind) {
            const kindOrder = (k: string | null) => (k === 'call' ? 0 : k === 'result' ? 1 : 2)
            return kindOrder(a.tool_kind) - kindOrder(b.tool_kind)
          }
        }
        return String(a.id).localeCompare(String(b.id))
      })
      const mapped: ChatItem[] = sorted.map((m) => {
        let role: ChatItem['role'] = m.role === 'assistant' ? 'assistant' : m.role === 'tool' ? 'tool' : 'user'
        let text = m.content
        if (m.role === 'tool') {
          if (m.tool_name && m.tool_kind) text = `[tool_${m.tool_kind}] ${m.tool_name}`
          else if (m.tool_name) text = `[tool] ${m.tool_name}`
          else text = '[tool]'
        }
        return {
          id: m.id,
          role,
          text,
          created_at: m.created_at,
          details: m.role === 'tool' ? m.tool : undefined,
          timings: m.metrics
            ? {
                asr: m.metrics.asr ?? undefined,
                llm_ttfb: m.metrics.llm1 ?? undefined,
                llm_total: m.metrics.llm ?? undefined,
                tts_first_audio: m.metrics.tts1 ?? undefined,
                total: m.metrics.total ?? undefined,
              }
            : undefined,
        }
      })
      setItems(mapped)
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  async function loadContainerStatus(cid?: string) {
    const id = cid || conversationId
    if (!id) return
    setContainerLoading(true)
    setContainerErr(null)
    try {
      const d = await apiGet<DataAgentStatus>(`/api/conversations/${id}/data-agent`)
      setContainerStatus(d)
    } catch (e: any) {
      setContainerErr(String(e?.message || e))
    } finally {
      setContainerLoading(false)
    }
  }

  async function loadFiles(cid?: string, recursive?: boolean, includeHidden?: boolean) {
    const id = cid || conversationId
    if (!id) return
    const rec = recursive ?? filesRecursive
    const hidden = includeHidden ?? filesHidden
    const params = new URLSearchParams()
    if (rec) params.set('recursive', '1')
    if (hidden) params.set('include_hidden', '1')
    setFilesLoading(true)
    setFilesErr(null)
    try {
      const d = await apiGet<ConversationFiles>(`/api/conversations/${id}/files?${params.toString()}`)
      setFiles(d)
      setFilesRecursive(rec)
      setFilesHidden(hidden)
    } catch (e: any) {
      setFilesErr(String(e?.message || e))
    } finally {
      setFilesLoading(false)
    }
  }

  const canInit = useMemo(() => stage === 'idle' || stage === 'disconnected' || stage === 'error', [stage])
  const canRecord = useMemo(() => {
    if (layout === 'whatsapp') return stage === 'idle' && !!conversationId
    return speak && stage === 'idle' && !!conversationId
  }, [layout, speak, stage, conversationId])

  useEffect(() => {
    const token = getBasicAuthToken()
    const authQuery = token ? `?auth=${encodeURIComponent(token)}` : ''
    const ws = new WebSocket(`${wsBase()}/ws/bots/${botId}/talk${authQuery}`)
    wsRef.current = ws
    playerRef.current = new WavQueuePlayer()
    setErr(null)
    setStage('disconnected')
    wsOpenPromiseRef.current = new Promise((resolve) => {
      wsOpenResolveRef.current = resolve
    })

    ws.onopen = () => {
      setErr(null)
      setStage('idle')
      wsOpenResolveRef.current?.(true)
    }
    ws.onclose = () => {
      setStage('disconnected')
      wsOpenResolveRef.current?.(false)
    }
    ws.onerror = () => {
      setStage('error')
      setErr('WebSocket error')
      wsOpenResolveRef.current?.(false)
    }
    ws.onmessage = async (ev) => {
      if (typeof ev.data !== 'string') return
      let msg: any
      try {
        msg = JSON.parse(ev.data)
      } catch {
        return
      }

      if (msg.type === 'status') {
        setStage(msg.stage || 'idle')
        if (msg.stage === 'idle') {
          finalizeTurn()
        }
        return
      }
      if (msg.type === 'conversation') {
        const cid = msg.conversation_id || msg.id
        if (cid) {
          const s = String(cid)
          setConversationId(s)
          void hydrateConversation(s)
        }
        return
      }
      if (msg.type === 'metrics') {
        const reqId = String(msg.req_id || '')
        const t = (msg.timings_ms || {}) as Timings
        if (reqId) timingsByReq.current[reqId] = t
        setLastTimings(t)
        return
      }
      if (msg.type === 'asr') {
        const text = String(msg.text || '').trim()
        if (text) {
        const pendingId = pendingUserIdRef.current
        if (pendingId) {
          setItems((prev) =>
            prev.map((it) => (it.id === pendingId ? { ...it, text, created_at: new Date().toISOString() } : it)),
          )
          pendingUserIdRef.current = null
        } else {
          setItems((prev) => [...prev, { id: makeId(), role: 'user', text, created_at: new Date().toISOString() }])
        }
        }
        return
      }
      if (msg.type === 'tool_call') {
        const details = normalizeToolEventForDisplay(msg)
        const toolItem = {
          id: makeId(),
          role: 'tool' as const,
          text: `[tool_call] ${msg.name}`,
          details,
          created_at: new Date().toISOString(),
        }
        const draftId = draftAssistantIdRef.current
        setItems((prev) => {
          if (!draftId) return [...prev, toolItem]
          const idx = prev.findIndex((it) => it.id === draftId)
          if (idx === -1) return [...prev, toolItem]
          return [...prev.slice(0, idx), toolItem, ...prev.slice(idx)]
        })
        return
      }
      if (msg.type === 'tool_result') {
        const details = normalizeToolEventForDisplay(msg)
        const toolItem = {
          id: makeId(),
          role: 'tool' as const,
          text: `[tool_result] ${msg.name}`,
          details,
          created_at: new Date().toISOString(),
        }
        const draftId = draftAssistantIdRef.current
        setItems((prev) => {
          if (!draftId) return [...prev, toolItem]
          const idx = prev.findIndex((it) => it.id === draftId)
          if (idx === -1) return [...prev, toolItem]
          return [...prev.slice(0, idx), toolItem, ...prev.slice(idx)]
        })
        return
      }
      if (msg.type === 'tool_progress') {
        const text = String(msg.text || '').trim()
        if (!text) return
        const key = `${String(msg.req_id || '')}:${String(msg.name || 'tool')}`
        if (!toolProgressIdRef.current[key]) toolProgressIdRef.current[key] = makeId()
        const toolId = toolProgressIdRef.current[key]
        setItems((prev) => {
          const draftId = draftAssistantIdRef.current
          const hasItem = prev.some((it) => it.id === toolId)
          if (!hasItem) {
            const toolItem = {
              id: toolId,
              role: 'tool' as const,
              text: `[tool_progress] ${text}`,
              created_at: new Date().toISOString(),
            }
            if (!draftId) return [...prev, toolItem]
            const idx = prev.findIndex((it) => it.id === draftId)
            if (idx === -1) return [...prev, toolItem]
            return [...prev.slice(0, idx), toolItem, ...prev.slice(idx)]
          }
          return prev.map((it) => (it.id === toolId ? { ...it, text: `${it.text}\n${text}` } : it))
        })
        return
      }
      if (msg.type === 'interim') {
        const text = String(msg.text || '').trim()
        if (!text) return
        setItems((prev) => [...prev, { id: makeId(), role: 'assistant', text, created_at: new Date().toISOString() }])
        return
      }
      if (msg.type === 'text_delta') {
        const delta = String(msg.delta || '')
        if (!delta) return
        if (!draftAssistantIdRef.current) draftAssistantIdRef.current = makeId()
        const draftId = draftAssistantIdRef.current
        setItems((prev) => {
          const hasDraft = prev.some((it) => it.id === draftId)
          if (!hasDraft)
            return [...prev, { id: draftId, role: 'assistant', text: delta, created_at: new Date().toISOString() }]
          return prev.map((it) => (it.id === draftId ? { ...it, text: it.text + delta } : it))
        })
        return
      }
      if (msg.type === 'audio_wav') {
        try {
          const bytes = b64ToBytes(String(msg.wav_base64 || ''))
          await playerRef.current?.playWavBytes(bytes)
        } catch (e: any) {
          setErr(`Audio playback failed: ${String(e?.message || e)}`)
        }
        return
      }
      if (msg.type === 'error') {
        const errorText = String(msg.error || 'Unknown error')
        setErr(errorText)
        setStage('error')
        setItems((prev) => [
          ...prev,
          { id: makeId(), role: 'assistant', text: `Error: ${errorText}`, created_at: new Date().toISOString() },
        ])
        return
      }
      if (msg.type === 'done') {
        const reqId = String(msg.req_id || '')
        const t = reqId ? timingsByReq.current[reqId] : undefined
        const doneText = String(msg.text || '')
        const draftId = draftAssistantIdRef.current
        setItems((prev) => {
          const hasDraft = draftId ? prev.some((it) => it.id === draftId) : false
          if (doneText.trim() && (!draftId || !hasDraft)) {
            const newId = makeId()
            draftAssistantIdRef.current = newId
            return [
              ...prev,
              { id: newId, role: 'assistant', text: doneText, timings: t, created_at: new Date().toISOString() },
            ]
          }
          if (draftId && hasDraft) {
            return prev.map((it) => {
              if (it.id !== draftId) return it
              const fixedText = doneText.trim() && it.text.trim().length < doneText.trim().length ? doneText : it.text
              return { ...it, text: fixedText, timings: t ?? it.timings }
            })
          }
          return prev
        })
        finalizeTurn()
        return
      }
    }

    return () => {
      try {
        ws.close()
      } catch {
        // ignore
      }
      void playerRef.current?.close()
    }
  }, [botId])

  useEffect(() => {
    ignoreInitialConversationRef.current = false
  }, [initialConversationId])

  useEffect(() => {
    if (!initialConversationId) return
    if (ignoreInitialConversationRef.current) return
    if (conversationId === initialConversationId) return
    setErr(null)
    setConversationId(initialConversationId)
    setItems([])
    hydratedConvIdRef.current = null
    draftAssistantIdRef.current = null
    void hydrateConversation(initialConversationId)
  }, [initialConversationId, conversationId])

  useEffect(() => {
    if (!onConversationIdChange) return
    onConversationIdChange(conversationId)
  }, [conversationId, onConversationIdChange])

  useEffect(() => {
    if (!onStageChange) return
    onStageChange(stage)
  }, [stage, onStageChange])

  useEffect(() => {
    if (layout !== 'whatsapp') return
    if (startToken === undefined) return
    if (startTokenRef.current === undefined) {
      startTokenRef.current = startToken
      return
    }
    if (startToken === startTokenRef.current) return
    startTokenRef.current = startToken
    void initConversation()
  }, [startToken])

  useEffect(() => {
    if (hideWorkspace) setShowFilesPane(false)
  }, [hideWorkspace])

  useEffect(() => {
    if (!conversationId) return
    void loadContainerStatus(conversationId)
    const id = conversationId
    const t = setInterval(() => {
      void loadContainerStatus(id)
    }, 5000)
    return () => clearInterval(t)
  }, [conversationId])

  useEffect(() => {
    if (!showFilesPane || !conversationId) return
    void loadFiles()
  }, [showFilesPane, conversationId])

  useEffect(() => {
    const el = scrollerRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [items.length])

  async function initConversation() {
    ignoreInitialConversationRef.current = true
    if (!(await ensureWsOpen())) return
    const ws = wsRef.current
    if (!ws) return
    if (!canInit) return
    setErr(null)
    setConversationId(null)
    setItems([])
    hydratedConvIdRef.current = null
    draftAssistantIdRef.current = null
    const reqId = makeId()
    activeReqIdRef.current = reqId
    ws.send(JSON.stringify({ type: 'init', req_id: reqId, speak, test_flag: testFlag, debug }))
  }

  async function sendChat() {
    if (!(await ensureWsOpen())) return
    const ws = wsRef.current
    if (!ws) return
    if (!conversationId) {
      setErr('Start a conversation first.')
      return
    }
    const text = chatText.trim()
    if (!text) return
    if (stage !== 'idle') return
    setErr(null)
    setChatText('')
    const draftId = makeId()
    draftAssistantIdRef.current = draftId
    const now = new Date().toISOString()
    setItems((prev) => [
      ...prev,
      { id: makeId(), role: 'user', text, created_at: now },
      { id: draftId, role: 'assistant', text: '', created_at: now },
    ])
    const reqId = makeId()
    activeReqIdRef.current = reqId
    ws.send(
      JSON.stringify({
        type: 'chat',
        req_id: reqId,
        conversation_id: conversationId,
        speak,
        test_flag: testFlag,
        debug,
        text,
      }),
    )
  }

  async function startRecording() {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setErr('WebSocket not connected')
      return
    }
    if (!conversationId) {
      setErr('Start a conversation first.')
      return
    }
    if (!canRecord) return
    setErr(null)
    draftAssistantIdRef.current = null
    if (!pendingUserIdRef.current) {
      const userId = makeId()
      pendingUserIdRef.current = userId
      setItems((prev) => [
        ...prev,
        { id: userId, role: 'user', text: '…', created_at: new Date().toISOString() },
      ])
    }
    const reqId = makeId()
    activeReqIdRef.current = reqId
    wsRef.current.send(
      JSON.stringify({ type: 'start', req_id: reqId, conversation_id: conversationId, speak, test_flag: testFlag, debug }),
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
      if (!draftAssistantIdRef.current) {
        const draftId = makeId()
        draftAssistantIdRef.current = draftId
        setItems((prev) => [...prev, { id: draftId, role: 'assistant', text: '' }])
      }
      wsRef.current.send(JSON.stringify({ type: 'stop', req_id: reqId }))
    }
  }

  const recordBtn = recording ? (
    <button className="btn danger" onClick={() => void stopRecording()}>
      Stop recording
    </button>
  ) : (
    <button className="btn primary" onClick={() => void startRecording()} disabled={!canRecord}>
      Record
    </button>
  )

  const statusLabel = !containerStatus
    ? '—'
    : !containerStatus.docker_available
      ? 'docker unavailable'
      : containerStatus.exists
        ? containerStatus.running
          ? 'running'
          : containerStatus.status || 'stopped'
        : 'not started'
  const modalItems = (files?.items || []).filter((it) => it.path && it.path !== '.')
  const tree = useMemo(() => buildTree(modalItems), [modalItems])

  useEffect(() => {
    if (!showFilesPane) return
    if (Object.keys(expandedPaths).length) return
    const next: Record<string, boolean> = {}
    for (const child of tree.children) {
      if (child.is_dir) next[child.path] = true
    }
    setExpandedPaths(next)
  }, [showFilesPane, tree, expandedPaths])

  const messages = (
    <>
      {items.map((it) => {
        const role = it.role
        return (
          <div key={it.id} className={`msgRow ${role}`}>
            <div className={`avatar ${role}`} aria-hidden="true">
              <RoleIcon role={role} />
            </div>
            <div className={role === 'user' ? 'bubble user' : role === 'assistant' ? 'bubble assistant' : 'bubble tool'}>
              <div className="bubbleText">{it.text || '…'}</div>
              {role === 'assistant' && it.timings ? (
                <details className="details">
                  <summary>metrics</summary>
                  <div className="bubbleMeta">
                    ASR {fmtMs(it.timings.asr)} | LLM 1st {fmtMs(it.timings.llm_ttfb)} | LLM {fmtMs(it.timings.llm_total)} | TTS 1st{' '}
                    {fmtMs(it.timings.tts_first_audio)} | total {fmtMs(it.timings.total)}
                  </div>
                </details>
              ) : null}
              {it.details ? (
                <details className="details">
                  <summary>details</summary>
                  <pre className="pre">{JSON.stringify(it.details, null, 2)}</pre>
                </details>
              ) : null}
              {it.created_at ? <div className="bubbleTime" title={it.created_at}>{fmtTime(it.created_at)}</div> : null}
            </div>
          </div>
        )
      })}
    </>
  )

  const workspacePane =
    conversationId && showFilesPane ? (
      <aside className="filesPane">
        <div className="paneHeader">
          <div>
            <div className="paneTitle">Workspace</div>
            <div className="muted mono">container: {statusLabel}</div>
          </div>
          <button
            className="btn iconBtn"
            onClick={() => {
              void loadContainerStatus()
              void loadFiles()
            }}
            disabled={containerLoading || filesLoading}
            aria-label="Refresh"
          >
            {containerLoading || filesLoading ? <LoadingSpinner label="Refreshing" /> : '⟳'}
          </button>
        </div>
        {containerErr ? <div className="muted">{containerErr}</div> : null}
        {filesErr ? <div className="alert" style={{ marginTop: 8 }}>{filesErr}</div> : null}
        <div className="row gap" style={{ marginTop: 8, alignItems: 'center' }}>
          <label className="check">
            <input
              type="checkbox"
              checked={filesRecursive}
              onChange={(e) => void loadFiles(conversationId || undefined, e.target.checked, filesHidden)}
            />{' '}
            recursive
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={filesHidden}
              onChange={(e) => void loadFiles(conversationId || undefined, filesRecursive, e.target.checked)}
            />{' '}
            hidden
          </label>
          <div className="spacer" />
          <div className="muted mono">{files?.items ? `${modalItems.length} items` : '—'}</div>
        </div>
        {!files ? (
          <div className="muted" style={{ marginTop: 10 }}>
            <LoadingSpinner />
          </div>
        ) : (
          <div className="tree">
            {modalItems.length === 0 ? (
              <div className="muted">No files found.</div>
            ) : (
              renderTree(tree, 0, expandedPaths, setExpandedPaths)
            )}
          </div>
        )}
      </aside>
    ) : null

  const micButton = recording ? (
    <button className="iconBtn danger" onClick={() => void stopRecording()} title="Stop recording">
      Stop
    </button>
  ) : (
    <button className="iconBtn" onClick={() => void startRecording()} disabled={!canRecord} title="Record">
      <MicrophoneIcon />
    </button>
  )
  const voiceToggle = (
    <button
      className={`iconBtn ${speak ? 'active' : ''}`}
      onClick={() => setSpeak((v) => !v)}
      title={speak ? 'Voice output on' : 'Voice output off'}
    >
      {speak ? <SpeakerWaveIcon /> : <SpeakerXMarkIcon />}
    </button>
  )
  const uploadButton = (
    <div className="iconBtnWrap" title={allowUploads ? 'Upload files' : uploadDisabledReason || 'Upload files'}>
      <button
        className="iconBtn"
        disabled={!allowUploads || uploading}
        onClick={() => {
          if (!allowUploads) return
          setUploadMenuOpen((v) => !v)
        }}
      >
        {uploading ? <LoadingSpinner /> : <PaperClipIcon />}
      </button>
      {uploadMenuOpen && allowUploads ? (
        <div className="uploadMenu">
          <button
            type="button"
            onMouseDown={(e) => {
              e.preventDefault()
              setUploadMenuOpen(false)
              uploadRef.current?.click()
            }}
          >
            Upload files
          </button>
          <button
            type="button"
            onMouseDown={(e) => {
              e.preventDefault()
              setUploadMenuOpen(false)
              uploadFolderRef.current?.click()
            }}
          >
            Upload folder
          </button>
        </div>
      ) : null}
    </div>
  )

  if (layout === 'whatsapp') {
    return (
      <div className="assistantChat">
        {err ? <div className="alert">{err}</div> : null}
        <div className={`assistantSplit ${hideWorkspace || !conversationId || !showFilesPane ? 'full' : ''}`}>
          <div className="assistantChatPane">
            <div className="chatArea" ref={scrollerRef}>
              {messages}
            </div>
            <div className="chatComposerBar">
              {uploadButton}
              <input
                ref={uploadRef}
                type="file"
                multiple
                style={{ display: 'none' }}
                onChange={(e) => {
                  const files = e.target.files
                  if (files && files.length > 0 && onUploadFiles) {
                    onUploadFiles(files)
                  }
                  e.currentTarget.value = ''
                }}
              />
              <input
                ref={uploadFolderRef}
                type="file"
                multiple
                style={{ display: 'none' }}
                {...({ webkitdirectory: 'true', directory: 'true' } as any)}
                onChange={(e) => {
                  const files = e.target.files
                  if (files && files.length > 0 && onUploadFiles) {
                    onUploadFiles(files)
                  }
                  e.currentTarget.value = ''
                }}
              />
              {voiceToggle}
              {micButton}
              <div className="assistantComposerInput">
                <textarea
                  value={chatText}
                  onChange={(e) => setChatText(e.target.value)}
                  placeholder="Type a message…"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      void sendChat()
                    }
                  }}
                  disabled={!conversationId || stage !== 'idle'}
                />
              </div>
              <button className="btn primary" onClick={() => void sendChat()} disabled={!conversationId || stage !== 'idle' || !chatText.trim()}>
                Send
              </button>
            </div>
          </div>
          {hideWorkspace ? null : workspacePane}
        </div>
      </div>
    )
  }

  return (
    <section className="card">
      <div className="cardTitleRow">
        <div>
          <div className="cardTitle">Conversation</div>
          <div className="muted">Assistant speaks first; record only when you press “Record”.</div>
        </div>
        <div className="pill accent">status: {stage}</div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <div className="row gap">
        <label className="check">
          <input type="checkbox" checked={speak} onChange={(e) => setSpeak(e.target.checked)} /> Speak
        </label>
        <label className="check">
          <input type="checkbox" checked={testFlag} onChange={(e) => setTestFlag(e.target.checked)} /> test
        </label>
        <label className="check">
          <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} /> debug
        </label>
        <button className="btn" onClick={() => void initConversation()} disabled={!canInit}>
          {conversationId ? 'New conversation' : 'Start conversation'}
        </button>
        <button className="btn" onClick={() => setShowFilesPane((p) => !p)} disabled={!conversationId}>
          {showFilesPane ? 'Hide workspace' : 'Show workspace'}
        </button>
        <div className="spacer" />
        {speak ? recordBtn : null}
        {speak && recording ? <div className="recDot" title="Recording" /> : null}
      </div>

      {!speak ? (
        <div className="row gap" style={{ marginTop: 10 }}>
          <input
            value={chatText}
            onChange={(e) => setChatText(e.target.value)}
            placeholder="Type a message…"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void sendChat()
              }
            }}
            disabled={!conversationId || stage !== 'idle'}
          />
          <button className="btn primary" onClick={() => void sendChat()} disabled={!conversationId || stage !== 'idle' || !chatText.trim()}>
            Send
          </button>
        </div>
      ) : null}

      <details className="accordion" style={{ marginTop: 10 }}>
        <summary>Latency & metrics</summary>
        <div className="muted mono">
          ASR {fmtMs(lastTimings.asr)} | LLM 1st {fmtMs(lastTimings.llm_ttfb)} | LLM {fmtMs(lastTimings.llm_total)} | TTS 1st{' '}
          {fmtMs(lastTimings.tts_first_audio)} | total {fmtMs(lastTimings.total)}
        </div>
      </details>

      <div className={`chatSplit ${!conversationId || !showFilesPane ? 'full' : ''}`}>
        {workspacePane}

        <div className="chatPane">
          <div className="chat" ref={scrollerRef}>
            {messages}
          </div>
        </div>
      </div>
    </section>
  )
}

function normalizeToolEventForDisplay(msg: any): any {
  if (!msg || typeof msg !== 'object') return msg
  const out: any = { ...msg }
  // The server sends `arguments_json` as a JSON string. Parse it for readability so the
  // UI renders nested JSON instead of an escaped string.
  if (typeof out.arguments_json === 'string') {
    const s = out.arguments_json.trim()
    if ((s.startsWith('{') && s.endsWith('}')) || (s.startsWith('[') && s.endsWith(']'))) {
      try {
        out.arguments = JSON.parse(s)
        out.arguments_json_raw = out.arguments_json
        delete out.arguments_json
      } catch {
        // ignore
      }
    }
  }
  return out
}

function fmtBytes(n: number): string {
  if (!Number.isFinite(n)) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return i === 0 ? `${v.toFixed(0)} ${units[i]}` : `${v.toFixed(1)} ${units[i]}`
}

function RoleIcon({ role }: { role: ChatItem['role'] }) {
  if (role === 'assistant') {
    return <CpuChipIcon aria-hidden="true" />
  }
  if (role === 'tool') {
    return <WrenchScrewdriverIcon aria-hidden="true" />
  }
  return <UserIcon aria-hidden="true" />
}

function fmtTime(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

type TreeNode = {
  name: string
  path: string
  is_dir: boolean
  children: TreeNode[]
  size_bytes?: number | null
  mtime?: string
  download_url?: string | null
}

function buildTree(items: ConversationFiles['items']): TreeNode {
  const root: TreeNode = { name: '', path: '', is_dir: true, children: [] }
  const nodes = new Map<string, TreeNode>()
  nodes.set('', root)

  function ensureNode(path: string, name: string, isDir: boolean, parent: TreeNode): TreeNode {
    const existing = nodes.get(path)
    if (existing) {
      if (isDir) existing.is_dir = true
      return existing
    }
    const node: TreeNode = { name, path, is_dir: isDir, children: [] }
    nodes.set(path, node)
    parent.children.push(node)
    return node
  }

  for (const item of items) {
    const parts = item.path.split('/').filter(Boolean)
    let cur = root
    let curPath = ''
    for (let i = 0; i < parts.length; i += 1) {
      const part = parts[i]
      curPath = curPath ? `${curPath}/${part}` : part
      const isLast = i === parts.length - 1
      const node = ensureNode(curPath, part, isLast ? item.is_dir : true, cur)
      if (isLast) {
        node.is_dir = item.is_dir
        node.size_bytes = item.size_bytes
        node.mtime = item.mtime
        node.download_url = item.download_url ?? null
      }
      cur = node
    }
  }

  function sortTree(node: TreeNode): void {
    node.children.sort((a, b) => {
      if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1
      return a.name.localeCompare(b.name)
    })
    node.children.forEach(sortTree)
  }

  sortTree(root)
  return root
}

function renderTree(
  node: TreeNode,
  depth: number,
  expanded: Record<string, boolean>,
  setExpanded: Dispatch<SetStateAction<Record<string, boolean>>>,
): React.ReactNode {
  if (!node.children.length && node.path === '') return null
  if (node.path === '') {
    return node.children.map((child) => renderTree(child, depth, expanded, setExpanded))
  }
  const isOpen = !!expanded[node.path]
  const indent = depth * 16
  const href = node.download_url ? new URL(node.download_url, BACKEND_URL).toString() : ''
  const size = node.is_dir ? '—' : node.size_bytes === null ? '—' : fmtBytes(node.size_bytes || 0)
  const mtime = node.mtime ? new Date(node.mtime).toLocaleString() : '—'
  const fallbackName = node.name || node.path.split('/').filter(Boolean).pop() || node.path || '(root)'
  const nameNode = node.is_dir ? (
    <div className="treeName mono">{fallbackName ? `${fallbackName}/` : fallbackName}</div>
  ) : href ? (
    <a className="treeName mono link" href={href} title={fallbackName}>
      {fallbackName}
    </a>
  ) : (
    <div className="treeName mono">{fallbackName}</div>
  )
  return (
    <div key={node.path}>
      <div className="treeRow" style={{ paddingLeft: indent }}>
        {node.is_dir ? (
          <button
            className="btn ghost treeToggle"
            onClick={() => setExpanded((p) => ({ ...p, [node.path]: !isOpen }))}
            aria-label={isOpen ? 'Collapse folder' : 'Expand folder'}
          >
            {isOpen ? 'v' : '>'}
          </button>
        ) : (
          <span className="treeSpacer" />
        )}
        <div className="treeMain">
          {nameNode}
          <div className="treeMetaRow">
            <span className="treeMeta mono">{size}</span>
            <span className="treeMeta mono">{mtime}</span>
          </div>
        </div>
      </div>
      {node.is_dir && isOpen ? node.children.map((child) => renderTree(child, depth + 1, expanded, setExpanded)) : null}
    </div>
  )
}
