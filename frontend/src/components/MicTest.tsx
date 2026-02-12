import { useEffect, useLayoutEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import {
  CpuChipIcon,
  MicrophoneIcon,
  PaperClipIcon,
  SpeakerWaveIcon,
  SpeakerXMarkIcon,
  UserIcon,
  WrenchScrewdriverIcon,
} from '@heroicons/react/24/solid'
import { apiGet, apiPost, BACKEND_URL, downloadFile } from '../api/client'
import { getBasicAuthToken } from '../auth'
import { createRecorder, type Recorder } from '../audio/recorder'
import { WavQueuePlayer } from '../audio/player'
import { fmtMs } from '../utils/format'
import type { DataAgentStatus, ConversationFiles, ConversationMessage, Citation } from '../types'
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
  local?: boolean
  citations?: Citation[]
}

type MessageCursor = {
  created_at: string
  id: string
}

const PAGE_SIZE = 10

function getOldestCursor(items: ChatItem[]): MessageCursor | null {
  let oldest: MessageCursor | null = null
  for (const it of items) {
    if (!it.created_at || !it.id) continue
    if (!oldest) {
      oldest = { created_at: it.created_at, id: it.id }
      continue
    }
    if (it.created_at < oldest.created_at) {
      oldest = { created_at: it.created_at, id: it.id }
      continue
    }
    if (it.created_at === oldest.created_at && String(it.id) < String(oldest.id)) {
      oldest = { created_at: it.created_at, id: it.id }
    }
  }
  return oldest
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `id_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`
}

function normalizeCitations(citations?: Citation[]): Citation[] {
  if (!Array.isArray(citations)) return []
  const seen = new Set<string>()
  const out: Citation[] = []
  for (const c of citations) {
    if (!c || typeof c.url !== 'string' || !c.url.trim()) continue
    const key = `${c.url}|${c.title || ''}`
    if (seen.has(key)) continue
    seen.add(key)
    out.push(c)
  }
  return out
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

export type ChatCacheEntry = {
  items: ChatItem[]
  lastAt?: string
  lastId?: string
}

export default function MicTest({
  botId,
  initialConversationId,
  layout = 'default',
  startToken,
  onConversationIdChange,
  onStageChange,
  cache,
  onCacheUpdate,
  onSync,
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
  cache?: Record<string, ChatCacheEntry>
  onCacheUpdate?: (conversationId: string, entry: ChatCacheEntry) => void
  onSync?: () => void
  hideWorkspace?: boolean
  allowUploads?: boolean
  uploadDisabledReason?: string
  uploading?: boolean
  onUploadFiles?: (files: FileList) => void
}) {
  const [connectionStage, setConnectionStage] = useState<Stage>('disconnected')
  const [stageByConversationId, setStageByConversationId] = useState<Record<string, Stage>>({})
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
  const [filesMsg, setFilesMsg] = useState<string | null>(null)
  const [filesLoading, setFilesLoading] = useState(false)
  const [filesRecursive, setFilesRecursive] = useState(true)
  const [filesHidden, setFilesHidden] = useState(false)
  const [showToolMessages, setShowToolMessages] = useState(false)
  const [showWsSettings, setShowWsSettings] = useState(false)
  const [expandedPaths, setExpandedPaths] = useState<Record<string, boolean>>({})
  const [items, setItems] = useState<ChatItem[]>([])
  const [recording, setRecording] = useState(false)
  const [lastTimings, setLastTimings] = useState<Timings>({})
  const [chatText, setChatText] = useState('')
  const [uploadMenuOpen, setUploadMenuOpen] = useState(false)
  const [isVisible, setIsVisible] = useState(
    typeof document !== 'undefined' ? document.visibilityState === 'visible' : true,
  )
  const [loadingOlder, setLoadingOlder] = useState(false)
  const [hasMore, setHasMore] = useState(true)
  const [oldestCursor, setOldestCursor] = useState<MessageCursor | null>(null)

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
  const hydrateSeqRef = useRef(0)
  const ignoreInitialConversationRef = useRef(false)
  const startTokenRef = useRef<number | undefined>(startToken)
  const cacheRef = useRef<Record<string, ChatCacheEntry> | undefined>(cache)
  const conversationIdRef = useRef<string | null>(null)
  const markReadTimerRef = useRef<number | null>(null)
  const reqToConversationRef = useRef<Record<string, string>>({})
  const pendingInitReqIdRef = useRef<string | null>(null)
  const isNearBottomRef = useRef(true)
  const pendingScrollAdjustRef = useRef<{ prevHeight: number; prevTop: number } | null>(null)
  const cacheAppliedConvRef = useRef<string | null>(null)
  const downloadMsgTimerRef = useRef<number | null>(null)
  const autoScrollLockRef = useRef(false)
  const lastScrollTopRef = useRef(0)
  const interimIdRef = useRef<string | null>(null)

  const activeStage = useMemo(() => {
    if (connectionStage === 'disconnected' || connectionStage === 'error') return connectionStage
    if (!conversationId) return connectionStage === 'idle' ? 'idle' : connectionStage
    return stageByConversationId[conversationId] || 'idle'
  }, [connectionStage, conversationId, stageByConversationId])

  function mapConversationMessage(m: ConversationMessage): ChatItem {
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
      citations: Array.isArray(m.citations) ? m.citations : undefined,
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
  }

  function isActiveConversationForReq(reqId?: string | null) {
    const id = reqId ? reqToConversationRef.current[reqId] : null
    if (!id) return true
    return id === conversationIdRef.current
  }

  function mergeItems(prev: ChatItem[], next: ChatItem[]) {
    const serverCounts = new Map<string, number>()
    next.forEach((it) => {
      const key = `${it.role}::${(it.text || '').trim()}`
      serverCounts.set(key, (serverCounts.get(key) || 0) + 1)
    })
    const filteredPrev = prev.filter((it) => {
      if (!it.local) return true
      if (it.role === 'tool') return true
      const key = `${it.role}::${(it.text || '').trim()}`
      const remaining = serverCounts.get(key) || 0
      if (remaining > 0) {
        serverCounts.set(key, remaining - 1)
        return false
      }
      return true
    })
    const merged = [...filteredPrev, ...next]
    const seen = new Set<string>()
    const unique = merged.filter((it) => {
      if (seen.has(it.id)) return false
      seen.add(it.id)
      return true
    })
    unique.sort((a, b) => {
      const at = new Date(a.created_at || '').getTime()
      const bt = new Date(b.created_at || '').getTime()
      if (at !== bt) return at - bt
      return String(a.id).localeCompare(String(b.id))
    })
    return unique
  }

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

  useEffect(() => {
    cacheRef.current = cache
  }, [cache])

  useEffect(() => {
    const onVis = () => setIsVisible(document.visibilityState === 'visible')
    document.addEventListener('visibilitychange', onVis)
    return () => document.removeEventListener('visibilitychange', onVis)
  }, [])

  useEffect(() => {
    setHasMore(true)
    setOldestCursor(null)
    cacheAppliedConvRef.current = null
    isNearBottomRef.current = true
    interimIdRef.current = null
  }, [conversationId])

  useEffect(() => {
    conversationIdRef.current = conversationId
    if (conversationId && !stageByConversationId[conversationId]) {
      setConversationStage(conversationId, 'idle')
    }
  }, [conversationId, stageByConversationId])

  function setConversationStage(cid: string, next: Stage) {
    setStageByConversationId((prev) => ({ ...prev, [cid]: next }))
  }

  function clearInterim() {
    const id = interimIdRef.current
    if (!id) return
    setItems((prev) => prev.filter((it) => it.id !== id))
    interimIdRef.current = null
  }

  function finalizeTurn(reqId?: string) {
    if (reqId && activeReqIdRef.current !== reqId) return
    activeReqIdRef.current = null
    draftAssistantIdRef.current = null
    clearInterim()
    toolProgressIdRef.current = {}
    if (pendingUserIdRef.current) {
      const pendingId = pendingUserIdRef.current
      setItems((prev) => prev.filter((it) => it.id !== pendingId || it.text.trim()))
      pendingUserIdRef.current = null
    }
  }

  async function hydrateConversation(cid: string) {
    if (!cid) return
    const seq = ++hydrateSeqRef.current
    const isCurrent = () => hydrateSeqRef.current === seq
    if (cache && cache[cid]?.items?.length && isCurrent()) {
      setItems(cache[cid].items)
      setOldestCursor(getOldestCursor(cache[cid].items))
    }
    try {
      const d = await apiGet<{ messages: ConversationMessage[] }>(
        `/api/conversations/${cid}/messages?limit=${PAGE_SIZE}&order=desc&include_tools=1`,
      )
      if (!isCurrent()) return
      const raw = Array.isArray(d.messages) ? d.messages : []
      const mapped: ChatItem[] = raw.map(mapConversationMessage).reverse()
      setItems(mapped)
      setHasMore(raw.length === PAGE_SIZE)
      setOldestCursor(getOldestCursor(mapped))
      draftAssistantIdRef.current = null
      interimIdRef.current = null
      if (onCacheUpdate) {
        const lastAt = mapped.length ? mapped[mapped.length - 1].created_at : undefined
        onCacheUpdate(cid, { items: mapped, lastAt })
      }
    } catch (e: any) {
      if (!isCurrent()) return
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

  async function handleFileDownload(downloadUrl: string, filename: string) {
    setFilesErr(null)
    setFilesMsg(null)
    try {
      await downloadFile(downloadUrl, filename || 'download')
      setFilesMsg(`Downloaded ${filename || 'file'}.`)
      if (downloadMsgTimerRef.current) window.clearTimeout(downloadMsgTimerRef.current)
      downloadMsgTimerRef.current = window.setTimeout(() => setFilesMsg(null), 2500)
    } catch (e: any) {
      setFilesErr(String(e?.message || e))
    }
  }

  const canInit = useMemo(
    () => connectionStage === 'idle' || connectionStage === 'disconnected' || connectionStage === 'error',
    [connectionStage],
  )
  const canRecord = useMemo(() => {
    if (layout === 'whatsapp') return connectionStage === 'idle' && activeStage === 'idle' && !!conversationId
    return speak && connectionStage === 'idle' && activeStage === 'idle' && !!conversationId
  }, [layout, speak, conversationId, activeStage, connectionStage])

  useEffect(() => {
    return () => {
      if (downloadMsgTimerRef.current) window.clearTimeout(downloadMsgTimerRef.current)
    }
  }, [])

  useEffect(() => {
    const token = getBasicAuthToken()
    const authQuery = token ? `?auth=${encodeURIComponent(token)}` : ''
    const ws = new WebSocket(`${wsBase()}/ws/bots/${botId}/talk${authQuery}`)
    wsRef.current = ws
    playerRef.current = new WavQueuePlayer()
    setErr(null)
    setConnectionStage('disconnected')
    wsOpenPromiseRef.current = new Promise((resolve) => {
      wsOpenResolveRef.current = resolve
    })

    ws.onopen = () => {
      setErr(null)
      setConnectionStage('idle')
      wsOpenResolveRef.current?.(true)
    }
    ws.onclose = () => {
      setConnectionStage('disconnected')
      wsOpenResolveRef.current?.(false)
    }
    ws.onerror = () => {
      setConnectionStage('error')
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
        const reqId = String(msg.req_id || '')
        const convId = reqId ? reqToConversationRef.current[reqId] : conversationIdRef.current
        if (convId) {
          setConversationStage(convId, msg.stage || 'idle')
        }
        if (msg.stage === 'idle') {
          finalizeTurn(reqId)
        }
        return
      }
      if (msg.type === 'conversation') {
        const cid = msg.conversation_id || msg.id
        if (cid) {
          const reqId = String(msg.req_id || '')
          const pendingReq = pendingInitReqIdRef.current
          if (pendingReq && reqId && reqId !== pendingReq) return
          if (!pendingReq && conversationIdRef.current) return
          const s = String(cid)
          setConversationId(s)
          void hydrateConversation(s)
          if (pendingInitReqIdRef.current) {
            reqToConversationRef.current[pendingInitReqIdRef.current] = s
            pendingInitReqIdRef.current = null
          } else if (reqId) {
            reqToConversationRef.current[reqId] = s
          }
          setConversationStage(s, 'idle')
        }
        return
      }
      if (msg.type === 'metrics') {
        const reqId = String(msg.req_id || '')
        if (!isActiveConversationForReq(reqId)) return
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
            prev.map((it) => (it.id === pendingId ? { ...it, text, created_at: new Date().toISOString(), local: true } : it)),
          )
          pendingUserIdRef.current = null
        } else {
          setItems((prev) => [
            ...prev,
            { id: makeId(), role: 'user', text, created_at: new Date().toISOString(), local: true },
          ])
        }
        }
        return
      }
      if (msg.type === 'tool_call') {
        const reqId = String(msg.req_id || '')
        if (!isActiveConversationForReq(reqId)) return
        const details = normalizeToolEventForDisplay(msg)
        const toolItem = {
          id: makeId(),
          role: 'tool' as const,
          text: `[tool_call] ${msg.name}`,
          details,
          created_at: new Date().toISOString(),
          local: true,
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
        const reqId = String(msg.req_id || '')
        if (!isActiveConversationForReq(reqId)) return
        const details = normalizeToolEventForDisplay(msg)
        const toolItem = {
          id: makeId(),
          role: 'tool' as const,
          text: `[tool_result] ${msg.name}`,
          details,
          created_at: new Date().toISOString(),
          local: true,
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
        const reqId = String(msg.req_id || '')
        if (!isActiveConversationForReq(reqId)) return
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
            local: true,
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
        if (!interimIdRef.current) interimIdRef.current = makeId()
        const interimId = interimIdRef.current
        setItems((prev) => {
          const hasInterim = prev.some((it) => it.id === interimId)
          const nextItem = { id: interimId, role: 'assistant' as const, text, created_at: new Date().toISOString(), local: true }
          if (!hasInterim) return [...prev, nextItem]
          return prev.map((it) => (it.id === interimId ? { ...it, text, created_at: nextItem.created_at, local: true } : it))
        })
        return
      }
      if (msg.type === 'text_delta') {
        const reqId = String(msg.req_id || '')
        if (!isActiveConversationForReq(reqId)) return
        const delta = String(msg.delta || '')
        if (!delta) return
        clearInterim()
        if (!draftAssistantIdRef.current) draftAssistantIdRef.current = makeId()
        const draftId = draftAssistantIdRef.current
        setItems((prev) => {
          const hasDraft = prev.some((it) => it.id === draftId)
          if (!hasDraft)
            return [
              ...prev,
              { id: draftId, role: 'assistant', text: delta, created_at: new Date().toISOString(), local: true },
            ]
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
        setConnectionStage('error')
        clearInterim()
        setItems((prev) => [
          ...prev,
          { id: makeId(), role: 'assistant', text: `Error: ${errorText}`, created_at: new Date().toISOString(), local: true },
        ])
        return
      }
      if (msg.type === 'done') {
        const reqId = String(msg.req_id || '')
        if (!isActiveConversationForReq(reqId)) return
        const t = reqId ? timingsByReq.current[reqId] : undefined
        const doneText = String(msg.text || '')
        const msgCitations = Array.isArray(msg.citations) ? (msg.citations as Citation[]) : undefined
        const doneAt = new Date().toISOString()
        const draftId = draftAssistantIdRef.current
        setItems((prev) => {
          const hasDraft = draftId ? prev.some((it) => it.id === draftId) : false
          if (doneText.trim() && (!draftId || !hasDraft)) {
            const newId = makeId()
            draftAssistantIdRef.current = newId
            return [
              ...prev,
              {
                id: newId,
                role: 'assistant',
                text: doneText,
                timings: t,
                created_at: doneAt,
                local: true,
                citations: msgCitations,
              },
            ]
          }
          if (draftId && hasDraft) {
            return prev.map((it) => {
              if (it.id !== draftId) return it
              const fixedText = doneText.trim() && it.text.trim().length < doneText.trim().length ? doneText : it.text
              return {
                ...it,
                text: fixedText,
                timings: t ?? it.timings,
                citations: msgCitations ?? it.citations,
                created_at: doneAt,
              }
            })
          }
          return prev
        })
        if (reqId) {
          const convId = reqToConversationRef.current[reqId]
          if (convId) setConversationStage(convId, 'idle')
        } else if (conversationIdRef.current) {
          setConversationStage(conversationIdRef.current, 'idle')
        }
        finalizeTurn(reqId)
        onSync?.()
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
    if (cache && cache[initialConversationId]?.items?.length) {
      setItems(cache[initialConversationId].items)
      setOldestCursor(getOldestCursor(cache[initialConversationId].items))
    } else {
      setItems([])
      setOldestCursor(null)
    }
    draftAssistantIdRef.current = null
    void hydrateConversation(initialConversationId)
  }, [initialConversationId, conversationId])

  useEffect(() => {
    if (!onConversationIdChange) return
    onConversationIdChange(conversationId)
  }, [conversationId, onConversationIdChange])

  useEffect(() => {
    if (!onStageChange) return
    onStageChange(activeStage)
  }, [activeStage, onStageChange])

  useEffect(() => {
    if (!conversationId || !onCacheUpdate) return
    const lastAt = items.length ? items[items.length - 1]?.created_at : undefined
    const lastId = items.length ? items[items.length - 1]?.id : undefined
    onCacheUpdate(conversationId, { items, lastAt, lastId })
  }, [items, conversationId, onCacheUpdate])

  useEffect(() => {
    if (!conversationId || !isVisible) return
    if (!items.length) return
    if (markReadTimerRef.current) window.clearTimeout(markReadTimerRef.current)
    markReadTimerRef.current = window.setTimeout(() => {
      void apiPost(`/api/conversations/${conversationId}/read`, {})
    }, 400)
    return () => {
      if (markReadTimerRef.current) {
        window.clearTimeout(markReadTimerRef.current)
        markReadTimerRef.current = null
      }
    }
  }, [items.length, conversationId, isVisible])

  useEffect(() => {
    if (!conversationId) return
    const wsConnected = connectionStage !== 'disconnected' && connectionStage !== 'error'
    const shouldPoll = !wsConnected || !isVisible
    if (!shouldPoll) return
    let canceled = false
    const fetchDelta = async () => {
      const entry = cacheRef.current?.[conversationId]
      if (!entry?.lastAt) return
      try {
        const since = encodeURIComponent(entry.lastAt)
        const sinceId = entry.lastId ? `&since_id=${encodeURIComponent(entry.lastId)}` : ''
        const d = await apiGet<{ messages: ConversationMessage[] }>(
          `/api/conversations/${conversationId}/messages?since=${since}${sinceId}&include_tools=1`,
        )
        if (!d?.messages?.length) return
        const delta = d.messages.map(mapConversationMessage)
        if (canceled) return
        setItems((prev) => mergeItems(prev, delta))
      } catch {
        // ignore
      }
    }
    void fetchDelta()
    const t = window.setInterval(() => {
      void fetchDelta()
    }, 30000)
    return () => {
      canceled = true
      window.clearInterval(t)
    }
  }, [conversationId, connectionStage, isVisible])

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
    if (cacheAppliedConvRef.current !== conversationId && cache && cache[conversationId]?.items?.length) {
      setItems(cache[conversationId].items)
      setOldestCursor(getOldestCursor(cache[conversationId].items))
      cacheAppliedConvRef.current = conversationId
    }
    if (hideWorkspace || !isVisible) return
    void loadContainerStatus(conversationId)
    const id = conversationId
    const t = setInterval(() => {
      void loadContainerStatus(id)
    }, 5000)
    return () => clearInterval(t)
  }, [conversationId, hideWorkspace, isVisible])

  useEffect(() => {
    if (!showFilesPane || !conversationId) return
    void loadFiles()
  }, [showFilesPane, conversationId])

  useLayoutEffect(() => {
    const el = scrollerRef.current
    if (!el) return
    const pending = pendingScrollAdjustRef.current
    if (pending) {
      const nextHeight = el.scrollHeight
      el.scrollTop = pending.prevTop + (nextHeight - pending.prevHeight)
      pendingScrollAdjustRef.current = null
      return
    }
    if (isNearBottomRef.current && !autoScrollLockRef.current) {
      el.scrollTop = el.scrollHeight
    }
  }, [items.length])

  useEffect(() => {
    const el = scrollerRef.current
    if (!el) return
    const onScroll = () => {
      const nearBottom = el.scrollHeight - (el.scrollTop + el.clientHeight) < 80
      isNearBottomRef.current = nearBottom
      const prevTop = lastScrollTopRef.current
      lastScrollTopRef.current = el.scrollTop
      if (nearBottom) {
        autoScrollLockRef.current = false
      } else if (el.scrollTop < prevTop) {
        autoScrollLockRef.current = true
      }
      const nearTop = el.scrollTop <= 40
      if (nearTop && hasMore && !loadingOlder) {
        void loadOlder()
      }
    }
    el.addEventListener('scroll', onScroll)
    return () => el.removeEventListener('scroll', onScroll)
  }, [hasMore, loadingOlder, conversationId])


  async function loadOlder() {
    if (!conversationId || loadingOlder || !oldestCursor) return
    const el = scrollerRef.current
    if (el) {
      pendingScrollAdjustRef.current = { prevHeight: el.scrollHeight, prevTop: el.scrollTop }
    }
    setLoadingOlder(true)
    let didAdd = false
    try {
      const before = encodeURIComponent(oldestCursor.created_at)
      const beforeId = encodeURIComponent(oldestCursor.id)
      const d = await apiGet<{ messages: ConversationMessage[] }>(
        `/api/conversations/${conversationId}/messages?before=${before}&before_id=${beforeId}&limit=${PAGE_SIZE}&order=desc&include_tools=1`,
      )
      const raw = Array.isArray(d.messages) ? d.messages : []
      const mapped = raw.map(mapConversationMessage).reverse()
      if (mapped.length) {
        setItems((prev) => mergeItems(prev, mapped))
        setOldestCursor(getOldestCursor(mapped) || oldestCursor)
        didAdd = true
      }
      if (raw.length < PAGE_SIZE) setHasMore(false)
    } catch {
      // ignore
    } finally {
      setLoadingOlder(false)
      if (!didAdd) pendingScrollAdjustRef.current = null
    }
  }

  async function initConversation() {
    ignoreInitialConversationRef.current = true
    if (!(await ensureWsOpen())) return
    const ws = wsRef.current
    if (!ws) return
    if (!canInit) return
    setErr(null)
    setConversationId(null)
    setItems([])
    draftAssistantIdRef.current = null
    const reqId = makeId()
    activeReqIdRef.current = reqId
    pendingInitReqIdRef.current = reqId
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
    if (activeStage !== 'idle') return
    setErr(null)
    setChatText('')
    const draftId = makeId()
    draftAssistantIdRef.current = draftId
    const now = new Date().toISOString()
    setItems((prev) => [
      ...prev,
      { id: makeId(), role: 'user', text, created_at: now, local: true },
      { id: draftId, role: 'assistant', text: '', created_at: now, local: true },
    ])
    const reqId = makeId()
    activeReqIdRef.current = reqId
    reqToConversationRef.current[reqId] = conversationId
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
    onSync?.()
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
        { id: userId, role: 'user', text: '…', created_at: new Date().toISOString(), local: true },
      ])
    }
    const reqId = makeId()
    activeReqIdRef.current = reqId
    reqToConversationRef.current[reqId] = conversationId
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
  const wsStatusLabel = connectionStage === 'idle' ? 'connected' : connectionStage
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

  const displayItems = showToolMessages ? items : items.filter((it) => it.role !== 'tool')
  const loadingDots = (
    <span className="loadingDots" aria-label="Loading">
      <span>.</span>
      <span>.</span>
      <span>.</span>
    </span>
  )
  const messages = (
    <>
      {displayItems.map((it) => {
        const role = it.role
        const citations = role === 'assistant' ? normalizeCitations(it.citations) : []
        const hasText = Boolean(it.text && it.text.trim())
        const bubbleText = hasText ? it.text : role === 'assistant' ? loadingDots : '…'
        return (
          <div key={it.id} className={`msgRow ${role}`}>
            <div className={`avatar ${role}`} aria-hidden="true">
              <RoleIcon role={role} />
            </div>
            <div className={role === 'user' ? 'bubble user' : role === 'assistant' ? 'bubble assistant' : 'bubble tool'}>
              <div className="bubbleText">{bubbleText}</div>
              {role === 'assistant' && citations.length ? (
                <div className="citationBlock">
                  <div className="citationTitle">Sources</div>
                  <ol className="citationList">
                    {citations.map((c, idx) => (
                      <li key={`${c.url}-${idx}`}>
                        <a className="citationLink" href={c.url} target="_blank" rel="noreferrer">
                          {c.title || c.url}
                        </a>
                      </li>
                    ))}
                  </ol>
                </div>
              ) : null}
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
        {filesMsg ? (
          <div
            className="alert"
            style={{ marginTop: 8, borderColor: 'rgba(80, 200, 160, 0.4)', background: 'rgba(80, 200, 160, 0.1)' }}
          >
            {filesMsg}
          </div>
        ) : null}
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
              renderTree(tree, 0, expandedPaths, setExpandedPaths, handleFileDownload)
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
            <div className="row gap" style={{ alignItems: 'center' }}>
              <div className="pill" title={`stage: ${activeStage}`}>
                ws: {wsStatusLabel}
              </div>
              <div className="spacer" />
              <button
                className="btn ghost"
                type="button"
                onClick={() => setShowWsSettings((v) => !v)}
                aria-pressed={showWsSettings}
              >
                WS settings: {showWsSettings ? 'on' : 'off'}
              </button>
              <button
                className="btn ghost"
                type="button"
                onClick={() => setShowToolMessages((v) => !v)}
                aria-pressed={showToolMessages}
              >
                More info: {showToolMessages ? 'on' : 'off'}
              </button>
            </div>
            {showWsSettings ? (
              <div className="row gap" style={{ marginTop: 6 }}>
                <label className="check">
                  <input type="checkbox" checked={testFlag} onChange={(e) => setTestFlag(e.target.checked)} /> test
                </label>
                <label className="check">
                  <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} /> debug
                </label>
              </div>
            ) : null}
            <div className="chatArea" ref={scrollerRef}>
              {loadingOlder || hasMore ? (
                <div className="muted" style={{ padding: '8px 0', textAlign: 'center' }}>
                  {loadingOlder ? (
                    <LoadingSpinner label="Loading older messages" />
                  ) : (
                    <button className="btn ghost" onClick={() => void loadOlder()} disabled={!hasMore}>
                      Load earlier messages
                    </button>
                  )}
                </div>
              ) : null}
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
                  disabled={!conversationId}
                />
              </div>
              <button
                className="btn primary"
                onClick={() => void sendChat()}
                disabled={!conversationId || activeStage !== 'idle' || !chatText.trim()}
              >
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
        <div className="pill accent">status: {activeStage}</div>
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
        <button className="btn ghost" type="button" onClick={() => setShowToolMessages((v) => !v)} aria-pressed={showToolMessages}>
          More info: {showToolMessages ? 'on' : 'off'}
        </button>
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
            disabled={!conversationId}
          />
          <button
            className="btn primary"
            onClick={() => void sendChat()}
            disabled={!conversationId || activeStage !== 'idle' || !chatText.trim()}
          >
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
            {loadingOlder || hasMore ? (
              <div className="muted" style={{ padding: '8px 0', textAlign: 'center' }}>
                {loadingOlder ? (
                  <LoadingSpinner label="Loading older messages" />
                ) : (
                  <button className="btn ghost" onClick={() => void loadOlder()} disabled={!hasMore}>
                    Load earlier messages
                  </button>
                )}
              </div>
            ) : null}
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
  onDownload?: (downloadUrl: string, filename: string) => void,
): React.ReactNode {
  if (!node.children.length && node.path === '') return null
  if (node.path === '') {
    return node.children.map((child) => renderTree(child, depth, expanded, setExpanded, onDownload))
  }
  const isOpen = !!expanded[node.path]
  const indent = depth * 16
  const canDownload = Boolean(onDownload && node.download_url && !node.is_dir)
  const size = node.is_dir ? '—' : node.size_bytes === null ? '—' : fmtBytes(node.size_bytes || 0)
  const mtime = node.mtime ? new Date(node.mtime).toLocaleString() : '—'
  const fallbackName = node.name || node.path.split('/').filter(Boolean).pop() || node.path || '(root)'
  const nameNode = node.is_dir ? (
    <div className="treeName mono">{fallbackName ? `${fallbackName}/` : fallbackName}</div>
  ) : canDownload ? (
    <button
      type="button"
      className="btn linkBtn treeName mono link"
      title={fallbackName}
      onClick={() => onDownload?.(node.download_url || '', fallbackName)}
    >
      {fallbackName}
    </button>
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
      {node.is_dir && isOpen
        ? node.children.map((child) => renderTree(child, depth + 1, expanded, setExpanded, onDownload))
        : null}
    </div>
  )
}
