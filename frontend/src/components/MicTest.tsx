import { useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import { apiGet, BACKEND_URL } from '../api/client'
import { getBasicAuthToken } from '../auth'
import { createRecorder, type Recorder } from '../audio/recorder'
import { WavQueuePlayer } from '../audio/player'
import { fmtMs } from '../utils/format'
import type { ConversationDetail, DataAgentStatus, ConversationFiles } from '../types'

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

export default function MicTest({ botId, initialConversationId }: { botId: string; initialConversationId?: string }) {
  const [stage, setStage] = useState<Stage>('disconnected')
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [speak, setSpeak] = useState(true)
  const [testFlag, setTestFlag] = useState(true)
  const [debug, setDebug] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [containerStatus, setContainerStatus] = useState<DataAgentStatus | null>(null)
  const [containerErr, setContainerErr] = useState<string | null>(null)
  const [containerLoading, setContainerLoading] = useState(false)
  const [showFilesModal, setShowFilesModal] = useState(false)
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

  const wsRef = useRef<WebSocket | null>(null)
  const recorderRef = useRef<Recorder | null>(null)
  const playerRef = useRef<WavQueuePlayer | null>(null)
  const activeReqIdRef = useRef<string | null>(null)
  const recordingRef = useRef(false)
  const draftAssistantIdRef = useRef<string | null>(null)
  const toolProgressIdRef = useRef<Record<string, string>>({})
  const timingsByReq = useRef<Record<string, Timings>>({})
  const scrollerRef = useRef<HTMLDivElement | null>(null)
  const hydratedConvIdRef = useRef<string | null>(null)

  async function hydrateConversation(cid: string) {
    if (!cid) return
    if (hydratedConvIdRef.current === cid) return
    try {
      const d = await apiGet<ConversationDetail>(`/api/conversations/${cid}`)
      hydratedConvIdRef.current = cid
      draftAssistantIdRef.current = null
      const mapped: ChatItem[] = d.messages.map((m) => {
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
  const canRecord = useMemo(() => speak && stage === 'idle' && !!conversationId, [speak, stage, conversationId])

  useEffect(() => {
    const token = getBasicAuthToken()
    const authQuery = token ? `?auth=${encodeURIComponent(token)}` : ''
    const ws = new WebSocket(`${wsBase()}/ws/bots/${botId}/talk${authQuery}`)
    wsRef.current = ws
    playerRef.current = new WavQueuePlayer()
    setStage('idle')

    ws.onopen = () => {
      setErr(null)
      setStage('idle')
    }
    ws.onclose = () => {
      setStage('disconnected')
    }
    ws.onerror = () => {
      setStage('error')
      setErr('WebSocket error')
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
          activeReqIdRef.current = null
          draftAssistantIdRef.current = null
          toolProgressIdRef.current = {}
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
        setItems((prev) => [...prev, { id: makeId(), role: 'user', text }])
        }
        return
      }
      if (msg.type === 'tool_call') {
        const details = normalizeToolEventForDisplay(msg)
        setItems((prev) => [
          ...prev,
          { id: makeId(), role: 'tool', text: `[tool_call] ${msg.name}`, details },
        ])
        return
      }
      if (msg.type === 'tool_result') {
        const details = normalizeToolEventForDisplay(msg)
        setItems((prev) => [
          ...prev,
          { id: makeId(), role: 'tool', text: `[tool_result] ${msg.name}`, details },
        ])
        return
      }
      if (msg.type === 'tool_progress') {
        const text = String(msg.text || '').trim()
        if (!text) return
        const key = `${String(msg.req_id || '')}:${String(msg.name || 'tool')}`
        if (!toolProgressIdRef.current[key]) toolProgressIdRef.current[key] = makeId()
        const toolId = toolProgressIdRef.current[key]
        setItems((prev) => {
          const hasItem = prev.some((it) => it.id === toolId)
          if (!hasItem) {
            return [...prev, { id: toolId, role: 'tool', text: `[tool_progress] ${text}` }]
          }
          return prev.map((it) => (it.id === toolId ? { ...it, text: `${it.text}\n${text}` } : it))
        })
        return
      }
      if (msg.type === 'interim') {
        const text = String(msg.text || '').trim()
        if (!text) return
        setItems((prev) => [...prev, { id: makeId(), role: 'assistant', text }])
        return
      }
      if (msg.type === 'text_delta') {
        const delta = String(msg.delta || '')
        if (!delta) return
        if (!draftAssistantIdRef.current) draftAssistantIdRef.current = makeId()
        const draftId = draftAssistantIdRef.current
        setItems((prev) => {
          const hasDraft = prev.some((it) => it.id === draftId)
          if (!hasDraft) return [...prev, { id: draftId, role: 'assistant', text: delta }]
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
        setErr(String(msg.error || 'Unknown error'))
        setStage('error')
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
            return [...prev, { id: newId, role: 'assistant', text: doneText, timings: t }]
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
    if (!initialConversationId) return
    if (conversationId) return
    setErr(null)
    setConversationId(initialConversationId)
    setItems([])
    hydratedConvIdRef.current = null
    draftAssistantIdRef.current = null
    void hydrateConversation(initialConversationId)
  }, [initialConversationId, conversationId])

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
    if (!showFilesModal) return
    void loadFiles()
  }, [showFilesModal])

  useEffect(() => {
    const el = scrollerRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [items.length])

  async function initConversation() {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setErr('WebSocket not connected')
      return
    }
    if (!canInit) return
    setErr(null)
    setConversationId(null)
    setItems([])
    hydratedConvIdRef.current = null
    draftAssistantIdRef.current = null
    const reqId = makeId()
    activeReqIdRef.current = reqId
    wsRef.current.send(JSON.stringify({ type: 'init', req_id: reqId, speak, test_flag: testFlag, debug }))
  }

  async function sendChat() {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setErr('WebSocket not connected')
      return
    }
    if (!conversationId) {
      setErr('Start a conversation first.')
      return
    }
    const text = chatText.trim()
    if (!text) return
    if (stage !== 'idle') return
    setErr(null)
    setChatText('')
    draftAssistantIdRef.current = null
    setItems((prev) => [...prev, { id: makeId(), role: 'user', text }])
    const reqId = makeId()
    activeReqIdRef.current = reqId
    wsRef.current.send(
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
  const modalItems = (files?.items || []).filter((it) => it.path)
  const tree = useMemo(() => buildTree(modalItems), [modalItems])

  useEffect(() => {
    if (!showFilesModal) return
    if (Object.keys(expandedPaths).length) return
    const next: Record<string, boolean> = {}
    for (const child of tree.children) {
      if (child.is_dir) next[child.path] = true
    }
    setExpandedPaths(next)
  }, [showFilesModal, tree, expandedPaths])

  return (
    <section className="card">
      <div className="cardTitleRow">
        <div>
          <div className="cardTitle">Test Conve</div>
          <div className="muted">Assistant speaks first; record only when you press “Record”.</div>
        </div>
        <div className="pill">status: {stage}</div>
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

      <div className="muted mono">
        conversation: {conversationId || '(none)'} | latency: ASR {fmtMs(lastTimings.asr)} | LLM 1st {fmtMs(lastTimings.llm_ttfb)} |
        TTS 1st {fmtMs(lastTimings.tts_first_audio)} | total {fmtMs(lastTimings.total)}
      </div>
      {conversationId ? (
        <div className="row gap" style={{ marginTop: 6, alignItems: 'center' }}>
          <div className="muted mono">container: {statusLabel}</div>
          {containerErr ? <div className="muted">{containerErr}</div> : null}
          <button className="btn iconBtn" onClick={() => void loadContainerStatus()} disabled={containerLoading} aria-label="Refresh">
            {containerLoading ? '…' : '↻'}
          </button>
          <button className="btn" onClick={() => setShowFilesModal(true)}>
            Files
          </button>
        </div>
      ) : null}

      <div className="chat" ref={scrollerRef}>
        {items.map((it) => (
          <div key={it.id} className={it.role === 'user' ? 'bubble user' : it.role === 'assistant' ? 'bubble assistant' : 'bubble tool'}>
            <div className="bubbleText">{it.text || '…'}</div>
            {it.timings ? (
              <div className="bubbleMeta">
                ASR {fmtMs(it.timings.asr)} | LLM 1st {fmtMs(it.timings.llm_ttfb)} | LLM {fmtMs(it.timings.llm_total)} | TTS 1st{' '}
                {fmtMs(it.timings.tts_first_audio)} | total {fmtMs(it.timings.total)}
              </div>
            ) : null}
            {it.details ? (
              <details className="details">
                <summary>details</summary>
                <pre className="pre">{JSON.stringify(it.details, null, 2)}</pre>
              </details>
            ) : null}
          </div>
        ))}
      </div>
      {showFilesModal ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow">
              <div>
                <div className="cardTitle">Container files</div>
                <div className="muted">Workspace files for this conversation.</div>
              </div>
              <div className="row gap">
                <button className="btn iconBtn" onClick={() => void loadFiles()} disabled={filesLoading} aria-label="Refresh">
                  {filesLoading ? '…' : '↻'}
                </button>
                <button className="btn" onClick={() => setShowFilesModal(false)}>
                  Close
                </button>
              </div>
            </div>
            {filesErr ? <div className="alert">{filesErr}</div> : null}
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
                Loading…
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
          </div>
        </div>
      ) : null}
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
        <div className="treeName mono">{node.is_dir ? `${node.name}/` : node.name}</div>
        <div className="treeMeta mono">{size}</div>
        <div className="treeMeta mono">{mtime}</div>
        <div className="treeAction">
          {!node.is_dir && href ? (
            <a className="btn" href={href}>
              Download
            </a>
          ) : (
            <span className="muted">—</span>
          )}
        </div>
      </div>
      {node.is_dir && isOpen ? node.children.map((child) => renderTree(child, depth + 1, expanded, setExpanded)) : null}
    </div>
  )
}
