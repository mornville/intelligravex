import { type MouseEvent as ReactMouseEvent, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  ArrowTopRightOnSquareIcon,
  ArrowsPointingInIcon,
  ArrowsPointingOutIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  Cog6ToothIcon,
  CpuChipIcon,
  EyeIcon,
  EyeSlashIcon,
  MicrophoneIcon,
  PaperClipIcon,
  UserIcon,
  WrenchScrewdriverIcon,
} from '@heroicons/react/24/solid'
import LoadingSpinner from '../components/LoadingSpinner'
import MarkdownText from '../components/MarkdownText'
import { apiDelete, apiGet, apiPost } from '../api/client'
import type {
  ConversationFiles,
  ConversationMessage,
  DataAgentStatus,
  GroupBot,
  GroupConversationDetail,
  GroupConversationSummary,
} from '../types'
import { fmtIso } from '../utils/format'

type MentionState = {
  active: boolean
  query: string
  index: number
  start: number
  end: number
}

const PAGE_SIZE = 10
const WORKSPACE_WIDTH_KEY = 'igx_workspace_width'
const WORKSPACE_COLLAPSED_KEY = 'igx_workspace_collapsed'
const DEFAULT_WORKSPACE_WIDTH = 320
const MIN_WORKSPACE_WIDTH = 240
const MAX_WORKSPACE_WIDTH = 900
const SIDEBAR_WIDTH_KEY = 'igx_sidebar_width'
const SIDEBAR_COLLAPSED_KEY = 'igx_sidebar_collapsed'
const DEFAULT_SIDEBAR_WIDTH = 320
const MIN_SIDEBAR_WIDTH = 220
const MAX_SIDEBAR_WIDTH = 520

function clampWorkspaceWidth(next: number) {
  if (Number.isNaN(next)) return DEFAULT_WORKSPACE_WIDTH
  return Math.min(MAX_WORKSPACE_WIDTH, Math.max(MIN_WORKSPACE_WIDTH, next))
}

function clampSidebarWidth(next: number) {
  if (Number.isNaN(next)) return DEFAULT_SIDEBAR_WIDTH
  return Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, next))
}

type MessageCursor = {
  created_at: string
  id: string
}

function getOldestCursor(items: ConversationMessage[]): MessageCursor | null {
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

export default function GroupConversationPage() {
  const { groupId } = useParams()
  const nav = useNavigate()
  const [data, setData] = useState<GroupConversationDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [loadingOlder, setLoadingOlder] = useState(false)
  const [hasMore, setHasMore] = useState(true)
  const [oldestCursor, setOldestCursor] = useState<MessageCursor | null>(null)
  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)
  const [sendErr, setSendErr] = useState<string | null>(null)
  const [resetting, setResetting] = useState(false)
  const [resetErr, setResetErr] = useState<string | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [deleteErr, setDeleteErr] = useState<string | null>(null)
  const [workingBots, setWorkingBots] = useState<Record<string, string>>({})
  const [mention, setMention] = useState<MentionState | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const nearBottomRef = useRef(true)
  const pendingScrollAdjustRef = useRef<{ prevHeight: number; prevTop: number } | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const autoScrollLockRef = useRef(false)
  const lastScrollTopRef = useRef(0)
  const markReadTimerRef = useRef<number | null>(null)
  const [groupList, setGroupList] = useState<GroupConversationSummary[]>([])
  const [groupListErr, setGroupListErr] = useState<string | null>(null)
  const [groupListLoading, setGroupListLoading] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [workspaceCollapsed, setWorkspaceCollapsed] = useState(false)
  const [workspaceWidth, setWorkspaceWidth] = useState(DEFAULT_WORKSPACE_WIDTH)
  const workspaceResizeActiveRef = useRef(false)
  const workspaceResizeStartXRef = useRef(0)
  const workspaceResizeStartWidthRef = useRef(DEFAULT_WORKSPACE_WIDTH)
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_SIDEBAR_WIDTH)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const sidebarResizeActiveRef = useRef(false)
  const sidebarResizeStartXRef = useRef(0)
  const sidebarResizeStartWidthRef = useRef(DEFAULT_SIDEBAR_WIDTH)
  const [workspaceIDEOpen, setWorkspaceIDEOpen] = useState(false)
  const [workspaceIDEFull, setWorkspaceIDEFull] = useState(false)
  const [resizeMode, setResizeMode] = useState<'workspace' | 'sidebar' | null>(null)
  const [agentStatus, setAgentStatus] = useState<DataAgentStatus | null>(null)
  const [agentErr, setAgentErr] = useState<string | null>(null)
  const [files, setFiles] = useState<ConversationFiles | null>(null)
  const [filesErr, setFilesErr] = useState<string | null>(null)
  const [filesLoading, setFilesLoading] = useState(false)

  function isNotFoundError(e: any) {
    return String(e?.message || e).includes('HTTP 404')
  }

  function clearConversationState() {
    setAgentStatus(null)
    setAgentErr(null)
    setWorkspaceIDEOpen(false)
    setWorkspaceIDEFull(false)
    setFiles(null)
    setFilesErr(null)
    setMessages([])
    setHasMore(true)
    setOldestCursor(null)
  }

  useEffect(() => {
    if (typeof window === 'undefined') return
    const savedWidth = window.localStorage.getItem(WORKSPACE_WIDTH_KEY)
    const parsedWidth = savedWidth ? Number(savedWidth) : Number.NaN
    if (!Number.isNaN(parsedWidth)) {
      setWorkspaceWidth(clampWorkspaceWidth(parsedWidth))
    }
    const savedCollapsed = window.localStorage.getItem(WORKSPACE_COLLAPSED_KEY)
    if (savedCollapsed === '1') {
      setWorkspaceCollapsed(true)
    }
    const savedSidebar = window.localStorage.getItem(SIDEBAR_WIDTH_KEY)
    const parsedSidebar = savedSidebar ? Number(savedSidebar) : Number.NaN
    if (!Number.isNaN(parsedSidebar)) {
      setSidebarWidth(clampSidebarWidth(parsedSidebar))
    }
    const savedSidebarCollapsed = window.localStorage.getItem(SIDEBAR_COLLAPSED_KEY)
    if (savedSidebarCollapsed === '1') {
      setSidebarCollapsed(true)
    }
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(WORKSPACE_WIDTH_KEY, String(workspaceWidth))
  }, [workspaceWidth])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(WORKSPACE_COLLAPSED_KEY, workspaceCollapsed ? '1' : '0')
  }, [workspaceCollapsed])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth))
  }, [sidebarWidth])

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(SIDEBAR_COLLAPSED_KEY, sidebarCollapsed ? '1' : '0')
  }, [sidebarCollapsed])

  function wsBase() {
    try {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      return `${proto}//${window.location.host}`
    } catch {
      return 'ws://127.0.0.1:8000'
    }
  }

  function mergeMessages(prev: ConversationMessage[], next: ConversationMessage[]) {
    const seen = new Set(prev.map((m) => m.id))
    const merged = [...prev]
    next.forEach((m) => {
      if (!seen.has(m.id)) {
        merged.push(m)
        seen.add(m.id)
      }
    })
    merged.sort((a, b) => {
      const at = new Date(a.created_at || '').getTime()
      const bt = new Date(b.created_at || '').getTime()
      if (at !== bt) return at - bt
      return String(a.id).localeCompare(String(b.id))
    })
    return merged
  }

  async function loadMessagesLatest(id: string) {
    try {
      const d = await apiGet<{ messages: ConversationMessage[] }>(
        `/api/group-conversations/${id}/messages?limit=${PAGE_SIZE}&order=desc`,
      )
      const raw = Array.isArray(d.messages) ? d.messages : []
      const mapped = raw.reverse()
      setMessages(mapped)
      setHasMore(raw.length === PAGE_SIZE)
      setOldestCursor(getOldestCursor(mapped))
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  async function loadOlderMessages() {
    if (!groupId || loadingOlder || !oldestCursor) return
    const el = scrollRef.current
    if (el) {
      pendingScrollAdjustRef.current = { prevHeight: el.scrollHeight, prevTop: el.scrollTop }
    }
    setLoadingOlder(true)
    let didAdd = false
    try {
      const before = encodeURIComponent(oldestCursor.created_at)
      const beforeId = encodeURIComponent(oldestCursor.id)
      const d = await apiGet<{ messages: ConversationMessage[] }>(
        `/api/group-conversations/${groupId}/messages?before=${before}&before_id=${beforeId}&limit=${PAGE_SIZE}&order=desc`,
      )
      const raw = Array.isArray(d.messages) ? d.messages : []
      const mapped = raw.reverse()
      if (mapped.length) {
        setMessages((prev) => mergeMessages(prev, mapped))
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

  useEffect(() => {
    if (!groupId) return
    setMessages([])
    setHasMore(true)
    setOldestCursor(null)
    nearBottomRef.current = true
    void (async () => {
      setLoading(true)
      setErr(null)
      try {
        const d = await apiGet<GroupConversationDetail>(`/api/group-conversations/${groupId}?include_messages=false`)
        setData(d)
        await loadMessagesLatest(groupId)
      } catch (e: any) {
        setErr(String(e?.message || e))
      } finally {
        setLoading(false)
      }
    })()
  }, [groupId])

  useEffect(() => {
    if (!groupId || !messages.length) return
    if (markReadTimerRef.current) window.clearTimeout(markReadTimerRef.current)
    markReadTimerRef.current = window.setTimeout(() => {
      void apiPost(`/api/conversations/${groupId}/read`, {})
    }, 400)
    return () => {
      if (markReadTimerRef.current) {
        window.clearTimeout(markReadTimerRef.current)
        markReadTimerRef.current = null
      }
    }
  }, [messages.length, groupId])

  useEffect(() => {
    void (async () => {
      setGroupListLoading(true)
      setGroupListErr(null)
      try {
        const g = await apiGet<{ items: GroupConversationSummary[] }>('/api/group-conversations')
        setGroupList(g.items)
      } catch (e: any) {
        setGroupListErr(String(e?.message || e))
      } finally {
        setGroupListLoading(false)
      }
    })()
  }, [])

  useEffect(() => {
    if (!groupId) return
    void (async () => {
      setAgentErr(null)
      try {
        const s = await apiGet<DataAgentStatus>(`/api/conversations/${groupId}/data-agent`)
        setAgentStatus(s)
      } catch (e: any) {
        if (isNotFoundError(e)) {
          clearConversationState()
          return
        }
        setAgentErr(String(e?.message || e))
      }
    })()
  }, [groupId])

  useEffect(() => {
    if (!groupId) return
    void (async () => {
      setFilesLoading(true)
      setFilesErr(null)
      try {
        const f = await apiGet<ConversationFiles>(`/api/conversations/${groupId}/files?path=`)
        setFiles(f)
      } catch (e: any) {
        if (isNotFoundError(e)) {
          clearConversationState()
          return
        }
        setFilesErr(String(e?.message || e))
      } finally {
        setFilesLoading(false)
      }
    })()
  }, [groupId])

  useEffect(() => {
    if (!groupId) return
    const ws = new WebSocket(`${wsBase()}/ws/groups/${groupId}`)
    wsRef.current = ws
    ws.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return
      let payload: any
      try {
        payload = JSON.parse(ev.data)
      } catch {
        return
      }
      if (payload.type === 'message' && payload.message) {
        setMessages((prev) => mergeMessages(prev, [payload.message]))
      }
      if (payload.type === 'status') {
        setWorkingBots((prev) => {
          const next = { ...prev }
          if (payload.state === 'working') {
            next[payload.bot_id] = payload.bot_name || 'assistant'
          } else {
            delete next[payload.bot_id]
          }
          return next
        })
      }
      if (payload.type === 'reset') {
        setWorkingBots({})
        setMessages([])
        setHasMore(true)
        setOldestCursor(null)
      }
    }
    ws.onclose = () => {
      wsRef.current = null
    }
    return () => {
      try {
        ws.close()
      } catch {
        // ignore
      }
    }
  }, [groupId])

  useLayoutEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const pending = pendingScrollAdjustRef.current
    if (pending) {
      const nextHeight = el.scrollHeight
      el.scrollTop = pending.prevTop + (nextHeight - pending.prevHeight)
      pendingScrollAdjustRef.current = null
      return
    }
    if (nearBottomRef.current && !autoScrollLockRef.current) {
      el.scrollTop = el.scrollHeight
    }
  }, [messages.length, groupId])

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const onScroll = () => {
      const nearBottom = el.scrollHeight - (el.scrollTop + el.clientHeight) < 80
      nearBottomRef.current = nearBottom
      const prevTop = lastScrollTopRef.current
      lastScrollTopRef.current = el.scrollTop
      if (nearBottom) {
        autoScrollLockRef.current = false
      } else if (el.scrollTop < prevTop) {
        autoScrollLockRef.current = true
      }
      const nearTop = el.scrollTop <= 40
      if (nearTop && hasMore && !loadingOlder) {
        void loadOlderMessages()
      }
    }
    el.addEventListener('scroll', onScroll)
    return () => el.removeEventListener('scroll', onScroll)
  }, [hasMore, loadingOlder, groupId])


  const bots = data?.conversation.group_bots || []
  const defaultBot = useMemo(() => {
    if (!data?.conversation) return null
    return bots.find((b) => b.id === data.conversation.default_bot_id) || null
  }, [bots, data?.conversation])
  const lastByBot = useMemo(() => {
    const out: Record<string, ConversationMessage | undefined> = {}
    for (const b of bots) {
      const msg = [...messages].reverse().find((m) => m.sender_bot_id === b.id)
      out[b.id] = msg
    }
    return out
  }, [bots, messages])
  const workspaceEnabled = Boolean(agentStatus?.exists || agentStatus?.running)
  const workspacePorts = agentStatus?.ports || []
  const workspaceIdePort = agentStatus?.ide_port || workspacePorts[0]?.host || 0
  const workspaceIdeUrl = groupId ? `/ide/${groupId}/` : ''
  const workspacePortsExhausted = Boolean(agentStatus?.running && !workspaceIdePort)
  const visibleFiles = (files?.items || []).filter((f) => !(f.is_dir && (f.path === '' || f.path === '.'))).slice(0, 6)

  const mentionOptions = useMemo(() => {
    if (!mention?.active) return []
    const q = mention.query.toLowerCase()
    const filtered = bots.filter((b) => b.slug.toLowerCase().startsWith(q))
    return filtered.length > 0 ? filtered : bots
  }, [mention, bots])

  function updateMentionState(nextText: string) {
    const el = inputRef.current
    if (!el) {
      setMention(null)
      return
    }
    const pos = el.selectionStart || 0
    const before = nextText.slice(0, pos)
    const at = before.lastIndexOf('@')
    if (at === -1) {
      setMention(null)
      return
    }
    if (at > 0 && !/\s/.test(before[at - 1])) {
      setMention(null)
      return
    }
    const query = before.slice(at + 1)
    if (!/^[a-zA-Z0-9_-]{0,48}$/.test(query)) {
      setMention(null)
      return
    }
    setMention({ active: true, query, index: 0, start: at, end: pos })
  }

  function applyMention(bot: GroupBot) {
    if (!mention) return
    const before = text.slice(0, mention.start)
    const after = text.slice(mention.end)
    const insert = `@${bot.slug} `
    const next = `${before}${insert}${after}`
    setText(next)
    setMention(null)
    requestAnimationFrame(() => {
      const el = inputRef.current
      if (!el) return
      const nextPos = before.length + insert.length
      el.focus()
      el.setSelectionRange(nextPos, nextPos)
    })
  }

  function beginWorkspaceResize(e: ReactMouseEvent<HTMLDivElement>) {
    if (workspaceCollapsed) return
    e.preventDefault()
    workspaceResizeActiveRef.current = true
    setResizeMode('workspace')
    workspaceResizeStartXRef.current = e.clientX
    workspaceResizeStartWidthRef.current = workspaceWidth
    const handleMove = (ev: MouseEvent) => {
      if (!workspaceResizeActiveRef.current) return
      const delta = workspaceResizeStartXRef.current - ev.clientX
      const next = clampWorkspaceWidth(workspaceResizeStartWidthRef.current + delta)
      setWorkspaceWidth(next)
    }
    const handleUp = () => {
      workspaceResizeActiveRef.current = false
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('mouseup', handleUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      setResizeMode(null)
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    window.addEventListener('mousemove', handleMove)
    window.addEventListener('mouseup', handleUp)
  }

  function beginSidebarResize(e: ReactMouseEvent<HTMLDivElement>) {
    if (sidebarCollapsed) return
    e.preventDefault()
    sidebarResizeActiveRef.current = true
    setResizeMode('sidebar')
    sidebarResizeStartXRef.current = e.clientX
    sidebarResizeStartWidthRef.current = sidebarWidth
    const handleMove = (ev: MouseEvent) => {
      if (!sidebarResizeActiveRef.current) return
      const delta = ev.clientX - sidebarResizeStartXRef.current
      const next = clampSidebarWidth(sidebarResizeStartWidthRef.current + delta)
      setSidebarWidth(next)
    }
    const handleUp = () => {
      sidebarResizeActiveRef.current = false
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('mouseup', handleUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      setResizeMode(null)
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    window.addEventListener('mousemove', handleMove)
    window.addEventListener('mouseup', handleUp)
  }

  async function sendMessage() {
    if (!groupId || !text.trim()) return
    if (!data?.conversation?.default_bot_id) {
      setSendErr('Pick a default assistant before starting the conversation.')
      return
    }
    setSending(true)
    setSendErr(null)
    try {
      await apiPost<GroupConversationDetail>(`/api/group-conversations/${groupId}/messages`, {
        text,
        sender_role: 'user',
        sender_name: 'User',
      })
      setText('')
      setMention(null)
      requestAnimationFrame(() => inputRef.current?.focus())
    } catch (e: any) {
      setSendErr(String(e?.message || e))
    } finally {
      setSending(false)
    }
  }

  async function resetConversation() {
    if (!groupId) return
    if (!window.confirm('Reset this group chat? This clears messages for the group and all individual logs.')) {
      return
    }
    setResetting(true)
    setResetErr(null)
    try {
      await apiPost(`/api/group-conversations/${groupId}/reset`, {})
      const d = await apiGet<GroupConversationDetail>(`/api/group-conversations/${groupId}?include_messages=false`)
      setData(d)
      setMessages([])
      setHasMore(true)
      setOldestCursor(null)
      await loadMessagesLatest(groupId)
    } catch (e: any) {
      setResetErr(String(e?.message || e))
    } finally {
      setResetting(false)
    }
  }

  async function deleteConversation() {
    if (!groupId) return
    if (!window.confirm('Delete this group chat? This will remove the group and all individual logs.')) {
      return
    }
    setDeleting(true)
    setDeleteErr(null)
    try {
      await apiDelete(`/api/group-conversations/${groupId}`)
      nav('/conversations')
    } catch (e: any) {
      setDeleteErr(String(e?.message || e))
    } finally {
      setDeleting(false)
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (mention?.active && mentionOptions.length > 0) {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setMention((prev) => (prev ? { ...prev, index: (prev.index + 1) % mentionOptions.length } : prev))
        return
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        setMention((prev) =>
          prev ? { ...prev, index: (prev.index - 1 + mentionOptions.length) % mentionOptions.length } : prev,
        )
        return
      }
      if (e.key === 'Enter') {
        e.preventDefault()
        applyMention(mentionOptions[mention.index] || mentionOptions[0])
        return
      }
      if (e.key === 'Escape') {
        e.preventDefault()
        setMention(null)
        return
      }
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void sendMessage()
    }
  }

  return (
    <div
      className={`chatLayout ${workspaceEnabled ? 'withWorkspace' : ''}`}
      style={{ ['--sidebar-width' as any]: `${(sidebarCollapsed ? 64 : sidebarWidth)}px` }}
    >
      {resizeMode ? <div className="resizeShield" /> : null}
      <aside className={`chatSidebar ${sidebarCollapsed ? 'collapsed' : ''}`}>
        <div className="sidebarResizeHandle" onMouseDown={beginSidebarResize} />
        <div className="chatSidebarHeader">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <div className="chatBrand">
              <span className="chatBrandDot" />
              GravexStudio
            </div>
            <button
              className="collapseBtn"
              onClick={() => setSidebarCollapsed((v) => !v)}
              aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {sidebarCollapsed ? <ChevronRightIcon /> : <ChevronLeftIcon />}
            </button>
          </div>
          <input className="chatSearch" placeholder="Search assistants or conversations" />
        </div>
        <div className="chatFilters">
          <span className="chatPill active">All</span>
          <span className="chatPill">Assistants</span>
          <span className="chatPill">Groups (Beta)</span>
        </div>
        <div className="chatSectionLabel">Assistants</div>
        <div className="chatSidebarScroll">
          {bots.map((b) => {
            const lastMsg = lastByBot[b.id]
            return (
              <div key={b.id} className={`assistantCard ${b.id === defaultBot?.id ? 'active' : ''}`}>
                <div className="assistantHeader">
                  <div className="assistantAvatar">{b.name.slice(0, 2).toUpperCase()}</div>
                  <div>
                    <strong>{b.name}</strong>
                    <div className="assistantMeta">1 conversation · {workingBots[b.id] ? '1 running task' : 'idle'}</div>
                  </div>
                  {b.id === defaultBot?.id ? <span className="assistantBadge">Default</span> : null}
                </div>
                <div className="assistantConversationRow">
                  <strong>{data?.conversation.title || 'Conversation'}</strong>
                  {lastMsg?.content ? lastMsg.content : 'No recent messages.'}
                </div>
              </div>
            )
          })}
        </div>
      </aside>

      <main className="chatMain">
        <div className="chatHeader">
          <div>
            <h2>{data?.conversation.title || 'Group chat'}</h2>
            <div className="muted">@{bots.map((b) => b.slug).join(' @')}</div>
          </div>
          <div className="chatHeaderActions">
            <div className="conversationSwitch">
              <select
                value={groupId}
                onChange={(e) => {
                  const nextId = e.target.value
                  if (nextId) nav(`/groups/${nextId}`)
                }}
              >
                {groupListLoading ? <option>Loading…</option> : null}
                {groupList.map((g) => (
                  <option key={g.id} value={g.id}>
                    {g.title || 'Group chat'}
                  </option>
                ))}
              </select>
              <button className="btn navPill" onClick={() => nav('/conversations')}>
                New conversation
              </button>
            </div>
            <div className="settingsWrapper">
              <button className="settingsBtn" onClick={() => setSettingsOpen((v) => !v)}>
                <Cog6ToothIcon />
              </button>
              {settingsOpen ? (
                <div className="settingsMenu">
                  <button className="settingsItem" onClick={() => nav('/dashboard')}>
                    Dashboard
                  </button>
                  <button className="settingsItem" onClick={() => nav('/keys')}>
                    Keys
                  </button>
                  <button className="settingsItem" onClick={() => nav('/developer')}>
                    Developer
                  </button>
                  <button className="settingsItem" onClick={() => nav('/bots')}>
                    Assistants
                  </button>
                  <button className="settingsItem" disabled={resetting} onClick={() => void resetConversation()}>
                    {resetting ? 'Resetting…' : 'Reset chat'}
                  </button>
                  <button className="settingsItem danger" disabled={deleting} onClick={() => void deleteConversation()}>
                    {deleting ? 'Deleting…' : 'Delete group'}
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        </div>

        {groupListErr ? <div className="alert">{groupListErr}</div> : null}
        {err ? <div className="alert">{err}</div> : null}
        {resetErr ? <div className="alert">{resetErr}</div> : null}
        {deleteErr ? <div className="alert">{deleteErr}</div> : null}

        <div className="chatShell">
          <div className="chatArea" ref={scrollRef}>
            {loading || loadingOlder || hasMore ? (
              <div className="muted" style={{ padding: '8px 0', textAlign: 'center' }}>
                {loading || loadingOlder ? (
                  <LoadingSpinner label="Loading older messages" />
                ) : (
                  <button className="btn ghost" onClick={() => void loadOlderMessages()} disabled={!hasMore}>
                    Load earlier messages
                  </button>
                )}
              </div>
            ) : null}
            {messages.map((m) => (
              <GroupMessageRow key={m.id} m={m} />
            ))}
          </div>
          <div className="chatComposerBar">
            <button className="iconBtn" title="Record">
              <MicrophoneIcon />
            </button>
            <div className="composerInput">
              {mention?.active && mentionOptions.length > 0 ? (
                <div className="mentionMenu">
                  {mentionOptions.map((b, idx) => (
                    <button
                      key={b.id}
                      type="button"
                      className={`mentionItem ${idx === mention.index ? 'active' : ''}`}
                      onMouseDown={(e) => {
                        e.preventDefault()
                        applyMention(b)
                      }}
                    >
                      <span>@{b.slug}</span>
                      <span className="muted">{b.name}</span>
                    </button>
                  ))}
                </div>
              ) : null}
              <textarea
                ref={inputRef}
                placeholder="Message the group (use @slug to mention an assistant)"
                value={text}
                onChange={(e) => {
                  const next = e.target.value
                  setText(next)
                  updateMentionState(next)
                }}
                onKeyDown={handleKeyDown}
                onClick={() => updateMentionState(text)}
                onKeyUp={() => updateMentionState(text)}
                rows={2}
              />
            </div>
            <button className="iconBtn" title="Attach">
              <PaperClipIcon />
            </button>
            <button
              className="btn primary"
              disabled={sending || !text.trim() || !data?.conversation?.default_bot_id}
              onClick={() => void sendMessage()}
            >
              {sending ? <LoadingSpinner label="Sending" /> : 'Send'}
            </button>
          </div>
          {Object.keys(workingBots).length ? (
            <div className="muted chatWorking">Working: {Object.values(workingBots).join(', ')}</div>
          ) : null}
          {sendErr ? <div className="alert">{sendErr}</div> : null}
        </div>
      </main>

      {workspaceEnabled ? (
        <aside
          className={`chatWorkspace ${workspaceCollapsed ? 'collapsed' : ''}`}
          style={!workspaceCollapsed ? { width: workspaceWidth } : undefined}
        >
          <div className="workspaceResizeHandle" onMouseDown={beginWorkspaceResize} />
          <div className={`workspaceHeader ${workspaceIDEFull ? 'fullIde' : ''}`}>
            <div>
              <h3>Workspace</h3>
              {!workspaceIDEFull ? <div className="muted">Container: {agentStatus?.running ? 'running' : 'idle'}</div> : null}
            </div>
            <button
              className="collapseBtn"
              onClick={() => setWorkspaceCollapsed((v) => !v)}
              aria-label={workspaceCollapsed ? 'Expand workspace' : 'Collapse workspace'}
              title={workspaceCollapsed ? 'Expand workspace' : 'Collapse workspace'}
            >
              {workspaceCollapsed ? <ChevronLeftIcon /> : <ChevronRightIcon />}
            </button>
          </div>
          <div className={`workspaceBody ${workspaceIDEFull ? 'fullIde' : ''}`}>
            {agentErr ? <div className="alert">{agentErr}</div> : null}
            {!workspaceIDEFull ? (
              <div className="workspaceCard">
                <div className="workspaceTitle">Status</div>
                <div className="workspaceRow">
                  <strong>Runtime</strong>
                  <span>{agentStatus?.exists ? 'Isolated Workspace' : '-'}</span>
                </div>
                <div className="workspaceRow">
                  <strong>Status</strong>
                  <span>{agentStatus?.running ? 'running' : agentStatus?.status || 'idle'}</span>
                </div>
                <div className="workspaceRow">
                  <strong>Container</strong>
                  <span>{agentStatus?.container_name || '-'}</span>
                </div>
              </div>
            ) : null}
            <div className={`workspaceCard ideCard ${workspaceIDEFull ? 'full' : ''}`}>
              {!workspaceIDEFull ? <div className="workspaceTitle">IDE (VS Code)</div> : null}
              {!agentStatus?.running ? (
                <div className="muted">Start the Isolated Workspace to enable the IDE.</div>
              ) : !workspaceIdePort ? (
                <div className={workspacePortsExhausted ? 'alert' : 'muted'}>
                  {workspacePortsExhausted
                    ? 'No available Isolated Workspace ports. Delete old Isolated Workspace containers from the Developer panel to free ports.'
                    : 'IDE port not assigned yet.'}
                </div>
              ) : (
                <>
                  {!workspaceIDEFull ? (
                    <div className="workspaceRow">
                      <strong>Port</strong>
                      <span className="mono">{workspaceIdePort}</span>
                    </div>
                  ) : null}
                  <div className="row" style={{ gap: 8, marginTop: 8 }}>
                    <a
                      className="iconBtn small"
                      href={workspaceIdeUrl}
                      target="_blank"
                      rel="noreferrer"
                      aria-label="Open IDE in new tab"
                      title="Open IDE in new tab"
                    >
                      <ArrowTopRightOnSquareIcon />
                    </a>
                    <button
                      className="iconBtn small"
                      onClick={() => setWorkspaceIDEOpen((v) => !v)}
                      aria-label={workspaceIDEOpen ? 'Hide IDE' : 'Embed IDE'}
                      title={workspaceIDEOpen ? 'Hide IDE' : 'Embed IDE'}
                    >
                      {workspaceIDEOpen ? <EyeSlashIcon /> : <EyeIcon />}
                    </button>
                    <button
                      className="iconBtn small"
                      onClick={() => {
                        setWorkspaceIDEFull((v) => !v)
                        if (!workspaceIDEOpen) setWorkspaceIDEOpen(true)
                      }}
                      aria-label={workspaceIDEFull ? 'Exit full screen' : 'Full screen'}
                      title={workspaceIDEFull ? 'Exit full screen' : 'Full screen'}
                    >
                      {workspaceIDEFull ? <ArrowsPointingInIcon /> : <ArrowsPointingOutIcon />}
                    </button>
                  </div>
                  {workspaceIDEOpen ? (
                    <iframe
                      title="Workspace IDE"
                      src={workspaceIdeUrl}
                      className="ideFrame"
                      style={{ height: workspaceIDEFull ? '100%' : 520, marginTop: 10 }}
                    />
                  ) : null}
                </>
              )}
            </div>
            {!workspaceIDEFull ? (
              <div className="workspaceCard">
                <div className="workspaceTitle">Files</div>
                {filesLoading ? (
                  <div className="muted">
                    <LoadingSpinner />
                  </div>
                ) : filesErr ? (
                  <div className="alert">{filesErr}</div>
                ) : (
                  <div className="workspaceFiles">
                    {visibleFiles.length === 0 ? <div className="muted">No files yet.</div> : null}
                    {visibleFiles.map((f) => (
                      <div key={f.path} className="workspaceRow">
                        <strong>{f.name}</strong>
                        <span>{f.is_dir ? 'folder' : f.size_bytes ? `${f.size_bytes} B` : '-'}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </aside>
      ) : null}
    </div>
  )
}

function GroupMessageRow({ m }: { m: ConversationMessage }) {
  const isUser = m.role === 'user'
  const isAssistant = m.role === 'assistant'
  const isTool = m.role === 'tool' || m.role === 'system'
  const cls = isUser ? 'bubble user' : isAssistant ? 'bubble assistant' : 'bubble tool'
  const rowCls = isUser ? 'msgRow user' : 'msgRow'
  const sender = m.sender_name || (isUser ? 'You' : isAssistant ? 'Assistant' : 'System')
  const label = `~${sender}`
  const body = isTool ? (m.tool_name ? `${m.tool_name}` : m.content) : m.content
  const bubbleStyle = isAssistant ? assistantBubbleStyle(m.sender_bot_id || sender) : undefined
  const citations = isAssistant ? normalizeCitations(m.citations) : []

  return (
    <div className={rowCls}>
      <div className={`avatar ${isAssistant ? 'assistant' : isTool ? 'tool' : 'user'}`}>
        {isUser ? <UserIcon /> : isTool ? <WrenchScrewdriverIcon /> : <CpuChipIcon />}
      </div>
      <div className={cls} style={bubbleStyle}>
        <div className="bubbleMeta" style={{ marginBottom: 6 }}>
          <span>{label}</span> <span className="muted">• {fmtIso(m.created_at)}</span>
        </div>
        <div className="bubbleText">
          <MarkdownText content={String(body || '')} />
        </div>
        {isAssistant && citations.length ? (
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
        {isTool ? (
          <details className="details">
            <summary>details</summary>
            <pre className="pre">{m.content}</pre>
          </details>
        ) : null}
      </div>
    </div>
  )
}

function normalizeCitations(citations?: ConversationMessage['citations']) {
  if (!Array.isArray(citations)) return []
  const seen = new Set<string>()
  return citations.filter((c) => {
    if (!c || typeof c.url !== 'string' || !c.url.trim()) return false
    const key = `${c.url}|${c.title || ''}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

function assistantBubbleStyle(key: string): React.CSSProperties {
  const palette = [
    { bg: 'rgba(129, 140, 248, 0.18)', border: 'rgba(129, 140, 248, 0.42)', glow: 'rgba(99, 102, 241, 0.2)' },
    { bg: 'rgba(56, 189, 248, 0.16)', border: 'rgba(56, 189, 248, 0.42)', glow: 'rgba(14, 165, 233, 0.18)' },
    { bg: 'rgba(74, 222, 128, 0.16)', border: 'rgba(74, 222, 128, 0.38)', glow: 'rgba(34, 197, 94, 0.18)' },
    { bg: 'rgba(251, 191, 36, 0.16)', border: 'rgba(251, 191, 36, 0.38)', glow: 'rgba(245, 158, 11, 0.18)' },
    { bg: 'rgba(244, 114, 182, 0.16)', border: 'rgba(244, 114, 182, 0.38)', glow: 'rgba(236, 72, 153, 0.18)' },
    { bg: 'rgba(167, 139, 250, 0.16)', border: 'rgba(167, 139, 250, 0.38)', glow: 'rgba(139, 92, 246, 0.18)' },
  ]
  let hash = 0
  for (let i = 0; i < key.length; i += 1) {
    hash = (hash * 31 + key.charCodeAt(i)) | 0
  }
  const idx = Math.abs(hash) % palette.length
  const color = palette[idx]
  return {
    ['--assistant-bg' as any]: color.bg,
    ['--assistant-border' as any]: color.border,
    ['--assistant-glow' as any]: color.glow,
  }
}
