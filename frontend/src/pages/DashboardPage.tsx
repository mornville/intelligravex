import { type Dispatch, type MouseEvent as ReactMouseEvent, type SetStateAction, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { apiDelete, apiGet, apiPost, BACKEND_URL, downloadFile } from '../api/client'
import { authHeader } from '../auth'
import LoadingSpinner from '../components/LoadingSpinner'
import MicTest, { type ChatCacheEntry } from '../components/MicTest'
import MarkdownText from '../components/MarkdownText'
import KeysPage from './KeysPage'
import DeveloperPage from './DeveloperPage'
import BotSettingsModal from '../components/BotSettingsModal'
import SelectField from '../components/SelectField'
import InlineHelpTip from '../components/InlineHelpTip'
import type {
  Bot,
  ConversationMessage,
  ConversationSummary,
  DataAgentStatus,
  GroupBot,
  GroupConversationDetail,
  GroupConversationSummary,
  ConversationFiles,
  HostAction,
  Options,
  WidgetConfig,
} from '../types'
import { fmtIso } from '../utils/format'
import { formatLocalModelToolSupport } from '../utils/localModels'
import { formatProviderLabel, orderProviderList } from '../utils/llmProviders'
import { useChatgptOauth } from '../hooks/useChatgptOauth'
import { useEscapeClose } from '../hooks/useEscapeClose'
import {
  ArrowsPointingInIcon,
  ArrowsPointingOutIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  Cog6ToothIcon,
  CpuChipIcon,
  EyeIcon,
  EyeSlashIcon,
  ArrowTopRightOnSquareIcon,
  PaperClipIcon,
  PlusIcon,
  TrashIcon,
  UserIcon,
  WrenchScrewdriverIcon,
  XMarkIcon,
} from '@heroicons/react/24/solid'

type FilterTab = 'all' | 'assistants' | 'groups'

type MentionState = {
  active: boolean
  query: string
  index: number
  start: number
  end: number
}

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

const PAGE_SIZE = 10

type MessageCursor = {
  created_at: string
  id: string
}

type FileTreeNode = {
  name: string
  path: string
  is_dir: boolean
  children: FileTreeNode[]
  size_bytes?: number | null
  mtime?: string
  download_url?: string | null
}

function getOldestGroupCursor(items: ConversationMessage[]): MessageCursor | null {
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

function fmtBytes(n: number): string {
  if (!Number.isFinite(n)) return '-'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return i === 0 ? `${v.toFixed(0)} ${units[i]}` : `${v.toFixed(1)} ${units[i]}`
}

function buildFileTree(items: ConversationFiles['items']): FileTreeNode {
  const root: FileTreeNode = { name: '', path: '', is_dir: true, children: [] }
  const nodes = new Map<string, FileTreeNode>()
  nodes.set('', root)

  function ensureNode(path: string, name: string, isDir: boolean, parent: FileTreeNode): FileTreeNode {
    const existing = nodes.get(path)
    if (existing) {
      if (isDir) existing.is_dir = true
      return existing
    }
    const node: FileTreeNode = { name, path, is_dir: isDir, children: [] }
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

  function sortTree(node: FileTreeNode): void {
    node.children.sort((a, b) => {
      if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1
      return a.name.localeCompare(b.name)
    })
    node.children.forEach(sortTree)
  }

  sortTree(root)
  return root
}

function renderFileTree(
  node: FileTreeNode,
  depth: number,
  expanded: Record<string, boolean>,
  setExpanded: Dispatch<SetStateAction<Record<string, boolean>>>,
  onDownload?: (downloadUrl: string, filename: string) => void,
): React.ReactNode {
  if (!node.children.length && node.path === '') return null
  if (node.path === '') {
    return node.children.map((child) => renderFileTree(child, depth, expanded, setExpanded, onDownload))
  }
  const isOpen = !!expanded[node.path]
  const indent = depth * 16
  const canDownload = Boolean(onDownload && node.download_url && !node.is_dir)
  const size = node.is_dir ? '-' : node.size_bytes === null ? '-' : fmtBytes(node.size_bytes || 0)
  const mtime = node.mtime ? new Date(node.mtime).toLocaleString() : '-'
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
        ? node.children.map((child) => renderFileTree(child, depth + 1, expanded, setExpanded, onDownload))
        : null}
    </div>
  )
}

export default function DashboardPage() {
  const [bots, setBots] = useState<Bot[]>([])
  const [groups, setGroups] = useState<GroupConversationSummary[]>([])
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [options, setOptions] = useState<Options | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)
  const [widgetBotId, setWidgetBotId] = useState<string | null>(null)
  const [widgetErr, setWidgetErr] = useState<string | null>(null)
  const [widgetSaving, setWidgetSaving] = useState(false)
  const [singleTabBlocked, setSingleTabBlocked] = useState(false)
  const [singleTabMessage, setSingleTabMessage] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [tab, setTab] = useState<FilterTab>('all')

  const [selectedType, setSelectedType] = useState<'assistant' | 'group'>('assistant')
  const [selectedBotId, setSelectedBotId] = useState<string | null>(null)
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null)

  const [botConversations, setBotConversations] = useState<ConversationSummary[]>([])
  const [botConvLoading, setBotConvLoading] = useState(false)
  const [botConvErr, setBotConvErr] = useState<string | null>(null)
  const [selectedConversationId, setSelectedConversationId] = useState<string>('')

  const [groupDetail, setGroupDetail] = useState<GroupConversationDetail | null>(null)
  const [groupLoading, setGroupLoading] = useState(false)
  const [groupErr, setGroupErr] = useState<string | null>(null)
  const [groupMessages, setGroupMessages] = useState<ConversationMessage[]>([])
  const [groupLoadingOlder, setGroupLoadingOlder] = useState(false)
  const [groupHasMore, setGroupHasMore] = useState(true)
  const [groupOldestCursor, setGroupOldestCursor] = useState<MessageCursor | null>(null)
  const [groupText, setGroupText] = useState('')
  const [groupSendErr, setGroupSendErr] = useState<string | null>(null)
  const [mention, setMention] = useState<MentionState | null>(null)
  const groupInputRef = useRef<HTMLTextAreaElement | null>(null)
  const groupScrollRef = useRef<HTMLDivElement | null>(null)
  const groupNearBottomRef = useRef(true)
  const groupPendingScrollAdjustRef = useRef<{ prevHeight: number; prevTop: number } | null>(null)
  const groupMarkReadTimerRef = useRef<number | null>(null)
  const groupUploadRef = useRef<HTMLInputElement | null>(null)
  const groupUploadFolderRef = useRef<HTMLInputElement | null>(null)
  const [groupUploadMenuOpen, setGroupUploadMenuOpen] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const [workingBots, setWorkingBots] = useState<Record<string, string>>({})
  const chatgptOauth = useChatgptOauth()

  const [settingsOpen, setSettingsOpen] = useState(false)
  const [showKeysModal, setShowKeysModal] = useState(false)
  const [showDeveloperModal, setShowDeveloperModal] = useState(false)
  const [showSettingsModal, setShowSettingsModal] = useState(false)
  const [settingsTab, setSettingsTab] = useState<'llm' | 'asr' | 'tts' | 'agent' | 'host' | 'tools'>('llm')
  const [resettingOnboarding, setResettingOnboarding] = useState(false)
  const [showCreateAssistant, setShowCreateAssistant] = useState(false)
  const [creatingAssistant, setCreatingAssistant] = useState(false)
  const [assistantErr, setAssistantErr] = useState<string | null>(null)
  function llmModels(provider: string, o: Options | null, fallback: string): string[] {
    if (provider === 'local') {
      const local = (o?.local_models || []).map((m) => m.id)
      const combined = [fallback, ...local].filter(Boolean)
      return Array.from(new Set(combined))
    }
    const base = provider === 'openrouter' ? o?.openrouter_models || [] : o?.openai_models || []
    if (!base.length) return [fallback]
    return base.includes(fallback) ? base : [fallback, ...base]
  }
  const [newBot, setNewBot] = useState({
    name: '',
    llm_provider: 'chatgpt',
    openai_model: 'o4-mini',
    openai_asr_model: 'gpt-4o-mini-transcribe',
    web_search_model: 'gpt-4o-mini',
    codex_model: 'gpt-5.1-codex-mini',
    system_prompt: 'You are a fast, helpful voice assistant. Keep answers concise unless asked.',
    language: 'en',
    openai_tts_model: 'gpt-4o-mini-tts',
    openai_tts_voice: 'alloy',
    openai_tts_speed: 1.0,
    start_message_mode: 'llm' as const,
    start_message_text: '',
  })
  const newBotLocalModel =
    (options?.local_models || []).find((m) => m.id === newBot.openai_model) || null
  const newBotVoiceRequiresOpenAI = newBot.llm_provider === 'chatgpt'
  const newBotNeedsChatgptAuth = newBot.llm_provider === 'chatgpt' && !chatgptOauth.ready
  const [showCreateGroup, setShowCreateGroup] = useState(false)
  const [groupTitle, setGroupTitle] = useState('')
  const [groupFilter, setGroupFilter] = useState('')
  const [groupSelected, setGroupSelected] = useState<string[]>([])
  const [groupDefaultId, setGroupDefaultId] = useState('')
  const [groupSaving, setGroupSaving] = useState(false)
  const [groupSaveErr, setGroupSaveErr] = useState<string | null>(null)
  const [startToken, setStartToken] = useState(0)
  const [refreshToken, setRefreshToken] = useState(0)
  const [isVisible, setIsVisible] = useState(
    typeof document !== 'undefined' ? document.visibilityState === 'visible' : true,
  )
  const [assistantConversationId, setAssistantConversationId] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadErr, setUploadErr] = useState<string | null>(null)
  const [previewByConversationId, setPreviewByConversationId] = useState<Record<string, string>>({})
  const [unseenByConversationId, setUnseenByConversationId] = useState<Record<string, number>>({})
  const [assistantStage, setAssistantStage] = useState<'disconnected' | 'idle' | 'init' | 'recording' | 'asr' | 'llm' | 'tts' | 'error'>('idle')
  const selectionRef = useRef<{ type: 'assistant' | 'group'; groupId: string | null; convId: string } | null>(null)
  const [chatCache, setChatCache] = useState<Record<string, ChatCacheEntry>>({})
  const bcRef = useRef<BroadcastChannel | null>(null)
  const reloadListsInFlight = useRef(false)
  const hostActionsInFlight = useRef(false)

  const [workspaceStatus, setWorkspaceStatus] = useState<DataAgentStatus | null>(null)
  const [workspaceErr, setWorkspaceErr] = useState<string | null>(null)
  const [workspaceIDEOpen, setWorkspaceIDEOpen] = useState(false)
  const [workspaceIDEFull, setWorkspaceIDEFull] = useState(false)
  const [workspaceWidth, setWorkspaceWidth] = useState(DEFAULT_WORKSPACE_WIDTH)
  const [workspaceCollapsed, setWorkspaceCollapsed] = useState(false)
  const workspaceResizeActiveRef = useRef(false)
  const workspaceResizeStartXRef = useRef(0)
  const workspaceResizeStartWidthRef = useRef(DEFAULT_WORKSPACE_WIDTH)
  const [resizeMode, setResizeMode] = useState<'workspace' | 'sidebar' | null>(null)
  const [sidebarWidth, setSidebarWidth] = useState(DEFAULT_SIDEBAR_WIDTH)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const sidebarResizeActiveRef = useRef(false)
  const sidebarResizeStartXRef = useRef(0)
  const sidebarResizeStartWidthRef = useRef(DEFAULT_SIDEBAR_WIDTH)
  const [files, setFiles] = useState<ConversationFiles | null>(null)
  const [filesErr, setFilesErr] = useState<string | null>(null)
  const [filesMsg, setFilesMsg] = useState<string | null>(null)
  const [filesLoading, setFilesLoading] = useState(false)
  const [filesRecursive, setFilesRecursive] = useState(true)
  const [filesHidden, setFilesHidden] = useState(false)
  const [filesExpanded, setFilesExpanded] = useState<Record<string, boolean>>({})
  const [hostActions, setHostActions] = useState<HostAction[]>([])
  const [hostActionsErr, setHostActionsErr] = useState<string | null>(null)
  const filesRequestSeqRef = useRef(0)
  const filesLoadingSeqRef = useRef(0)
  const filesNonSilentInFlightRef = useRef(false)
  const filesMsgTimerRef = useRef<number | null>(null)

  const anyModalOpen =
    showKeysModal || showDeveloperModal || showSettingsModal || showCreateAssistant || showCreateGroup

  useEscapeClose(() => {
    if (showKeysModal) setShowKeysModal(false)
    if (showDeveloperModal) setShowDeveloperModal(false)
    if (showSettingsModal) setShowSettingsModal(false)
    if (showCreateAssistant) setShowCreateAssistant(false)
    if (showCreateGroup) setShowCreateGroup(false)
  }, anyModalOpen)

  function isNotFoundError(e: any) {
    return String(e?.message || e).includes('HTTP 404')
  }

  function clearAssistantConversation() {
    setAssistantConversationId(null)
    setSelectedConversationId('')
    setBotConversations([])
  }

  function clearGroupConversation() {
    setSelectedGroupId(null)
    setGroupDetail(null)
    setGroupMessages([])
    setGroupHasMore(true)
    setGroupOldestCursor(null)
  }

  function clearActiveConversationState() {
    setWorkspaceStatus(null)
    setWorkspaceErr(null)
    setWorkspaceIDEOpen(false)
    setWorkspaceIDEFull(false)
    setFiles(null)
    setFilesErr(null)
    setFilesMsg(null)
    setFilesExpanded({})
    setHostActions([])
    setHostActionsErr(null)
    if (filesMsgTimerRef.current) {
      window.clearTimeout(filesMsgTimerRef.current)
      filesMsgTimerRef.current = null
    }
    if (selectedType === 'group') {
      clearGroupConversation()
    } else {
      clearAssistantConversation()
    }
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

  useEffect(() => {
    void (async () => {
      setLoading(true)
      setErr(null)
      try {
        const [b, g, c, o, w] = await Promise.all([
          apiGet<{ items: Bot[] }>('/api/bots'),
          apiGet<{ items: GroupConversationSummary[] }>('/api/group-conversations'),
          apiGet<{ items: ConversationSummary[] }>(`/api/conversations?page=1&page_size=200`),
          apiGet<Options>('/api/options'),
          apiGet<WidgetConfig>('/api/widget-config'),
        ])
        setBots(b.items)
        setGroups(g.items)
        setConversations(c.items)
        setOptions(o)
        if (o?.default_llm_provider || o?.default_llm_model) {
          setNewBot((p) => ({
            ...p,
            llm_provider: (o?.default_llm_provider || p.llm_provider) as any,
            openai_model: o?.default_llm_model || p.openai_model,
          }))
        }
        setWidgetBotId(w?.bot_id || null)
        const initialUnseen: Record<string, number> = {}
        c.items.forEach((item) => {
          initialUnseen[item.id] = item.unread_count || 0
        })
        g.items.forEach((item) => {
          initialUnseen[item.id] = item.unread_count || 0
        })
        setUnseenByConversationId(initialUnseen)
        let latestType: 'assistant' | 'group' | null = null
        let latestBotId: string | null = null
        let latestGroupId: string | null = null
        let latestTs = ''
        c.items.forEach((item) => {
          if (!latestTs || item.updated_at > latestTs) {
            latestTs = item.updated_at
            latestType = 'assistant'
            latestBotId = item.bot_id
            latestGroupId = null
          }
        })
        g.items.forEach((item) => {
          if (!latestTs || item.updated_at > latestTs) {
            latestTs = item.updated_at
            latestType = 'group'
            latestGroupId = item.id
            latestBotId = null
          }
        })
        if (latestType === 'group' && latestGroupId) {
          setSelectedType('group')
          setSelectedGroupId(latestGroupId)
        } else if (latestType === 'assistant' && latestBotId) {
          setSelectedType('assistant')
          setSelectedBotId(latestBotId)
        } else if (b.items.length) {
          setSelectedType('assistant')
          setSelectedBotId(b.items[0].id)
        } else if (g.items.length) {
          setSelectedType('group')
          setSelectedGroupId(g.items[0].id)
        }
      } catch (e: any) {
        setErr(String(e?.message || e))
      } finally {
        setLoading(false)
      }
    })()
  }, [])

  async function updateWidgetBot(nextBotId: string | null) {
    setWidgetSaving(true)
    setWidgetErr(null)
    try {
      const res = await apiPost<WidgetConfig>('/api/widget-config', { bot_id: nextBotId })
      setWidgetBotId(res?.bot_id || null)
    } catch (e: any) {
      setWidgetErr(String(e?.message || e))
    } finally {
      setWidgetSaving(false)
    }
  }

  useEffect(() => {
    const bc = new BroadcastChannel('gravex-single-tab')
    bcRef.current = bc
    const tabId = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
    const startedAt = Date.now()
    const announce = () => bc.postMessage({ type: 'hello', tabId, startedAt })
    bc.onmessage = (ev) => {
      const data = ev?.data
      if (!data || data.tabId === tabId) return
      if (data.type === 'hello') {
        if (data.startedAt < startedAt || (data.startedAt === startedAt && data.tabId < tabId)) {
          setSingleTabBlocked(true)
          setSingleTabMessage('Another GravexStudio tab is active. Please close this tab.')
        } else {
          bc.postMessage({ type: 'deny', tabId, target: data.tabId, startedAt })
        }
      } else if (data.type === 'deny' && data.target === tabId) {
        setSingleTabBlocked(true)
        setSingleTabMessage('Another GravexStudio tab is active. Please close this tab.')
      }
    }
    announce()
    const t = window.setInterval(announce, 5000)
    return () => {
      try {
        window.clearInterval(t)
        bc.close()
      } catch {
        // ignore
      }
    }
  }, [])

  useEffect(() => {
    const onVis = () => setIsVisible(document.visibilityState === 'visible')
    document.addEventListener('visibilitychange', onVis)
    return () => document.removeEventListener('visibilitychange', onVis)
  }, [])

  const q = query.trim().toLowerCase()
  const filteredBots = useMemo(() => {
    if (!q) return bots
    return bots.filter(
      (b) =>
        b.name.toLowerCase().includes(q) ||
        b.openai_model.toLowerCase().includes(q) ||
        (b.llm_provider || 'openai').toLowerCase().includes(q),
    )
  }, [bots, q])
  const filteredGroups = useMemo(() => {
    if (!q) return groups
    return groups.filter((g) => g.title.toLowerCase().includes(q))
  }, [groups, q])

  const showBots = tab !== 'groups'
  const showGroups = tab !== 'assistants'
  const filteredBotsForGroup = useMemo(() => {
    const q = groupFilter.trim().toLowerCase()
    if (!q) return bots
    return bots.filter((b) => b.name.toLowerCase().includes(q))
  }, [bots, groupFilter])
  const selectedBots = useMemo(() => bots.filter((b) => groupSelected.includes(b.id)), [bots, groupSelected])
  const selectedSlugMap = useMemo(() => buildSlugMap(selectedBots), [selectedBots])
  useEffect(() => {
    if (!groupSelected.length) {
      if (groupDefaultId) setGroupDefaultId('')
      return
    }
    if (!groupDefaultId || !groupSelected.includes(groupDefaultId)) {
      setGroupDefaultId(groupSelected[0])
    }
  }, [groupSelected, groupDefaultId])

  const latestConversationByBot = useMemo(() => {
    const map = new Map<string, ConversationSummary>()
    for (const c of conversations) {
      const prev = map.get(c.bot_id)
      if (!prev || c.updated_at > prev.updated_at) {
        map.set(c.bot_id, c)
      }
    }
    return map
  }, [conversations])

  const selectedBot = useMemo(() => {
    if (!selectedBotId) return null
    return bots.find((b) => b.id === selectedBotId) || null
  }, [bots, selectedBotId])

  useEffect(() => {
    if (!selectedBotId) return
    const immediate = conversations
      .filter((conv) => conv.bot_id === selectedBotId)
      .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
      .slice(0, 50)
    const fallbackLatest = immediate[0] || latestConversationByBot.get(selectedBotId)
    const initialId = fallbackLatest?.id || ''
    setBotConversations(immediate)
    setSelectedConversationId(initialId)
    setAssistantConversationId(null)
    void (async () => {
      setBotConvLoading(true)
      setBotConvErr(null)
      try {
        const c = await apiGet<{ items: ConversationSummary[] }>(
          `/api/conversations?page=1&page_size=50&bot_id=${selectedBotId}`,
        )
        const sorted = [...(c.items || [])].sort((a, b) => b.updated_at.localeCompare(a.updated_at))
        const finalList = sorted.length ? sorted : immediate
        setBotConversations(finalList)
        const nextId = finalList[0]?.id || fallbackLatest?.id || ''
        setSelectedConversationId((prev) => (prev === initialId ? nextId : prev))
      } catch (e: any) {
        setBotConvErr(String(e?.message || e))
      } finally {
        setBotConvLoading(false)
      }
    })()
  }, [selectedBotId])

  useEffect(() => {
    selectionRef.current = { type: selectedType, groupId: selectedGroupId, convId: selectedConversationId }
  }, [selectedType, selectedGroupId, selectedConversationId])

  useEffect(() => {
    if (selectedType !== 'assistant') return
    if (!assistantConversationId) return
    setSelectedConversationId((prev) => (prev === assistantConversationId ? prev : assistantConversationId))
  }, [assistantConversationId, selectedType])

  function clipPreview(text: string, max = 90) {
    const cleaned = (text || '').replace(/\s+/g, ' ').trim()
    if (!cleaned) return ''
    if (cleaned.length <= max) return cleaned
    return `${cleaned.slice(0, max - 3)}...`
  }

  function mergeGroupMessages(prev: ConversationMessage[], next: ConversationMessage[]) {
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

  useEffect(() => {
    if (selectedType !== 'group' || !selectedGroupId) return
    setGroupMessages([])
    setGroupHasMore(true)
    setGroupOldestCursor(null)
    groupNearBottomRef.current = true
    void (async () => {
      setGroupLoading(true)
      setGroupErr(null)
      try {
        const d = await apiGet<GroupConversationDetail>(
          `/api/group-conversations/${selectedGroupId}?include_messages=false`,
        )
        setGroupDetail(d)
        await loadGroupMessagesLatest(selectedGroupId)
      } catch (e: any) {
        setGroupErr(String(e?.message || e))
      } finally {
        setGroupLoading(false)
      }
    })()
  }, [selectedGroupId, selectedType])

  useLayoutEffect(() => {
    const el = groupScrollRef.current
    if (!el) return
    const pending = groupPendingScrollAdjustRef.current
    if (pending) {
      const nextHeight = el.scrollHeight
      el.scrollTop = pending.prevTop + (nextHeight - pending.prevHeight)
      groupPendingScrollAdjustRef.current = null
      return
    }
    if (groupNearBottomRef.current) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
    }
  }, [groupMessages.length, selectedGroupId])

  useEffect(() => {
    const el = groupScrollRef.current
    if (!el) return
    const onScroll = () => {
      const nearBottom = el.scrollHeight - (el.scrollTop + el.clientHeight) < 80
      groupNearBottomRef.current = nearBottom
      const nearTop = el.scrollTop <= 40
      if (nearTop && groupHasMore && !groupLoadingOlder) {
        void loadGroupMessagesOlder()
      }
    }
    el.addEventListener('scroll', onScroll)
    return () => el.removeEventListener('scroll', onScroll)
  }, [groupHasMore, groupLoadingOlder, selectedGroupId])

  useEffect(() => {
    if (selectedType !== 'group' || !selectedGroupId) return
    if (!groupMessages.length || !isVisible) return
    if (groupMarkReadTimerRef.current) window.clearTimeout(groupMarkReadTimerRef.current)
    groupMarkReadTimerRef.current = window.setTimeout(() => {
      void apiPost(`/api/conversations/${selectedGroupId}/read`, {})
    }, 400)
    return () => {
      if (groupMarkReadTimerRef.current) {
        window.clearTimeout(groupMarkReadTimerRef.current)
        groupMarkReadTimerRef.current = null
      }
    }
  }, [groupMessages.length, selectedGroupId, selectedType, isVisible])


  useEffect(() => {
    const active = selectedType === 'group' ? selectedGroupId : selectedConversationId
    if (!active) return
    setUnseenByConversationId((prev) => ({ ...prev, [active]: 0 }))
  }, [selectedType, selectedGroupId, selectedConversationId])

  useEffect(() => {
    if (!selectedBotId) return
    if (selectedConversationId) return
    const latest = latestConversationByBot.get(selectedBotId)
    if (latest?.id) {
      setSelectedConversationId(latest.id)
    }
  }, [selectedBotId, selectedConversationId, latestConversationByBot])

  function wsBase() {
    try {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      return `${proto}//${window.location.host}`
    } catch {
      return 'ws://127.0.0.1:8000'
    }
  }

  useEffect(() => {
    if (selectedType !== 'group' || !selectedGroupId) return
    const ws = new WebSocket(`${wsBase()}/ws/groups/${selectedGroupId}`)
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
        setGroupMessages((prev) => mergeGroupMessages(prev, [payload.message]))
        if (selectedGroupId) {
          setUnseenByConversationId((prev) => ({ ...prev, [selectedGroupId]: 0 }))
          const preview = clipPreview(payload.message.content || '')
          if (preview) {
            setPreviewByConversationId((prev) => ({ ...prev, [selectedGroupId]: preview }))
          }
        }
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
        setGroupMessages([])
        setGroupHasMore(true)
        setGroupOldestCursor(null)
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
  }, [selectedGroupId, selectedType])

  const groupBots = groupDetail?.conversation.group_bots || []

  const mentionOptions = useMemo(() => {
    if (!mention?.active) return []
    const q = mention.query.toLowerCase()
    const filtered = groupBots.filter((b) => b.slug.toLowerCase().startsWith(q))
    return filtered.length > 0 ? filtered : groupBots
  }, [mention, groupBots])

  function updateMentionState(nextText: string) {
    const el = groupInputRef.current
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
    const before = groupText.slice(0, mention.start)
    const after = groupText.slice(mention.end)
    const insert = `@${bot.slug} `
    const next = `${before}${insert}${after}`
    setGroupText(next)
    setMention(null)
    requestAnimationFrame(() => {
      const el = groupInputRef.current
      if (!el) return
      const nextPos = before.length + insert.length
      el.focus()
      el.setSelectionRange(nextPos, nextPos)
    })
  }

  async function loadGroupMessagesLatest(groupId: string) {
    try {
      const d = await apiGet<{ messages: ConversationMessage[] }>(
        `/api/group-conversations/${groupId}/messages?limit=${PAGE_SIZE}&order=desc`,
      )
      const raw = Array.isArray(d.messages) ? d.messages : []
      const mapped = raw.reverse()
      setGroupMessages(mapped)
      setGroupHasMore(raw.length === PAGE_SIZE)
      setGroupOldestCursor(getOldestGroupCursor(mapped))
    } catch (e: any) {
      setGroupErr(String(e?.message || e))
    }
  }

  async function loadGroupMessagesOlder() {
    if (!selectedGroupId || groupLoadingOlder || !groupOldestCursor) return
    const el = groupScrollRef.current
    if (el) {
      groupPendingScrollAdjustRef.current = { prevHeight: el.scrollHeight, prevTop: el.scrollTop }
    }
    setGroupLoadingOlder(true)
    let didAdd = false
    try {
      const before = encodeURIComponent(groupOldestCursor.created_at)
      const beforeId = encodeURIComponent(groupOldestCursor.id)
      const d = await apiGet<{ messages: ConversationMessage[] }>(
        `/api/group-conversations/${selectedGroupId}/messages?before=${before}&before_id=${beforeId}&limit=${PAGE_SIZE}&order=desc`,
      )
      const raw = Array.isArray(d.messages) ? d.messages : []
      const mapped = raw.reverse()
      if (mapped.length) {
        setGroupMessages((prev) => mergeGroupMessages(prev, mapped))
        setGroupOldestCursor(getOldestGroupCursor(mapped) || groupOldestCursor)
        didAdd = true
      }
      if (raw.length < PAGE_SIZE) setGroupHasMore(false)
    } catch {
      // ignore
    } finally {
      setGroupLoadingOlder(false)
      if (!didAdd) groupPendingScrollAdjustRef.current = null
    }
  }

  async function sendGroupMessage() {
    if (!selectedGroupId || !groupText.trim()) return
    if (!groupDetail?.conversation.default_bot_id) {
      setGroupSendErr('Pick a default assistant before starting the group chat.')
      return
    }
    setGroupSendErr(null)
    try {
      await apiPost(`/api/group-conversations/${selectedGroupId}/messages`, {
        text: groupText,
        sender_role: 'user',
        sender_name: 'User',
      })
      setGroupText('')
      setMention(null)
      requestAnimationFrame(() => groupInputRef.current?.focus())
    } catch (e: any) {
      setGroupSendErr(String(e?.message || e))
    }
  }

  const visibleGroupMessages = groupMessages.filter((m) => m.role !== 'tool' && m.role !== 'system')
  const activeConversationId = selectedType === 'group' ? selectedGroupId : assistantConversationId
  const activeBotId =
    selectedType === 'group'
      ? groupDetail?.conversation.default_bot_id || groups.find((g) => g.id === selectedGroupId)?.default_bot_id || null
      : selectedBotId
  const activeBot = bots.find((b) => b.id === activeBotId) || null
  const canUpload = Boolean(activeConversationId && activeBot?.enable_data_agent)
  const uploadDisabledReason = !activeBot?.enable_data_agent
    ? 'Enable Isolated Workspace to upload files.'
    : !activeConversationId
      ? 'Start a conversation to upload files.'
      : ''
  useEffect(() => {
    if (!activeConversationId) {
      setWorkspaceStatus(null)
      setFiles(null)
      return
    }
    setFilesExpanded({})
    void loadWorkspaceStatus(activeConversationId)
    void loadFiles(activeConversationId, filesRecursive, filesHidden)
  }, [activeConversationId])

  useEffect(() => {
    if (!activeConversationId) return
    setUnseenByConversationId((prev) => ({ ...prev, [activeConversationId]: 0 }))
  }, [activeConversationId])

  useEffect(() => {
    if (!activeConversationId) {
      setHostActions([])
      return
    }
    if (!isVisible) return
    void loadHostActions(activeConversationId)
    const id = window.setInterval(() => {
      void loadHostActions(activeConversationId)
    }, 5000)
    return () => window.clearInterval(id)
  }, [activeConversationId, isVisible])

  async function loadWorkspaceStatus(convId?: string | null) {
    const id = convId || activeConversationId
    if (!id) return
    setWorkspaceErr(null)
    try {
      const s = await apiGet<DataAgentStatus>(`/api/conversations/${id}/data-agent`)
      setWorkspaceStatus(s)
    } catch (e: any) {
      if (isNotFoundError(e) && id === activeConversationId) {
        clearActiveConversationState()
        return
      }
      setWorkspaceErr(String(e?.message || e))
    }
  }

  async function loadFiles(convId?: string | null, recursive?: boolean, includeHidden?: boolean, silent?: boolean) {
    const id = convId || activeConversationId
    if (!id) return
    if (silent && filesNonSilentInFlightRef.current) return
    const rec = recursive ?? filesRecursive
    const hidden = includeHidden ?? filesHidden
    const params = new URLSearchParams()
    if (rec) params.set('recursive', '1')
    if (hidden) params.set('include_hidden', '1')
    const seq = ++filesRequestSeqRef.current
    const loadingSeq = !silent ? ++filesLoadingSeqRef.current : 0
    if (!silent) {
      filesNonSilentInFlightRef.current = true
      setFilesLoading(true)
      setFilesErr(null)
    }
    try {
      const f = await apiGet<ConversationFiles>(`/api/conversations/${id}/files?${params.toString()}`)
      if (seq !== filesRequestSeqRef.current) return
      setFiles(f)
      setFilesRecursive(rec)
      setFilesHidden(hidden)
      if (silent) setFilesErr(null)
    } catch (e: any) {
      if (seq !== filesRequestSeqRef.current) return
      if (isNotFoundError(e) && id === activeConversationId) {
        clearActiveConversationState()
        return
      }
      setFilesErr(String(e?.message || e))
    } finally {
      if (!silent && loadingSeq === filesLoadingSeqRef.current) {
        filesNonSilentInFlightRef.current = false
        setFilesLoading(false)
      }
    }
  }

  async function handleFileDownload(downloadUrl: string, filename: string) {
    setFilesErr(null)
    setFilesMsg(null)
    try {
      await downloadFile(downloadUrl, filename || 'download')
      setFilesMsg(`Downloaded ${filename || 'file'}.`)
      if (filesMsgTimerRef.current) window.clearTimeout(filesMsgTimerRef.current)
      filesMsgTimerRef.current = window.setTimeout(() => setFilesMsg(null), 2500)
    } catch (e: any) {
      setFilesErr(String(e?.message || e))
    }
  }

  useEffect(() => {
    if (!activeConversationId || !isVisible) return
    const id = window.setInterval(() => {
      void loadWorkspaceStatus(activeConversationId)
    }, 15000)
    return () => window.clearInterval(id)
  }, [activeConversationId, isVisible])

  useEffect(() => {
    if (!activeConversationId || !isVisible || !workspaceStatus?.running) return
    void loadFiles(activeConversationId, filesRecursive, filesHidden, true)
    const id = window.setInterval(() => {
      void loadFiles(activeConversationId, filesRecursive, filesHidden, true)
    }, 15000)
    return () => window.clearInterval(id)
  }, [activeConversationId, filesHidden, filesRecursive, isVisible, workspaceStatus?.running])

  useEffect(() => {
    return () => {
      if (filesMsgTimerRef.current) {
        window.clearTimeout(filesMsgTimerRef.current)
      }
    }
  }, [])

  const fileItems = (files?.items || []).filter((it) => it.path && it.path !== '.')
  const fileTree = useMemo(() => buildFileTree(fileItems), [fileItems])
  const workspaceStatusLabel = !activeConversationId
    ? '-'
    : !workspaceStatus
      ? 'loading'
      : !workspaceStatus.docker_available
        ? 'docker unavailable'
        : workspaceStatus.exists
          ? workspaceStatus.running
            ? 'running'
            : workspaceStatus.status || 'stopped'
          : 'not started'
  const workspacePorts = workspaceStatus?.ports || []
  const workspaceIdePort = workspaceStatus?.ide_port || workspacePorts[0]?.host || 0
  const workspaceIdeUrl = activeConversationId ? `/ide/${activeConversationId}/` : ''
  const workspacePortsExhausted = Boolean(activeConversationId && workspaceStatus?.running && !workspaceIdePort)

  function conversationTitle(summary: ConversationSummary) {
    const raw = String(summary.first_user_message || '').trim()
    if (!raw) return 'Conversation'
    return raw.length > 30 ? `${raw.slice(0, 30)}...` : raw
  }

  useEffect(() => {
    if (!fileItems.length) return
    if (Object.keys(filesExpanded).length) return
    const next: Record<string, boolean> = {}
    for (const child of fileTree.children) {
      if (child.is_dir) next[child.path] = true
    }
    setFilesExpanded(next)
  }, [fileTree, fileItems.length, filesExpanded])

  async function reloadLists() {
    if (reloadListsInFlight.current) return
    reloadListsInFlight.current = true
    try {
      const [b, g, c] = await Promise.all([
        apiGet<{ items: Bot[] }>('/api/bots'),
        apiGet<{ items: GroupConversationSummary[] }>('/api/group-conversations'),
        apiGet<{ items: ConversationSummary[] }>(`/api/conversations?page=1&page_size=200`),
      ])
      setBots(b.items)
      setGroups(g.items)
      setConversations(c.items)
      if (selectedBotId && !b.items.some((bot) => bot.id === selectedBotId)) {
        clearAssistantConversation()
        setSelectedBotId(b.items[0]?.id || null)
      }
      if (selectedGroupId && !g.items.some((group) => group.id === selectedGroupId)) {
        if (selectedType === 'group') {
          clearGroupConversation()
        } else {
          setSelectedGroupId(null)
        }
      }
      const sel = selectionRef.current
      const active = sel?.type === 'group' ? sel?.groupId : sel?.convId
      const nextUnseen: Record<string, number> = {}
      c.items.forEach((item) => {
        const count = item.unread_count || 0
        nextUnseen[item.id] = item.id === active ? 0 : count
      })
      g.items.forEach((item) => {
        const count = item.unread_count || 0
        nextUnseen[item.id] = item.id === active ? 0 : count
      })
      setUnseenByConversationId(nextUnseen)
    } catch {
      // ignore
    } finally {
      reloadListsInFlight.current = false
    }
  }

  function handleCacheUpdate(conversationId: string, entry: ChatCacheEntry) {
    if (!conversationId) return
    setChatCache((prev) => ({ ...prev, [conversationId]: entry }))
    if (entry?.items?.length) {
      const last = [...entry.items].reverse().find((m) => m.role !== 'tool')
      if (last?.text) {
        setPreviewByConversationId((prev) => ({ ...prev, [conversationId]: clipPreview(last.text) }))
      }
    }
  }

  function notifySync() {
    // no-op when single-tab only
  }

  useEffect(() => {
    if (loading) return
    if (!isVisible) return
    const id = window.setInterval(() => {
      void reloadLists()
    }, 10000)
    return () => window.clearInterval(id)
  }, [loading, isVisible])

  async function refreshFiles(convId?: string | null) {
    const id = convId || activeConversationId
    if (!id) return
    await loadFiles(id, filesRecursive, filesHidden)
  }

  async function loadHostActions(convId?: string) {
    const id = convId || activeConversationId
    if (!id) return
    if (hostActionsInFlight.current) return
    hostActionsInFlight.current = true
    setHostActionsErr(null)
    try {
      const res = await apiGet<{ items: HostAction[] }>(`/api/conversations/${id}/host-actions`)
      setHostActions(res.items || [])
    } catch (e: any) {
      if (isNotFoundError(e) && id === activeConversationId) {
        clearActiveConversationState()
        return
      }
      setHostActionsErr(String(e?.message || e))
    } finally {
      hostActionsInFlight.current = false
    }
  }

  function hostActionRequiresApproval(action: HostAction) {
    const botId = action.requested_by_bot_id || null
    const sourceBot = botId ? bots.find((b) => b.id === botId) : activeBot
    return Boolean(sourceBot?.require_host_action_approval)
  }

  async function runHostAction(action: HostAction) {
    try {
      await apiPost(`/api/host-actions/${action.id}/run`, {})
      await loadHostActions()
      setRefreshToken((v) => v + 1)
    } catch (e: any) {
      setHostActionsErr(String(e?.message || e))
    }
  }

  function appendFiles(form: FormData, list: FileList) {
    Array.from(list).forEach((f) => {
      const rel = (f as any).webkitRelativePath || ''
      if (rel) form.append('files', f, rel)
      else form.append('files', f)
    })
  }

  async function uploadConversationFiles(list: FileList) {
    if (!activeConversationId || !list || list.length === 0) return
    const form = new FormData()
    appendFiles(form, list)
    setUploading(true)
    setUploadErr(null)
    try {
      const res = await fetch(`${BACKEND_URL}/api/conversations/${activeConversationId}/files/upload`, {
        method: 'POST',
        headers: { ...authHeader() },
        body: form,
      })
      if (!res.ok) {
        let msg = `HTTP ${res.status}`
        try {
          const j = await res.json()
          if (j && typeof j === 'object' && 'detail' in j) msg = String((j as any).detail)
        } catch {
          // ignore
        }
        throw new Error(msg)
      }
      await refreshFiles(activeConversationId)
    } catch (e: any) {
      setUploadErr(String(e?.message || e))
    } finally {
      setUploading(false)
    }
  }

  async function createAssistant() {
    if (!newBot.name.trim()) return
    setCreatingAssistant(true)
    setAssistantErr(null)
    try {
      await apiPost('/api/bots', { ...newBot })
      setShowCreateAssistant(false)
      setNewBot((p) => ({ ...p, name: '' }))
      await reloadLists()
    } catch (e: any) {
      setAssistantErr(String(e?.message || e))
    } finally {
      setCreatingAssistant(false)
    }
  }

  async function deleteAssistantById(botId: string) {
    if (!botId) return
    const target = bots.find((b) => b.id === botId)
    const ok = window.confirm(`Delete assistant \"${target?.name || 'assistant'}\"? This removes its conversations.`)
    if (!ok) return
    try {
      await apiDelete(`/api/bots/${botId}`)
      if (selectedType === 'assistant' && selectedBotId === botId) {
        clearAssistantConversation()
      }
      await reloadLists()
      if (selectedBotId === botId) {
        const next = bots.filter((b) => b.id !== botId)[0]
        setSelectedBotId(next?.id || null)
      }
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  async function deleteConversation() {
    if (!selectedConversationId) return
    const conv = botConversations.find((c) => c.id === selectedConversationId)
    const title = conv ? conversationTitle(conv) : 'Conversation'
    const ok = window.confirm(`Delete conversation \"${title}\"? This removes its messages.`)
    if (!ok) return
    try {
      await apiDelete(`/api/conversations/${selectedConversationId}`)
      clearAssistantConversation()
      await reloadLists()
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  function openSettingsTab(tab: 'llm' | 'asr' | 'tts' | 'agent' | 'host' | 'tools') {
    setSettingsTab(tab)
    setShowSettingsModal(true)
  }

  async function createGroup() {
    if (!groupTitle.trim() || groupSelected.length === 0 || !groupDefaultId) return
    setGroupSaving(true)
    setGroupSaveErr(null)
    try {
      const payload = await apiPost<GroupConversationDetail>('/api/group-conversations', {
        title: groupTitle.trim(),
        bot_ids: groupSelected,
        default_bot_id: groupDefaultId,
      })
      setShowCreateGroup(false)
      setGroupTitle('')
      setGroupFilter('')
      setGroupSelected([])
      setGroupDefaultId('')
      await reloadLists()
      setSelectedType('group')
      setSelectedGroupId(payload.conversation.id)
    } catch (e: any) {
      setGroupSaveErr(String(e?.message || e))
    } finally {
      setGroupSaving(false)
    }
  }

  async function deleteGroup() {
    if (!selectedGroupId) return
    const target = groups.find((g) => g.id === selectedGroupId)
    const ok = window.confirm(`Delete group \"${target?.title || 'group'}\"? This removes its logs.`)
    if (!ok) return
    try {
      await apiDelete(`/api/group-conversations/${selectedGroupId}`)
      await reloadLists()
      const next = groups.filter((g) => g.id !== selectedGroupId)[0]
      setSelectedGroupId(next?.id || null)
      if (!next && bots[0]) {
        setSelectedType('assistant')
        setSelectedBotId(bots[0].id)
      }
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  async function resetGroupConversation() {
    if (!selectedGroupId) return
    const ok = window.confirm('Reset this group chat? This clears the group thread and individual logs.')
    if (!ok) return
    try {
      const res = await apiPost<GroupConversationDetail>(`/api/group-conversations/${selectedGroupId}/reset`, {})
      setGroupDetail(res)
      setGroupMessages([])
      setGroupHasMore(true)
      setGroupOldestCursor(null)
      void loadGroupMessagesLatest(selectedGroupId)
      setWorkingBots({})
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  async function resetOnboarding() {
    const ok = window.confirm('Remove all keys and restart onboarding? This clears stored API keys and client keys.')
    if (!ok) return
    setResettingOnboarding(true)
    try {
      await apiPost('/api/onboarding/reset', {})
      window.location.reload()
    } catch (e: any) {
      window.alert(String(e?.message || e))
    } finally {
      setResettingOnboarding(false)
    }
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

  function toggleSelected(botId: string) {
    setGroupSelected((prev) => (prev.includes(botId) ? prev.filter((id) => id !== botId) : [...prev, botId]))
  }

  if (singleTabBlocked) {
    return (
      <div className="singleTabBlock">
        <div className="singleTabCard">
          <div className="chatBrand">
            <span className="chatBrandDot" />
            GravexStudio
          </div>
          <div className="cardTitle">Single‑tab mode</div>
          <div className="muted">
            {singleTabMessage || 'Another GravexStudio tab is active. Please close this tab.'}
          </div>
          <button className="btn" onClick={() => window.location.reload()}>
            Reload
          </button>
        </div>
      </div>
    )
  }

  return (
    <div
      className="chatLayout withWorkspace"
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
          <input
            className="chatSearch"
            placeholder="Search assistants or conversations"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <div className="chatFilters">
          <button className={`chatPill ${tab === 'all' ? 'active' : ''}`} onClick={() => setTab('all')}>
            All
          </button>
          <button className={`chatPill ${tab === 'assistants' ? 'active' : ''}`} onClick={() => setTab('assistants')}>
            Assistants
          </button>
          <button className={`chatPill ${tab === 'groups' ? 'active' : ''}`} onClick={() => setTab('groups')}>
            Groups
          </button>
        </div>
        <div className="chatSidebarScroll">
          {loading ? (
            <div className="muted">
              <LoadingSpinner />
            </div>
          ) : (
            <>
              {err ? <div className="alert">{err}</div> : null}
              {widgetErr ? <div className="alert">{widgetErr}</div> : null}
              {showBots ? (
                <>
                  <div className="chatSectionLabelRow">
                    <div className="chatSectionLabel">Assistants</div>
                    <button className="iconBtn small" onClick={() => setShowCreateAssistant(true)} title="New assistant">
                      <PlusIcon />
                    </button>
                  </div>
                  {filteredBots.map((b) => {
                    const latest = latestConversationByBot.get(b.id)
                    const preview = latest?.id
                      ? previewByConversationId[latest.id] || latest?.last_message_preview || ''
                      : ''
                    const unseen = latest?.id
                      ? unseenByConversationId[latest.id] ?? latest?.unread_count ?? 0
                      : 0
                    const isActive = selectedType === 'assistant' && selectedBotId === b.id
                    const showTyping = isActive && ['init', 'recording', 'asr', 'llm', 'tts'].includes(assistantStage)
                    return (
                      <button
                        key={b.id}
                        className={`waAssistantCard ${selectedType === 'assistant' && selectedBotId === b.id ? 'active' : ''}`}
                        onClick={() => {
                          setSelectedType('assistant')
                          if (selectedBotId !== b.id) setSelectedConversationId('')
                          setSelectedBotId(b.id)
                        }}
                      >
                        <div className="waAssistantHeader">
                          <div className="waAvatar">{b.name.slice(0, 2).toUpperCase()}</div>
                          <div className="waAssistantTitle">
                            <div className="waAssistantName">{b.name}</div>
                            <div className="waConversationRow">
                              {preview
                                ? preview
                                : latest
                                  ? unseen > 0
                                    ? 'New messages'
                                    : 'No messages yet'
                                  : 'No conversations yet'}
                            </div>
                          </div>
                          <button
                            className="iconBtn small danger"
                            title="Delete assistant"
                            onClick={(e) => {
                              e.stopPropagation()
                              void deleteAssistantById(b.id)
                            }}
                          >
                            <TrashIcon />
                          </button>
                          {showTyping ? <span className="waTyping">typing…</span> : unseen > 0 ? <span className="waUnreadBadge">{unseen}</span> : null}
                        </div>
                      </button>
                    )
                  })}
                </>
              ) : null}

              {showGroups ? (
                <>
                  <div className="chatSectionLabelRow">
                    <div className="chatSectionLabel">Groups</div>
                    <button className="iconBtn small" onClick={() => setShowCreateGroup(true)} title="New group">
                      <PlusIcon />
                    </button>
                  </div>
                  {filteredGroups.map((g) => {
                    const preview = previewByConversationId[g.id] || g.last_message_preview || ''
                    const unseen = unseenByConversationId[g.id] ?? g.unread_count ?? 0
                    const showTyping =
                      selectedType === 'group' && selectedGroupId === g.id && Object.keys(workingBots).length > 0
                    return (
                      <button
                        key={g.id}
                        className={`waAssistantCard ${selectedType === 'group' && selectedGroupId === g.id ? 'active' : ''}`}
                        onClick={() => {
                          setSelectedType('group')
                          setSelectedGroupId(g.id)
                        }}
                      >
                        <div className="waAssistantHeader">
                          <div className="waAvatar">{g.title.slice(0, 2).toUpperCase()}</div>
                          <div className="waAssistantTitle">
                            <div className="waAssistantName">{g.title || 'Group chat'}</div>
                            <div className="waConversationRow">
                              {preview ? preview : unseen > 0 ? 'New messages' : 'No messages yet'}
                            </div>
                          </div>
                          {showTyping ? <span className="waTyping">typing…</span> : unseen > 0 ? <span className="waUnreadBadge">{unseen}</span> : null}
                        </div>
                      </button>
                    )
                  })}
                </>
              ) : null}
            </>
          )}
        </div>
      </aside>

      <main className="chatMain">
        <div className="chatHeader">
          <div>
            <h2>
              {selectedType === 'group'
                ? groupDetail?.conversation.title || 'Group chat'
                : bots.find((b) => b.id === selectedBotId)?.name || 'Assistant'}
            </h2>
            <div className="muted">
              {selectedType === 'group'
                ? `@${groupBots.map((b) => b.slug).join(' @')}`
                : (() => {
                    const sel = bots.find((b) => b.id === selectedBotId)
                    if (!sel) return ''
                    return `${sel.openai_model} · ${formatProviderLabel(sel.llm_provider || 'openai')}`
                  })()}
            </div>
          </div>
          <div className="chatHeaderActions">
            {selectedType === 'group' ? (
              <select
                value={selectedGroupId || ''}
                onChange={(e) => setSelectedGroupId(e.target.value)}
                className="conversationSelect"
              >
                {groups.map((g) => (
                  <option key={g.id} value={g.id}>
                    {g.title || 'Group chat'}
                  </option>
                ))}
              </select>
            ) : (
              <select
                value={selectedConversationId}
                onChange={(e) => setSelectedConversationId(e.target.value)}
                className="conversationSelect"
              >
                {botConvLoading ? <option>Loading…</option> : null}
                {botConversations.length === 0 ? <option value="">No conversations</option> : null}
                {botConversations.map((c) => (
                  <option key={c.id} value={c.id}>
                    {conversationTitle(c)} · {fmtIso(c.updated_at)}
                  </option>
                ))}
              </select>
            )}
            {selectedType === 'assistant' ? (
              <button
                className={`btn navPill ${!selectedConversationId ? 'attentionBtn' : ''}`}
                onClick={() => {
                  setStartToken((t) => t + 1)
                }}
              >
                New conversation
              </button>
            ) : null}
            {selectedType === 'assistant' && selectedBotId ? (
              <button
                className={`btn ${widgetBotId === selectedBotId ? 'primary' : ''}`}
                onClick={() => void updateWidgetBot(selectedBotId)}
                disabled={widgetSaving || widgetBotId === selectedBotId}
                title="Set this assistant as the mic overlay"
              >
                {widgetBotId === selectedBotId ? 'Widget enabled' : 'Enable widget'}
              </button>
            ) : null}
            {selectedType === 'group' ? (
              <button className="btn" onClick={() => void resetGroupConversation()}>
                Reset chat
              </button>
            ) : null}
            {selectedType === 'assistant' ? (
              <button
                className="iconBtn danger"
                title="Delete conversation"
                onClick={() => void deleteConversation()}
                disabled={!selectedConversationId}
              >
                <TrashIcon />
              </button>
            ) : null}
            {selectedType === 'group' ? (
              <button className="iconBtn danger" title="Delete group" onClick={() => void deleteGroup()}>
                <TrashIcon />
              </button>
            ) : null}
            <div className="settingsWrapper">
              <button className="settingsBtn" onClick={() => setSettingsOpen((v) => !v)}>
                <Cog6ToothIcon />
              </button>
              {settingsOpen ? (
                <div className="settingsMenu">
                  <button
                    className="settingsItem"
                    onClick={() => {
                      setSettingsOpen(false)
                      setShowKeysModal(true)
                    }}
                  >
                    Keys
                  </button>
                  <button
                    className="settingsItem"
                    onClick={() => {
                      setSettingsOpen(false)
                      setShowDeveloperModal(true)
                    }}
                  >
                    Developer
                  </button>
                  <button
                    className="settingsItem danger"
                    disabled={resettingOnboarding}
                    onClick={() => {
                      setSettingsOpen(false)
                      void resetOnboarding()
                    }}
                  >
                    {resettingOnboarding ? 'Resetting onboarding…' : 'Reset onboarding'}
                  </button>
                </div>
              ) : null}
            </div>
          </div>
        </div>

        <div className="chatShell">
          {selectedType === 'group' ? (
            <>
              {groupErr ? <div className="alert">{groupErr}</div> : null}
              {groupLoading ? (
                <div className="muted">
                  <LoadingSpinner />
                </div>
              ) : (
                <div className="chatArea" ref={groupScrollRef}>
                  {groupLoadingOlder || groupHasMore ? (
                    <div className="muted" style={{ padding: '8px 0', textAlign: 'center' }}>
                      {groupLoadingOlder ? (
                        <LoadingSpinner label="Loading older messages" />
                      ) : (
                        <button className="btn ghost" onClick={() => void loadGroupMessagesOlder()} disabled={!groupHasMore}>
                          Load earlier messages
                        </button>
                      )}
                    </div>
                  ) : null}
                  {visibleGroupMessages.map((m) => (
                    <GroupMessageRow key={m.id} m={m} />
                  ))}
                </div>
              )}
              {uploadErr ? <div className="alert">{uploadErr}</div> : null}
              <div className="chatComposerBar">
                <div className="iconBtnWrap" title={canUpload ? 'Upload files' : uploadDisabledReason || 'Upload files'}>
                  <button
                    className="iconBtn"
                    onClick={() => {
                      if (!canUpload) return
                      setGroupUploadMenuOpen((v) => !v)
                    }}
                    disabled={!canUpload || uploading}
                  >
                    {uploading ? <LoadingSpinner /> : <PaperClipIcon />}
                  </button>
                  {groupUploadMenuOpen && canUpload ? (
                    <div className="uploadMenu">
                      <button
                        type="button"
                        onMouseDown={(e) => {
                          e.preventDefault()
                          setGroupUploadMenuOpen(false)
                          groupUploadRef.current?.click()
                        }}
                      >
                        Upload files
                      </button>
                      <button
                        type="button"
                        onMouseDown={(e) => {
                          e.preventDefault()
                          setGroupUploadMenuOpen(false)
                          groupUploadFolderRef.current?.click()
                        }}
                      >
                        Upload folder
                      </button>
                    </div>
                  ) : null}
                </div>
                <input
                  ref={groupUploadRef}
                  type="file"
                  multiple
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const files = e.target.files
                    if (files && files.length > 0) {
                      void uploadConversationFiles(files)
                    }
                    e.currentTarget.value = ''
                  }}
                />
                <input
                  ref={groupUploadFolderRef}
                  type="file"
                  multiple
                  style={{ display: 'none' }}
                  {...({ webkitdirectory: 'true', directory: 'true' } as any)}
                  onChange={(e) => {
                    const files = e.target.files
                    if (files && files.length > 0) {
                      void uploadConversationFiles(files)
                    }
                    e.currentTarget.value = ''
                  }}
                />
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
                    ref={groupInputRef}
                    placeholder="Message the group (use @slug to mention an assistant)"
                    value={groupText}
                    onChange={(e) => {
                      const next = e.target.value
                      setGroupText(next)
                      updateMentionState(next)
                    }}
                    onKeyDown={(e) => {
                      if (mention?.active && mentionOptions.length > 0) {
                        if (e.key === 'ArrowDown') {
                          e.preventDefault()
                          setMention((prev) =>
                            prev ? { ...prev, index: (prev.index + 1) % mentionOptions.length } : prev,
                          )
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
                      }
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault()
                        void sendGroupMessage()
                      }
                    }}
                    rows={2}
                  />
                </div>
                <button className="btn primary" onClick={() => void sendGroupMessage()} disabled={!groupText.trim()}>
                  Send
                </button>
              </div>
              {Object.keys(workingBots).length ? (
                <div className="muted chatWorking">Working: {Object.values(workingBots).join(', ')}</div>
              ) : null}
              {groupSendErr ? <div className="alert">{groupSendErr}</div> : null}
            </>
          ) : (
            selectedBotId ? (
              <MicTest
                key={selectedBotId}
                botId={selectedBotId}
                initialConversationId={selectedConversationId || undefined}
                layout="whatsapp"
                startToken={startToken}
                refreshToken={refreshToken}
                onConversationIdChange={setAssistantConversationId}
                onStageChange={setAssistantStage}
                cache={chatCache}
                onCacheUpdate={handleCacheUpdate}
                onSync={notifySync}
                hideWorkspace
                allowUploads={canUpload}
                uploadDisabledReason={uploadDisabledReason}
                uploading={uploading}
                onUploadFiles={uploadConversationFiles}
                hostActions={hostActions}
                hostActionRequiresApproval={hostActionRequiresApproval}
                hostActionsErr={hostActionsErr}
                onRunHostAction={(action) => void runHostAction(action)}
                settingsTab={settingsTab}
                onOpenSettingsTab={openSettingsTab}
                isolatedEnabled={!!selectedBot?.enable_data_agent}
              />
            ) : (
              <div className="muted" style={{ padding: '16px 20px' }}>
                Select an assistant to begin.
              </div>
            )
          )}
          {selectedType === 'assistant' && uploadErr ? <div className="alert">{uploadErr}</div> : null}
          {selectedType === 'assistant' && botConvErr ? <div className="alert">{botConvErr}</div> : null}
        </div>
      </main>

      <aside
        className={`chatWorkspace ${workspaceCollapsed ? 'collapsed' : ''}`}
        style={!workspaceCollapsed ? { width: workspaceWidth } : undefined}
      >
          <div className="workspaceResizeHandle" onMouseDown={beginWorkspaceResize} />
          <div className={`workspaceHeader ${workspaceIDEFull ? 'fullIde' : ''}`}>
            <div>
              <h3>Workspace</h3>
              {!workspaceIDEFull ? <div className="muted">Container: {workspaceStatusLabel}</div> : null}
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
            {workspaceErr ? <div className="alert">{workspaceErr}</div> : null}
            {!workspaceIDEFull ? (
              <div className="workspaceCard">
                <div className="workspaceTitle">Status</div>
                <div className="workspaceRow">
                  <strong>Runtime</strong>
                  <span>
                    {workspaceStatus?.exists ? 'Isolated Workspace' : workspaceStatus?.docker_available ? 'Not started' : '-'}
                  </span>
                </div>
                <div className="workspaceRow">
                  <strong>Status</strong>
                  <span>{workspaceStatusLabel}</span>
                </div>
                <div className="workspaceRow">
                  <strong>Container</strong>
                  {workspaceStatus?.container_name ? (
                    <span className="truncate" title={workspaceStatus.container_name}>
                      {workspaceStatus.container_name}
                    </span>
                  ) : (
                    <span>-</span>
                  )}
                </div>
              </div>
            ) : null}
            <div className={`workspaceCard ideCard ${workspaceIDEFull ? 'full' : ''}`}>
              {!workspaceIDEFull ? <div className="workspaceTitle">IDE (VS Code)</div> : null}
              {!activeConversationId ? (
                <div className="muted">Select a conversation to enable the IDE.</div>
              ) : !workspaceStatus?.running ? (
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
              <>
                <div className="workspaceCard">
                  <div className="row" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
                    <div className="workspaceTitle">Files</div>
                    <button
                      className="btn iconBtn"
                      onClick={() => void loadFiles(activeConversationId || undefined, filesRecursive, filesHidden)}
                      disabled={!activeConversationId || filesLoading}
                      aria-label="Refresh files"
                    >
                      {filesLoading ? <LoadingSpinner label="Refreshing" /> : '⟳'}
                    </button>
                  </div>
                  {filesErr ? <div className="alert" style={{ marginTop: 8 }}>{filesErr}</div> : null}
                  {filesMsg ? (
                    <div
                      className="alert"
                      style={{ marginTop: 8, borderColor: 'rgba(80, 200, 160, 0.4)', background: 'rgba(80, 200, 160, 0.1)' }}
                    >
                      {filesMsg}
                    </div>
                  ) : null}
                  <div className="row" style={{ marginTop: 8, alignItems: 'center' }}>
                    <label className="check">
                      <input
                        type="checkbox"
                        checked={filesRecursive}
                        disabled={!activeConversationId}
                        onChange={(e) => void loadFiles(activeConversationId || undefined, e.target.checked, filesHidden)}
                      />{' '}
                      recursive
                    </label>
                    <label className="check">
                      <input
                        type="checkbox"
                        checked={filesHidden}
                        disabled={!activeConversationId}
                        onChange={(e) => void loadFiles(activeConversationId || undefined, filesRecursive, e.target.checked)}
                      />{' '}
                      hidden
                    </label>
                    <div className="spacer" />
                    <div className="muted mono">{files?.items ? `${fileItems.length} items` : '-'}</div>
                  </div>
                  {filesLoading ? (
                    <div className="muted" style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
                      <LoadingSpinner label="Loading files" />
                      <span>Loading files…</span>
                    </div>
                  ) : (
                    <div className="workspaceFiles" style={{ marginTop: 8 }}>
                      {!files ? <div className="muted">No workspace yet.</div> : null}
                      {files && fileItems.length === 0 ? <div className="muted">No files yet.</div> : null}
                      {files && fileItems.length > 0 ? (
                        <div className="tree">
                          {renderFileTree(fileTree, 0, filesExpanded, setFilesExpanded, handleFileDownload)}
                        </div>
                      ) : null}
                    </div>
                  )}
                </div>
              </>
            ) : null}
          </div>
      </aside>
      {showKeysModal ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow modalSticky">
              <div>
                <div className="cardTitle">Keys</div>
                <div className="muted">Manage API and client keys.</div>
              </div>
              <button className="iconBtn modalCloseBtn" onClick={() => setShowKeysModal(false)} aria-label="Close">
                <XMarkIcon />
              </button>
            </div>
            <KeysPage />
          </div>
        </div>
      ) : null}

      {showDeveloperModal ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow modalSticky">
              <div>
                <div className="cardTitle">Developer</div>
                <div className="muted">Monitor and stop Isolated Workspace containers.</div>
              </div>
              <button className="iconBtn modalCloseBtn" onClick={() => setShowDeveloperModal(false)} aria-label="Close">
                <XMarkIcon />
              </button>
            </div>
            <DeveloperPage />
          </div>
        </div>
      ) : null}

      {showSettingsModal ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            {selectedBotId ? (
              <BotSettingsModal
                botId={selectedBotId}
                onClose={() => setShowSettingsModal(false)}
                activeTab={settingsTab}
                onBotUpdate={(updated) =>
                  setBots((prev) => prev.map((b) => (b.id === updated.id ? { ...b, ...updated } : b)))
                }
              />
            ) : (
              <>
                <div className="cardTitleRow modalSticky">
                  <div>
                    <div className="cardTitle">Settings</div>
                    <div className="muted">Select an assistant to edit settings.</div>
                  </div>
                  <button className="iconBtn modalCloseBtn" onClick={() => setShowSettingsModal(false)} aria-label="Close">
                    <XMarkIcon />
                  </button>
                </div>
                <div className="muted">Pick an assistant from the left panel, then reopen Settings.</div>
              </>
            )}
          </div>
        </div>
      ) : null}

      {showCreateAssistant ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow modalSticky">
              <div>
                <div className="cardTitle">New assistant</div>
                <div className="muted">Set a name and choose the core models.</div>
              </div>
              <button className="iconBtn modalCloseBtn" onClick={() => setShowCreateAssistant(false)} aria-label="Close">
                <XMarkIcon />
              </button>
            </div>
            {assistantErr ? <div className="alert">{assistantErr}</div> : null}
            <div className="formRow">
              <label>Name</label>
              <input value={newBot.name} onChange={(e) => setNewBot((p) => ({ ...p, name: e.target.value }))} />
            </div>
            <div className="formRow">
              <label>Provider</label>
              <SelectField
                value={newBot.llm_provider}
                onChange={(e) => {
                  const next = e.target.value
                  const models = llmModels(next, options, newBot.openai_model)
                  setNewBot((p) => ({
                    ...p,
                    llm_provider: next,
                    openai_model: models.includes(p.openai_model) ? p.openai_model : models[0] || p.openai_model,
                  }))
                }}
              >
                {orderProviderList(options?.llm_providers || ['openai', 'openrouter', 'local']).map((p) => (
                  <option value={p} key={p}>
                    {formatProviderLabel(p)}
                  </option>
                ))}
              </SelectField>
            </div>
            <div className="formRowGrid2">
              <div className="formRow">
                <label>LLM model</label>
                {newBot.llm_provider === 'local' ? (
                  <>
                    <input
                      list="local-models-new"
                      value={newBot.openai_model}
                      onChange={(e) => setNewBot((p) => ({ ...p, openai_model: e.target.value }))}
                    />
                    <datalist id="local-models-new">
                      {(options?.local_models || []).map((m) => (
                        <option value={m.id} key={m.id}>
                          {m.name}
                        </option>
                      ))}
                    </datalist>
                    <div className="muted" style={{ marginTop: 6 }}>
                      {formatLocalModelToolSupport(newBotLocalModel)}
                    </div>
                  </>
                ) : (
                  <SelectField value={newBot.openai_model} onChange={(e) => setNewBot((p) => ({ ...p, openai_model: e.target.value }))}>
                    {llmModels(newBot.llm_provider, options, newBot.openai_model).map((m) => (
                      <option value={m} key={m}>
                        {m}
                      </option>
                    ))}
                  </SelectField>
                )}
              </div>
              <div className="formRow">
                <label>
                  ASR model
                  {newBotVoiceRequiresOpenAI ? <InlineHelpTip text="Requires OpenAI API key." /> : null}
                </label>
                <SelectField
                  value={newBot.openai_asr_model}
                  onChange={(e) => setNewBot((p) => ({ ...p, openai_asr_model: e.target.value }))}
                  disabled={newBotVoiceRequiresOpenAI}
                >
                  {(options?.openai_asr_models || [newBot.openai_asr_model]).map((m) => (
                    <option value={m} key={m}>
                      {m}
                    </option>
                  ))}
                </SelectField>
                {newBotVoiceRequiresOpenAI ? (
                  <div className="muted" style={{ marginTop: 6 }}>
                    ASR disabled for ChatGPT OAuth. Add an OpenAI API key to enable it.
                  </div>
                ) : null}
              </div>
            </div>
            {newBotNeedsChatgptAuth ? (
              <div className="alert" style={{ marginTop: 12 }}>
                <div style={{ marginBottom: 8 }}>Sign in with ChatGPT to create an assistant with this provider.</div>
                {chatgptOauth.error ? <div className="muted" style={{ marginBottom: 8 }}>{chatgptOauth.error}</div> : null}
                <div className="row gap">
                  <button className="btn primary" onClick={() => void chatgptOauth.start()} disabled={chatgptOauth.busy}>
                    {chatgptOauth.busy ? 'Starting…' : 'Sign in with ChatGPT'}
                  </button>
                  {chatgptOauth.authUrl ? (
                    <button
                      className="btn"
                      onClick={() => {
                        const url = chatgptOauth.authUrl
                        if (url) window.open(url, '_blank', 'noopener,noreferrer')
                      }}
                    >
                      Open login
                    </button>
                  ) : null}
                </div>
                {chatgptOauth.authState ? <div className="muted" style={{ marginTop: 8 }}>Waiting for approval…</div> : null}
              </div>
            ) : null}
            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button
                className="btn primary"
                onClick={() => void createAssistant()}
                disabled={creatingAssistant || !newBot.name.trim() || newBotNeedsChatgptAuth}
              >
                {creatingAssistant ? 'Creating…' : 'Create assistant'}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {showCreateGroup ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow modalSticky">
              <div>
                <div className="cardTitle">New group</div>
                <div className="muted">Pick assistants and a default.</div>
              </div>
              <button className="iconBtn modalCloseBtn" onClick={() => setShowCreateGroup(false)} aria-label="Close">
                <XMarkIcon />
              </button>
            </div>
            {groupSaveErr ? <div className="alert">{groupSaveErr}</div> : null}
            <div className="formRow">
              <label>Group title</label>
              <input value={groupTitle} onChange={(e) => setGroupTitle(e.target.value)} />
            </div>
            <div className="formRow">
              <label>Filter assistants</label>
              <input value={groupFilter} onChange={(e) => setGroupFilter(e.target.value)} />
            </div>
            <div className="formRow">
              <label>Assistants</label>
              <div className="row wrap" style={{ gap: 8 }}>
                {filteredBotsForGroup.map((b) => (
                  <button
                    key={b.id}
                    className={`pill ${groupSelected.includes(b.id) ? 'accent' : ''}`}
                    onClick={() => toggleSelected(b.id)}
                  >
                    {b.name}
                  </button>
                ))}
              </div>
            </div>
            <div className="formRow">
              <label>Default assistant</label>
              <SelectField value={groupDefaultId} onChange={(e) => setGroupDefaultId(e.target.value)}>
                {selectedBots.length === 0 ? <option value="">Select assistants first</option> : null}
                {selectedBots.map((b) => (
                  <option key={b.id} value={b.id}>
                    {b.name} (@{selectedSlugMap[b.id]})
                  </option>
                ))}
              </SelectField>
            </div>
            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button className="btn primary" onClick={() => void createGroup()} disabled={groupSaving || !groupTitle.trim()}>
                {groupSaving ? 'Creating…' : 'Create group'}
              </button>
            </div>
          </div>
        </div>
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
        <div className="bubbleTime" title={m.created_at}>{fmtTime(m.created_at)}</div>
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

function fmtTime(iso: string): string {
  try {
    const d = new Date(iso)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
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

function buildSlugMap(bots: Bot[]): Record<string, string> {
  const used = new Set<string>()
  const map: Record<string, string> = {}
  for (const b of bots) {
    const base = b.name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'assistant'
    let slug = base
    let i = 2
    while (used.has(slug)) {
      slug = `${base}-${i}`
      i += 1
    }
    used.add(slug)
    map[b.id] = slug
  }
  return map
}
