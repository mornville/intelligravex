import { useEffect, useMemo, useRef, useState } from 'react'
import { apiDelete, apiGet, apiPost, BACKEND_URL } from '../api/client'
import { authHeader } from '../auth'
import LoadingSpinner from '../components/LoadingSpinner'
import MicTest from '../components/MicTest'
import KeysPage from './KeysPage'
import DeveloperPage from './DeveloperPage'
import BotSettingsModal from '../components/BotSettingsModal'
import SelectField from '../components/SelectField'
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
} from '../types'
import { fmtIso } from '../utils/format'
import {
  Cog6ToothIcon,
  CpuChipIcon,
  PaperClipIcon,
  PlusIcon,
  TrashIcon,
  UserIcon,
  WrenchScrewdriverIcon,
} from '@heroicons/react/24/solid'

type FilterTab = 'all' | 'assistants' | 'groups'

type MentionState = {
  active: boolean
  query: string
  index: number
  start: number
  end: number
}

export default function DashboardPage() {
  const [bots, setBots] = useState<Bot[]>([])
  const [groups, setGroups] = useState<GroupConversationSummary[]>([])
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [options, setOptions] = useState<Options | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)
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
  const [groupText, setGroupText] = useState('')
  const [groupSendErr, setGroupSendErr] = useState<string | null>(null)
  const [mention, setMention] = useState<MentionState | null>(null)
  const groupInputRef = useRef<HTMLTextAreaElement | null>(null)
  const groupUploadRef = useRef<HTMLInputElement | null>(null)
  const groupUploadFolderRef = useRef<HTMLInputElement | null>(null)
  const [groupUploadMenuOpen, setGroupUploadMenuOpen] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const [workingBots, setWorkingBots] = useState<Record<string, string>>({})

  const [settingsOpen, setSettingsOpen] = useState(false)
  const [showKeysModal, setShowKeysModal] = useState(false)
  const [showDeveloperModal, setShowDeveloperModal] = useState(false)
  const [showSettingsModal, setShowSettingsModal] = useState(false)
  const [showCreateAssistant, setShowCreateAssistant] = useState(false)
  const [creatingAssistant, setCreatingAssistant] = useState(false)
  const [assistantErr, setAssistantErr] = useState<string | null>(null)
  const [newBot, setNewBot] = useState({
    name: '',
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
  const [showCreateGroup, setShowCreateGroup] = useState(false)
  const [groupTitle, setGroupTitle] = useState('')
  const [groupFilter, setGroupFilter] = useState('')
  const [groupSelected, setGroupSelected] = useState<string[]>([])
  const [groupDefaultId, setGroupDefaultId] = useState('')
  const [groupSaving, setGroupSaving] = useState(false)
  const [groupSaveErr, setGroupSaveErr] = useState<string | null>(null)
  const [startToken, setStartToken] = useState(0)
  const [assistantConversationId, setAssistantConversationId] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadErr, setUploadErr] = useState<string | null>(null)
  const [previewByConversationId, setPreviewByConversationId] = useState<Record<string, string>>({})
  const [unseenByConversationId, setUnseenByConversationId] = useState<Record<string, number>>({})
  const [lastUpdatedByConversationId, setLastUpdatedByConversationId] = useState<Record<string, string>>({})
  const [assistantStage, setAssistantStage] = useState<'disconnected' | 'idle' | 'init' | 'recording' | 'asr' | 'llm' | 'tts' | 'error'>('idle')
  const lastUpdatedRef = useRef<Record<string, string>>({})
  const unseenRef = useRef<Record<string, number>>({})
  const selectionRef = useRef<{ type: 'assistant' | 'group'; groupId: string | null; convId: string } | null>(null)

  const [workspaceStatus, setWorkspaceStatus] = useState<DataAgentStatus | null>(null)
  const [workspaceErr, setWorkspaceErr] = useState<string | null>(null)
  const [files, setFiles] = useState<ConversationFiles | null>(null)
  const [filesErr, setFilesErr] = useState<string | null>(null)
  const [filesLoading, setFilesLoading] = useState(false)
  const [hostActions, setHostActions] = useState<HostAction[]>([])
  const [hostActionsErr, setHostActionsErr] = useState<string | null>(null)
  const [hostActionsLoading, setHostActionsLoading] = useState(false)

  useEffect(() => {
    void (async () => {
      setLoading(true)
      setErr(null)
      try {
        const [b, g, c, o] = await Promise.all([
          apiGet<{ items: Bot[] }>('/api/bots'),
          apiGet<{ items: GroupConversationSummary[] }>('/api/group-conversations'),
          apiGet<{ items: ConversationSummary[] }>(`/api/conversations?page=1&page_size=200`),
          apiGet<Options>('/api/options'),
        ])
        setBots(b.items)
        setGroups(g.items)
        setConversations(c.items)
        setOptions(o)
        const initialUpdated: Record<string, string> = {}
        c.items.forEach((item) => {
          initialUpdated[item.id] = item.updated_at
        })
        g.items.forEach((item) => {
          initialUpdated[item.id] = item.updated_at
        })
        setLastUpdatedByConversationId(initialUpdated)
        lastUpdatedRef.current = initialUpdated
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

  const q = query.trim().toLowerCase()
  const filteredBots = useMemo(() => {
    if (!q) return bots
    return bots.filter((b) => b.name.toLowerCase().includes(q) || b.openai_model.toLowerCase().includes(q))
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

  useEffect(() => {
    if (!selectedBotId) return
    const fallbackLatest = latestConversationByBot.get(selectedBotId)
    setSelectedConversationId(fallbackLatest?.id || '')
    setAssistantConversationId(null)
    setBotConversations([])
    void (async () => {
      setBotConvLoading(true)
      setBotConvErr(null)
      try {
        const c = await apiGet<{ items: ConversationSummary[] }>(
          `/api/conversations?page=1&page_size=50&bot_id=${selectedBotId}`,
        )
        const sorted = [...(c.items || [])].sort((a, b) => b.updated_at.localeCompare(a.updated_at))
        const fallback = conversations
          .filter((conv) => conv.bot_id === selectedBotId)
          .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
          .slice(0, 50)
        const finalList = sorted.length ? sorted : fallback
        setBotConversations(finalList)
        const nextId = finalList[0]?.id || fallbackLatest?.id || ''
        setSelectedConversationId(nextId)
        if (nextId) {
          setUnseenByConversationId((prev) => ({ ...prev, [nextId]: 0 }))
        }
      } catch (e: any) {
        setBotConvErr(String(e?.message || e))
      } finally {
        setBotConvLoading(false)
      }
    })()
  }, [selectedBotId])

  useEffect(() => {
    lastUpdatedRef.current = lastUpdatedByConversationId
  }, [lastUpdatedByConversationId])

  useEffect(() => {
    unseenRef.current = unseenByConversationId
  }, [unseenByConversationId])

  useEffect(() => {
    selectionRef.current = { type: selectedType, groupId: selectedGroupId, convId: selectedConversationId }
  }, [selectedType, selectedGroupId, selectedConversationId])

  function clipPreview(text: string, max = 90) {
    const cleaned = (text || '').replace(/\s+/g, ' ').trim()
    if (!cleaned) return ''
    if (cleaned.length <= max) return cleaned
    return `${cleaned.slice(0, max - 3)}...`
  }

  useEffect(() => {
    if (loading) return
    const ids = new Set<string>()
    if (showBots) {
      filteredBots.forEach((b) => {
        const latest = latestConversationByBot.get(b.id)
        if (latest?.id) ids.add(latest.id)
      })
    }
    if (showGroups) {
      filteredGroups.forEach((g) => {
        if (g.id) ids.add(g.id)
      })
    }
    const missing = Array.from(ids).filter((id) => !(id in previewByConversationId))
    if (!missing.length) return
    let canceled = false
    void (async () => {
      const results: Record<string, string> = {}
      await Promise.all(
        missing.map(async (id) => {
          try {
            const d = await apiGet<{ messages: ConversationMessage[] }>(`/api/conversations/${id}`)
            const msgs = Array.isArray(d.messages) ? d.messages : []
            for (let i = msgs.length - 1; i >= 0; i -= 1) {
              const m = msgs[i]
              if (m.role !== 'tool' && m.role !== 'system') {
                results[id] = clipPreview(m.content || '')
                break
              }
            }
            if (!(id in results)) results[id] = ''
          } catch {
            results[id] = ''
          }
        }),
      )
      if (canceled) return
      setPreviewByConversationId((prev) => ({ ...prev, ...results }))
    })()
    return () => {
      canceled = true
    }
  }, [loading, showBots, showGroups, filteredBots, filteredGroups, latestConversationByBot, previewByConversationId])

  useEffect(() => {
    if (selectedType !== 'group' || !selectedGroupId) return
    void (async () => {
      setGroupLoading(true)
      setGroupErr(null)
      try {
        const d = await apiGet<GroupConversationDetail>(`/api/group-conversations/${selectedGroupId}`)
        setGroupDetail(d)
      } catch (e: any) {
        setGroupErr(String(e?.message || e))
      } finally {
        setGroupLoading(false)
      }
    })()
  }, [selectedGroupId, selectedType])

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
        setGroupDetail((prev) => {
          if (!prev) return prev
          const exists = prev.messages.some((m) => m.id === payload.message.id)
          if (exists) return prev
          const next = [...prev.messages, payload.message]
          next.sort((a, b) => a.created_at.localeCompare(b.created_at))
          return { ...prev, messages: next }
        })
        if (selectedGroupId) {
          const ts = payload.message.created_at || new Date().toISOString()
          setLastUpdatedByConversationId((prev) => ({ ...prev, [selectedGroupId]: ts }))
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

  useEffect(() => {
    const convId = selectedType === 'group' ? selectedGroupId : assistantConversationId
    if (!convId) {
      setWorkspaceStatus(null)
      setFiles(null)
      return
    }
    void (async () => {
      setWorkspaceErr(null)
      try {
        const s = await apiGet<DataAgentStatus>(`/api/conversations/${convId}/data-agent`)
        setWorkspaceStatus(s)
      } catch (e: any) {
        setWorkspaceErr(String(e?.message || e))
      }
    })()
  }, [selectedType, selectedGroupId, assistantConversationId])

  useEffect(() => {
    const convId = selectedType === 'group' ? selectedGroupId : assistantConversationId
    if (!convId) return
    void (async () => {
      setFilesLoading(true)
      setFilesErr(null)
      try {
        const f = await apiGet<ConversationFiles>(`/api/conversations/${convId}/files?path=`)
        setFiles(f)
      } catch (e: any) {
        setFilesErr(String(e?.message || e))
      } finally {
        setFilesLoading(false)
      }
    })()
  }, [selectedType, selectedGroupId, assistantConversationId])

  const groupMessages = (groupDetail?.messages || []).filter((m) => m.role !== 'tool' && m.role !== 'system')
  const activeConversationId = selectedType === 'group' ? selectedGroupId : assistantConversationId
  const activeBotId =
    selectedType === 'group'
      ? groupDetail?.conversation.default_bot_id || groups.find((g) => g.id === selectedGroupId)?.default_bot_id || null
      : selectedBotId
  const activeBot = bots.find((b) => b.id === activeBotId) || null
  const canUpload = Boolean(activeConversationId && activeBot?.enable_data_agent)
  const uploadDisabledReason = !activeBot?.enable_data_agent
    ? 'Enable Data Agent to upload files.'
    : !activeConversationId
      ? 'Start a conversation to upload files.'
      : ''
  const hostActionsEnabled =
    selectedType === 'group'
      ? groupBots.some((b) => bots.find((bot) => bot.id === b.id)?.enable_host_actions)
      : Boolean(activeBot?.enable_host_actions)

  useEffect(() => {
    if (!activeConversationId) {
      setHostActions([])
      return
    }
    void loadHostActions(activeConversationId)
    const id = window.setInterval(() => {
      void loadHostActions(activeConversationId)
    }, 5000)
    return () => window.clearInterval(id)
  }, [activeConversationId])

  const visibleFiles = (files?.items || []).filter((f) => !(f.is_dir && (f.path === '' || f.path === '.'))).slice(0, 6)

  async function reloadLists() {
    try {
      const [b, g, c] = await Promise.all([
        apiGet<{ items: Bot[] }>('/api/bots'),
        apiGet<{ items: GroupConversationSummary[] }>('/api/group-conversations'),
        apiGet<{ items: ConversationSummary[] }>(`/api/conversations?page=1&page_size=200`),
      ])
      setBots(b.items)
      setGroups(g.items)
      setConversations(c.items)
      const sel = selectionRef.current
      const active = sel?.type === 'group' ? sel?.groupId : sel?.convId
      const nextUpdated: Record<string, string> = { ...lastUpdatedRef.current }
      const nextUnseen: Record<string, number> = { ...unseenRef.current }
      const updatedPreviewIds: string[] = []
      c.items.forEach((item) => {
        const prev = nextUpdated[item.id]
        if (prev && item.updated_at > prev) {
          if (item.id !== active) {
            nextUnseen[item.id] = (nextUnseen[item.id] || 0) + 1
          } else {
            nextUnseen[item.id] = 0
          }
          updatedPreviewIds.push(item.id)
        }
        nextUpdated[item.id] = item.updated_at
      })
      g.items.forEach((item) => {
        const prev = nextUpdated[item.id]
        if (prev && item.updated_at > prev) {
          if (item.id !== active) {
            nextUnseen[item.id] = (nextUnseen[item.id] || 0) + 1
          } else {
            nextUnseen[item.id] = 0
          }
          updatedPreviewIds.push(item.id)
        }
        nextUpdated[item.id] = item.updated_at
      })
      if (updatedPreviewIds.length) {
        setPreviewByConversationId((prev) => {
          const next = { ...prev }
          updatedPreviewIds.forEach((id) => {
            delete next[id]
          })
          return next
        })
      }
      setLastUpdatedByConversationId(nextUpdated)
      setUnseenByConversationId(nextUnseen)
    } catch {
      // ignore
    }
  }

  useEffect(() => {
    if (loading) return
    const id = window.setInterval(() => {
      void reloadLists()
    }, 10000)
    return () => window.clearInterval(id)
  }, [loading])

  async function refreshFiles(convId?: string) {
    const id = convId || activeConversationId
    if (!id) return
    setFilesLoading(true)
    setFilesErr(null)
    try {
      const f = await apiGet<ConversationFiles>(`/api/conversations/${id}/files?path=`)
      setFiles(f)
    } catch (e: any) {
      setFilesErr(String(e?.message || e))
    } finally {
      setFilesLoading(false)
    }
  }

  async function loadHostActions(convId?: string) {
    const id = convId || activeConversationId
    if (!id) return
    setHostActionsLoading(true)
    setHostActionsErr(null)
    try {
      const res = await apiGet<{ items: HostAction[] }>(`/api/conversations/${id}/host-actions`)
      setHostActions(res.items || [])
    } catch (e: any) {
      setHostActionsErr(String(e?.message || e))
    } finally {
      setHostActionsLoading(false)
    }
  }

  async function runHostAction(actionId: string) {
    try {
      await apiPost(`/api/host-actions/${actionId}/run`, {})
      await loadHostActions()
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

  async function deleteAssistant() {
    if (!selectedBotId) return
    const target = bots.find((b) => b.id === selectedBotId)
    const ok = window.confirm(`Delete assistant \"${target?.name || 'assistant'}\"? This removes its conversations.`)
    if (!ok) return
    try {
      await apiDelete(`/api/bots/${selectedBotId}`)
      await reloadLists()
      const next = bots.filter((b) => b.id !== selectedBotId)[0]
      setSelectedBotId(next?.id || null)
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
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
      setWorkingBots({})
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  function toggleSelected(botId: string) {
    setGroupSelected((prev) => (prev.includes(botId) ? prev.filter((id) => id !== botId) : [...prev, botId]))
  }

  return (
    <div className="chatLayout withWorkspace">
      <aside className="chatSidebar">
        <div className="chatSidebarHeader">
          <div className="chatBrand">
            <span className="chatBrandDot" />
            GravexStudio
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
                    const preview = latest?.id ? previewByConversationId[latest.id] : ''
                    const unseen = latest?.id ? unseenByConversationId[latest.id] || 0 : 0
                    const isActive = selectedType === 'assistant' && selectedBotId === b.id
                    const showTyping = isActive && ['init', 'recording', 'asr', 'llm', 'tts'].includes(assistantStage)
                    return (
                      <button
                        key={b.id}
                        className={`waAssistantCard ${selectedType === 'assistant' && selectedBotId === b.id ? 'active' : ''}`}
                        onClick={() => {
                          setSelectedType('assistant')
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
                                  ? 'No messages yet'
                                  : 'No conversations yet'}
                            </div>
                          </div>
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
                    const preview = previewByConversationId[g.id] || ''
                    const unseen = unseenByConversationId[g.id] || 0
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
                            <div className="waConversationRow">{preview || 'No messages yet'}</div>
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
                : bots.find((b) => b.id === selectedBotId)?.openai_model || ''}
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
                    Conversation · {fmtIso(c.updated_at)}
                  </option>
                ))}
              </select>
            )}
            {selectedType === 'assistant' ? (
              <button
                className="btn navPill"
                onClick={() => {
                  setStartToken((t) => t + 1)
                }}
              >
                New conversation
              </button>
            ) : null}
            {selectedType === 'group' ? (
              <button className="btn" onClick={() => void resetGroupConversation()}>
                Reset chat
              </button>
            ) : null}
            {selectedType === 'assistant' ? (
              <button className="iconBtn danger" title="Delete assistant" onClick={() => void deleteAssistant()}>
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
                  <button className="settingsItem" onClick={() => setSettingsOpen(false)}>
                    Dashboard
                  </button>
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
                    className="settingsItem"
                    onClick={() => {
                      setSettingsOpen(false)
                      setShowSettingsModal(true)
                    }}
                  >
                    Settings
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
                <div className="chatArea">
                  {groupMessages.map((m) => (
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
                onConversationIdChange={setAssistantConversationId}
                onStageChange={setAssistantStage}
                hideWorkspace
                allowUploads={canUpload}
                uploadDisabledReason={uploadDisabledReason}
                uploading={uploading}
                onUploadFiles={uploadConversationFiles}
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

      <aside className="chatWorkspace">
          <div className="workspaceHeader">
            <div>
              <h3>Workspace</h3>
              <div className="muted">Container: {workspaceStatus?.running ? 'running' : 'idle'}</div>
            </div>
          </div>
          <div className="workspaceBody">
            {workspaceErr ? <div className="alert">{workspaceErr}</div> : null}
            <div className="workspaceCard">
              <div className="workspaceTitle">Status</div>
              <div className="workspaceRow">
                <strong>Runtime</strong>
                <span>{workspaceStatus?.exists ? 'Data Agent' : '—'}</span>
              </div>
              <div className="workspaceRow">
                <strong>Status</strong>
                <span>{workspaceStatus?.running ? 'running' : workspaceStatus?.status || 'idle'}</span>
              </div>
              <div className="workspaceRow">
                <strong>Container</strong>
                {workspaceStatus?.container_name ? (
                  <span className="truncate" title={workspaceStatus.container_name}>
                    {workspaceStatus.container_name}
                  </span>
                ) : (
                  <span>—</span>
                )}
              </div>
            </div>
            <div className="workspaceCard">
              <div className="workspaceTitle">Action queue</div>
              {!hostActionsEnabled ? (
                <div className="muted">Enable Host Actions in settings to queue actions.</div>
              ) : hostActionsLoading ? (
                <div className="muted">
                  <LoadingSpinner label="Loading actions" />
                </div>
              ) : hostActionsErr ? (
                <div className="alert">{hostActionsErr}</div>
              ) : hostActions.length === 0 ? (
                <div className="muted">No pending actions.</div>
              ) : (
                <div className="workspaceFiles">
                  {hostActions.map((a) => {
                    const label = formatHostActionLabel(a)
                    return (
                      <div key={a.id} className="workspaceRow" style={{ alignItems: 'flex-start', gap: 10 }}>
                        <div style={{ flex: 1 }}>
                          <strong>{label.title}</strong>
                          <div className="muted" title={label.detail}>
                            {label.detail}
                          </div>
                          <div className="muted" style={{ marginTop: 2 }}>
                            {a.status}
                          </div>
                        </div>
                        <div>
                          {a.status === 'pending' ? (
                            <button className="btn" onClick={() => void runHostAction(a.id)}>
                              Run
                            </button>
                          ) : null}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
            <div className="workspaceCard">
              <div className="workspaceTitle">Files</div>
              {filesLoading ? (
                <div className="muted" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <LoadingSpinner label="Loading files" />
                  <span>Loading files…</span>
                </div>
              ) : filesErr ? (
                <div className="alert">{filesErr}</div>
              ) : (
                <div className="workspaceFiles">
                  {!files ? <div className="muted">No workspace yet.</div> : null}
                  {files && visibleFiles.length === 0 ? <div className="muted">No files yet.</div> : null}
                  {visibleFiles.map((f) => (
                    <div key={f.path} className="workspaceRow">
                      <strong>{f.name}</strong>
                      <span>{f.is_dir ? 'folder' : f.size_bytes ? `${f.size_bytes} B` : '—'}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
      </aside>
      {showKeysModal ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow">
              <div>
                <div className="cardTitle">Keys</div>
                <div className="muted">Manage API and client keys.</div>
              </div>
              <button className="btn" onClick={() => setShowKeysModal(false)}>
                Close
              </button>
            </div>
            <KeysPage />
          </div>
        </div>
      ) : null}

      {showDeveloperModal ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow">
              <div>
                <div className="cardTitle">Developer</div>
                <div className="muted">Monitor and stop Data Agent containers.</div>
              </div>
              <button className="btn" onClick={() => setShowDeveloperModal(false)}>
                Close
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
              <BotSettingsModal botId={selectedBotId} onClose={() => setShowSettingsModal(false)} />
            ) : (
              <>
                <div className="cardTitleRow">
                  <div>
                    <div className="cardTitle">Settings</div>
                    <div className="muted">Select an assistant to edit settings.</div>
                  </div>
                  <button className="btn" onClick={() => setShowSettingsModal(false)}>
                    Close
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
            <div className="cardTitleRow">
              <div>
                <div className="cardTitle">New assistant</div>
                <div className="muted">Set a name and choose the core models.</div>
              </div>
              <button className="btn" onClick={() => setShowCreateAssistant(false)}>
                Close
              </button>
            </div>
            {assistantErr ? <div className="alert">{assistantErr}</div> : null}
            <div className="formRow">
              <label>Name</label>
              <input value={newBot.name} onChange={(e) => setNewBot((p) => ({ ...p, name: e.target.value }))} />
            </div>
            <div className="formRowGrid2">
              <div className="formRow">
                <label>OpenAI model</label>
                <SelectField value={newBot.openai_model} onChange={(e) => setNewBot((p) => ({ ...p, openai_model: e.target.value }))}>
                  {(options?.openai_models || [newBot.openai_model]).map((m) => (
                    <option value={m} key={m}>
                      {m}
                    </option>
                  ))}
                </SelectField>
              </div>
              <div className="formRow">
                <label>ASR model</label>
                <SelectField
                  value={newBot.openai_asr_model}
                  onChange={(e) => setNewBot((p) => ({ ...p, openai_asr_model: e.target.value }))}
                >
                  {(options?.openai_asr_models || [newBot.openai_asr_model]).map((m) => (
                    <option value={m} key={m}>
                      {m}
                    </option>
                  ))}
                </SelectField>
              </div>
            </div>
            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button className="btn primary" onClick={() => void createAssistant()} disabled={creatingAssistant || !newBot.name.trim()}>
                {creatingAssistant ? 'Creating…' : 'Create assistant'}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {showCreateGroup ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow">
              <div>
                <div className="cardTitle">New group</div>
                <div className="muted">Pick assistants and a default.</div>
              </div>
              <button className="btn" onClick={() => setShowCreateGroup(false)}>
                Close
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

  return (
    <div className={rowCls}>
      <div className={`avatar ${isAssistant ? 'assistant' : isTool ? 'tool' : 'user'}`}>
        {isUser ? <UserIcon /> : isTool ? <WrenchScrewdriverIcon /> : <CpuChipIcon />}
      </div>
      <div className={cls} style={bubbleStyle}>
        <div className="bubbleMeta" style={{ marginBottom: 6 }}>
          <span>{label}</span> <span className="muted">• {fmtIso(m.created_at)}</span>
        </div>
        <div className="bubbleText">{body}</div>
        <div className="bubbleTime" title={m.created_at}>{fmtTime(m.created_at)}</div>
      </div>
    </div>
  )
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

function formatHostActionLabel(action: HostAction): { title: string; detail: string } {
  const payload = action.payload || {}
  switch (action.action_type) {
    case 'run_shell':
      return {
        title: 'Shell command',
        detail: String(payload.command || ''),
      }
    case 'run_applescript':
      return {
        title: 'AppleScript',
        detail: String(payload.script || ''),
      }
    default:
      return { title: action.action_type || 'Host action', detail: '' }
  }
}
