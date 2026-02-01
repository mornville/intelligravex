import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { CpuChipIcon, UserIcon, WrenchScrewdriverIcon } from '@heroicons/react/24/solid'
import LoadingSpinner from '../components/LoadingSpinner'
import { apiGet, apiPost } from '../api/client'
import type { ConversationMessage, GroupBot, GroupConversationDetail } from '../types'
import { fmtIso } from '../utils/format'

type MentionState = {
  active: boolean
  query: string
  index: number
  start: number
  end: number
}

export default function GroupConversationPage() {
  const { groupId } = useParams()
  const nav = useNavigate()
  const [data, setData] = useState<GroupConversationDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)
  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)
  const [sendErr, setSendErr] = useState<string | null>(null)
  const [resetting, setResetting] = useState(false)
  const [resetErr, setResetErr] = useState<string | null>(null)
  const [workingBots, setWorkingBots] = useState<Record<string, string>>({})
  const [mention, setMention] = useState<MentionState | null>(null)
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  function wsBase() {
    try {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      return `${proto}//${window.location.host}`
    } catch {
      return 'ws://127.0.0.1:8000'
    }
  }

  useEffect(() => {
    if (!groupId) return
    void (async () => {
      setLoading(true)
      setErr(null)
      try {
        const d = await apiGet<GroupConversationDetail>(`/api/group-conversations/${groupId}`)
        setData(d)
      } catch (e: any) {
        setErr(String(e?.message || e))
      } finally {
        setLoading(false)
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
        setData((prev) => {
          if (!prev) return prev
          const exists = prev.messages.some((m) => m.id === payload.message.id)
          if (exists) return prev
          const next = [...prev.messages, payload.message]
          next.sort((a, b) => a.created_at.localeCompare(b.created_at))
          return { ...prev, messages: next }
        })
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
        setData((prev) => (prev ? { ...prev, messages: [] } : prev))
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

  const bots = data?.conversation.group_bots || []
  const defaultBot = useMemo(() => {
    if (!data?.conversation) return null
    return bots.find((b) => b.id === data.conversation.default_bot_id) || null
  }, [bots, data?.conversation])

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
      const d = await apiGet<GroupConversationDetail>(`/api/group-conversations/${groupId}`)
      setData(d)
    } catch (e: any) {
      setResetErr(String(e?.message || e))
    } finally {
      setResetting(false)
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
    <div className="page">
      <div className="pageHeader">
        <div>
          <h1>{data?.conversation.title || 'Group chat'}</h1>
          <div className="muted">
            Default assistant: {defaultBot ? `${defaultBot.name} (@${defaultBot.slug})` : '—'}
          </div>
        </div>
        <div className="row gap">
          <button className="btn" disabled={resetting} onClick={() => void resetConversation()}>
            {resetting ? <LoadingSpinner label="Resetting" /> : 'Reset chat'}
          </button>
          <button className="btn" onClick={() => nav('/conversations')}>
            Back
          </button>
        </div>
      </div>

      {resetErr ? <div className="alert">{resetErr}</div> : null}
      {err ? <div className="alert">{err}</div> : null}

      {loading ? (
        <div className="muted">
          <LoadingSpinner />
        </div>
      ) : (
        <>
          <section className="card">
            <div className="cardTitleRow">
              <div className="cardTitle">Assistants</div>
            </div>
            <div className="groupMemberRow">
              {bots.map((b) => (
                <span key={b.id} className={`pill ${b.id === data?.conversation.default_bot_id ? 'accent' : ''}`}>
                  @{b.slug}
                </span>
              ))}
            </div>
          </section>

          {data?.conversation.individual_conversations?.length ? (
            <section className="card">
              <div className="cardTitleRow">
                <div className="cardTitle">Individual logs</div>
              </div>
              <div className="groupMemberRow">
                {data.conversation.individual_conversations.map((item) => {
                  const bot = bots.find((b) => b.id === item.bot_id)
                  return (
                    <button
                      key={item.conversation_id}
                      type="button"
                      className="btn ghost"
                      onClick={() => nav(`/conversations/${item.conversation_id}`)}
                    >
                      {bot ? bot.name : item.bot_id}
                    </button>
                  )
                })}
              </div>
            </section>
          ) : null}

          <section className="card">
            <div className="cardTitle">Conversation</div>
            <div className="chat">
              {(data?.messages || []).map((m) => (
                <GroupMessageRow key={m.id} m={m} />
              ))}
            </div>
            {Object.keys(workingBots).length ? (
              <div className="muted" style={{ marginTop: 8 }}>
                Working: {Object.values(workingBots).join(', ')}
              </div>
            ) : null}
            <div className="chatComposer">
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
                  rows={3}
                />
              </div>
              <button
                className="btn primary"
                disabled={sending || !text.trim() || !data?.conversation?.default_bot_id}
                onClick={() => void sendMessage()}
              >
                {sending ? <LoadingSpinner label="Sending" /> : 'Send'}
              </button>
            </div>
            {sendErr ? <div className="alert">{sendErr}</div> : null}
          </section>
        </>
      )}
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
