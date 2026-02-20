import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { apiGet, apiPost } from '../api/client'
import SelectField from '../components/SelectField'
import LoadingSpinner from '../components/LoadingSpinner'
import type { Bot, ConversationSummary, GroupConversationDetail, GroupConversationSummary } from '../types'
import { fmtIso, fmtUsd } from '../utils/format'

export default function ConversationsPage() {
  const DEMO_TITLE = 'Demo: PM → SDE1 → SDE2'
  const nav = useNavigate()
  const [items, setItems] = useState<ConversationSummary[]>([])
  const [groups, setGroups] = useState<GroupConversationSummary[]>([])
  const [bots, setBots] = useState<Bot[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [groupErr, setGroupErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [groupLoading, setGroupLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)
  const [total, setTotal] = useState(0)
  const [filterBotId, setFilterBotId] = useState<string>('')
  const [filterTest, setFilterTest] = useState<string>('') // "", "true", "false"
  const [showGroupModal, setShowGroupModal] = useState(false)
  const [groupTitle, setGroupTitle] = useState('')
  const [groupFilter, setGroupFilter] = useState('')
  const [groupSelected, setGroupSelected] = useState<string[]>([])
  const [groupDefaultInput, setGroupDefaultInput] = useState('')
  const [groupDefaultId, setGroupDefaultId] = useState('')
  const [groupSaving, setGroupSaving] = useState(false)
  const [groupSaveErr, setGroupSaveErr] = useState<string | null>(null)
  const [demoSeen, setDemoSeen] = useState(true)

  const pages = useMemo(() => Math.max(1, Math.ceil(total / pageSize)), [total, pageSize])
  const selectedBots = useMemo(() => bots.filter((b) => groupSelected.includes(b.id)), [bots, groupSelected])
  const selectedSlugMap = useMemo(() => buildSlugMap(selectedBots), [selectedBots])

  async function reload() {
    setLoading(true)
    setErr(null)
    try {
      const qs = new URLSearchParams()
      qs.set('page', String(page))
      qs.set('page_size', String(pageSize))
      if (filterBotId) qs.set('bot_id', filterBotId)
      if (filterTest) qs.set('test_flag', filterTest)
      const [c, b] = await Promise.all([
        apiGet<{ items: ConversationSummary[]; total: number }>(`/api/conversations?${qs.toString()}`),
        apiGet<{ items: Bot[] }>('/api/bots'),
      ])
      setItems(c.items)
      setTotal(c.total)
      setBots(b.items)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setLoading(false)
    }
  }

  async function reloadGroups() {
    setGroupLoading(true)
    setGroupErr(null)
    try {
      const g = await apiGet<{ items: GroupConversationSummary[] }>('/api/group-conversations')
      setGroups(g.items)
    } catch (e: any) {
      setGroupErr(String(e?.message || e))
    } finally {
      setGroupLoading(false)
    }
  }

  useEffect(() => {
    void reload()
  }, [page, pageSize, filterBotId, filterTest])

  useEffect(() => {
    void reloadGroups()
  }, [])

  useEffect(() => {
    try {
      const seen = window.localStorage.getItem('gravex_demo_seen')
      setDemoSeen(Boolean(seen))
    } catch {
      setDemoSeen(true)
    }
  }, [])

  useEffect(() => {
    if (groupDefaultId && !groupSelected.includes(groupDefaultId)) {
      setGroupDefaultId('')
      setGroupDefaultInput('')
    }
  }, [groupDefaultId, groupSelected])

  function toggleSelected(botId: string) {
    setGroupSelected((prev) => (prev.includes(botId) ? prev.filter((id) => id !== botId) : [...prev, botId]))
  }

  function updateDefaultInput(next: string) {
    setGroupDefaultInput(next)
    const value = next.trim().replace(/^@/, '').toLowerCase()
    const match = selectedBots.find((b) => selectedSlugMap[b.id] === value)
    setGroupDefaultId(match ? match.id : '')
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
      setShowGroupModal(false)
      setGroupTitle('')
      setGroupFilter('')
      setGroupSelected([])
      setGroupDefaultInput('')
      setGroupDefaultId('')
      void reloadGroups()
      nav(`/groups/${payload.conversation.id}`)
    } catch (e: any) {
      setGroupSaveErr(String(e?.message || e))
    } finally {
      setGroupSaving(false)
    }
  }

  const filteredBots = useMemo(() => {
    const q = groupFilter.trim().toLowerCase()
    if (!q) return bots
    return bots.filter((b) => b.name.toLowerCase().includes(q))
  }, [bots, groupFilter])

  const defaultOptions = selectedBots.map((b) => ({
    id: b.id,
    slug: selectedSlugMap[b.id],
    name: b.name,
  }))

  const canCreateGroup = Boolean(groupTitle.trim()) && groupSelected.length > 0 && Boolean(groupDefaultId)

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h1>Conversations</h1>
          <div className="muted">Stored conversations (paginated).</div>
        </div>
        <div className="row gap">
          <SelectField wrapClassName="compact" value={filterBotId} onChange={(e) => (setPage(1), setFilterBotId(e.target.value))}>
            <option value="">All bots</option>
            {bots.map((b) => (
              <option key={b.id} value={b.id}>
                {b.name}
              </option>
            ))}
          </SelectField>
          <SelectField wrapClassName="compact" value={filterTest} onChange={(e) => (setPage(1), setFilterTest(e.target.value))}>
            <option value="">All</option>
            <option value="true">test=true</option>
            <option value="false">test=false</option>
          </SelectField>
          <SelectField wrapClassName="compact" value={pageSize} onChange={(e) => (setPage(1), setPageSize(Number(e.target.value)))}>
            {[25, 50, 100, 200].map((n) => (
              <option key={n} value={n}>
                {n}/page
              </option>
            ))}
          </SelectField>
        </div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <section className="card">
        <div className="cardTitleRow">
          <div className="cardTitle">Group chats</div>
          <button className="btn primary" onClick={() => setShowGroupModal(true)}>
            New group
          </button>
        </div>
        {groupErr ? <div className="alert">{groupErr}</div> : null}
        {groupLoading ? (
          <div className="muted">
            <LoadingSpinner />
          </div>
        ) : groups.length === 0 ? (
          <div className="muted">No group chats yet.</div>
        ) : (
          <div className="groupCardGrid">
            {groups.map((g) => {
              const defaultBot = g.group_bots.find((b) => b.id === g.default_bot_id)
              const isDemo = g.title === DEMO_TITLE
              const showDemoPulse = isDemo && !demoSeen
              return (
                <button
                  key={g.id}
                  type="button"
                  className={`groupCard ${isDemo ? 'demo' : ''} ${showDemoPulse ? 'demoPulse' : ''}`}
                  onClick={() => {
                    if (isDemo && !demoSeen) {
                      try {
                        window.localStorage.setItem('gravex_demo_seen', '1')
                      } catch {
                        // ignore
                      }
                      setDemoSeen(true)
                    }
                    nav(`/groups/${g.id}`)
                  }}
                >
                  <div className="groupTitle">{g.title}</div>
                  <div className="groupMeta">
                    Default: {defaultBot ? `${defaultBot.name} (@${defaultBot.slug})` : '-'} • {g.group_bots.length} assistants
                  </div>
                  <div className="groupMembers">
                    {g.group_bots.map((b) => (
                      <span key={b.id} className="pill">
                        @{b.slug}
                      </span>
                    ))}
                    {isDemo ? <span className="pill accent">Demo</span> : null}
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </section>

      <section className="card">
        <div className="cardTitleRow">
          <div className="cardTitle">List</div>
          <div className="muted">
            page {page}/{pages} • total {total}
          </div>
        </div>

        {loading ? (
          <div className="muted">
            <LoadingSpinner />
          </div>
        ) : items.length === 0 ? (
          <div className="muted">No conversations.</div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Created</th>
                <th>Bot</th>
                <th>test</th>
                <th style={{ textAlign: 'right' }}>Cost</th>
                <th style={{ textAlign: 'right' }}>Open</th>
              </tr>
            </thead>
            <tbody>
              {items.map((c) => (
                <tr key={c.id}>
                  <td>{fmtIso(c.created_at)}</td>
                  <td>{c.bot_name || c.bot_id}</td>
                  <td>{c.test_flag ? 'true' : 'false'}</td>
                  <td className="mono" style={{ textAlign: 'right' }}>
                    {fmtUsd(c.cost_usd_est)}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <Link className="btn ghost" to={`/bots/${c.bot_id}?conversation_id=${c.id}`}>
                      Open
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <div className="row gap" style={{ justifyContent: 'space-between', marginTop: 12 }}>
          <button className="btn" disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))}>
            Prev
          </button>
          <div className="muted">
            Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, total)} of {total}
          </div>
          <button className="btn" disabled={page >= pages} onClick={() => setPage((p) => Math.min(pages, p + 1))}>
            Next
          </button>
        </div>
      </section>

      {showGroupModal ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow">
              <div>
                <div className="cardTitle">New group chat</div>
                <div className="muted">Pick assistants and set a default before starting.</div>
              </div>
              <button className="btn" onClick={() => setShowGroupModal(false)}>
                Close
              </button>
            </div>

            {groupSaveErr ? <div className="alert">{groupSaveErr}</div> : null}

            <div className="formRow">
              <label>Title</label>
              <input value={groupTitle} onChange={(e) => setGroupTitle(e.target.value)} placeholder="e.g. Build & review loop" />
            </div>

            <div className="formRow">
              <label>Assistants</label>
              <input
                value={groupFilter}
                onChange={(e) => setGroupFilter(e.target.value)}
                placeholder="Filter assistants"
              />
              <div className="groupBotList">
                {filteredBots.map((b) => (
                  <label key={b.id} className="groupBotRow">
                    <input type="checkbox" checked={groupSelected.includes(b.id)} onChange={() => toggleSelected(b.id)} />
                    <span>{b.name}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="formRow">
              <label>Default assistant (@slug with autocomplete)</label>
              <input
                list="group-default-list"
                value={groupDefaultInput}
                onChange={(e) => updateDefaultInput(e.target.value)}
                placeholder="@assistant-slug"
              />
              <datalist id="group-default-list">
                {defaultOptions.map((b) => (
                  <option key={b.id} value={`@${b.slug}`}>
                    {b.name}
                  </option>
                ))}
              </datalist>
              <div className="muted">A default assistant is required to start the group conversation.</div>
            </div>

            <div className="groupMembers" style={{ marginTop: 10 }}>
              {defaultOptions.map((b) => (
                <span key={b.id} className={`pill ${b.id === groupDefaultId ? 'accent' : ''}`}>
                  @{b.slug}
                </span>
              ))}
            </div>

            <div className="formActions row gap" style={{ justifyContent: 'flex-end' }}>
              <button className="btn" onClick={() => setShowGroupModal(false)}>
                Cancel
              </button>
              <button className="btn primary" disabled={!canCreateGroup || groupSaving} onClick={() => void createGroup()}>
                {groupSaving ? <LoadingSpinner label="Creating" /> : 'Create group'}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

function slugifyName(name: string): string {
  const base = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return base || 'assistant'
}

function buildSlugMap(bots: Bot[]): Record<string, string> {
  const used = new Set<string>()
  const map: Record<string, string> = {}
  for (const b of bots) {
    const base = slugifyName(b.name)
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
