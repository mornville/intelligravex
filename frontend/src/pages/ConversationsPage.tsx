import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { apiGet } from '../api/client'
import type { Bot, ConversationSummary } from '../types'
import { fmtIso, fmtUsd } from '../utils/format'

export default function ConversationsPage() {
  const [items, setItems] = useState<ConversationSummary[]>([])
  const [bots, setBots] = useState<Bot[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)
  const [total, setTotal] = useState(0)
  const [filterBotId, setFilterBotId] = useState<string>('')
  const [filterTest, setFilterTest] = useState<string>('') // "", "true", "false"

  const pages = useMemo(() => Math.max(1, Math.ceil(total / pageSize)), [total, pageSize])

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

  useEffect(() => {
    void reload()
  }, [page, pageSize, filterBotId, filterTest])

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h1>Conversations</h1>
          <div className="muted">Stored conversations (paginated).</div>
        </div>
        <div className="row gap">
          <select value={filterBotId} onChange={(e) => (setPage(1), setFilterBotId(e.target.value))}>
            <option value="">All bots</option>
            {bots.map((b) => (
              <option key={b.id} value={b.id}>
                {b.name}
              </option>
            ))}
          </select>
          <select value={filterTest} onChange={(e) => (setPage(1), setFilterTest(e.target.value))}>
            <option value="">All</option>
            <option value="true">test=true</option>
            <option value="false">test=false</option>
          </select>
          <select value={pageSize} onChange={(e) => (setPage(1), setPageSize(Number(e.target.value)))}>
            {[25, 50, 100, 200].map((n) => (
              <option key={n} value={n}>
                {n}/page
              </option>
            ))}
          </select>
        </div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <section className="card">
        <div className="cardTitleRow">
          <div className="cardTitle">List</div>
          <div className="muted">
            page {page}/{pages} • total {total}
          </div>
        </div>

        {loading ? (
          <div className="muted">Loading…</div>
        ) : items.length === 0 ? (
          <div className="muted">No conversations.</div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Created</th>
                <th>UUID</th>
                <th>Bot</th>
                <th>test</th>
                <th style={{ textAlign: 'right' }}>Cost</th>
                <th style={{ textAlign: 'right' }}>Resume</th>
              </tr>
            </thead>
            <tbody>
              {items.map((c) => (
                <tr key={c.id}>
                  <td>{fmtIso(c.created_at)}</td>
                  <td className="mono">
                    <Link className="link" to={`/conversations/${c.id}`}>
                      {c.id}
                    </Link>
                  </td>
                  <td>{c.bot_name || c.bot_id}</td>
                  <td>{c.test_flag ? 'true' : 'false'}</td>
                  <td className="mono" style={{ textAlign: 'right' }}>
                    {fmtUsd(c.cost_usd_est)}
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <Link className="btn ghost" to={`/bots/${c.bot_id}?conversation_id=${c.id}`}>
                      Continue
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
    </div>
  )
}
