import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { apiGet, downloadFile } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'
import type { ConversationDetail, ConversationMessage, ConversationFiles, DataAgentStatus } from '../types'
import { fmtIso, fmtMs, fmtUsd } from '../utils/format'

function safeJson(s: string): any {
  try {
    return JSON.parse(s)
  } catch {
    return null
  }
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

export default function ConversationDetailPage() {
  const { conversationId } = useParams()
  const nav = useNavigate()
  const [data, setData] = useState<ConversationDetail | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [agentStatus, setAgentStatus] = useState<DataAgentStatus | null>(null)
  const [agentErr, setAgentErr] = useState<string | null>(null)
  const [agentLoading, setAgentLoading] = useState(false)
  const [files, setFiles] = useState<ConversationFiles | null>(null)
  const [filesErr, setFilesErr] = useState<string | null>(null)
  const [filesDownloadErr, setFilesDownloadErr] = useState<string | null>(null)
  const [filesDownloadMsg, setFilesDownloadMsg] = useState<string | null>(null)
  const [filesLoading, setFilesLoading] = useState(false)
  const [filesPath, setFilesPath] = useState('')
  const [filesRecursive, setFilesRecursive] = useState(false)
  const [filesHidden, setFilesHidden] = useState(false)
  const downloadMsgTimerRef = useRef<number | null>(null)

  useEffect(() => {
    if (!conversationId) return
    void (async () => {
      setErr(null)
      try {
        const d = await apiGet<ConversationDetail>(`/api/conversations/${conversationId}`)
        setData(d)
      } catch (e: any) {
        setErr(String(e?.message || e))
      }
    })()
  }, [conversationId])

  async function loadAgentStatus() {
    if (!conversationId) return
    setAgentLoading(true)
    setAgentErr(null)
    try {
      const d = await apiGet<DataAgentStatus>(`/api/conversations/${conversationId}/data-agent`)
      setAgentStatus(d)
    } catch (e: any) {
      setAgentErr(String(e?.message || e))
    } finally {
      setAgentLoading(false)
    }
  }

  async function loadFiles(nextPath?: string, nextRecursive?: boolean, nextHidden?: boolean) {
    if (!conversationId) return
    const path = nextPath ?? filesPath
    const recursive = nextRecursive ?? filesRecursive
    const includeHidden = nextHidden ?? filesHidden
    const params = new URLSearchParams()
    if (path) params.set('path', path)
    if (recursive) params.set('recursive', '1')
    if (includeHidden) params.set('include_hidden', '1')
    setFilesLoading(true)
    setFilesErr(null)
    setFilesDownloadErr(null)
    try {
      const d = await apiGet<ConversationFiles>(`/api/conversations/${conversationId}/files?${params.toString()}`)
      setFiles(d)
      setFilesPath(d.path || path)
      setFilesRecursive(recursive)
      setFilesHidden(includeHidden)
    } catch (e: any) {
      setFilesErr(String(e?.message || e))
    } finally {
      setFilesLoading(false)
    }
  }

  useEffect(() => {
    if (!conversationId) return
    void loadAgentStatus()
    void loadFiles('')
  }, [conversationId])

  useEffect(() => {
    return () => {
      if (downloadMsgTimerRef.current) {
        window.clearTimeout(downloadMsgTimerRef.current)
      }
    }
  }, [])

  async function handleDownload(downloadUrl: string | null | undefined, filename: string) {
    if (!downloadUrl) return
    setFilesDownloadErr(null)
    setFilesDownloadMsg(null)
    try {
      await downloadFile(downloadUrl, filename || 'download')
      setFilesDownloadMsg(`Downloaded ${filename || 'file'}.`)
      if (downloadMsgTimerRef.current) window.clearTimeout(downloadMsgTimerRef.current)
      downloadMsgTimerRef.current = window.setTimeout(() => setFilesDownloadMsg(null), 2500)
    } catch (e: any) {
      setFilesDownloadErr(String(e?.message || e))
    }
  }

  const conv = data?.conversation
  const statusLabel = !agentStatus
    ? 'loading'
    : !agentStatus.docker_available
      ? 'docker unavailable'
      : agentStatus.exists
        ? agentStatus.running
          ? 'running'
          : agentStatus.status || 'stopped'
        : 'not started'
  const visibleItems = (files?.items || []).filter((it) => !(it.is_dir && (it.path === '' || it.path === '.')))

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h1>Conversation</h1>
          <div className="muted">
            bot: {conv?.bot_name || conv?.bot_id} • {conv?.test_flag ? 'test' : 'prod'}
          </div>
        </div>
        <div className="row gap">
          {conv ? (
            <button className="btn" onClick={() => nav(`/bots/${conv.bot_id}?conversation_id=${conv.id}`)}>
              Continue
            </button>
          ) : null}
          <button className="btn" onClick={() => nav('/conversations')}>
            Back
          </button>
        </div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      {!data || !conv ? (
        <div className="muted">
          <LoadingSpinner />
        </div>
      ) : (
        <>
          <section className="card">
            <div className="cardTitleRow">
              <div className="cardTitle">Summary</div>
            </div>
            <div className="summaryGrid">
              <div className="summaryItem">
                <div className="muted">Created</div>
                <div>{fmtIso(conv.created_at)}</div>
              </div>
              <div className="summaryItem">
                <div className="muted">Updated</div>
                <div>{fmtIso(conv.updated_at)}</div>
              </div>
              <div className="summaryItem">
                <div className="muted">Tokens</div>
                <div className="mono">
                  in {conv.llm_input_tokens_est} • out {conv.llm_output_tokens_est}
                </div>
              </div>
              <div className="summaryItem">
                <div className="muted">Cost</div>
                <div className="mono">{fmtUsd(conv.cost_usd_est)}</div>
              </div>
              <div className="summaryItem">
                <div className="muted">Last latency</div>
                <div className="mono">
                  ASR {fmtMs(conv.last_asr_ms)} • LLM 1st {fmtMs(conv.last_llm_ttfb_ms)} • LLM {fmtMs(conv.last_llm_total_ms)} •
                  TTS 1st {fmtMs(conv.last_tts_first_audio_ms)} • total {fmtMs(conv.last_total_ms)}
                </div>
              </div>
              <div className="summaryItem">
                <div className="muted">Container</div>
                <div className="mono">{statusLabel}</div>
              </div>
            </div>

            <details className="details" style={{ marginTop: 10 }}>
              <summary>Metadata</summary>
              <pre className="pre">{JSON.stringify(safeJson(conv.metadata_json) || {}, null, 2)}</pre>
            </details>
          </section>

          <section className="card">
            <div className="cardTitleRow">
              <div>
                <div className="cardTitle">Isolated Workspace container</div>
                <div className="muted">Per-conversation runtime for tools and files.</div>
              </div>
              <button className="btn" onClick={() => void loadAgentStatus()} disabled={agentLoading}>
                {agentLoading ? <LoadingSpinner label="Refreshing" /> : 'Refresh'}
              </button>
            </div>
            {agentErr ? <div className="alert">{agentErr}</div> : null}
            {!agentStatus ? (
              <div className="muted">
                <LoadingSpinner />
              </div>
            ) : (
              <div className="summaryGrid">
                <div className="summaryItem">
                  <div className="muted">Status</div>
                  <div className="mono">{statusLabel}</div>
                </div>
                <div className="summaryItem">
                  <div className="muted">Container name</div>
                  <div className="mono">{agentStatus.container_name || '—'}</div>
                </div>
                <div className="summaryItem">
                  <div className="muted">Container id</div>
                  <div className="mono">{agentStatus.container_id || '—'}</div>
                </div>
                <div className="summaryItem">
                  <div className="muted">Workspace</div>
                  <div className="mono">{agentStatus.workspace_dir || '—'}</div>
                </div>
                <div className="summaryItem">
                  <div className="muted">Session id</div>
                  <div className="mono">{agentStatus.session_id || '—'}</div>
                </div>
                <div className="summaryItem">
                  <div className="muted">Started</div>
                  <div className="mono">{agentStatus.started_at ? fmtIso(agentStatus.started_at) : '—'}</div>
                </div>
              </div>
            )}
          </section>

          <section className="card">
            <div className="cardTitleRow">
              <div>
                <div className="cardTitle">Container files</div>
                <div className="muted">Browse the data-agent workspace and download outputs.</div>
              </div>
              <button className="btn" onClick={() => void loadFiles()} disabled={filesLoading}>
                {filesLoading ? <LoadingSpinner label="Refreshing" /> : 'Refresh'}
              </button>
            </div>
            {filesErr || filesDownloadErr ? <div className="alert">{filesErr || filesDownloadErr}</div> : null}
            {filesDownloadMsg ? (
              <div
                className="alert"
                style={{ borderColor: 'rgba(80, 200, 160, 0.4)', background: 'rgba(80, 200, 160, 0.1)' }}
              >
                {filesDownloadMsg}
              </div>
            ) : null}
            <div className="row gap" style={{ marginTop: 10, alignItems: 'center' }}>
              <label className="muted">Path</label>
              <input
                value={filesPath}
                onChange={(e) => setFilesPath(e.target.value)}
                placeholder="(root)"
              />
              <button
                className="btn"
                onClick={() => void loadFiles(filesPath)}
                disabled={filesLoading}
              >
                Go
              </button>
              <button
                className="btn"
                onClick={() => {
                  const parts = filesPath.split('/').filter(Boolean)
                  parts.pop()
                  const next = parts.join('/')
                  setFilesPath(next)
                  void loadFiles(next)
                }}
                disabled={filesLoading || !filesPath}
              >
                Up
              </button>
              <label className="check">
                <input
                  type="checkbox"
                  checked={filesRecursive}
                  onChange={(e) => void loadFiles(filesPath, e.target.checked, filesHidden)}
                />{' '}
                recursive
              </label>
              <label className="check">
                <input
                  type="checkbox"
                  checked={filesHidden}
                  onChange={(e) => void loadFiles(filesPath, filesRecursive, e.target.checked)}
                />{' '}
                hidden
              </label>
              <div className="spacer" />
              <div className="muted mono">
                {files?.items ? `${visibleItems.length} items` : '—'}
              </div>
            </div>
            {!files ? (
              <div className="muted" style={{ marginTop: 10 }}>
                <LoadingSpinner />
              </div>
            ) : (
              <table className="table" style={{ marginTop: 12 }}>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Size</th>
                    <th>Modified</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleItems.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="muted">
                        No files found.
                      </td>
                    </tr>
                  ) : (
                    visibleItems.map((it) => {
                      const size = it.size_bytes === null ? '—' : fmtBytes(it.size_bytes)
                      return (
                        <tr key={`${it.path}_${it.name}`}>
                          <td className="mono">
                            {it.is_dir ? `${it.path || it.name}/` : it.path || it.name}
                          </td>
                          <td className="mono">{size}</td>
                          <td className="mono">{fmtIso(it.mtime)}</td>
                          <td>
                            {it.is_dir ? (
                              <button className="btn" onClick={() => void loadFiles(it.path, false, filesHidden)}>
                                Open
                              </button>
                            ) : it.download_url ? (
                              <button
                                type="button"
                                className="btn"
                                onClick={() => void handleDownload(it.download_url, it.name || 'download')}
                              >
                                Download
                              </button>
                            ) : (
                              <span className="muted">—</span>
                            )}
                          </td>
                        </tr>
                      )
                    })
                  )}
                </tbody>
              </table>
            )}
          </section>

          <section className="card">
            <div className="cardTitle">Messages</div>
            <div className="chat">
              {data.messages.map((m) => (
                <MessageRow key={m.id} m={m} />
              ))}
            </div>
          </section>
        </>
      )}
    </div>
  )
}

function MessageRow({ m }: { m: ConversationMessage }) {
  const cls = m.role === 'user' ? 'bubble user' : m.role === 'assistant' ? 'bubble assistant' : 'bubble tool'
  const showMetrics =
    m.metrics.asr !== null ||
    m.metrics.llm1 !== null ||
    m.metrics.llm !== null ||
    m.metrics.tts1 !== null ||
    m.metrics.total !== null ||
    m.metrics.in !== null ||
    m.metrics.out !== null ||
    m.metrics.cost !== null

  const label = m.role === 'tool' ? (m.tool_kind ? `tool ${m.tool_kind}` : 'tool') : m.role
  const citations = m.role === 'assistant' ? normalizeCitations(m.citations) : []

  return (
    <div className={cls}>
      <div className="bubbleMeta" style={{ marginBottom: 6 }}>
        <span className="pill accent">{label}</span> <span className="muted">{fmtIso(m.created_at)}</span>
      </div>
      <div className="bubbleText">{m.role === 'tool' ? (m.tool_name ? `${m.tool_name}` : 'tool') : m.content}</div>
      {m.role === 'assistant' && citations.length ? (
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
      {showMetrics ? (
        <div className="bubbleMeta mono">
          tok in {m.metrics.in ?? '—'} • out {m.metrics.out ?? '—'} • cost {m.metrics.cost ?? '—'} • ASR {fmtMs(m.metrics.asr)} •
          LLM 1st {fmtMs(m.metrics.llm1)} • LLM {fmtMs(m.metrics.llm)} • TTS 1st {fmtMs(m.metrics.tts1)} • total {fmtMs(m.metrics.total)}
        </div>
      ) : null}
      {m.role === 'tool' ? (
        <details className="details">
          <summary>details</summary>
          <pre className="pre">{m.content}</pre>
        </details>
      ) : null}
    </div>
  )
}
