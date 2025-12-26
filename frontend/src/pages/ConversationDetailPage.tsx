import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { apiGet } from '../api/client'
import type { ConversationDetail, ConversationMessage } from '../types'
import { fmtIso, fmtMs, fmtUsd } from '../utils/format'

function safeJson(s: string): any {
  try {
    return JSON.parse(s)
  } catch {
    return null
  }
}

export default function ConversationDetailPage() {
  const { conversationId } = useParams()
  const nav = useNavigate()
  const [data, setData] = useState<ConversationDetail | null>(null)
  const [err, setErr] = useState<string | null>(null)

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

  const conv = data?.conversation

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h1>Conversation</h1>
          <div className="muted">
            bot: {conv?.bot_name || conv?.bot_id} • {conv?.test_flag ? 'test' : 'prod'}
          </div>
        </div>
        <button className="btn" onClick={() => nav('/conversations')}>
          Back
        </button>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      {!data || !conv ? (
        <div className="muted">Loading…</div>
      ) : (
        <>
          <section className="card">
            <div className="cardTitleRow">
              <div className="cardTitle">Summary</div>
              <div className="muted mono">{conv.id}</div>
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
            </div>

            <details className="details" style={{ marginTop: 10 }}>
              <summary>Metadata</summary>
              <pre className="pre">{JSON.stringify(safeJson(conv.metadata_json) || {}, null, 2)}</pre>
            </details>
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

  return (
    <div className={cls}>
      <div className="bubbleMeta" style={{ marginBottom: 6 }}>
        <span className="pill">{label}</span> <span className="muted">{fmtIso(m.created_at)}</span>
      </div>
      <div className="bubbleText">{m.role === 'tool' ? (m.tool_name ? `${m.tool_name}` : 'tool') : m.content}</div>
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

