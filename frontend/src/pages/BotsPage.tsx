import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { apiDelete, apiGet, apiPost } from '../api/client'
import SelectField from '../components/SelectField'
import LoadingSpinner from '../components/LoadingSpinner'
import type { Bot, Options } from '../types'
import { fmtIso } from '../utils/format'
import { TrashIcon } from '@heroicons/react/24/solid'

export default function BotsPage() {
  const nav = useNavigate()
  const [bots, setBots] = useState<Bot[]>([])
  const [options, setOptions] = useState<Options | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [showCreateModal, setShowCreateModal] = useState(false)

  const defaultPrompt = useMemo(
    () => 'You are a fast, helpful voice assistant. Keep answers concise unless asked.',
    [],
  )

  function llmModels(provider: string, o: Options | null, fallback: string): string[] {
    const base = provider === 'openrouter' ? o?.openrouter_models || [] : o?.openai_models || []
    if (!base.length) return [fallback]
    return base.includes(fallback) ? base : [fallback, ...base]
  }

  const [newBot, setNewBot] = useState({
    name: '',
    llm_provider: 'openai',
    openai_model: 'o4-mini',
    openai_asr_model: 'gpt-4o-mini-transcribe',
    web_search_model: 'gpt-4o-mini',
    codex_model: 'gpt-5.1-codex-mini',
    system_prompt: defaultPrompt,
    language: 'en',
    openai_tts_model: 'gpt-4o-mini-tts',
    openai_tts_voice: 'alloy',
    openai_tts_speed: 1.0,
    start_message_mode: 'llm' as const,
    start_message_text: '',
  })

  async function reload() {
    setLoading(true)
    setErr(null)
    try {
      const [b, o] = await Promise.all([
        apiGet<{ items: Bot[] }>('/api/bots'),
        apiGet<Options>('/api/options'),
      ])
      setBots(b.items)
      setOptions(o)
      setNewBot((p) => ({
        ...p,
        openai_model: (() => {
          const provider = p.llm_provider || 'openai'
          const models = llmModels(provider, o, p.openai_model)
          return models.includes(p.openai_model) ? p.openai_model : models[0] || p.openai_model
        })(),
        openai_asr_model: o.openai_asr_models?.includes(p.openai_asr_model)
          ? p.openai_asr_model
          : o.openai_asr_models?.[0] || p.openai_asr_model,
        web_search_model: o.openai_models.includes(p.web_search_model)
          ? p.web_search_model
          : o.openai_models[0] || p.web_search_model,
        codex_model: o.openai_models.includes(p.codex_model) ? p.codex_model : o.openai_models[0] || p.codex_model,
      }))
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void reload()
  }, [])

  async function onCreate() {
    if (!newBot.name.trim()) return
    setCreating(true)
    setErr(null)
    try {
      const payload: any = { ...newBot }
      const created = await apiPost<Bot>('/api/bots', payload)
      setShowCreateModal(false)
      nav(`/bots/${created.id}`)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setCreating(false)
    }
  }

  async function onDelete(bot: Bot) {
    const ok = window.confirm(
      `Delete bot "${bot.name}"?\n\nThis will also delete its conversations and messages.\nThis is NOT reversible.`,
    )
    if (!ok) return
    setErr(null)
    try {
      await apiDelete(`/api/bots/${bot.id}`)
      await reload()
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h1>Assistants</h1>
          <div className="muted">Manage assistants and track their performance.</div>
        </div>
        <button className="btn primary" onClick={() => setShowCreateModal(true)}>
          New assistant
        </button>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      {loading ? (
        <div className="muted">
          <LoadingSpinner />
        </div>
      ) : bots.length === 0 ? (
        <div className="assistantEmpty">
          <button className="assistantCreatePill" onClick={() => setShowCreateModal(true)}>
            + Create new assistant
          </button>
        </div>
      ) : (
        <div className="assistantsGrid">
          {bots.map((b) => (
            <Link className="card assistantCard" to={`/bots/${b.id}`} key={b.id}>
              <div className="assistantHeader">
                <div>
                  <div className="assistantName">{b.name}</div>
                  <div className="muted">Model: {b.openai_model} · {(b.llm_provider || 'openai')}</div>
                </div>
                <button
                  className="btn iconBtn danger"
                  onClick={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    onDelete(b)
                  }}
                  aria-label="Delete assistant"
                  title="Delete assistant"
                >
                  <TrashIcon aria-hidden="true" />
                </button>
              </div>
              <div className="assistantStats">
                <div className="assistantStat">
                  <div className="muted">Conversations</div>
                  <div>{b.stats?.conversations ?? 0}</div>
                </div>
                <div className="assistantStat">
                  <div className="muted">Tokens</div>
                  <div className="mono">
                    {(b.stats?.input_tokens ?? 0).toLocaleString()} / {(b.stats?.output_tokens ?? 0).toLocaleString()}
                  </div>
                </div>
                <div className="assistantStat">
                  <div className="muted">Cost</div>
                  <div className="mono">${(b.stats?.cost_usd ?? 0).toFixed(6)}</div>
                </div>
                <div className="assistantStat">
                  <div className="muted">Latency</div>
                  <div className="mono">
                    {(b.stats?.avg_llm_ttfb_ms ?? null) === null ? '—' : `${b.stats?.avg_llm_ttfb_ms}ms`} /{' '}
                    {(b.stats?.avg_llm_total_ms ?? null) === null ? '—' : `${b.stats?.avg_llm_total_ms}ms`}
                  </div>
                </div>
              </div>
              <div className="assistantFooter">
                <div className="muted">Updated {fmtIso(b.updated_at)}</div>
              </div>
            </Link>
          ))}
        </div>
      )}

      {showCreateModal ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow">
              <div>
                <div className="cardTitle">New assistant</div>
                <div className="muted">Set a name and choose the core models.</div>
              </div>
              <button className="btn" onClick={() => setShowCreateModal(false)}>
                Close
              </button>
            </div>
            <details className="accordion" open>
              <summary>Core settings</summary>
              <div className="formRow">
                <label>Name</label>
                <input
                  value={newBot.name}
                  onChange={(e) => setNewBot((p) => ({ ...p, name: e.target.value }))}
                  placeholder="e.g. Sales Assistant"
                />
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
                  {(options?.llm_providers || ['openai', 'openrouter']).map((p) => (
                    <option value={p} key={p}>
                      {p}
                    </option>
                  ))}
                </SelectField>
              </div>
              <div className="formRow">
                <label>LLM model</label>
                <SelectField value={newBot.openai_model} onChange={(e) => setNewBot((p) => ({ ...p, openai_model: e.target.value }))}>
                  {llmModels(newBot.llm_provider, options, newBot.openai_model).map((m) => (
                    <option value={m} key={m}>
                      {m}
                    </option>
                  ))}
                </SelectField>
              </div>
              <div className="formRow">
                <label>System prompt</label>
                <textarea
                  value={newBot.system_prompt}
                  onChange={(e) => setNewBot((p) => ({ ...p, system_prompt: e.target.value }))}
                  rows={6}
                />
              </div>
            </details>

            <details className="accordion" style={{ marginTop: 10 }}>
              <summary>Advanced models</summary>
              <div className="formRow">
                <label>Web search model</label>
                <SelectField value={newBot.web_search_model} onChange={(e) => setNewBot((p) => ({ ...p, web_search_model: e.target.value }))}>
                  {(options?.openai_models || ['gpt-4o-mini']).map((m) => (
                    <option value={m} key={m}>
                      {m}
                    </option>
                  ))}
                </SelectField>
                <div className="muted">Used for web_search filtering + summarization.</div>
              </div>
              <div className="formRow">
                <label>Codex model</label>
                <SelectField value={newBot.codex_model} onChange={(e) => setNewBot((p) => ({ ...p, codex_model: e.target.value }))}>
                  {(options?.openai_models || ['gpt-5.1-codex-mini']).map((m) => (
                    <option value={m} key={m}>
                      {m}
                    </option>
                  ))}
                </SelectField>
                <div className="muted">Used for “use Codex for response” HTTP integrations.</div>
              </div>
            </details>

            <details className="accordion" style={{ marginTop: 10 }}>
              <summary>Voice & ASR</summary>
              <div className="formRowGrid2">
                <div className="formRow">
                  <label>ASR language</label>
                  <SelectField value={newBot.language} onChange={(e) => setNewBot((p) => ({ ...p, language: e.target.value }))}>
                    {(options?.languages || ['en']).map((l) => (
                      <option value={l} key={l}>
                        {l}
                      </option>
                    ))}
                  </SelectField>
                </div>
                <div className="formRow">
                  <label>ASR model</label>
                  <SelectField value={newBot.openai_asr_model} onChange={(e) => setNewBot((p) => ({ ...p, openai_asr_model: e.target.value }))}>
                    {(options?.openai_asr_models || ['gpt-4o-mini-transcribe']).map((m) => (
                      <option value={m} key={m}>
                        {m}
                      </option>
                    ))}
                  </SelectField>
                </div>
              </div>

              <div className="formRowGrid2">
                <div className="formRow">
                  <label>OpenAI TTS model</label>
                  <SelectField value={newBot.openai_tts_model} onChange={(e) => setNewBot((p) => ({ ...p, openai_tts_model: e.target.value }))}>
                    {(options?.openai_tts_models?.length ? options.openai_tts_models : ['gpt-4o-mini-tts']).map((m) => (
                      <option value={m} key={m}>
                        {m}
                      </option>
                    ))}
                  </SelectField>
                </div>
                <div className="formRow">
                  <label>OpenAI voice</label>
                  <SelectField value={newBot.openai_tts_voice} onChange={(e) => setNewBot((p) => ({ ...p, openai_tts_voice: e.target.value }))}>
                    {(options?.openai_tts_voices?.length ? options.openai_tts_voices : ['alloy']).map((v) => (
                      <option value={v} key={v}>
                        {v}
                      </option>
                    ))}
                  </SelectField>
                </div>
              </div>
              <div className="formRow">
                <label>OpenAI speed</label>
                <input
                  type="number"
                  step="0.1"
                  min="0.25"
                  max="4"
                  value={newBot.openai_tts_speed}
                  onChange={(e) => setNewBot((p) => ({ ...p, openai_tts_speed: Number(e.target.value) }))}
                />
              </div>
            </details>

            <div className="row formActions" style={{ justifyContent: 'flex-end' }}>
              <button className="btn primary" onClick={onCreate} disabled={creating || !newBot.name.trim()}>
                {creating ? 'Creating…' : 'Create assistant'}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
