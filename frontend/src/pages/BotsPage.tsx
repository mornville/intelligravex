import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { apiDelete, apiGet, apiPost } from '../api/client'
import SelectField from '../components/SelectField'
import LoadingSpinner from '../components/LoadingSpinner'
import InlineHelpTip from '../components/InlineHelpTip'
import type { Bot, Options } from '../types'
import { fmtIso } from '../utils/format'
import { formatLocalModelToolSupport } from '../utils/localModels'
import { formatProviderLabel, orderProviderList } from '../utils/llmProviders'
import { useChatgptOauth } from '../hooks/useChatgptOauth'
import { TrashIcon, XMarkIcon } from '@heroicons/react/24/solid'
import { useEscapeClose } from '../hooks/useEscapeClose'

export default function BotsPage() {
  const nav = useNavigate()
  const [bots, setBots] = useState<Bot[]>([])
  const [options, setOptions] = useState<Options | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [showCreateModal, setShowCreateModal] = useState(false)
  useEscapeClose(() => setShowCreateModal(false), showCreateModal)
  const chatgptOauth = useChatgptOauth()

  const defaultPrompt = useMemo(
    () => 'You are a fast, helpful voice assistant. Keep answers concise unless asked.',
    [],
  )

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
  const voiceRequiresOpenAI = newBot.llm_provider === 'chatgpt'
  const newBotNeedsChatgptAuth = newBot.llm_provider === 'chatgpt' && !chatgptOauth.ready

  const newBotLocalModel =
    (options?.local_models || []).find((m) => m.id === newBot.openai_model) || null

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
      if (o?.default_llm_provider || o?.default_llm_model) {
        setNewBot((p) => ({
          ...p,
          llm_provider: (o?.default_llm_provider || p.llm_provider) as any,
          openai_model: o?.default_llm_model || p.openai_model,
        }))
      }
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
                  <div className="muted">Model: {b.openai_model} · {formatProviderLabel(b.llm_provider || 'openai')}</div>
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
                    {(b.stats?.avg_llm_ttfb_ms ?? null) === null ? '-' : `${b.stats?.avg_llm_ttfb_ms}ms`} /{' '}
                    {(b.stats?.avg_llm_total_ms ?? null) === null ? '-' : `${b.stats?.avg_llm_total_ms}ms`}
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
            <div className="cardTitleRow modalSticky">
              <div>
                <div className="cardTitle">New assistant</div>
                <div className="muted">Set a name and choose the core models.</div>
              </div>
              <button className="iconBtn modalCloseBtn" onClick={() => setShowCreateModal(false)} aria-label="Close">
                <XMarkIcon />
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
                  {orderProviderList(options?.llm_providers || ['openai', 'openrouter', 'local']).map((p) => (
                    <option value={p} key={p}>
                      {formatProviderLabel(p)}
                    </option>
                  ))}
                </SelectField>
              </div>
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
                  <label>
                    ASR language
                    {voiceRequiresOpenAI ? <InlineHelpTip text="Requires OpenAI API key." /> : null}
                  </label>
                  <SelectField
                    value={newBot.language}
                    onChange={(e) => setNewBot((p) => ({ ...p, language: e.target.value }))}
                    disabled={voiceRequiresOpenAI}
                  >
                    {(options?.languages || ['en']).map((l) => (
                      <option value={l} key={l}>
                        {l}
                      </option>
                    ))}
                  </SelectField>
                </div>
                <div className="formRow">
                  <label>
                    ASR model
                    {voiceRequiresOpenAI ? <InlineHelpTip text="Requires OpenAI API key." /> : null}
                  </label>
                  <SelectField
                    value={newBot.openai_asr_model}
                    onChange={(e) => setNewBot((p) => ({ ...p, openai_asr_model: e.target.value }))}
                    disabled={voiceRequiresOpenAI}
                  >
                    {(options?.openai_asr_models || ['gpt-4o-mini-transcribe']).map((m) => (
                      <option value={m} key={m}>
                        {m}
                      </option>
                    ))}
                  </SelectField>
                </div>
              </div>
              {voiceRequiresOpenAI ? <div className="muted">ASR disabled for ChatGPT OAuth. Add an OpenAI API key to enable it.</div> : null}

              <div className="formRowGrid2">
                <div className="formRow">
                  <label>
                    OpenAI TTS model
                    {voiceRequiresOpenAI ? <InlineHelpTip text="Requires OpenAI API key." /> : null}
                  </label>
                  <SelectField
                    value={newBot.openai_tts_model}
                    onChange={(e) => setNewBot((p) => ({ ...p, openai_tts_model: e.target.value }))}
                    disabled={voiceRequiresOpenAI}
                  >
                    {(options?.openai_tts_models?.length ? options.openai_tts_models : ['gpt-4o-mini-tts']).map((m) => (
                      <option value={m} key={m}>
                        {m}
                      </option>
                    ))}
                  </SelectField>
                </div>
                <div className="formRow">
                  <label>
                    OpenAI voice
                    {voiceRequiresOpenAI ? <InlineHelpTip text="Requires OpenAI API key." /> : null}
                  </label>
                  <SelectField
                    value={newBot.openai_tts_voice}
                    onChange={(e) => setNewBot((p) => ({ ...p, openai_tts_voice: e.target.value }))}
                    disabled={voiceRequiresOpenAI}
                  >
                    {(options?.openai_tts_voices?.length ? options.openai_tts_voices : ['alloy']).map((v) => (
                      <option value={v} key={v}>
                        {v}
                      </option>
                    ))}
                  </SelectField>
                </div>
              </div>
            <div className="formRow">
              <label>
                OpenAI speed
                {voiceRequiresOpenAI ? <InlineHelpTip text="Requires OpenAI API key." /> : null}
              </label>
              <input
                type="number"
                step="0.1"
                min="0.25"
                max="4"
                value={newBot.openai_tts_speed}
                onChange={(e) => setNewBot((p) => ({ ...p, openai_tts_speed: Number(e.target.value) }))}
                disabled={voiceRequiresOpenAI}
              />
            </div>
              {voiceRequiresOpenAI ? <div className="muted">TTS disabled for ChatGPT OAuth. Add an OpenAI API key to enable it.</div> : null}
            </details>

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

            <div className="row formActions" style={{ justifyContent: 'flex-end' }}>
              <button
                className="btn primary"
                onClick={onCreate}
                disabled={creating || !newBot.name.trim() || newBotNeedsChatgptAuth}
              >
                {creating ? 'Creating…' : 'Create assistant'}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
