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

  const defaultPrompt = useMemo(
    () => 'You are a fast, helpful voice assistant. Keep answers concise unless asked.',
    [],
  )

  const [newBot, setNewBot] = useState({
    name: '',
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
        openai_model: o.openai_models.includes(p.openai_model) ? p.openai_model : o.openai_models[0] || p.openai_model,
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
          <h1>Bots</h1>
          <div className="muted">Create, configure, and test voice bots.</div>
        </div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <div className="grid2">
        <section className="card">
          <div className="cardTitle">New bot</div>
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
              <label>OpenAI model</label>
              <SelectField value={newBot.openai_model} onChange={(e) => setNewBot((p) => ({ ...p, openai_model: e.target.value }))}>
                {(options?.openai_models || ['gpt-4o']).map((m) => (
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
          <div className="row formActions">
            <button className="btn primary" onClick={onCreate} disabled={creating || !newBot.name.trim()}>
              {creating ? 'Creating…' : 'Create bot'}
            </button>
          </div>
        </section>

        <section className="card">
          <div className="cardTitle">Existing bots</div>
          {loading ? (
            <div className="muted">
              <LoadingSpinner />
            </div>
          ) : bots.length === 0 ? (
            <div className="muted">No bots yet.</div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Model</th>
                  <th>Conversations</th>
                  <th>Tokens</th>
                  <th>Cost</th>
                  <th>Latency</th>
                  <th>Updated</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {bots.map((b) => (
                  <tr key={b.id}>
                    <td>
                      <Link className="link" to={`/bots/${b.id}`}>
                        {b.name}
                      </Link>
                    </td>
                    <td className="mono">{b.openai_model}</td>
                    <td>{b.stats?.conversations ?? 0}</td>
                    <td className="mono">
                      {(b.stats?.input_tokens ?? 0).toLocaleString()} / {(b.stats?.output_tokens ?? 0).toLocaleString()}
                    </td>
                    <td className="mono">${(b.stats?.cost_usd ?? 0).toFixed(6)}</td>
                    <td className="mono">
                      {(b.stats?.avg_llm_ttfb_ms ?? null) === null ? '—' : `${b.stats?.avg_llm_ttfb_ms}ms`} /{' '}
                      {(b.stats?.avg_llm_total_ms ?? null) === null ? '—' : `${b.stats?.avg_llm_total_ms}ms`}
                    </td>
                    <td>{fmtIso(b.updated_at)}</td>
                    <td style={{ textAlign: 'right' }}>
                      <button
                        className="btn iconBtn danger"
                        onClick={() => onDelete(b)}
                        aria-label="Delete bot"
                        title="Delete bot"
                      >
                        <TrashIcon aria-hidden="true" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </div>
    </div>
  )
}
