import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { apiDelete, apiGet, apiPost } from '../api/client'
import type { Bot, Options } from '../types'
import { fmtIso } from '../utils/format'

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
    web_search_model: 'gpt-4o-mini',
    codex_model: 'gpt-5.1-codex-mini',
    system_prompt: defaultPrompt,
    language: 'en',
    tts_language: 'en',
    tts_vendor: 'openai_tts',
    whisper_model: 'small',
    whisper_device: 'auto',
    xtts_model: 'tts_models/multilingual/multi-dataset/xtts_v2',
    speaker_id: '',
    speaker_wav: '',
    openai_tts_model: 'gpt-4o-mini-tts',
    openai_tts_voice: 'alloy',
    openai_tts_speed: 1.0,
    openai_key_id: '',
    tts_split_sentences: false,
    tts_chunk_min_chars: 20,
    tts_chunk_max_chars: 120,
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
      payload.speaker_id = payload.speaker_id.trim() || null
      payload.speaker_wav = payload.speaker_wav.trim() || null
      payload.openai_key_id = payload.openai_key_id.trim() || null
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
        <h1>Bots</h1>
        <div className="muted">Create, configure, and test voice bots.</div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <div className="grid2">
        <section className="card">
          <div className="cardTitle">New bot</div>
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
            <select
              value={newBot.openai_model}
              onChange={(e) => setNewBot((p) => ({ ...p, openai_model: e.target.value }))}
            >
              {(options?.openai_models || ['gpt-4o']).map((m) => (
                <option value={m} key={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
          <div className="formRow">
            <label>Web search model</label>
            <select
              value={newBot.web_search_model}
              onChange={(e) => setNewBot((p) => ({ ...p, web_search_model: e.target.value }))}
            >
              {(options?.openai_models || ['gpt-4o-mini']).map((m) => (
                <option value={m} key={m}>
                  {m}
                </option>
              ))}
            </select>
            <div className="muted">Used for web_search filtering + summarization.</div>
          </div>
          <div className="formRow">
            <label>Codex model</label>
            <select value={newBot.codex_model} onChange={(e) => setNewBot((p) => ({ ...p, codex_model: e.target.value }))}>
              {(options?.openai_models || ['gpt-5.1-codex-mini']).map((m) => (
                <option value={m} key={m}>
                  {m}
                </option>
              ))}
            </select>
            <div className="muted">Used for “use Codex for response” HTTP integrations.</div>
          </div>
          <div className="formRow">
            <label>System prompt</label>
            <textarea
              value={newBot.system_prompt}
              onChange={(e) => setNewBot((p) => ({ ...p, system_prompt: e.target.value }))}
              rows={6}
            />
          </div>
          <div className="formRowGrid2">
            <div className="formRow">
              <label>ASR language</label>
              <select value={newBot.language} onChange={(e) => setNewBot((p) => ({ ...p, language: e.target.value }))}>
                {(options?.languages || ['en']).map((l) => (
                  <option value={l} key={l}>
                    {l}
                  </option>
                ))}
              </select>
            </div>
            <div className="formRow">
              <label>TTS vendor</label>
              <select value={newBot.tts_vendor} onChange={(e) => setNewBot((p) => ({ ...p, tts_vendor: e.target.value }))}>
                {(options?.tts_vendors?.length ? options.tts_vendors : ['openai_tts']).map((v) => (
                  <option value={v} key={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {newBot.tts_vendor === 'xtts_local' ? (
            <div className="formRow">
              <label>TTS language</label>
              <select
                value={newBot.tts_language}
                onChange={(e) => setNewBot((p) => ({ ...p, tts_language: e.target.value }))}
              >
                {(options?.languages || ['en']).map((l) => (
                  <option value={l} key={l}>
                    {l}
                  </option>
                ))}
              </select>
            </div>
          ) : null}

          {newBot.tts_vendor === 'openai_tts' ? (
            <>
              <div className="formRowGrid2">
                <div className="formRow">
                  <label>OpenAI TTS model</label>
                  <select
                    value={newBot.openai_tts_model}
                    onChange={(e) => setNewBot((p) => ({ ...p, openai_tts_model: e.target.value }))}
                  >
                    {(options?.openai_tts_models?.length ? options.openai_tts_models : ['gpt-4o-mini-tts']).map((m) => (
                      <option value={m} key={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="formRow">
                  <label>OpenAI voice</label>
                  <select
                    value={newBot.openai_tts_voice}
                    onChange={(e) => setNewBot((p) => ({ ...p, openai_tts_voice: e.target.value }))}
                  >
                    {(options?.openai_tts_voices?.length ? options.openai_tts_voices : ['alloy']).map((v) => (
                      <option value={v} key={v}>
                        {v}
                      </option>
                    ))}
                  </select>
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
            </>
          ) : null}
          <div className="formRowGrid2">
            <div className="formRow">
              <label>Whisper model</label>
              <select
                value={newBot.whisper_model}
                onChange={(e) => setNewBot((p) => ({ ...p, whisper_model: e.target.value }))}
              >
                {(options?.whisper_models || ['small']).map((m) => (
                  <option value={m} key={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <div className="formRow">
              <label>Whisper device</label>
              <select
                value={newBot.whisper_device}
                onChange={(e) => setNewBot((p) => ({ ...p, whisper_device: e.target.value }))}
              >
                {(options?.whisper_devices || ['auto']).map((d) => (
                  <option value={d} key={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
          </div>
          {newBot.tts_vendor === 'xtts_local' ? (
            <>
              <div className="formRow">
                <label>XTTS v2 model</label>
                <select value={newBot.xtts_model} onChange={(e) => setNewBot((p) => ({ ...p, xtts_model: e.target.value }))}>
                  {(options?.xtts_models || ['tts_models/multilingual/multi-dataset/xtts_v2']).map((m) => (
                    <option value={m} key={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div className="formRowGrid2">
                <div className="formRow">
                  <label>Speaker ID (optional)</label>
                  <input
                    value={newBot.speaker_id}
                    onChange={(e) => setNewBot((p) => ({ ...p, speaker_id: e.target.value }))}
                    placeholder="(auto)"
                  />
                </div>
                <div className="formRow">
                  <label>Speaker WAV path (optional)</label>
                  <input
                    value={newBot.speaker_wav}
                    onChange={(e) => setNewBot((p) => ({ ...p, speaker_wav: e.target.value }))}
                    placeholder="/path/to/voice.wav"
                  />
                </div>
              </div>
            </>
          ) : null}
          <div className="row">
            <button className="btn primary" onClick={onCreate} disabled={creating || !newBot.name.trim()}>
              {creating ? 'Creating…' : 'Create bot'}
            </button>
          </div>
        </section>

        <section className="card">
          <div className="cardTitle">Existing bots</div>
          {loading ? (
            <div className="muted">Loading…</div>
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
                      <div className="muted mono">{b.id}</div>
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
                      <button className="btn danger ghost" onClick={() => onDelete(b)}>
                        Delete
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
