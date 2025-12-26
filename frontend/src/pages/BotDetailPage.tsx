import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { apiDelete, apiGet, apiPut } from '../api/client'
import MicTest from '../components/MicTest'
import type { ApiKey, Bot, Options } from '../types'

type TtsMeta = { speakers: string[]; languages: string[] }

export default function BotDetailPage() {
  const { botId } = useParams()
  const nav = useNavigate()
  const [bot, setBot] = useState<Bot | null>(null)
  const [keys, setKeys] = useState<ApiKey[]>([])
  const [options, setOptions] = useState<Options | null>(null)
  const [ttsMeta, setTtsMeta] = useState<TtsMeta | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)

  const speakerChoices = useMemo(() => {
    const s = ttsMeta?.speakers || []
    return ['(auto)', ...s]
  }, [ttsMeta?.speakers])

  async function reload() {
    if (!botId) return
    setErr(null)
    try {
      const [b, k, o] = await Promise.all([
        apiGet<Bot>(`/api/bots/${botId}`),
        apiGet<{ items: ApiKey[] }>(`/api/keys?provider=openai`),
        apiGet<Options>(`/api/options`),
      ])
      setBot(b)
      setKeys(k.items)
      setOptions(o)
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  useEffect(() => {
    void reload()
  }, [botId])

  useEffect(() => {
    if (!bot?.xtts_model) return
    void (async () => {
      try {
        const meta = await apiGet<TtsMeta>(`/api/tts/meta?model_name=${encodeURIComponent(bot.xtts_model)}`)
        setTtsMeta(meta)
      } catch {
        setTtsMeta({ speakers: [], languages: [] })
      }
    })()
  }, [bot?.xtts_model])

  async function save(patch: Record<string, unknown>) {
    if (!botId) return
    setSaving(true)
    setErr(null)
    try {
      const updated = await apiPut<Bot>(`/api/bots/${botId}`, patch)
      setBot(updated)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setSaving(false)
    }
  }

  async function onDelete() {
    if (!bot) return
    const ok = window.confirm(
      `Delete bot "${bot.name}"?\n\nThis will also delete its conversations and messages.\nThis is NOT reversible.`,
    )
    if (!ok) return
    try {
      await apiDelete(`/api/bots/${bot.id}`)
      nav('/bots')
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  if (!botId) return null

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h1>{bot?.name || 'Bot'}</h1>
          <div className="muted mono">UUID: {botId}</div>
        </div>
        <div className="row gap">
          <button className="btn danger ghost" onClick={() => void onDelete()} disabled={!bot}>
            Delete bot
          </button>
          <button className="btn" onClick={() => nav('/bots')}>
            Back
          </button>
        </div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <div className="grid2">
        <section className="card">
          <div className="cardTitleRow">
            <div className="cardTitle">Configuration</div>
            {saving ? <div className="pill">saving…</div> : <div className="pill">saved</div>}
          </div>

          {!bot ? (
            <div className="muted">Loading…</div>
          ) : (
            <>
              <div className="formRow">
                <label>Name</label>
                <input value={bot.name} onChange={(e) => setBot((p) => (p ? { ...p, name: e.target.value } : p))} />
              </div>
              <div className="formRow">
                <label>OpenAI model</label>
                <select
                  value={bot.openai_model}
                  onChange={(e) => void save({ openai_model: e.target.value })}
                >
                  {(options?.openai_models || [bot.openai_model]).map((m) => (
                    <option value={m} key={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div className="formRow">
                <label>OpenAI key</label>
                <select
                  value={bot.openai_key_id || ''}
                  onChange={(e) => void save({ openai_key_id: e.target.value || null })}
                >
                  <option value="">(use env OPENAI_API_KEY)</option>
                  {keys.map((k) => (
                    <option value={k.id} key={k.id}>
                      {k.name} — {k.hint}
                    </option>
                  ))}
                </select>
              </div>
              <div className="formRow">
                <label>System prompt</label>
                <textarea
                  value={bot.system_prompt}
                  onChange={(e) => setBot((p) => (p ? { ...p, system_prompt: e.target.value } : p))}
                  rows={10}
                />
                <div className="row">
                  <button className="btn" onClick={() => void save({ name: bot.name, system_prompt: bot.system_prompt })}>
                    Save prompt/name
                  </button>
                </div>
              </div>

              <div className="cardSubTitle">Conversation start message</div>
              <div className="formRowGrid2">
                <div className="formRow">
                  <label>Mode</label>
                  <select value={bot.start_message_mode} onChange={(e) => void save({ start_message_mode: e.target.value })}>
                    <option value="static">Static</option>
                    <option value="llm">LLM-generated</option>
                  </select>
                </div>
                <div className="formRow">
                  <label>Static start message (optional)</label>
                  <input
                    value={bot.start_message_text}
                    placeholder="(empty = will be generated by LLM)"
                    onChange={(e) => setBot((p) => (p ? { ...p, start_message_text: e.target.value } : p))}
                  />
                  <div className="row">
                    <button className="btn" onClick={() => void save({ start_message_text: bot.start_message_text })}>
                      Save start message
                    </button>
                  </div>
                </div>
              </div>

              <div className="formRowGrid2">
                <div className="formRow">
                  <label>ASR language</label>
                  <select value={bot.language} onChange={(e) => void save({ language: e.target.value })}>
                    {(options?.languages || [bot.language]).map((l) => (
                      <option value={l} key={l}>
                        {l}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="formRow">
                  <label>TTS language</label>
                  <select value={bot.tts_language} onChange={(e) => void save({ tts_language: e.target.value })}>
                    {(ttsMeta?.languages?.length ? ttsMeta.languages : options?.languages || [bot.tts_language]).map((l) => (
                      <option value={l} key={l}>
                        {l}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="formRowGrid2">
                <div className="formRow">
                  <label>Whisper model</label>
                  <select value={bot.whisper_model} onChange={(e) => void save({ whisper_model: e.target.value })}>
                    {(options?.whisper_models || [bot.whisper_model]).map((m) => (
                      <option value={m} key={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="formRow">
                  <label>Whisper device</label>
                  <select value={bot.whisper_device} onChange={(e) => void save({ whisper_device: e.target.value })}>
                    {(options?.whisper_devices || [bot.whisper_device]).map((d) => (
                      <option value={d} key={d}>
                        {d}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="formRow">
                <label>XTTS v2 model</label>
                <select value={bot.xtts_model} onChange={(e) => void save({ xtts_model: e.target.value })}>
                  {(options?.xtts_models || [bot.xtts_model]).map((m) => (
                    <option value={m} key={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>

              <div className="formRowGrid2">
                <div className="formRow">
                  <label>Speaker ID (optional)</label>
                  <select
                    value={bot.speaker_id || '(auto)'}
                    onChange={(e) => void save({ speaker_id: e.target.value === '(auto)' ? null : e.target.value })}
                  >
                    {speakerChoices.map((s) => (
                      <option value={s} key={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="formRow">
                  <label>Speaker WAV path (optional)</label>
                  <input
                    value={bot.speaker_wav || ''}
                    placeholder="/path/to/voice.wav"
                    onChange={(e) => setBot((p) => (p ? { ...p, speaker_wav: e.target.value } : p))}
                  />
                  <div className="row">
                    <button className="btn" onClick={() => void save({ speaker_wav: bot.speaker_wav || null })}>
                      Save WAV path
                    </button>
                  </div>
                </div>
              </div>

              <div className="formRowGrid2">
                <div className="formRow">
                  <label>TTS chunk min chars</label>
                  <input
                    type="number"
                    value={bot.tts_chunk_min_chars}
                    onChange={(e) => void save({ tts_chunk_min_chars: Number(e.target.value) })}
                  />
                </div>
                <div className="formRow">
                  <label>TTS chunk max chars</label>
                  <input
                    type="number"
                    value={bot.tts_chunk_max_chars}
                    onChange={(e) => void save({ tts_chunk_max_chars: Number(e.target.value) })}
                  />
                </div>
              </div>
            </>
          )}
        </section>

        <MicTest botId={botId} />
      </div>
    </div>
  )
}

