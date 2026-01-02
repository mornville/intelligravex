import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { apiDelete, apiGet, apiPost, apiPut } from '../api/client'
import MicTest from '../components/MicTest'
import type { ApiKey, Bot, IntegrationTool, Options, SystemTool } from '../types'

type TtsMeta = { speakers: string[]; languages: string[] }

function HelpTip({ children }: { children: ReactNode }) {
  return (
    <span className="helpTipWrap" tabIndex={0} role="button" aria-label="Show example">
      <span className="helpTipIcon">?</span>
      <span className="helpTipBubble">{children}</span>
    </span>
  )
}

export default function BotDetailPage() {
  const { botId } = useParams()
  const nav = useNavigate()
  const [bot, setBot] = useState<Bot | null>(null)
  const [keys, setKeys] = useState<ApiKey[]>([])
  const [options, setOptions] = useState<Options | null>(null)
  const [ttsMeta, setTtsMeta] = useState<TtsMeta | null>(null)
  const [tools, setTools] = useState<IntegrationTool[]>([])
  const [systemTools, setSystemTools] = useState<SystemTool[]>([])
  const [showToolModal, setShowToolModal] = useState(false)
  const [toolForm, setToolForm] = useState({
    id: '',
    name: '',
    description: '',
    url: '',
    method: 'GET',
    args_required_csv: '',
    headers_template_json: '{}',
    headers_template_json_masked: '',
    headers_configured: false,
    request_body_template: '{}',
    parameters_schema_json: '',
    response_mapper_json: '{}',
    static_reply_template: '',
  })
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
      const t = await apiGet<{ items: IntegrationTool[]; system_tools?: SystemTool[] }>(`/api/bots/${botId}/tools`)
      setTools(t.items)
      setSystemTools(t.system_tools || [])
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  useEffect(() => {
    void reload()
  }, [botId])

  useEffect(() => {
    if (!bot || !bot.xtts_model || bot.tts_vendor !== 'xtts_local') {
      setTtsMeta(null)
      return
    }
    void (async () => {
      try {
        const meta = await apiGet<TtsMeta>(`/api/tts/meta?model_name=${encodeURIComponent(bot.xtts_model)}`)
        setTtsMeta(meta)
      } catch {
        setTtsMeta({ speakers: [], languages: [] })
      }
    })()
  }, [bot?.xtts_model, bot?.tts_vendor])

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

  function openNewTool() {
    setToolForm({
      id: '',
      name: '',
      description: '',
      url: '',
      method: (options?.http_methods?.[0] || 'GET') as any,
      args_required_csv: '',
      headers_template_json: '{}',
      headers_template_json_masked: '',
      headers_configured: false,
      request_body_template: '{}',
      parameters_schema_json: '',
      response_mapper_json: '{}',
      static_reply_template: '',
    })
    setShowToolModal(true)
  }

  function openEditTool(t: IntegrationTool) {
    setToolForm({
      id: t.id,
      name: t.name,
      description: t.description || '',
      url: t.url,
      method: t.method || 'GET',
      args_required_csv: (t.args_required || []).join(', '),
      // Write-only: never hydrate secrets back into the UI.
      headers_template_json: '',
      headers_template_json_masked: t.headers_template_json_masked || '',
      headers_configured: Boolean(t.headers_configured),
      request_body_template: t.request_body_template || '{}',
      parameters_schema_json: t.parameters_schema_json || '',
      response_mapper_json: t.response_mapper_json || '{}',
      static_reply_template: t.static_reply_template || '',
    })
    setShowToolModal(true)
  }

  async function saveTool() {
    if (!botId) return
    const name = toolForm.name.trim()
    const url = toolForm.url.trim()
    if (!name || !url) {
      setErr('Tool name and url are required.')
      return
    }
    setErr(null)
    try {
      const args_required = toolForm.args_required_csv
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
      if (toolForm.id) {
        const patch: any = {
          name,
          description: toolForm.description,
          url,
          method: toolForm.method,
          args_required,
          request_body_template: toolForm.request_body_template || '{}',
          parameters_schema_json: toolForm.parameters_schema_json || '',
          response_mapper_json: toolForm.response_mapper_json || '{}',
          static_reply_template: toolForm.static_reply_template || '',
        }
        if (toolForm.headers_template_json.trim()) {
          patch.headers_template_json = toolForm.headers_template_json
        }
        await apiPut<IntegrationTool>(`/api/bots/${botId}/tools/${toolForm.id}`, patch)
      } else {
        await apiPost<IntegrationTool>(`/api/bots/${botId}/tools`, {
          name,
          description: toolForm.description,
          url,
          method: toolForm.method,
          args_required,
          headers_template_json: toolForm.headers_template_json || '{}',
          request_body_template: toolForm.request_body_template || '{}',
          parameters_schema_json: toolForm.parameters_schema_json || '',
          response_mapper_json: toolForm.response_mapper_json || '{}',
          static_reply_template: toolForm.static_reply_template || '',
        })
      }
      setShowToolModal(false)
      await reload()
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  async function deleteTool(t: IntegrationTool) {
    if (!botId) return
    const ok = window.confirm(`Delete integration "${t.name}"?\n\nThis is NOT reversible.`)
    if (!ok) return
    setErr(null)
    try {
      await apiDelete(`/api/bots/${botId}/tools/${t.id}`)
      await reload()
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

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
                  <label>TTS vendor</label>
                  <select value={bot.tts_vendor} onChange={(e) => void save({ tts_vendor: e.target.value })}>
                    {(options?.tts_vendors?.length ? options.tts_vendors : [bot.tts_vendor]).map((v) => (
                      <option value={v} key={v}>
                        {v}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {bot.tts_vendor === 'xtts_local' ? (
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
              ) : null}

              {bot.tts_vendor === 'openai_tts' ? (
                <>
                  <div className="formRowGrid2">
                    <div className="formRow">
                      <label>OpenAI TTS model</label>
                      <select value={bot.openai_tts_model} onChange={(e) => void save({ openai_tts_model: e.target.value })}>
                        {(options?.openai_tts_models?.length ? options.openai_tts_models : [bot.openai_tts_model]).map((m) => (
                          <option value={m} key={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="formRow">
                      <label>OpenAI voice</label>
                      <select value={bot.openai_tts_voice} onChange={(e) => void save({ openai_tts_voice: e.target.value })}>
                        {(options?.openai_tts_voices?.length ? options.openai_tts_voices : [bot.openai_tts_voice]).map((v) => (
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
                      value={bot.openai_tts_speed}
                      onChange={(e) => void save({ openai_tts_speed: Number(e.target.value) })}
                    />
                  </div>
                </>
              ) : null}

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

              {bot.tts_vendor === 'xtts_local' ? (
                <>
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
                </>
              ) : null}

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

              {systemTools.length ? (
                <>
                  <div className="cardSubTitle">System tools (default)</div>
                  <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
                    <div className="muted">Built-in tools available to every bot (not editable here).</div>
                  </div>
                  <table className="table" style={{ marginBottom: 16 }}>
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {systemTools.map((t) => (
                        <tr key={t.name}>
                          <td className="mono">{t.name}</td>
                          <td className="muted" style={{ maxWidth: 520 }}>
                            {t.description}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : null}

              <div className="cardSubTitle">Integrations (HTTP tools)</div>
              <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8 }}>
                <div className="muted">
                  Use variables like <span className="mono">{'{{.firstName}}'}</span> in prompts and tool next_reply.
                </div>
                <button className="btn primary" onClick={openNewTool}>
                  Add integration
                </button>
              </div>
              {tools.length === 0 ? (
                <div className="muted">No integrations yet.</div>
              ) : (
                <table className="table">
                  <thead>
                    <tr>
                      <th>Name</th>
                      <th>Method</th>
                      <th>URL</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {tools.map((t) => (
                      <tr key={t.id}>
                        <td className="mono">{t.name}</td>
                        <td className="mono">{t.method}</td>
                        <td className="mono" style={{ maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {t.url}
                        </td>
                        <td style={{ textAlign: 'right' }}>
                          <div className="row" style={{ justifyContent: 'flex-end' }}>
                            <button className="btn" onClick={() => openEditTool(t)}>
                              Edit
                            </button>
                            <button className="btn danger ghost" onClick={() => void deleteTool(t)}>
                              Delete
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </>
          )}
        </section>

        <MicTest botId={botId} />
      </div>

      {showToolModal ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow">
              <div className="cardTitle">{toolForm.id ? 'Edit integration' : 'New integration'}</div>
              <button className="btn" onClick={() => setShowToolModal(false)}>
                Close
              </button>
            </div>
            <div className="formRow">
              <label>
                Tool name (used by LLM){' '}
                <HelpTip>
                  <div className="tipTitle">Example</div>
                  <pre className="tipPre">{'get_user'}</pre>
                </HelpTip>
              </label>
              <input
                value={toolForm.name}
                onChange={(e) => setToolForm((p) => ({ ...p, name: e.target.value }))}
                placeholder=""
              />
            </div>
            <div className="formRow">
              <label>
                Description{' '}
                <HelpTip>
                  <div className="tipTitle">Example</div>
                  <pre className="tipPre">{'Fetch user profile (firstName, lastName, dob) by user_id.'}</pre>
                </HelpTip>
              </label>
              <input
                value={toolForm.description}
                onChange={(e) => setToolForm((p) => ({ ...p, description: e.target.value }))}
                placeholder=""
              />
            </div>
            <div className="formRowGrid2">
              <div className="formRow">
                <label>Method</label>
                <select value={toolForm.method} onChange={(e) => setToolForm((p) => ({ ...p, method: e.target.value }))}>
                  {(options?.http_methods || ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']).map((m) => (
                    <option value={m} key={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>
              <div className="formRow">
                <label>
                  URL{' '}
                  <HelpTip>
                    <div className="tipTitle">Examples</div>
                    <pre className="tipPre">
                      {'http://127.0.0.1:9001/v1/user?user_id={{args.user_id}}\n'}
                      {'http://127.0.0.1:9001/v1/doctors?city={{.city}}&specialty={{.specialty}}'}
                    </pre>
                  </HelpTip>
                </label>
                <input
                  value={toolForm.url}
                  onChange={(e) => setToolForm((p) => ({ ...p, url: e.target.value }))}
                  placeholder=""
                />
              </div>
            </div>
            <div className="formRow">
              <label>
                Required args (comma-separated){' '}
                <HelpTip>
                  <div className="tipTitle">Examples</div>
                  <pre className="tipPre">{'sql\nuser_id, city, specialty'}</pre>
                </HelpTip>
              </label>
              <input
                value={toolForm.args_required_csv}
                onChange={(e) => setToolForm((p) => ({ ...p, args_required_csv: e.target.value }))}
                placeholder=""
              />
              <div className="muted">
                The LLM must call this tool with <span className="mono">{'{ "args": { ... }, "next_reply": "..." }'}</span>. These keys will be required
                inside <span className="mono">args</span>.
              </div>
            </div>
            <div className="formRow">
              <label>
                Args schema (JSON Schema, optional){' '}
                <HelpTip>
                  <div className="tipTitle">Notes</div>
                  <div className="tipText">
                    If set, this schema is used to guide the LLM to produce structured <span className="mono">args</span> (nested objects/arrays, enums,
                    etc). If empty, <span className="mono">Required args</span> will be used as simple string fields.
                  </div>
                </HelpTip>
              </label>
              <textarea
                value={toolForm.parameters_schema_json}
                onChange={(e) => setToolForm((p) => ({ ...p, parameters_schema_json: e.target.value }))}
                rows={6}
                placeholder='{"type":"object","properties":{}}'
              />
            </div>
            <div className="formRow">
              <label>
                Headers template (JSON, optional){' '}
                <HelpTip>
                  <div className="tipTitle">Examples</div>
                  <pre className="tipPre">
                    {'{\n  \"Authorization\": \"Bearer <PASTE_TOKEN_HERE>\"\n}\n\n'}
                    {'{\n  \"X-Client\": \"intelligravex\"\n}'}
                  </pre>
                </HelpTip>
              </label>
              <textarea
                value={toolForm.headers_template_json}
                onChange={(e) => setToolForm((p) => ({ ...p, headers_template_json: e.target.value }))}
                rows={3}
                placeholder=""
              />
              {toolForm.headers_template_json_masked ? (
                <div className="muted">
                  Current (masked): <span className="mono">{toolForm.headers_template_json_masked}</span>
                </div>
              ) : toolForm.headers_configured ? (
                <div className="muted">Current: configured (hidden)</div>
              ) : null}
              <div className="muted">
                For secrets, paste them directly here. This field is write-only and will be masked after save.
              </div>
            </div>
            <div className="formRow">
              <label>
                Request body template (JSON){' '}
                <HelpTip>
                  <div className="tipTitle">Example (POST)</div>
                  <pre className="tipPre">{'{\n  \"user_id\": \"{{args.user_id}}\",\n  \"dob\": \"{{.dob}}\"\n}'}</pre>
                </HelpTip>
              </label>
              <textarea
                value={toolForm.request_body_template}
                onChange={(e) => setToolForm((p) => ({ ...p, request_body_template: e.target.value }))}
                rows={4}
                placeholder=""
              />
              <div className="muted">
                You can reference metadata with <span className="mono">{'{{.path}}'}</span> and tool args with{' '}
                <span className="mono">{'{{args.user_id}}'}</span>.
              </div>
            </div>
            <div className="formRow">
              <label>
                Response mapper (JSON: metadata_key → template){' '}
                <HelpTip>
                  <div className="tipTitle">Example</div>
                  <pre className="tipPre">
                    {'{\n  \"firstName\": \"{{response.data.firstName}}\",\n  \"lastName\": \"{{response.data.lastName}}\",\n  \"dob\": \"{{response.data.dob}}\"\n}'}
                  </pre>
                </HelpTip>
              </label>
              <textarea
                value={toolForm.response_mapper_json}
                onChange={(e) => setToolForm((p) => ({ ...p, response_mapper_json: e.target.value }))}
                rows={5}
                placeholder=""
              />
              <div className="muted">
                Templates can reference the HTTP response JSON via <span className="mono">{'{{response...}}'}</span>.
              </div>
            </div>
            <div className="formRow">
              <label>
                Static reply template (optional, Jinja2){' '}
                <HelpTip>
                  <div className="tipTitle">Example</div>
                  <pre className="tipPre">
                    {'{% if meta.firstName %}\n'}
                    {'Hello {{meta.firstName}}. What can I do for you today?\n'}
                    {'{% else %}\n'}
                    {'Hello! What is your name?\n'}
                    {'{% endif %}\n'}
                  </pre>
                </HelpTip>
              </label>
              <textarea
                value={toolForm.static_reply_template}
                onChange={(e) => setToolForm((p) => ({ ...p, static_reply_template: e.target.value }))}
                rows={6}
                placeholder=""
              />
              <div className="muted">
                Supports <span className="mono">{'{% if %}'}</span> / <span className="mono">{'{% for %}'}</span>. Shorthand{' '}
                <span className="mono">{'{{.key}}'}</span> works (rewritten to <span className="mono">{'meta.key'}</span>).
              </div>
            </div>
            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button className="btn primary" onClick={() => void saveTool()}>
                Save
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
