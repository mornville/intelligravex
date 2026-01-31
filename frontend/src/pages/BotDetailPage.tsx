import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { apiDelete, apiGet, apiPost, apiPut } from '../api/client'
import MicTest from '../components/MicTest'
import SelectField from '../components/SelectField'
import LoadingSpinner from '../components/LoadingSpinner'
import type { Bot, ConversationSummary, IntegrationTool, Options, SystemTool } from '../types'
import { TrashIcon } from '@heroicons/react/24/solid'

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
  const [searchParams] = useSearchParams()
  const [initialConversationId, setInitialConversationId] = useState<string | undefined>(
    searchParams.get('conversation_id') || undefined,
  )
  const [bot, setBot] = useState<Bot | null>(null)
  const [options, setOptions] = useState<Options | null>(null)
  const [tools, setTools] = useState<IntegrationTool[]>([])
  const [systemTools, setSystemTools] = useState<SystemTool[]>([])
  const [showToolModal, setShowToolModal] = useState(false)
  const [gitSshKeyPath, setGitSshKeyPath] = useState('')
  const [preferredRepoUrl, setPreferredRepoUrl] = useState('')
  const [preferredRepoCachePath, setPreferredRepoCachePath] = useState('')
  const [preferredRepoSourcePath, setPreferredRepoSourcePath] = useState('')
  const [toolForm, setToolForm] = useState({
    id: '',
    name: '',
    description: '',
    url: '',
    method: 'GET',
    enabled: true,
    use_codex_response: false,
    args_required_csv: '',
    headers_template_json: '{}',
    headers_template_json_masked: '',
    headers_configured: false,
    request_body_template: '{}',
    parameters_schema_json: '',
    response_schema_json: '',
    codex_prompt: '',
    postprocess_python: '',
    return_result_directly: false,
    response_mapper_json: '{}',
    pagination_json: '',
    static_reply_template: '',
  })

  const DEFAULT_CODEX_FILTER_PROMPT = `You are a data extraction agent.

Goal: filter the HTTP response JSON to only what is needed to answer what_to_search_for.

Rules:
- Prefer minimal fields (don’t copy the whole payload).
- Be deterministic and robust to missing fields.
- If multiple matches exist, return a list of candidates and keep it short.
- If no matches exist, return an empty result with a clear reason.
`
  const DEFAULT_PY_POSTPROCESS_HELP = `# Optional: run locally (no LLM). Use this to deterministically summarize/filter the HTTP response.
#
# You have these variables available:
# - response_json (dict/list/str): the parsed HTTP response
# - meta (dict): current conversation metadata
# - args (dict): tool args used for the HTTP call
# - fields_required (str)
# - why_api_was_called (str)
#
# Produce output by calling:
#   emit("your result text", metadata_patch={...})
#
# Notes:
# - Keep it stdlib-only for portability.
# - This runs on the server's Python; avoid Python 3.10+ only syntax like "str | None".
#
# Example:
# top = (response_json or {}).get("items") or []
# emit(f"Found {len(top)} results.")
`
  const [err, setErr] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [activeTab, setActiveTab] = useState<'llm' | 'asr' | 'tts' | 'agent' | 'tools'>('llm')
  const [showSettings, setShowSettings] = useState(false)

  function parseAuthJson(raw?: string): Record<string, any> | null {
    try {
      const obj = JSON.parse((raw || '').trim() || '{}')
      if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return null
      return obj
    } catch {
      return null
    }
  }

  async function reload() {
    if (!botId) return
    setErr(null)
    try {
      const [b, o] = await Promise.all([apiGet<Bot>(`/api/bots/${botId}`), apiGet<Options>(`/api/options`)])
      setBot(b)
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
    const obj = parseAuthJson(bot?.data_agent_auth_json)
    if (!obj) return
    const keyPath = String(obj.ssh_private_key_path || obj.ssh_key_path || '')
    setGitSshKeyPath(keyPath)
    const repoUrl = String(obj.preferred_repo_url || obj.git_preferred_repo_url || obj.git_repo_url || obj.preferred_repo || '')
    const repoCache = String(
      obj.preferred_repo_cache_path || obj.git_repo_cache_path || obj.preferred_repo_path || '',
    )
    const repoSource = String(
      obj.preferred_repo_source_path || obj.git_repo_source_path || obj.preferred_repo_working_path || '',
    )
    setPreferredRepoUrl(repoUrl)
    setPreferredRepoCachePath(repoCache)
    setPreferredRepoSourcePath(repoSource)
  }, [bot?.data_agent_auth_json])

  useEffect(() => {
    if (!botId) return
    const fromQuery = searchParams.get('conversation_id')
    if (fromQuery) {
      setInitialConversationId(fromQuery)
      return
    }
    let canceled = false
    void (async () => {
      try {
        const c = await apiGet<{ items: ConversationSummary[]; total: number }>(
          `/api/conversations?bot_id=${botId}&page=1&page_size=1`,
        )
        const latest = c.items?.[0]?.id
        if (!canceled) setInitialConversationId(latest || undefined)
      } catch {
        if (!canceled) setInitialConversationId(undefined)
      }
    })()
    return () => {
      canceled = true
    }
  }, [botId, searchParams])

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

  async function saveGitSshKeyPath() {
    if (!bot) return
    const base = parseAuthJson(bot.data_agent_auth_json) || {}
    const trimmed = gitSshKeyPath.trim()
    if (trimmed) {
      base.ssh_private_key_path = trimmed
      delete base.ssh_key_path
      base.git_auth_method = 'ssh'
    } else {
      delete base.ssh_private_key_path
      delete base.ssh_key_path
      if (String(base.git_auth_method || '').toLowerCase() === 'ssh') {
        delete base.git_auth_method
      }
    }
    const nextJson = JSON.stringify(base, null, 2)
    setBot({ ...bot, data_agent_auth_json: nextJson })
    await save({ data_agent_auth_json: nextJson })
  }

  async function savePreferredRepoCache() {
    if (!bot) return
    const base = parseAuthJson(bot.data_agent_auth_json) || {}
    const url = preferredRepoUrl.trim()
    const cachePath = preferredRepoCachePath.trim()
    const sourcePath = preferredRepoSourcePath.trim()
    if (url || cachePath) {
      if (url) base.preferred_repo_url = url
      else delete base.preferred_repo_url
      if (cachePath) base.preferred_repo_cache_path = cachePath
      else delete base.preferred_repo_cache_path
      if (sourcePath) base.preferred_repo_source_path = sourcePath
      else delete base.preferred_repo_source_path
    } else {
      delete base.preferred_repo_url
      delete base.preferred_repo_cache_path
      delete base.preferred_repo_source_path
      delete base.git_preferred_repo_url
      delete base.git_repo_url
      delete base.git_repo_cache_path
      delete base.preferred_repo
      delete base.preferred_repo_path
      delete base.git_repo_source_path
      delete base.preferred_repo_working_path
    }
    const nextJson = JSON.stringify(base, null, 2)
    setBot({ ...bot, data_agent_auth_json: nextJson })
    await save({ data_agent_auth_json: nextJson })
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
      enabled: true,
      use_codex_response: false,
      args_required_csv: '',
      headers_template_json: '{}',
      headers_template_json_masked: '',
      headers_configured: false,
      request_body_template: '{}',
      parameters_schema_json: '',
      response_schema_json: '',
      codex_prompt: '',
      postprocess_python: '',
      return_result_directly: false,
      response_mapper_json: '{}',
      pagination_json: '',
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
      enabled: Boolean(t.enabled ?? true),
      use_codex_response: Boolean(t.use_codex_response),
      args_required_csv: (t.args_required || []).join(', '),
      // Write-only: never hydrate secrets back into the UI.
      headers_template_json: '',
      headers_template_json_masked: t.headers_template_json_masked || '',
      headers_configured: Boolean(t.headers_configured),
      request_body_template: t.request_body_template || '{}',
      parameters_schema_json: t.parameters_schema_json || '',
      response_schema_json: t.response_schema_json || '',
      codex_prompt: t.codex_prompt || '',
      postprocess_python: t.postprocess_python || '',
      return_result_directly: Boolean(t.return_result_directly),
      response_mapper_json: t.response_mapper_json || '{}',
      pagination_json: t.pagination_json || '',
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
          enabled: Boolean(toolForm.enabled),
          use_codex_response: Boolean(toolForm.use_codex_response),
          args_required,
          request_body_template: toolForm.request_body_template || '{}',
          parameters_schema_json: toolForm.parameters_schema_json || '',
          response_schema_json: toolForm.response_schema_json || '',
          codex_prompt: toolForm.codex_prompt || '',
          postprocess_python: toolForm.postprocess_python || '',
          return_result_directly: Boolean(toolForm.return_result_directly),
          response_mapper_json: toolForm.response_mapper_json || '{}',
          pagination_json: toolForm.pagination_json || '',
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
          enabled: Boolean(toolForm.enabled),
          use_codex_response: Boolean(toolForm.use_codex_response),
          args_required,
          headers_template_json: toolForm.headers_template_json || '{}',
          request_body_template: toolForm.request_body_template || '{}',
          parameters_schema_json: toolForm.parameters_schema_json || '',
          response_schema_json: toolForm.response_schema_json || '',
          codex_prompt: toolForm.codex_prompt || '',
          postprocess_python: toolForm.postprocess_python || '',
          return_result_directly: Boolean(toolForm.return_result_directly),
          response_mapper_json: toolForm.response_mapper_json || '{}',
          pagination_json: toolForm.pagination_json || '',
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

  function toggleSystemTool(name: string, enabled: boolean) {
    setSystemTools((prev) => prev.map((t) => (t.name === name ? { ...t, enabled } : t)))
  }

  async function toggleIntegrationTool(t: IntegrationTool, enabled: boolean) {
    if (!botId) return
    setErr(null)
    try {
      await apiPut<IntegrationTool>(`/api/bots/${botId}/tools/${t.id}`, { enabled })
      setTools((prev) => prev.map((x) => (x.id === t.id ? { ...x, enabled } : x)))
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  async function saveSystemTools() {
    if (!bot) return
    const disabled = systemTools.filter((t) => t.enabled === false).map((t) => t.name)
    await save({ disabled_tools: disabled })
  }

  const systemToolsDirty = useMemo(() => {
    const a = new Set<string>(bot?.disabled_tools || [])
    const b = new Set<string>(systemTools.filter((t) => t.enabled === false).map((t) => t.name))
    if (a.size !== b.size) return true
    for (const x of a) if (!b.has(x)) return true
    return false
  }, [bot?.disabled_tools, systemTools])

  return (
    <div className="page">
      <div className="pageHeader">
        <div>
          <h1>{bot?.name || 'Bot'}</h1>
        </div>
        <div className="row gap">
          <button className="btn" onClick={() => setShowSettings(true)}>
            Settings
          </button>
          <button
            className="btn iconBtn danger"
            onClick={() => void onDelete()}
            disabled={!bot}
            aria-label="Delete bot"
            title="Delete bot"
          >
            <TrashIcon aria-hidden="true" />
          </button>
          <button className="btn" onClick={() => nav('/bots')}>
            Back
          </button>
        </div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <MicTest botId={botId} initialConversationId={initialConversationId} />

      {showSettings ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard settingsModal">
            <section className="card settingsCard">
          <div className="cardTitleRow">
            <div>
              <div className="cardTitle">Configuration</div>
              <div className="muted">Tune models, voice, and tools.</div>
            </div>
            <div className="row gap">
              {saving ? <div className="pill accent">saving…</div> : <div className="pill accent">saved</div>}
              <button className="btn" onClick={() => setShowSettings(false)}>
                Close
              </button>
            </div>
          </div>
          <div className="row gap settingsTabs" style={{ flexWrap: 'wrap' }}>
            <button className={activeTab === 'llm' ? 'btn primary' : 'btn'} onClick={() => setActiveTab('llm')}>
              LLM
            </button>
            <button className={activeTab === 'asr' ? 'btn primary' : 'btn'} onClick={() => setActiveTab('asr')}>
              ASR
            </button>
            <button className={activeTab === 'tts' ? 'btn primary' : 'btn'} onClick={() => setActiveTab('tts')}>
              TTS
            </button>
            <button className={activeTab === 'agent' ? 'btn primary' : 'btn'} onClick={() => setActiveTab('agent')}>
              Data Agent
            </button>
            <button className={activeTab === 'tools' ? 'btn primary' : 'btn'} onClick={() => setActiveTab('tools')}>
              Tools
            </button>
          </div>

          <div className="settingsContent">
            {!bot ? (
              <div className="muted">
                <LoadingSpinner />
              </div>
            ) : (
              <>
              {activeTab === 'llm' ? (
                <>
                  <div className="formRow">
                    <label>Name</label>
                    <input value={bot.name} onChange={(e) => setBot((p) => (p ? { ...p, name: e.target.value } : p))} />
                  </div>
                  <div className="formRow">
                    <label>OpenAI model</label>
                    <SelectField value={bot.openai_model} onChange={(e) => void save({ openai_model: e.target.value })}>
                      {(options?.openai_models || [bot.openai_model]).map((m) => (
                        <option value={m} key={m}>
                          {m}
                        </option>
                      ))}
                    </SelectField>
                  </div>
                  <details className="accordion" style={{ marginTop: 10 }}>
                    <summary>Advanced models</summary>
                  <div className="formRow">
                    <label>Web search model</label>
                    <SelectField
                      value={bot.web_search_model || bot.openai_model}
                      onChange={(e) => void save({ web_search_model: e.target.value })}
                    >
                      {(options?.openai_models || [bot.web_search_model || bot.openai_model]).map((m) => (
                        <option value={m} key={m}>
                          {m}
                        </option>
                      ))}
                    </SelectField>
                    <div className="muted">Used for web_search filtering + summarization.</div>
                  </div>
                  <div className="formRow">
                    <label>Codex model</label>
                    <SelectField
                      value={bot.codex_model || 'gpt-5.1-codex-mini'}
                      onChange={(e) => void save({ codex_model: e.target.value })}
                    >
                      {(options?.openai_models || [bot.codex_model || 'gpt-5.1-codex-mini']).map((m) => (
                        <option value={m} key={m}>
                          {m}
                        </option>
                      ))}
                    </SelectField>
                    <div className="muted">Used for “use Codex for response” HTTP integrations.</div>
                  </div>
                  <div className="formRow">
                    <label>Summary model</label>
                    <SelectField
                      value={bot.summary_model || 'gpt-5-nano'}
                      onChange={(e) => void save({ summary_model: e.target.value })}
                    >
                      {(options?.openai_models || [bot.summary_model || 'gpt-5-nano']).map((m) => (
                        <option value={m} key={m}>
                          {m}
                        </option>
                      ))}
                    </SelectField>
                    <div className="muted">Used to summarize older conversation history when context grows.</div>
                  </div>
                  <div className="formRow">
                    <label>History window (turns)</label>
                    <input
                      type="number"
                      min="1"
                      max="64"
                      value={bot.history_window_turns ?? 16}
                      onChange={(e) => void save({ history_window_turns: Number(e.target.value) })}
                    />
                    <div className="muted">Keep the last N user turns verbatim; older turns are summarized.</div>
                  </div>
                  </details>
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

                  <details className="accordion" style={{ marginTop: 10 }}>
                    <summary>Conversation start message</summary>
                    <div className="formRowGrid2">
                      <div className="formRow">
                        <label>Mode</label>
                        <SelectField value={bot.start_message_mode} onChange={(e) => void save({ start_message_mode: e.target.value })}>
                          <option value="static">Static</option>
                          <option value="llm">LLM-generated</option>
                        </SelectField>
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
                  </details>
                </>
              ) : null}

              {activeTab === 'asr' ? (
                <>
                  <div className="formRowGrid2">
                    <div className="formRow">
                      <label>ASR language</label>
                      <SelectField value={bot.language} onChange={(e) => void save({ language: e.target.value })}>
                        {(options?.languages || [bot.language]).map((l) => (
                          <option value={l} key={l}>
                            {l}
                          </option>
                        ))}
                      </SelectField>
                    </div>
                    <div className="formRow">
                      <label>ASR model</label>
                      <SelectField value={bot.openai_asr_model} onChange={(e) => void save({ openai_asr_model: e.target.value })}>
                        {(options?.openai_asr_models || [bot.openai_asr_model]).map((m) => (
                          <option value={m} key={m}>
                            {m}
                          </option>
                        ))}
                      </SelectField>
                    </div>
                  </div>
                </>
              ) : null}

              {activeTab === 'tts' ? (
                <>
                  <div className="formRowGrid2">
                    <div className="formRow">
                      <label>OpenAI TTS model</label>
                      <SelectField value={bot.openai_tts_model} onChange={(e) => void save({ openai_tts_model: e.target.value })}>
                        {(options?.openai_tts_models?.length ? options.openai_tts_models : [bot.openai_tts_model]).map((m) => (
                          <option value={m} key={m}>
                            {m}
                          </option>
                        ))}
                      </SelectField>
                    </div>
                    <div className="formRow">
                      <label>OpenAI voice</label>
                      <SelectField value={bot.openai_tts_voice} onChange={(e) => void save({ openai_tts_voice: e.target.value })}>
                        {(options?.openai_tts_voices?.length ? options.openai_tts_voices : [bot.openai_tts_voice]).map((v) => (
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
                      value={bot.openai_tts_speed}
                      onChange={(e) => void save({ openai_tts_speed: Number(e.target.value) })}
                    />
                  </div>
                </>
              ) : null}


              {activeTab === 'agent' ? (
                <>
                  <div className="muted" style={{ marginBottom: 8 }}>
                    Data Agent requires Docker. Install and start Docker before enabling.
                  </div>
                  <div className="formRow">
                    <label>Enable Data Agent</label>
                    <label className="checkRow">
                      <input
                        type="checkbox"
                        checked={Boolean(bot.enable_data_agent)}
                        onChange={(e) => void save({ enable_data_agent: e.target.checked })}
                      />
                      <span className="muted">Enable additional data-agent behaviors (configured later).</span>
                    </label>
                  </div>
                  {bot.enable_data_agent ? (
                    <>
                      <div className="formRow">
                        <label>Prewarm Data Agent on conversation start</label>
                        <label className="checkRow">
                          <input
                            type="checkbox"
                            checked={Boolean(bot.data_agent_prewarm_on_start)}
                            onChange={(e) => void save({ data_agent_prewarm_on_start: e.target.checked })}
                          />
                          <span className="muted">Starts the container + initializes a Codex session in the background.</span>
                        </label>
                      </div>
                      <div className="formRow">
                        <label>Prewarm prompt</label>
                        <textarea
                          value={bot.data_agent_prewarm_prompt || ''}
                          onChange={(e) => setBot((p) => (p ? { ...p, data_agent_prewarm_prompt: e.target.value } : p))}
                          rows={6}
                          placeholder="INIT / PREWARM:&#10;- Open and read: api_spec.json, auth.json, conversation_context.json.&#10;- Do NOT call external APIs.&#10;- Output ok=true and result_text='READY'."
                        />
                        <div className="row">
                          <button
                            className="btn"
                            onClick={() => void save({ data_agent_prewarm_prompt: bot.data_agent_prewarm_prompt || '' })}
                          >
                            Save prewarm prompt
                          </button>
                        </div>
                      </div>
                      <div className="formRow">
                        <label>Return Data Agent result directly</label>
                        <label className="checkRow">
                          <input
                            type="checkbox"
                            checked={Boolean(bot.data_agent_return_result_directly)}
                            onChange={(e) => void save({ data_agent_return_result_directly: e.target.checked })}
                          />
                          <span className="muted">Skip LLM rewrite; show Data Agent result_text as-is.</span>
                        </label>
                      </div>
                      <div className="formRow">
                        <label>Data Agent API spec</label>
                        <textarea
                          value={bot.data_agent_api_spec_text || ''}
                          onChange={(e) => setBot((p) => (p ? { ...p, data_agent_api_spec_text: e.target.value } : p))}
                          rows={8}
                          placeholder="Paste API spec JSON here (saved as api_spec.json in the agent workspace). The Data Agent will use only the APIs listed here."
                        />
                        <div className="row">
                          <button className="btn" onClick={() => void save({ data_agent_api_spec_text: bot.data_agent_api_spec_text || '' })}>
                            Save API spec
                          </button>
                        </div>
                      </div>
                      <div className="formRow">
                        <label>Data Agent API authorizations (JSON)</label>
                        <textarea
                          value={bot.data_agent_auth_json || '{}'}
                          onChange={(e) => setBot((p) => (p ? { ...p, data_agent_auth_json: e.target.value } : p))}
                          rows={6}
                          placeholder='{"Authorization":"Bearer ..."}'
                        />
                        <div className="muted">Stored as plaintext (not masked). Put only what you’re comfortable storing.</div>
                        <div className="row">
                          <button className="btn" onClick={() => void save({ data_agent_auth_json: bot.data_agent_auth_json || '{}' })}>
                            Save auth JSON
                          </button>
                        </div>
                      </div>
                      <div className="formRow">
                        <label>Git SSH key path (per bot)</label>
                        <input
                          value={gitSshKeyPath}
                          onChange={(e) => setGitSshKeyPath(e.target.value)}
                          placeholder="/Users/you/.ssh/id_ed25519"
                        />
                        <div className="muted">
                          Stored as a path in the bot's auth JSON. The key file must exist on the server host. Clear to
                          fall back to the saved GitHub token.
                        </div>
                        <div className="row">
                          <button className="btn" onClick={() => void saveGitSshKeyPath()}>
                            Save SSH key path
                          </button>
                        </div>
                      </div>
                      <div className="formRow">
                        <label>Preferred repo (URL)</label>
                        <input
                          value={preferredRepoUrl}
                          onChange={(e) => setPreferredRepoUrl(e.target.value)}
                          placeholder="git@github.com:org/repo.git"
                        />
                        <div className="muted">Used to prime a local mirror for faster clones in each conversation.</div>
                      </div>
                      <div className="formRow">
                        <label>Preferred repo cache path (host)</label>
                        <input
                          value={preferredRepoCachePath}
                          onChange={(e) => setPreferredRepoCachePath(e.target.value)}
                          placeholder="/Users/you/.igx_repo_cache/candorverse.git"
                        />
                        <div className="muted">
                          This is a shared bare mirror on the host. Each conversation will clone from it instead of the
                          network.
                        </div>
                      </div>
                      <div className="formRow">
                        <label>Preferred repo source path (host)</label>
                        <input
                          value={preferredRepoSourcePath}
                          onChange={(e) => setPreferredRepoSourcePath(e.target.value)}
                          placeholder="/Users/you/Desktop/candor/candorverse"
                        />
                        <div className="muted">
                          Optional: a local working repo to use as a reference source for fast clones per conversation.
                        </div>
                        <div className="row">
                          <button className="btn" onClick={() => void savePreferredRepoCache()}>
                            Save repo preferences
                          </button>
                        </div>
                      </div>
                      <div className="formRow">
                        <label>Data Agent system prompt</label>
                        <textarea
                          value={bot.data_agent_system_prompt || ''}
                          onChange={(e) => setBot((p) => (p ? { ...p, data_agent_system_prompt: e.target.value } : p))}
                          rows={6}
                          placeholder="Default: You are given a task (what_to_do), API spec, authorization tokens, and conversation context..."
                        />
                        <div className="row">
                          <button className="btn" onClick={() => void save({ data_agent_system_prompt: bot.data_agent_system_prompt || '' })}>
                            Save Data Agent prompt
                          </button>
                        </div>
                      </div>
                    </>
                  ) : null}
                </>
              ) : null}

              {activeTab === 'tools' && systemTools.length ? (
                <details className="accordion" style={{ marginBottom: 12 }} open>
                  <summary>System tools</summary>
                  <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8, marginTop: 8 }}>
                    <div className="muted">Toggle built-in tools per bot. Click “Update tools” to save.</div>
                    <button className="btn" onClick={() => void saveSystemTools()} disabled={!systemToolsDirty || saving}>
                      Update tools
                    </button>
                  </div>
                  <table className="table" style={{ marginBottom: 16 }}>
                    <thead>
                      <tr>
                        <th>Enabled</th>
                        <th>Name</th>
                        <th>Description</th>
                      </tr>
                    </thead>
                    <tbody>
                      {systemTools.map((t) => (
                        <tr key={t.name}>
                          <td>
                            <input
                              type="checkbox"
                              checked={Boolean(t.enabled ?? true)}
                              disabled={t.can_disable === false}
                              onChange={(e) => toggleSystemTool(t.name, e.target.checked)}
                            />
                          </td>
                          <td className="mono">{t.name}</td>
                          <td className="muted" style={{ maxWidth: 520 }}>
                            {t.description}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </details>
              ) : null}

              {activeTab === 'tools' ? (
                <details className="accordion" open>
                  <summary>Integrations (HTTP tools)</summary>
                  <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8, marginTop: 8 }}>
                    <div className="muted">
                      Use variables like <span className="mono">{'{{.firstName}}'}</span> in prompts and tool next_reply.{' '}
                      <HelpTip>
                        <div className="tipTitle">How integrations work</div>
                        <div className="tipText">
                          The LLM calls your tool with <span className="mono">{'{ "args": { ... } }'}</span>. The backend renders URL/body templates, calls the
                          HTTP API, maps selected fields into metadata, and returns a tool result.
                        </div>
                        <div className="tipText">
                          Pagination: if configured, the backend will fetch multiple pages and merge results (the LLM can pass{' '}
                          <span className="mono">max_items</span> to stop early).
                        </div>
                      </HelpTip>
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
                          <th>Enabled</th>
                          <th>Method</th>
                          <th>URL</th>
                          <th />
                        </tr>
                      </thead>
                      <tbody>
                        {tools.map((t) => (
                          <tr key={t.id}>
                            <td className="mono">{t.name}</td>
                            <td>
                              <input
                                type="checkbox"
                                checked={Boolean(t.enabled ?? true)}
                                onChange={(e) => void toggleIntegrationTool(t, e.target.checked)}
                              />
                            </td>
                            <td className="mono">{t.method}</td>
                            <td className="mono" style={{ maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {t.url}
                            </td>
                            <td style={{ textAlign: 'right' }}>
                              <div className="row" style={{ justifyContent: 'flex-end' }}>
                                <button className="btn" onClick={() => openEditTool(t)}>
                                  Edit
                                </button>
                                <button
                                  className="btn iconBtn danger"
                                  onClick={() => void deleteTool(t)}
                                  aria-label="Delete integration"
                                  title="Delete integration"
                                >
                                  <TrashIcon aria-hidden="true" />
                                </button>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </details>
              ) : null}
              </>
            )}
          </div>
        </section>
          </div>
        </div>
      ) : null}

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
                <SelectField value={toolForm.method} onChange={(e) => setToolForm((p) => ({ ...p, method: e.target.value }))}>
                  {(options?.http_methods || ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']).map((m) => (
                    <option value={m} key={m}>
                      {m}
                    </option>
                  ))}
                </SelectField>
              </div>
              <div className="formRow">
                <label>Enabled</label>
                <label className="row" style={{ justifyContent: 'flex-start', gap: 8 }}>
                  <input
                    type="checkbox"
                    checked={Boolean(toolForm.enabled)}
                    onChange={(e) => setToolForm((p) => ({ ...p, enabled: e.target.checked }))}
                  />
                  <span className="muted">When disabled, the LLM cannot call this integration.</span>
                </label>
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
                The LLM calls this tool with <span className="mono">{'{ "args": { ... }, "wait_reply": "..." }'}</span>. If “Use Codex for response” is
                enabled, the backend runs Codex and the main chat model rephrases the tool result (don’t rely on <span className="mono">next_reply</span>).
                These keys will be required inside <span className="mono">args</span>.
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
            {toolForm.use_codex_response ? (
              <div className="formRow">
                <label>
                  Response schema (JSON Schema, optional){' '}
                  <HelpTip>
                    <div className="tipTitle">What this does</div>
                    <div className="tipText">
                      If set, the backend uses this as the HTTP response JSON schema for the Codex agent (instead of deriving a schema from the live
                      response). The raw response payload is still saved to a temp file locally and is not sent to Codex. If this schema doesn’t match the
                      response, the backend warns and falls back to derived schema.
                    </div>
                  </HelpTip>
                </label>
                <textarea
                  value={toolForm.response_schema_json}
                  onChange={(e) => setToolForm((p) => ({ ...p, response_schema_json: e.target.value }))}
                  rows={6}
                  placeholder='{"type":"object","properties":{}}'
                />
              </div>
            ) : null}
            <div className="formRow">
              <label>
                Use Codex for response{' '}
                <HelpTip>
                  <div className="tipTitle">What this does</div>
	                  <div className="tipText">
	                    If enabled, the backend runs a Codex agent to post-process the HTTP response and stores its output as tool result for the main chat
	                    model to rephrase. <span className="mono">static_reply_template</span> (if set and non-empty) still takes priority.
	                  </div>
	                </HelpTip>
              </label>
              <label className="row" style={{ justifyContent: 'flex-start', gap: 8 }}>
                <input
                  type="checkbox"
                  checked={Boolean(toolForm.use_codex_response)}
                  onChange={(e) =>
                    setToolForm((p) => ({
                      ...p,
                      use_codex_response: e.target.checked,
                      codex_prompt:
                        e.target.checked && !String(p.codex_prompt || '').trim()
                          ? DEFAULT_CODEX_FILTER_PROMPT
                          : p.codex_prompt,
                    }))
                  }
                />
                <span className="muted">Use Codex to write the reply after this HTTP tool runs.</span>
              </label>
            </div>
            {toolForm.use_codex_response ? (
              <div className="formRow">
                <label>
                  Codex filter prompt (per tool){' '}
                  <HelpTip>
                    <div className="tipTitle">What this does</div>
                    <div className="tipText">
                      Extra instructions added to the Codex agent when filtering/summarizing this tool’s HTTP response. If empty, a default prompt is used.
                    </div>
                  </HelpTip>
                </label>
                <textarea
                  value={toolForm.codex_prompt}
                  onChange={(e) => setToolForm((p) => ({ ...p, codex_prompt: e.target.value }))}
                  rows={8}
                  placeholder={DEFAULT_CODEX_FILTER_PROMPT}
                />
              </div>
            ) : null}
            {toolForm.use_codex_response ? (
              <div className="formRow">
                <label>
                  Python post-processor (optional){' '}
                  <HelpTip>
                    <div className="tipTitle">What this does</div>
                    <div className="tipText">
                      If set, the backend runs this Python code locally (60s timeout) to summarize/filter the HTTP response and uses its output as the tool
                      result. This skips the Codex one-shot call.
                    </div>
                    <div className="tipText">
                      Use <span className="mono">emit("text", metadata_patch=&#123;...&#125;)</span> to return a result.
                    </div>
                  </HelpTip>
                </label>
                <textarea
                  value={toolForm.postprocess_python}
                  onChange={(e) => setToolForm((p) => ({ ...p, postprocess_python: e.target.value }))}
                  rows={10}
                  placeholder={DEFAULT_PY_POSTPROCESS_HELP}
                />
                <div className="muted">Tip: keep it stdlib-only (json/datetime/re) for portability.</div>
              </div>
            ) : null}
            {toolForm.use_codex_response ? (
              <div className="formRow">
                <label>
                  Return tool result directly{' '}
                  <HelpTip>
                    <div className="tipTitle">What this does</div>
                    <div className="tipText">
                      If enabled, the backend will return the post-processed result text (Python/Codex) directly as the assistant reply and will not run the
                      chat LLM again to rephrase it.
                    </div>
                  </HelpTip>
                </label>
                <label className="row" style={{ justifyContent: 'flex-start', gap: 8 }}>
                  <input
                    type="checkbox"
                    checked={Boolean(toolForm.return_result_directly)}
                    onChange={(e) => setToolForm((p) => ({ ...p, return_result_directly: e.target.checked }))}
                  />
                  <span className="muted">Skip rephrasing and show the tool output as-is.</span>
                </label>
              </div>
            ) : null}
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
                  Pagination (JSON, optional){' '}
                  <HelpTip>
                    <div className="tipTitle">Example (page + limit)</div>
                  <pre className="tipPre">
                    {'{\n  \"mode\": \"page_limit\",\n  \"items_path\": \"items\",\n  \"page_arg\": \"page\",\n  \"limit_arg\": \"limit\",\n  \"max_pages\": 5,\n  \"max_items_cap\": 5000\n}\n'}
                  </pre>
                  <div className="tipText">
                    If set, the backend will fetch multiple pages (up to <span className="mono">max_pages</span>) and merge the list at{' '}
                    <span className="mono">items_path</span>. The LLM can optionally pass <span className="mono">max_items</span> inside args to stop early.
                  </div>
                    <div className="tipText">This field must be valid JSON (no templates like <span className="mono">{'{{...}}'}</span>).</div>
                    <div className="tipText">
                      Important: your URL/body template must reference <span className="mono">args.page</span> and <span className="mono">args.limit</span>{' '}
                      (where your API expects them), otherwise every request will fetch the first page.
                    </div>
                  </HelpTip>
                </label>
              <textarea
                value={toolForm.pagination_json}
                onChange={(e) => setToolForm((p) => ({ ...p, pagination_json: e.target.value }))}
                rows={4}
                placeholder=""
              />
              <div className="muted">
                Set <span className="mono">items_path</span> to the JSON path that contains the returned list (e.g. <span className="mono">providers.rows</span>).
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
