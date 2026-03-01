import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { apiDelete, apiGet, apiPost, apiPut } from '../api/client'
import SelectField from './SelectField'
import LoadingSpinner from './LoadingSpinner'
import BotGitIntegrationsPanel from './BotGitIntegrationsPanel'
import type { Bot, DataAgentSetupStatus, IntegrationTool, Options, SystemTool } from '../types'
import { formatLocalModelToolSupport } from '../utils/localModels'
import { formatProviderLabel, orderProviderList } from '../utils/llmProviders'
import { useChatgptOauth } from '../hooks/useChatgptOauth'
import { useEscapeClose } from '../hooks/useEscapeClose'
import InlineHelpTip from './InlineHelpTip'
import { TrashIcon, XMarkIcon } from '@heroicons/react/24/solid'

function HelpTip({ children }: { children: ReactNode }) {
  return (
    <span className="helpTipWrap" tabIndex={0} role="button" aria-label="Show example">
      <span className="helpTipIcon">?</span>
      <span className="helpTipBubble">{children}</span>
    </span>
  )
}

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

const DEFAULT_TOOL_FORM = {
  id: '',
  name: '',
  description: '',
  url: '',
  method: 'GET',
  enabled: true,
  use_codex_response: false,
  args_required_csv: '',
  parameters_schema_json: '',
  response_schema_json: '',
  codex_prompt: '',
  postprocess_python: '',
  return_result_directly: false,
  headers_template_json: '{}',
  headers_template_json_masked: '',
  headers_configured: false,
  request_body_template: '{}',
  pagination_json: '',
  response_mapper_json: '{}',
  static_reply_template: '',
}

export default function BotSettingsModal({
  botId,
  onClose,
  activeTab,
  onBotUpdate,
}: {
  botId: string
  onClose: () => void
  activeTab?: 'llm' | 'asr' | 'tts' | 'agent' | 'host' | 'integrations' | 'tools'
  onBotUpdate?: (bot: Bot) => void
}) {
  const [bot, setBot] = useState<Bot | null>(null)
  const [options, setOptions] = useState<Options | null>(null)
  const [tools, setTools] = useState<IntegrationTool[]>([])
  const [systemTools, setSystemTools] = useState<SystemTool[]>([])
  const [showToolModal, setShowToolModal] = useState(false)
  const [toolForm, setToolForm] = useState({ ...DEFAULT_TOOL_FORM })
  const [err, setErr] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [agentSetup, setAgentSetup] = useState<DataAgentSetupStatus | null>(null)
  const [agentSetupBusy, setAgentSetupBusy] = useState(false)
  const [agentSetupPolling, setAgentSetupPolling] = useState(false)
  const [agentEnablePending, setAgentEnablePending] = useState(false)
  const [applyAgentEnable, setApplyAgentEnable] = useState(false)
  const activeTabValue = activeTab || 'llm'
  const chatgptOauth = useChatgptOauth()
  const [pendingChatgptModel, setPendingChatgptModel] = useState<string | null>(null)
  const [uiProvider, setUiProvider] = useState<string | null>(null)
  const selectedLocalModel =
    (options?.local_models || []).find((m) => m.id === (bot?.openai_model || '')) || null

  useEscapeClose(() => setShowToolModal(false), showToolModal)

  async function reload() {
    if (!botId) return
    setErr(null)
    try {
      const [b, o] = await Promise.all([apiGet<Bot>(`/api/bots/${botId}`), apiGet<Options>('/api/options')])
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
    if (!bot) return
    setUiProvider(bot.llm_provider || 'openai')
  }, [bot?.id, bot?.llm_provider])

  const llmProvider = (uiProvider || bot?.llm_provider || 'openai')
  const showChatgptNudge = !chatgptOauth.ready && (llmProvider === 'chatgpt' || !!pendingChatgptModel)
  const voiceRequiresOpenAI = llmProvider === 'chatgpt'

  async function save(patch: Record<string, unknown>) {
    if (!botId) return
    setSaving(true)
    setErr(null)
    try {
      const updated = await apiPut<Bot>(`/api/bots/${botId}`, patch)
      setBot(updated)
      onBotUpdate?.(updated)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setSaving(false)
    }
  }

  async function handleToggleDataAgent(next: boolean) {
    if (!botId) return
    if (!next) {
      setAgentEnablePending(false)
      setAgentSetupPolling(false)
      setApplyAgentEnable(false)
      setAgentSetup(null)
      await save({ enable_data_agent: false })
      return
    }
    setErr(null)
    setAgentSetupBusy(true)
    setAgentEnablePending(true)
    try {
      const res = await apiPost<DataAgentSetupStatus>('/api/data-agent/setup/ensure', {})
      setAgentSetup(res)
      if (!res.docker_available) {
        setAgentEnablePending(false)
        setErr(res.error || res.message || 'Docker is not available. Install Docker Desktop: https://www.docker.com/products/docker-desktop/')
        return
      }
      if (res.image_ready || res.status === 'ready') {
        setAgentEnablePending(false)
        setApplyAgentEnable(true)
        return
      }
      if (res.status === 'building') {
        setAgentSetupPolling(true)
        return
      }
      setAgentEnablePending(false)
      setErr(res.error || res.message || 'Failed to prepare Isolated Workspace.')
    } catch (e: any) {
      setAgentEnablePending(false)
      setErr(String(e?.message || e))
    } finally {
      setAgentSetupBusy(false)
    }
  }

  useEffect(() => {
    if (!agentSetupPolling || !agentEnablePending) return
    let canceled = false
    const tick = async () => {
      try {
        const res = await apiGet<DataAgentSetupStatus>('/api/data-agent/setup/status')
        if (canceled) return
        setAgentSetup(res)
        if (!res.docker_available) {
          setAgentSetupPolling(false)
          setAgentEnablePending(false)
          setErr(res.error || res.message || 'Docker is not available. Install Docker Desktop: https://www.docker.com/products/docker-desktop/')
          return
        }
        if (res.image_ready || res.status === 'ready') {
          setAgentSetupPolling(false)
          setAgentEnablePending(false)
          setApplyAgentEnable(true)
          return
        }
        if (res.status === 'error') {
          setAgentSetupPolling(false)
          setAgentEnablePending(false)
          setErr(res.error || res.message || 'Failed to build Isolated Workspace image.')
        }
      } catch (e: any) {
        if (canceled) return
        setAgentSetupPolling(false)
        setAgentEnablePending(false)
        setErr(String(e?.message || e))
      }
    }
    void tick()
    const id = window.setInterval(() => {
      void tick()
    }, 1200)
    return () => {
      canceled = true
      window.clearInterval(id)
    }
  }, [agentSetupPolling, agentEnablePending])

  useEffect(() => {
    if (!applyAgentEnable) return
    setApplyAgentEnable(false)
    void save({ enable_data_agent: true })
  }, [applyAgentEnable])

  useEffect(() => {
    if (!pendingChatgptModel || !chatgptOauth.ready) return
    void save({ llm_provider: 'chatgpt', openai_model: pendingChatgptModel })
    setPendingChatgptModel(null)
  }, [pendingChatgptModel, chatgptOauth.ready])

  function llmModels(provider: string, fallback: string): string[] {
    if (provider === 'local') {
      const local = (options?.local_models || []).map((m) => m.id)
      const combined = [fallback, ...local].filter(Boolean)
      return Array.from(new Set(combined))
    }
    const base = provider === 'openrouter' ? options?.openrouter_models || [] : options?.openai_models || []
    if (!base.length) return [fallback]
    return base.includes(fallback) ? base : [fallback, ...base]
  }

  function openNewTool() {
    setToolForm({ ...DEFAULT_TOOL_FORM })
    setShowToolModal(true)
  }

  function openEditTool(t: IntegrationTool) {
    setToolForm({
      id: t.id,
      name: t.name || '',
      description: t.description || '',
      url: t.url || '',
      method: t.method || 'GET',
      enabled: Boolean(t.enabled ?? true),
      use_codex_response: Boolean(t.use_codex_response ?? false),
      args_required_csv: (t.args_required || []).join(', '),
      parameters_schema_json: t.parameters_schema_json || '',
      response_schema_json: t.response_schema_json || '',
      codex_prompt: t.codex_prompt || '',
      postprocess_python: t.postprocess_python || '',
      return_result_directly: Boolean(t.return_result_directly ?? false),
      headers_template_json: t.headers_template_json || '{}',
      headers_template_json_masked: t.headers_template_json_masked || '',
      headers_configured: Boolean(t.headers_configured),
      request_body_template: t.request_body_template || '{}',
      pagination_json: t.pagination_json || '',
      response_mapper_json: t.response_mapper_json || '{}',
      static_reply_template: t.static_reply_template || '',
    })
    setShowToolModal(true)
  }

  async function saveTool() {
    if (!botId) return
    const name = toolForm.name.trim()
    const url = toolForm.url.trim()
    if (!name || !url) return
    setSaving(true)
    setErr(null)
    try {
      const args_required = toolForm.args_required_csv
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
      if (toolForm.id) {
        const patch: any = {
          name: toolForm.name.trim(),
          description: toolForm.description,
          url: toolForm.url.trim(),
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
          name: toolForm.name.trim(),
          description: toolForm.description,
          url: toolForm.url.trim(),
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
      await reload()
      setShowToolModal(false)
    } catch (e: any) {
      setErr(String(e?.message || e))
    } finally {
      setSaving(false)
    }
  }

  async function deleteTool(t: IntegrationTool) {
    if (!botId) return
    const ok = window.confirm(`Delete tool ${t.name}?`)
    if (!ok) return
    setErr(null)
    try {
      await apiDelete(`/api/bots/${botId}/tools/${t.id}`)
      await reload()
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  async function toggleIntegrationTool(t: IntegrationTool, enabled: boolean) {
    if (!botId) return
    setErr(null)
    try {
      await apiPut<IntegrationTool>(`/api/bots/${botId}/tools/${t.id}`, { enabled })
      await reload()
    } catch (e: any) {
      setErr(String(e?.message || e))
    }
  }

  async function saveSystemTools() {
    if (!bot) return
    const disabled = systemTools.filter((t) => t.enabled === false).map((t) => t.name)
    await save({ disabled_tools: disabled })
  }

  function toggleSystemTool(name: string, enabled: boolean) {
    setSystemTools((prev) => prev.map((t) => (t.name === name ? { ...t, enabled } : t)))
  }

  const systemToolsDirty = useMemo(() => {
    const a = new Set<string>(bot?.disabled_tools || [])
    const b = new Set<string>(systemTools.filter((t) => t.enabled === false).map((t) => t.name))
    if (a.size !== b.size) return true
    for (const x of a) if (!b.has(x)) return true
    return false
  }, [bot?.disabled_tools, systemTools])

  if (!botId) return null

  return (
    <>
      <div className="cardTitleRow modalSticky">
        <div>
          <div className="cardTitle">Configuration</div>
          <div className="muted">Tune models, voice, and tools.</div>
        </div>
        <div className="row gap">
          {saving ? <div className="pill accent">saving…</div> : <div className="pill accent">saved</div>}
          <button className="iconBtn modalCloseBtn" onClick={onClose} aria-label="Close">
            <XMarkIcon />
          </button>
        </div>
      </div>

      {err ? <div className="alert">{err}</div> : null}

      <div className="settingsContent">
        {!bot ? (
          <div className="muted">
            <LoadingSpinner />
          </div>
        ) : (
          <>
            {activeTabValue === 'llm' ? (
              <>
                <div className="formRow">
                  <label>Name</label>
                  <input value={bot.name} onChange={(e) => setBot((p) => (p ? { ...p, name: e.target.value } : p))} />
                </div>
                <div className="formRow">
                  <label>Provider</label>
                  <SelectField
                    value={llmProvider}
                    onChange={(e) => {
                      const next = e.target.value
                      const models = llmModels(next, bot.openai_model)
                      const nextModel = models.includes(bot.openai_model) ? bot.openai_model : models[0] || bot.openai_model
                      setUiProvider(next)
                      if (next === 'chatgpt' && !chatgptOauth.ready) {
                        setPendingChatgptModel(nextModel)
                        return
                      }
                      setPendingChatgptModel(null)
                      void save({ llm_provider: next, openai_model: nextModel })
                    }}
                  >
                    {orderProviderList(options?.llm_providers || ['openai', 'openrouter', 'local']).map((p) => (
                      <option value={p} key={p}>
                        {formatProviderLabel(p)}
                      </option>
                    ))}
                  </SelectField>
                </div>
                {showChatgptNudge ? (
                  <div className="alert" style={{ marginTop: 4 }}>
                    <div style={{ marginBottom: 8 }}>Sign in with ChatGPT to use this provider.</div>
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
                <div className="formRow">
                  <label>LLM model</label>
                  {llmProvider === 'local' ? (
                    <>
                      <input
                        list="local-models-settings"
                        value={bot.openai_model}
                        onChange={(e) => void save({ openai_model: e.target.value })}
                      />
                      <datalist id="local-models-settings">
                        {(options?.local_models || []).map((m) => (
                          <option value={m.id} key={m.id}>
                            {m.name}
                          </option>
                        ))}
                      </datalist>
                      <div className="muted" style={{ marginTop: 6 }}>
                        {formatLocalModelToolSupport(selectedLocalModel)}
                      </div>
                    </>
                  ) : (
                    <SelectField
                      value={pendingChatgptModel || bot.openai_model}
                      onChange={(e) => {
                        const next = e.target.value
                        if (llmProvider === 'chatgpt' && !chatgptOauth.ready) {
                          setPendingChatgptModel(next)
                          return
                        }
                        setPendingChatgptModel(null)
                        void save({ openai_model: next })
                      }}
                    >
                      {llmModels(llmProvider, bot.openai_model).map((m) => (
                        <option value={m} key={m}>
                          {m}
                        </option>
                      ))}
                    </SelectField>
                  )}
                </div>
                <div className="formRow">
                  <label>Summary model</label>
                  <SelectField
                    value={bot.summary_model || 'gpt-5-nano'}
                    onChange={(e) => void save({ summary_model: e.target.value })}
                  >
                    {llmModels(llmProvider, bot.summary_model || 'gpt-5-nano').map((m) => (
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

            {activeTabValue === 'asr' ? (
              <>
                <div className="formRowGrid2">
                  <div className="formRow">
                    <label>
                      ASR language
                      {voiceRequiresOpenAI ? <InlineHelpTip text="Requires OpenAI API key." /> : null}
                    </label>
                    <SelectField
                      value={bot.language}
                      onChange={(e) => void save({ language: e.target.value })}
                      disabled={voiceRequiresOpenAI}
                    >
                      {(options?.languages || [bot.language]).map((l) => (
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
                      value={bot.openai_asr_model}
                      onChange={(e) => void save({ openai_asr_model: e.target.value })}
                      disabled={voiceRequiresOpenAI}
                    >
                      {(options?.openai_asr_models || [bot.openai_asr_model]).map((m) => (
                        <option value={m} key={m}>
                          {m}
                        </option>
                      ))}
                    </SelectField>
                  </div>
                </div>
                {voiceRequiresOpenAI ? <div className="muted">ASR disabled for ChatGPT OAuth. Add an OpenAI API key to enable it.</div> : null}
              </>
            ) : null}

            {activeTabValue === 'tts' ? (
              <>
                <div className="formRowGrid2">
                  <div className="formRow">
                    <label>
                      OpenAI TTS model
                      {voiceRequiresOpenAI ? <InlineHelpTip text="Requires OpenAI API key." /> : null}
                    </label>
                    <SelectField
                      value={bot.openai_tts_model}
                      onChange={(e) => void save({ openai_tts_model: e.target.value })}
                      disabled={voiceRequiresOpenAI}
                    >
                      {(options?.openai_tts_models?.length ? options.openai_tts_models : [bot.openai_tts_model]).map((m) => (
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
                      value={bot.openai_tts_voice}
                      onChange={(e) => void save({ openai_tts_voice: e.target.value })}
                      disabled={voiceRequiresOpenAI}
                    >
                      {(options?.openai_tts_voices?.length ? options.openai_tts_voices : [bot.openai_tts_voice]).map((v) => (
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
                    value={bot.openai_tts_speed}
                    onChange={(e) => void save({ openai_tts_speed: Number(e.target.value) })}
                    disabled={voiceRequiresOpenAI}
                  />
                </div>
                {voiceRequiresOpenAI ? <div className="muted">TTS disabled for ChatGPT OAuth. Add an OpenAI API key to enable it.</div> : null}
              </>
            ) : null}

            {activeTabValue === 'agent' ? (
                <>
                  <div className="muted" style={{ marginBottom: 8 }}>
                    Isolated Workspace requires Docker. Install and start Docker before enabling.
                  </div>
                  <div className="muted" style={{ marginBottom: 8 }}>
                    GitHub/GitLab connection has moved to Connected apps.
                  </div>
                <div className="formRow">
                  <label>Enable Isolated Workspace</label>
                  <label className="checkRow">
                    <input
                      type="checkbox"
                      checked={Boolean(bot.enable_data_agent)}
                      disabled={saving || agentSetupBusy || agentSetupPolling}
                      onChange={(e) => void handleToggleDataAgent(e.target.checked)}
                    />
                    <span className="muted">Enable additional data-agent behaviors (configured later).</span>
                  </label>
                </div>
                {agentSetup ? (
                  <div className="muted" style={{ marginTop: 4 }}>
                    {agentSetup.message || (agentSetupPolling ? 'Building Isolated Workspace image...' : '')}
                    {agentSetupPolling && agentSetup.image ? ` (${agentSetup.image})` : ''}
                    {!agentSetup.docker_available ? (
                      <>
                        {' '}
                        <a href="https://www.docker.com/products/docker-desktop/" target="_blank" rel="noreferrer">
                          Download Docker Desktop
                        </a>
                      </>
                    ) : null}
                  </div>
                ) : null}
                {agentSetup?.logs?.length ? (
                  <div className="muted" style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: 12 }}>
                    {agentSetup.logs[agentSetup.logs.length - 1]}
                  </div>
                ) : null}
                {bot.enable_data_agent ? (
                  <>
                    <div className="formRowGrid2">
                      <div className="formRow">
                        <label>Codex model (Isolated Workspace)</label>
                        <SelectField
                          value={bot.data_agent_model || 'gpt-5.2'}
                          onChange={(e) => void save({ data_agent_model: e.target.value })}
                        >
                          {([bot.data_agent_model || 'gpt-5.2', ...(options?.openai_models || [])] as string[])
                            .filter(Boolean)
                            .filter((v, i, a) => a.indexOf(v) === i)
                            .map((m) => (
                              <option value={m} key={m}>
                                {m}
                              </option>
                            ))}
                        </SelectField>
                      </div>
                      <div className="formRow">
                        <label>Reasoning effort</label>
                        <SelectField
                          value={bot.data_agent_reasoning_effort || 'high'}
                          onChange={(e) => void save({ data_agent_reasoning_effort: e.target.value })}
                        >
                          {['low', 'medium', 'high'].map((r) => (
                            <option value={r} key={r}>
                              {r}
                            </option>
                          ))}
                        </SelectField>
                      </div>
                    </div>
                    <div className="formRow">
                      <label>Prewarm Isolated Workspace on conversation start</label>
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
                        placeholder="INIT / PREWARM:\n- Open and read: api_spec.json, auth.json, conversation_context.json.\n- Do NOT call external APIs.\n- Output ok=true and result_text='READY'."
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
                      <label>Return Isolated Workspace result directly</label>
                      <label className="checkRow">
                        <input
                          type="checkbox"
                          checked={Boolean(bot.data_agent_return_result_directly)}
                          onChange={(e) => void save({ data_agent_return_result_directly: e.target.checked })}
                        />
                        <span className="muted">Skip LLM rewrite; show Isolated Workspace result_text as-is.</span>
                      </label>
                    </div>
                    <div className="formRow">
                      <label>Isolated Workspace API spec</label>
                      <textarea
                        value={bot.data_agent_api_spec_text || ''}
                        onChange={(e) => setBot((p) => (p ? { ...p, data_agent_api_spec_text: e.target.value } : p))}
                        rows={8}
                        placeholder="Paste API spec JSON here (saved as api_spec.json in the agent workspace). The Isolated Workspace will use only the APIs listed here."
                      />
                      <div className="row">
                        <button className="btn" onClick={() => void save({ data_agent_api_spec_text: bot.data_agent_api_spec_text || '' })}>
                          Save API spec
                        </button>
                      </div>
                    </div>
                    <div className="formRow">
                      <label>Isolated Workspace API authorizations (JSON)</label>
                      <textarea
                        value={bot.data_agent_auth_json || '{}'}
                        onChange={(e) => setBot((p) => (p ? { ...p, data_agent_auth_json: e.target.value } : p))}
                        rows={6}
                        placeholder='{"Authorization":"Bearer ..."}'
                      />
                      <div className="muted">
                        Used by Isolated Workspace API calls. Git provider auth is managed in Connected apps.
                      </div>
                      <div className="row">
                        <button className="btn" onClick={() => void save({ data_agent_auth_json: bot.data_agent_auth_json || '{}' })}>
                          Save auth JSON
                        </button>
                      </div>
                    </div>
                    <div className="formRow">
                      <label>Isolated Workspace system prompt</label>
                      <textarea
                        value={bot.data_agent_system_prompt || ''}
                        onChange={(e) => setBot((p) => (p ? { ...p, data_agent_system_prompt: e.target.value } : p))}
                        rows={6}
                        placeholder="Default: You are given a task (what_to_do), API spec, authorization tokens, and conversation context..."
                      />
                      <div className="row">
                        <button className="btn" onClick={() => void save({ data_agent_system_prompt: bot.data_agent_system_prompt || '' })}>
                          Save Isolated Workspace prompt
                        </button>
                      </div>
                    </div>
                  </>
                ) : null}
              </>
            ) : null}

            {activeTabValue === 'integrations' && bot ? (
              <BotGitIntegrationsPanel bot={bot} save={save} saving={saving} />
            ) : null}

            {activeTabValue === 'host' && bot ? (
              <details className="accordion" open>
                <summary>Host actions (one‑click)</summary>
                <div className="muted" style={{ marginTop: 6 }}>
                  Allow this employee to request actions on your Mac. You can require approvals or auto-run actions.
                </div>
                <div className="row gap" style={{ marginTop: 10, flexWrap: 'wrap' }}>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={Boolean(bot.enable_host_actions)}
                      onChange={(e) => setBot((p) => (p ? { ...p, enable_host_actions: e.target.checked } : p))}
                    />
                    Enable host actions
                  </label>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={Boolean(bot.require_host_action_approval)}
                      disabled={!bot.enable_host_actions}
                      onChange={(e) =>
                        setBot((p) => (p ? { ...p, require_host_action_approval: e.target.checked } : p))
                      }
                    />
                    Require approval
                  </label>
                  <label className="check">
                    <input
                      type="checkbox"
                      checked={Boolean(bot.enable_host_shell)}
                      disabled={!bot.enable_host_actions}
                      onChange={(e) => setBot((p) => (p ? { ...p, enable_host_shell: e.target.checked } : p))}
                    />
                    Allow shell commands
                  </label>
                </div>
                <div className="row" style={{ marginTop: 10 }}>
                  <button
                    className="btn"
                    onClick={() =>
                      void save({
                        enable_host_actions: Boolean(bot.enable_host_actions),
                        enable_host_shell: Boolean(bot.enable_host_shell),
                        require_host_action_approval: Boolean(bot.require_host_action_approval),
                      })
                    }
                  >
                    Save host settings
                  </button>
                </div>
              </details>
            ) : null}

            {activeTabValue === 'tools' && systemTools.length ? (
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

            {activeTabValue === 'tools' ? (
              <details className="accordion" open>
                <summary>Integrations (HTTP tools)</summary>
                <div className="row" style={{ justifyContent: 'space-between', marginBottom: 8, marginTop: 8 }}>
                  <div className="muted">
                    Use variables like <span className="mono">{'{{.firstName}}'}</span> in prompts and tool next_reply.{' '}
                    <HelpTip>
                      <div className="tipTitle">How integrations work</div>
                      <div className="tipText">
                        The LLM calls your tool with <span className="mono">{`{ \"args\": { ... } }`}</span>. The backend renders URL/body templates, calls the
                        HTTP API, maps selected fields into metadata, and returns a tool result.
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

      {showToolModal ? (
        <div className="modalOverlay" role="dialog" aria-modal="true">
          <div className="modalCard">
            <div className="cardTitleRow modalSticky">
              <div className="cardTitle">{toolForm.id ? 'Edit integration' : 'New integration'}</div>
              <button className="iconBtn modalCloseBtn" onClick={() => setShowToolModal(false)} aria-label="Close">
                <XMarkIcon />
              </button>
            </div>
            <div className="formRow">
              <label>
                Tool name (used by LLM){' '}
                <HelpTip>
                  <div className="tipTitle">Example</div>
                  <pre className="tipPre">get_user</pre>
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
                  <pre className="tipPre">Fetch user profile (firstName, lastName, dob) by user_id.</pre>
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
            </div>
            <div className="formRow">
              <label>URL template</label>
              <input
                value={toolForm.url}
                onChange={(e) => setToolForm((p) => ({ ...p, url: e.target.value }))}
                placeholder="https://api.example.com/search?q={{.query}}"
              />
            </div>
            <div className="formRow">
              <label>Required args (comma-separated)</label>
              <input
                value={toolForm.args_required_csv}
                onChange={(e) => setToolForm((p) => ({ ...p, args_required_csv: e.target.value }))}
              />
              <div className="muted">
                The LLM calls this tool with <span className="mono">{`{ "args": { ... }, "wait_reply": "..." }`}</span>.
              </div>
            </div>
            <div className="formRow">
              <label>Parameters schema (JSON)</label>
              <textarea
                value={toolForm.parameters_schema_json}
                onChange={(e) => setToolForm((p) => ({ ...p, parameters_schema_json: e.target.value }))}
                rows={6}
              />
            </div>
            {toolForm.use_codex_response ? (
              <div className="formRow">
                <label>Response schema (JSON)</label>
                <textarea
                  value={toolForm.response_schema_json}
                  onChange={(e) => setToolForm((p) => ({ ...p, response_schema_json: e.target.value }))}
                  rows={6}
                />
              </div>
            ) : null}
            <div className="formRow">
              <label className="row" style={{ justifyContent: 'flex-start', gap: 10 }}>
                <input
                  type="checkbox"
                  checked={Boolean(toolForm.use_codex_response)}
                  onChange={(e) => setToolForm((p) => ({ ...p, use_codex_response: e.target.checked }))}
                />
                <span className="muted">Use Codex to write the reply after this HTTP tool runs.</span>
              </label>
            </div>
            {toolForm.use_codex_response ? (
              <div className="formRow">
                <label>
                  Codex filter prompt (per tool){' '}
                  <HelpTip>
                    <div className="tipTitle">Default</div>
                    <pre className="tipPre">{DEFAULT_CODEX_FILTER_PROMPT}</pre>
                  </HelpTip>
                </label>
                <textarea
                  value={toolForm.codex_prompt}
                  onChange={(e) => setToolForm((p) => ({ ...p, codex_prompt: e.target.value }))}
                  rows={8}
                />
              </div>
            ) : null}
            {toolForm.use_codex_response ? (
              <div className="formRow">
                <label>
                  Python postprocess (optional){' '}
                  <HelpTip>
                    <div className="tipTitle">Example</div>
                    <pre className="tipPre">{DEFAULT_PY_POSTPROCESS_HELP}</pre>
                  </HelpTip>
                </label>
                <textarea
                  value={toolForm.postprocess_python}
                  onChange={(e) => setToolForm((p) => ({ ...p, postprocess_python: e.target.value }))}
                  rows={10}
                />
              </div>
            ) : null}
            {toolForm.use_codex_response ? (
              <div className="formRow">
                <label className="row" style={{ justifyContent: 'flex-start', gap: 10 }}>
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
              <label>Headers template (JSON)</label>
              <textarea
                value={toolForm.headers_template_json}
                onChange={(e) => setToolForm((p) => ({ ...p, headers_template_json: e.target.value }))}
                rows={6}
              />
              {toolForm.headers_template_json_masked ? (
                <div className="muted">
                  Current (masked): <span className="mono">{toolForm.headers_template_json_masked}</span>
                </div>
              ) : toolForm.headers_configured ? (
                <div className="muted">Current headers are configured.</div>
              ) : null}
            </div>
            <div className="formRow">
              <label>Request body template (JSON)</label>
              <textarea
                value={toolForm.request_body_template}
                onChange={(e) => setToolForm((p) => ({ ...p, request_body_template: e.target.value }))}
                rows={6}
              />
              <div className="muted">
                You can reference metadata with <span className="mono">{'{{.path}}'}</span> and tool args with{' '}
                <span className="mono">{'{{.args.query}}'}</span>.
              </div>
            </div>
            <div className="formRow">
              <label>Pagination (JSON)</label>
              <textarea
                value={toolForm.pagination_json}
                onChange={(e) => setToolForm((p) => ({ ...p, pagination_json: e.target.value }))}
                rows={6}
              />
            </div>
            <div className="formRow">
              <label>Response mapper (JSON)</label>
              <textarea
                value={toolForm.response_mapper_json}
                onChange={(e) => setToolForm((p) => ({ ...p, response_mapper_json: e.target.value }))}
                rows={6}
              />
            </div>
            <div className="formRow">
              <label>Static reply template</label>
              <textarea
                value={toolForm.static_reply_template}
                onChange={(e) => setToolForm((p) => ({ ...p, static_reply_template: e.target.value }))}
                rows={6}
              />
            </div>
            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button className="btn primary" onClick={() => void saveTool()}>
                Save
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  )
}
