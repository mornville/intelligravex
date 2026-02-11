export type UUID = string

export type Bot = {
  id: UUID
  name: string
  llm_provider?: string
  openai_model: string
  openai_asr_model: string
  web_search_model: string
  codex_model: string
  summary_model?: string
  history_window_turns?: number
  enable_data_agent?: boolean
  data_agent_api_spec_text?: string
  data_agent_auth_json?: string
  data_agent_system_prompt?: string
  data_agent_return_result_directly?: boolean
  data_agent_prewarm_on_start?: boolean
  data_agent_prewarm_prompt?: string
  enable_host_actions?: boolean
  enable_host_shell?: boolean
  require_host_action_approval?: boolean
  system_prompt: string
  language: string
  openai_tts_model: string
  openai_tts_voice: string
  openai_tts_speed: number
  start_message_mode: 'llm' | 'static'
  start_message_text: string
  disabled_tools?: string[]
  stats?: {
    conversations: number
    input_tokens: number
    output_tokens: number
    cost_usd: number
    avg_llm_ttfb_ms?: number | null
    avg_llm_total_ms?: number | null
    avg_total_ms?: number | null
  }
  created_at: string
  updated_at: string
}

export type ClientKey = {
  id: UUID
  name: string
  hint: string
  allowed_origins: string
  allowed_bot_ids: string[]
  created_at: string
}

export type GitTokenInfo = {
  provider: string
  configured: boolean
  hint?: string
  created_at?: string
  updated_at?: string
  validated?: boolean
  warning?: string
}

export type ApiKey = {
  id: UUID
  provider: string
  name: string
  hint: string
  created_at: string
}

export type ConversationSummary = {
  id: UUID
  bot_id: UUID
  bot_name: string | null
  test_flag: boolean
  metadata_json: string
  llm_input_tokens_est: number
  llm_output_tokens_est: number
  cost_usd_est: number
  last_asr_ms: number | null
  last_llm_ttfb_ms: number | null
  last_llm_total_ms: number | null
  last_tts_first_audio_ms: number | null
  last_total_ms: number | null
  created_at: string
  updated_at: string
}

export type MessageMetrics = {
  in: number | null
  out: number | null
  cost: number | null
  asr: number | null
  llm1: number | null
  llm: number | null
  tts1: number | null
  total: number | null
}

export type Citation = {
  url: string
  title?: string | null
  start_index?: number | null
  end_index?: number | null
}

export type ConversationMessage = {
  id: UUID
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  created_at: string
  tool: Record<string, unknown> | null
  tool_name: string | null
  tool_kind: 'call' | 'result' | null
  sender_bot_id?: UUID | null
  sender_name?: string | null
  citations?: Citation[]
  metrics: MessageMetrics
}

export type ConversationDetail = {
  conversation: ConversationSummary
  bot: Bot
  messages: ConversationMessage[]
}

export type GroupBot = {
  id: UUID
  name: string
  slug: string
}

export type GroupConversationSummary = {
  id: UUID
  title: string
  default_bot_id: UUID
  group_bots: GroupBot[]
  created_at: string
  updated_at: string
}

export type GroupConversationDetail = {
  conversation: {
    id: UUID
    title: string
    default_bot_id: UUID
    default_bot_name: string | null
    group_bots: GroupBot[]
    individual_conversations?: { bot_id: UUID; conversation_id: UUID }[]
    created_at: string
    updated_at: string
  }
  messages: ConversationMessage[]
}

export type DataAgentStatus = {
  conversation_id: UUID
  docker_available: boolean
  exists: boolean
  running: boolean
  status?: string
  container_id?: string
  container_name?: string
  started_at?: string
  finished_at?: string
  workspace_dir?: string
  session_id?: string
  error?: string
}

export type HostAction = {
  id: UUID
  conversation_id: UUID
  requested_by_bot_id?: UUID | null
  requested_by_name?: string | null
  action_type: string
  payload: Record<string, any>
  status: string
  stdout?: string
  stderr?: string
  exit_code?: number | null
  error?: string
  created_at?: string
  updated_at?: string
  executed_at?: string | null
}

export type ConversationFileItem = {
  path: string
  name: string
  is_dir: boolean
  size_bytes: number | null
  mtime: string
  download_url?: string | null
}

export type ConversationFiles = {
  conversation_id: UUID
  bot_id: UUID
  bot_name: string
  external_id?: string | null
  workspace_dir: string
  path: string
  recursive: boolean
  items: ConversationFileItem[]
  max_items?: number
}

export type Options = {
  llm_providers?: string[]
  openai_models: string[]
  openrouter_models?: string[]
  local_models?: LocalModel[]
  default_llm_provider?: string
  default_llm_model?: string
  openai_asr_models: string[]
  languages: string[]
  openai_tts_models?: string[]
  openai_tts_voices?: string[]
  start_message_modes: string[]
  openai_pricing: Record<string, { input_per_1m: number; output_per_1m: number }>
  openrouter_pricing?: Record<string, { input_per_1m: number; output_per_1m: number }>
  http_methods?: string[]
}

export type LocalModel = {
  id: string
  name: string
  download_url?: string
  filename?: string
  size_gb?: number | null
  min_ram_gb?: number | null
  supports_tools?: boolean
  tool_support?: string | null
  recommended?: boolean
}

export type WidgetConfig = {
  bot_id: UUID | null
  bot_name?: string | null
  widget_mode?: 'mic' | 'text'
}

export type IntegrationTool = {
  id: UUID
  bot_id: UUID
  name: string
  description: string
  url: string
  method: string
  enabled?: boolean
  use_codex_response?: boolean
  args_required: string[]
  headers_template_json: string
  headers_template_json_masked?: string
  headers_configured?: boolean
  request_body_template: string
  parameters_schema_json?: string
  response_schema_json?: string
  codex_prompt?: string
  postprocess_python?: string
  return_result_directly?: boolean
  response_mapper_json: string
  pagination_json?: string
  static_reply_template: string
  created_at: string
  updated_at: string
}

export type SystemTool = {
  name: string
  description: string
  enabled?: boolean
  can_disable?: boolean
}
