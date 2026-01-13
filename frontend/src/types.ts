export type UUID = string

export type Bot = {
  id: UUID
  name: string
  openai_model: string
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
  openai_key_id: UUID | null
  system_prompt: string
  language: string
  tts_language: string
  tts_vendor: 'xtts_local' | 'openai_tts' | string
  whisper_model: string
  whisper_device: string
  xtts_model: string
  speaker_id: string | null
  speaker_wav: string | null
  openai_tts_model: string
  openai_tts_voice: string
  openai_tts_speed: number
  tts_split_sentences: boolean
  tts_chunk_min_chars: number
  tts_chunk_max_chars: number
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

export type ConversationMessage = {
  id: UUID
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string
  created_at: string
  tool: Record<string, unknown> | null
  tool_name: string | null
  tool_kind: 'call' | 'result' | null
  metrics: MessageMetrics
}

export type ConversationDetail = {
  conversation: ConversationSummary
  bot: Bot
  messages: ConversationMessage[]
}

export type Options = {
  openai_models: string[]
  whisper_models: string[]
  whisper_devices: string[]
  languages: string[]
  xtts_models: string[]
  openai_tts_models?: string[]
  openai_tts_voices?: string[]
  start_message_modes: string[]
  asr_vendors?: string[]
  tts_vendors?: string[]
  openai_pricing: Record<string, { input_per_1m: number; output_per_1m: number }>
  http_methods?: string[]
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
