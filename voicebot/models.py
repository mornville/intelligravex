from __future__ import annotations

import datetime as dt
from typing import Optional
from uuid import UUID, uuid4

from sqlmodel import Field, SQLModel


class ApiKey(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    provider: str = Field(index=True)  # e.g. "openai"
    name: str = Field(index=True)
    hint: str = Field(default="")  # masked preview for UI, e.g. sk-****abcd
    secret_ciphertext: bytes
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class Bot(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    name: str = Field(index=True)

    # LLM
    llm_provider: str = "openai"  # "openai" | "openrouter"
    openai_model: str = "o4-mini"
    web_search_model: str = "gpt-4o-mini"
    codex_model: str = "gpt-5.1-codex-mini"
    summary_model: str = "gpt-5-nano"
    history_window_turns: int = 16
    enable_data_agent: bool = Field(default=False)
    # Data Agent (Codex CLI in an isolated runtime, per conversation).
    # NOTE: auth is stored as plaintext JSON per user request (no masking); handle with care.
    data_agent_api_spec_text: str = Field(default="")
    data_agent_auth_json: str = Field(default="{}")
    data_agent_system_prompt: str = Field(default="")
    data_agent_return_result_directly: bool = Field(default=False)
    data_agent_prewarm_on_start: bool = Field(default=False)
    data_agent_prewarm_prompt: str = Field(default="")
    enable_host_actions: bool = Field(default=False)
    enable_host_shell: bool = Field(default=False)
    require_host_action_approval: bool = Field(default=False)
    # List of tool names disabled for this bot (JSON list stored as text).
    # Applies to system tools (e.g. web_search) and integration tools by name.
    disabled_tools_json: str = "[]"
    system_prompt: str = "You are a fast, helpful voice assistant. Keep answers concise unless asked."

    # ASR (OpenAI)
    language: str = "en"
    openai_asr_model: str = "gpt-4o-mini-transcribe"

    # TTS (OpenAI)
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    openai_tts_speed: float = 1.0

    # Conversation start message (assistant speaks first)
    start_message_mode: str = "llm"  # "static" | "llm"
    start_message_text: str = ""

    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class Conversation(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    bot_id: UUID = Field(foreign_key="bot.id", index=True)
    test_flag: bool = Field(default=True, index=True)
    is_group: bool = Field(default=False, index=True)
    group_title: str = Field(default="")
    group_bots_json: str = Field(default="[]")
    # External, client-provided stable conversation id (used for embeds).
    external_id: Optional[str] = Field(default=None, index=True)
    client_key_id: Optional[UUID] = Field(default=None, index=True)
    metadata_json: str = Field(default="{}")
    # LLM usage + cost (estimated)
    llm_input_tokens_est: int = Field(default=0)
    llm_output_tokens_est: int = Field(default=0)
    cost_usd_est: float = Field(default=0.0)

    # Latency (last turn, ms)
    last_asr_ms: Optional[int] = Field(default=None)
    last_llm_ttfb_ms: Optional[int] = Field(default=None)
    last_llm_total_ms: Optional[int] = Field(default=None)
    last_tts_first_audio_ms: Optional[int] = Field(default=None)
    last_total_ms: Optional[int] = Field(default=None)

    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)


class ConversationMessage(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    conversation_id: UUID = Field(foreign_key="conversation.id", index=True)
    role: str = Field(index=True)  # system/user/assistant
    content: str
    sender_bot_id: Optional[UUID] = Field(default=None, index=True)
    sender_name: Optional[str] = Field(default=None)
    # Optional per-message metrics (typically set on assistant messages)
    input_tokens_est: Optional[int] = Field(default=None)
    output_tokens_est: Optional[int] = Field(default=None)
    cost_usd_est: Optional[float] = Field(default=None)
    asr_ms: Optional[int] = Field(default=None)
    llm_ttfb_ms: Optional[int] = Field(default=None)
    llm_total_ms: Optional[int] = Field(default=None)
    tts_first_audio_ms: Optional[int] = Field(default=None)
    total_ms: Optional[int] = Field(default=None)
    citations_json: str = Field(default="[]")
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)


class HostAction(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    conversation_id: UUID = Field(foreign_key="conversation.id", index=True)
    requested_by_bot_id: Optional[UUID] = Field(default=None, index=True)
    requested_by_name: Optional[str] = Field(default=None)
    action_type: str = Field(default="")
    payload_json: str = Field(default="{}")
    status: str = Field(default="pending")
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    exit_code: Optional[int] = Field(default=None)
    error: str = Field(default="")
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    executed_at: Optional[dt.datetime] = Field(default=None, index=True)


class IntegrationTool(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    bot_id: UUID = Field(foreign_key="bot.id", index=True)

    # OpenAI tool name (unique per bot, enforced at app layer)
    name: str = Field(index=True)
    description: str = Field(default="")

    url: str
    method: str = Field(default="GET")  # GET/POST/PUT/PATCH/DELETE
    # List of required keys inside the LLM tool-call `args` object (stored as JSON list).
    args_required_json: str = Field(default="[]")
    # Optional request headers as a JSON template string.
    # Example: {"Authorization":"Bearer {{env.MEDICAL_SQL_API_TOKEN}}"}
    headers_template_json: str = Field(default="{}")
    request_body_template: str = Field(default="{}")  # JSON template string

    # JSON schema for tool call args (object schema). Stored as JSON string.
    parameters_schema_json: str = Field(default='{"type":"object","properties":{},"additionalProperties":true}')

    # Optional JSON schema for the integration HTTP response (object schema). Stored as JSON string.
    # Used by the Codex HTTP agent to generate extraction code without sending the full response payload to the model.
    response_schema_json: str = Field(default="")

    # Optional per-tool Codex prompt/instructions (applied when use_codex_response is enabled).
    codex_prompt: str = Field(default="")
    # Optional: deterministic post-processing script (runs locally instead of Codex one_shot when non-empty).
    postprocess_python: str = Field(default="")
    # If true, return codex_result_text directly as the assistant reply (skip follow-up LLM rephrase).
    return_result_directly: bool = Field(default=False)

    # Maps conversation metadata keys to template values using {{response...}} or {{.meta...}}.
    response_mapper_json: str = Field(default="{}")

    # Optional: if set, ignore LLM-provided next_reply and render this template instead (Jinja2).
    static_reply_template: str = Field(default="")

    # If true, backend uses the bot's codex_model to generate the post-tool reply (instead of trusting next_reply).
    use_codex_response: bool = Field(default=False)
    # If false, this tool is not exposed to the LLM and cannot be executed.
    enabled: bool = Field(default=True)
    # Optional pagination config (JSON). If set, backend will automatically fetch multiple pages
    # and merge results before mapping/Codex post-processing.
    pagination_json: str = Field(default="")

    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)


class AppSetting(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str = Field(default="")
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)


class ClientKey(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    name: str = Field(index=True)
    hint: str = Field(default="")  # masked preview, e.g. igx_****abcd
    secret_hash: str  # sha256 hex
    allowed_origins: str = Field(default="")  # comma-separated
    allowed_bot_ids_json: str = Field(default="[]")  # JSON list of UUID strings (empty = all)
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)


class GitToken(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    provider: str = Field(index=True)
    hint: str = Field(default="")
    token_ciphertext: bytes
    created_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)
    updated_at: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc), index=True)
