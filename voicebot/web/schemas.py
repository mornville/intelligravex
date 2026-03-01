from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    text: str
    speak: bool = True


class GroupConversationCreateRequest(BaseModel):
    title: str
    bot_ids: list[str]
    default_bot_id: str
    test_flag: bool = False
    swarm_config: Optional[dict[str, Any]] = None


class GroupSwarmConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    coordinator_mode: Optional[str] = None
    max_turns_per_run: Optional[int] = None
    max_parallel_bots: Optional[int] = None
    max_hops: Optional[int] = None
    allow_revisit: Optional[bool] = None


class GroupMessageRequest(BaseModel):
    text: str
    sender_role: str = "user"
    sender_bot_id: Optional[str] = None
    sender_name: Optional[str] = None


class TalkResponseEvent(BaseModel):
    type: str


class ApiKeyCreateRequest(BaseModel):
    provider: str = "openai"
    name: str
    secret: str


class ClientKeyCreateRequest(BaseModel):
    name: str
    allowed_origins: str = ""
    allowed_bot_ids: list[str] = []
    secret: Optional[str] = None


class GitTokenRequest(BaseModel):
    provider: str = "github"
    token: str


class WidgetConfigRequest(BaseModel):
    bot_id: Optional[str] = None
    widget_mode: Optional[str] = None


class OpenDashboardRequest(BaseModel):
    path: Optional[str] = None


class LocalSetupRequest(BaseModel):
    model_id: Optional[str] = None
    custom_url: Optional[str] = None
    custom_name: Optional[str] = None


class BotCreateRequest(BaseModel):
    name: str
    llm_provider: str = "openai"
    openai_model: str = "o4-mini"
    openai_asr_model: str = "gpt-4o-mini-transcribe"
    web_search_model: Optional[str] = None
    codex_model: Optional[str] = None
    summary_model: Optional[str] = None
    history_window_turns: Optional[int] = None
    enable_data_agent: bool = False
    data_agent_api_spec_text: str = ""
    data_agent_auth_json: str = "{}"
    data_agent_system_prompt: str = ""
    data_agent_return_result_directly: bool = False
    data_agent_prewarm_on_start: bool = False
    data_agent_prewarm_prompt: str = ""
    data_agent_model: str = "gpt-5.2"
    data_agent_reasoning_effort: str = "high"
    enable_host_actions: bool = False
    enable_host_shell: bool = False
    require_host_action_approval: bool = False
    system_prompt: str
    language: str = "en"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    openai_tts_speed: float = 1.0
    start_message_mode: str = "llm"
    start_message_text: str = ""


class BotUpdateRequest(BaseModel):
    name: Optional[str] = None
    llm_provider: Optional[str] = None
    openai_model: Optional[str] = None
    openai_asr_model: Optional[str] = None
    web_search_model: Optional[str] = None
    codex_model: Optional[str] = None
    summary_model: Optional[str] = None
    history_window_turns: Optional[int] = None
    enable_data_agent: Optional[bool] = None
    data_agent_api_spec_text: Optional[str] = None
    data_agent_auth_json: Optional[str] = None
    data_agent_system_prompt: Optional[str] = None
    data_agent_return_result_directly: Optional[bool] = None
    data_agent_prewarm_on_start: Optional[bool] = None
    data_agent_prewarm_prompt: Optional[str] = None
    data_agent_model: Optional[str] = None
    data_agent_reasoning_effort: Optional[str] = None
    enable_host_actions: Optional[bool] = None
    enable_host_shell: Optional[bool] = None
    require_host_action_approval: Optional[bool] = None
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    openai_tts_model: Optional[str] = None
    openai_tts_voice: Optional[str] = None
    openai_tts_speed: Optional[float] = None
    start_message_mode: Optional[str] = None
    start_message_text: Optional[str] = None
    disabled_tools: Optional[list[str]] = None


class IntegrationToolCreateRequest(BaseModel):
    name: str
    description: str = ""
    url: str
    method: str = "GET"
    use_codex_response: bool = False
    enabled: bool = True
    args_required: list[str] = []
    headers_template_json: str = "{}"
    request_body_template: str = "{}"
    parameters_schema_json: str = ""
    response_schema_json: str = ""
    codex_prompt: str = ""
    postprocess_python: str = ""
    return_result_directly: bool = False
    response_mapper_json: str = "{}"
    pagination_json: str = ""
    static_reply_template: str = ""


class IntegrationToolUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    method: Optional[str] = None
    use_codex_response: Optional[bool] = None
    enabled: Optional[bool] = None
    args_required: Optional[list[str]] = None
    headers_template_json: Optional[str] = None
    request_body_template: Optional[str] = None
    parameters_schema_json: Optional[str] = None
    response_schema_json: Optional[str] = None
    codex_prompt: Optional[str] = None
    postprocess_python: Optional[str] = None
    return_result_directly: Optional[bool] = None
    response_mapper_json: Optional[str] = None
    pagination_json: Optional[str] = None
    static_reply_template: Optional[str] = None
