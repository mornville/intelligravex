from __future__ import annotations

import datetime as dt

from sqlmodel import Session, select

from voicebot.models import Bot, IntegrationTool
from voicebot.store import create_bot, get_bot
from voicebot.local_runtime import LOCAL_RUNTIME
from voicebot.web.constants import (
    SHOWCASE_BOT_NAME,
    SHOWCASE_BOT_PROMPT,
    SHOWCASE_BOT_START_MESSAGE,
    SYSTEM_BOT_NAME,
    SYSTEM_BOT_PROMPT,
    SYSTEM_BOT_START_MESSAGE,
)
from voicebot.web.helpers.integration_utils import (
    parse_required_args_json,
)
from voicebot.web.helpers.settings import get_app_setting
from voicebot.web.helpers.settings import mask_headers_json, headers_configured


def get_or_create_system_bot(session: Session) -> Bot:
    stmt = select(Bot).where(Bot.name == SYSTEM_BOT_NAME).limit(1)
    bot = session.exec(stmt).first()
    if bot:
        updated = False
        if bot.system_prompt != SYSTEM_BOT_PROMPT:
            bot.system_prompt = SYSTEM_BOT_PROMPT
            updated = True
        if bot.start_message_mode != "static":
            bot.start_message_mode = "static"
            updated = True
        if bot.start_message_text != SYSTEM_BOT_START_MESSAGE:
            bot.start_message_text = SYSTEM_BOT_START_MESSAGE
            updated = True
        if bot.enable_data_agent:
            bot.enable_data_agent = False
            updated = True
        if not bot.enable_host_actions:
            bot.enable_host_actions = True
            updated = True
        if not bot.enable_host_shell:
            bot.enable_host_shell = True
            updated = True
        if bot.require_host_action_approval:
            bot.require_host_action_approval = False
            updated = True
        if bot.disabled_tools_json != "[]":
            bot.disabled_tools_json = "[]"
            updated = True
        if updated:
            bot.updated_at = dt.datetime.now(dt.timezone.utc)
            session.add(bot)
            session.commit()
            session.refresh(bot)
        return bot
    bot = Bot(
        name=SYSTEM_BOT_NAME,
        system_prompt=SYSTEM_BOT_PROMPT,
        start_message_mode="static",
        start_message_text=SYSTEM_BOT_START_MESSAGE,
        enable_data_agent=False,
        enable_host_actions=True,
        enable_host_shell=True,
        require_host_action_approval=False,
        disabled_tools_json="[]",
    )
    return create_bot(session, bot)


def _default_showcase_provider(session: Session) -> str:
    provider = (get_app_setting(session, "default_llm_provider") or "").strip().lower() or "openai"
    if provider not in ("openai", "openrouter", "local", "chatgpt"):
        return "openai"
    return provider


def _default_showcase_model(session: Session, provider: str) -> str:
    default_model = (get_app_setting(session, "default_llm_model") or "").strip()
    if default_model:
        return default_model
    if provider in ("openai", "chatgpt"):
        return "gpt-5.2"
    if provider == "local":
        models = LOCAL_RUNTIME.list_models()
        for item in models:
            if item.get("recommended"):
                return str(item.get("id") or "").strip()
        if models:
            return str(models[0].get("id") or "").strip()
        return "llama3.2-3b-instruct-q4_k_m"
    return "o4-mini"


def get_or_create_showcase_bot(session: Session) -> Bot:
    provider = _default_showcase_provider(session)
    model = _default_showcase_model(session, provider)
    stmt = select(Bot).where(Bot.name == SHOWCASE_BOT_NAME).limit(1)
    bot = session.exec(stmt).first()
    if not bot:
        legacy = session.exec(select(Bot).where(Bot.name == "GravexStudio Showcase").limit(1)).first()
        if legacy:
            legacy.name = SHOWCASE_BOT_NAME
            bot = legacy
    if bot:
        updated = False
        if bot.llm_provider != provider:
            bot.llm_provider = provider
            updated = True
        if bot.openai_model != model:
            bot.openai_model = model
            updated = True
        if bot.web_search_model != model:
            bot.web_search_model = model
            updated = True
        if bot.system_prompt != SHOWCASE_BOT_PROMPT:
            bot.system_prompt = SHOWCASE_BOT_PROMPT
            updated = True
        if bot.start_message_mode != "static":
            bot.start_message_mode = "static"
            updated = True
        if bot.start_message_text != SHOWCASE_BOT_START_MESSAGE:
            bot.start_message_text = SHOWCASE_BOT_START_MESSAGE
            updated = True
        if bot.enable_data_agent:
            bot.enable_data_agent = False
            updated = True
        if not bot.enable_host_actions:
            bot.enable_host_actions = True
            updated = True
        if not bot.enable_host_shell:
            bot.enable_host_shell = True
            updated = True
        if bot.require_host_action_approval:
            bot.require_host_action_approval = False
            updated = True
        if bot.disabled_tools_json != "[]":
            bot.disabled_tools_json = "[]"
            updated = True
        if updated:
            bot.updated_at = dt.datetime.now(dt.timezone.utc)
            session.add(bot)
            session.commit()
            session.refresh(bot)
        return bot
    bot = Bot(
        name=SHOWCASE_BOT_NAME,
        llm_provider=provider,
        openai_model=model,
        web_search_model=model,
        system_prompt=SHOWCASE_BOT_PROMPT,
        start_message_mode="static",
        start_message_text=SHOWCASE_BOT_START_MESSAGE,
        enable_data_agent=False,
        enable_host_actions=True,
        enable_host_shell=True,
        require_host_action_approval=False,
        disabled_tools_json="[]",
    )
    return create_bot(session, bot)


def bot_to_dict(bot: Bot, *, disabled_tool_names_fn, llm_provider_for_bot_fn) -> dict:
    disabled = disabled_tool_names_fn(bot)
    return {
        "id": str(bot.id),
        "name": bot.name,
        "llm_provider": llm_provider_for_bot_fn(bot),
        "openai_model": bot.openai_model,
        "web_search_model": getattr(bot, "web_search_model", bot.openai_model),
        "codex_model": getattr(bot, "codex_model", "gpt-5.1-codex-mini"),
        "summary_model": getattr(bot, "summary_model", "gpt-5-nano"),
        "history_window_turns": int(getattr(bot, "history_window_turns", 16) or 16),
        "enable_data_agent": bool(getattr(bot, "enable_data_agent", False)),
        "data_agent_api_spec_text": getattr(bot, "data_agent_api_spec_text", "") or "",
        "data_agent_auth_json": getattr(bot, "data_agent_auth_json", "") or "{}",
        "data_agent_system_prompt": getattr(bot, "data_agent_system_prompt", "") or "",
        "data_agent_return_result_directly": bool(getattr(bot, "data_agent_return_result_directly", False)),
        "data_agent_prewarm_on_start": bool(getattr(bot, "data_agent_prewarm_on_start", False)),
        "data_agent_prewarm_prompt": getattr(bot, "data_agent_prewarm_prompt", "") or "",
        "data_agent_model": getattr(bot, "data_agent_model", "gpt-5.2") or "gpt-5.2",
        "data_agent_reasoning_effort": getattr(bot, "data_agent_reasoning_effort", "high") or "high",
        "enable_host_actions": bool(getattr(bot, "enable_host_actions", False)),
        "enable_host_shell": bool(getattr(bot, "enable_host_shell", False)),
        "require_host_action_approval": bool(getattr(bot, "require_host_action_approval", False)),
        "disabled_tools": sorted(disabled),
        "system_prompt": bot.system_prompt,
        "language": bot.language,
        "openai_asr_model": getattr(bot, "openai_asr_model", "gpt-4o-mini-transcribe"),
        "openai_tts_model": bot.openai_tts_model,
        "openai_tts_voice": bot.openai_tts_voice,
        "openai_tts_speed": float(bot.openai_tts_speed),
        "start_message_mode": bot.start_message_mode,
        "start_message_text": bot.start_message_text,
        "created_at": bot.created_at.isoformat(),
        "updated_at": bot.updated_at.isoformat(),
    }


def tool_to_dict(t: IntegrationTool) -> dict:
    return {
        "id": str(t.id),
        "bot_id": str(t.bot_id),
        "name": t.name,
        "description": t.description,
        "url": t.url,
        "method": t.method,
        "enabled": bool(getattr(t, "enabled", True)),
        "args_required": parse_required_args_json(getattr(t, "args_required_json", "[]")),
        "headers_template_json": "{}",
        "headers_template_json_masked": mask_headers_json(t.headers_template_json),
        "headers_configured": headers_configured(t.headers_template_json),
        "request_body_template": t.request_body_template,
        "parameters_schema_json": t.parameters_schema_json,
        "response_schema_json": getattr(t, "response_schema_json", "") or "",
        "codex_prompt": getattr(t, "codex_prompt", "") or "",
        "postprocess_python": getattr(t, "postprocess_python", "") or "",
        "return_result_directly": bool(getattr(t, "return_result_directly", False)),
        "response_mapper_json": t.response_mapper_json,
        "pagination_json": getattr(t, "pagination_json", "") or "",
        "static_reply_template": t.static_reply_template,
        "use_codex_response": bool(getattr(t, "use_codex_response", False)),
        "created_at": t.created_at.isoformat(),
        "updated_at": t.updated_at.isoformat(),
    }
