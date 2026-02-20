from __future__ import annotations

import asyncio
import base64
import datetime as dt
import html
import json
import logging
import mimetypes
import os
import queue
import re
import shlex
import secrets
import shutil
import subprocess
import sys
import threading
import tempfile
import time
import webbrowser
from functools import lru_cache, partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from urllib.parse import quote as _url_quote, parse_qsl
from uuid import UUID

import httpx

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlmodel import Session, delete, select
from sqlalchemy import and_, or_

from voicebot.asr.openai_asr import OpenAIASR
from voicebot.config import Settings
from voicebot.crypto import CryptoError, build_hint, get_crypto_box
from voicebot.data_agent.docker_runner import (
    DEFAULT_DATA_AGENT_SYSTEM_PROMPT,
    container_name_for_conversation,
    default_workspace_dir_for_conversation,
    docker_available,
    ensure_conversation_container,
    ensure_image_pulled,
    get_container_ports,
    get_container_status,
    list_data_agent_containers,
    run_container_command,
    run_data_agent,
    stop_data_agent_container,
)
from voicebot.db import init_db, make_engine
from voicebot.downloads import create_download_token, is_allowed_download_path, load_download_token
from voicebot.llm.codex_http_agent import run_codex_export_from_paths, run_codex_http_agent_one_shot, run_codex_http_agent_one_shot_from_paths
from voicebot.llm.codex_saved_runs import append_saved_run_index, find_saved_run
from voicebot.llm.openai_compat_llm import OpenAICompatLLM
from voicebot.llm.openai_llm import CitationEvent, Message, OpenAILLM, ToolCall
from voicebot.llm.openrouter_llm import OpenRouterLLM
from voicebot.local_runtime import LOCAL_RUNTIME
from voicebot.models import AppSetting, Bot, Conversation, ConversationMessage, HostAction, IntegrationTool
from voicebot.store import (
    add_message,
    add_message_with_metrics,
    bots_aggregate_metrics,
    count_conversations,
    count_unread_by_conversation,
    create_bot,
    create_client_key,
    create_conversation,
    create_integration_tool,
    create_key,
    delete_bot,
    delete_client_key,
    delete_integration_tool,
    delete_key,
    decrypt_provider_key,
    get_bot,
    get_client_key,
    get_conversation,
    get_git_token,
    get_integration_tool,
    get_integration_tool_by_name,
    get_or_create_conversation_by_external_id,
    list_bots,
    list_client_keys,
    list_conversations,
    list_integration_tools,
    list_keys,
    list_messages,
    mark_conversation_read,
    merge_conversation_metadata,
    update_bot,
    update_conversation_metrics,
    update_integration_tool,
    upsert_git_token,
    verify_client_key,
)
from voicebot.tools.data_agent import give_command_to_data_agent_tool_def
from voicebot.tools.http_request import http_request_tool_def
from voicebot.tools.set_metadata import set_metadata_tool_def, set_variable_tool_def
from voicebot.tools.web_search import web_search_tool_def
from voicebot.tts.openai_tts import OpenAITTS
from voicebot.utils.python_postprocess import run_python_postprocessor
from voicebot.utils.template import eval_template_value, render_jinja_template, render_template, safe_json_loads
from voicebot.utils.text import SentenceChunker
from voicebot.utils.tokens import ModelPrice, estimate_cost_usd, estimate_messages_tokens, estimate_text_tokens
from voicebot.web.constants import (
    SHOWCASE_BOT_NAME,
    SHOWCASE_BOT_PROMPT,
    SHOWCASE_BOT_START_MESSAGE,
    SYSTEM_BOT_NAME,
    SYSTEM_BOT_PROMPT,
    SYSTEM_BOT_START_MESSAGE,
    WIDGET_BOT_KEY,
    WIDGET_MODE_KEY,
)
from voicebot.web.deps import get_session, require_crypto
from voicebot.web.helpers import audio as audio_helpers
from voicebot.web.helpers import bots as bot_helpers
from voicebot.web.helpers import conversation_init as conversation_init_helpers
from voicebot.web.helpers import data_agent as data_agent_helpers
from voicebot.web.helpers import files as files_helpers
from voicebot.web.helpers import git as git_helpers
from voicebot.web.helpers import group as group_helpers
from voicebot.web.helpers import history as history_helpers
from voicebot.web.helpers import integration_exec as integration_exec_helpers
from voicebot.web.helpers import integration_utils as integration_utils_helpers
from voicebot.web.helpers import llm as llm_helpers
from voicebot.web.helpers import llm_keys as llm_keys_helpers
from voicebot.web.helpers import public_access as public_access_helpers
from voicebot.web.helpers import settings as settings_helpers
from voicebot.web.helpers import tools as tools_helpers
from voicebot.web.helpers import ws_utils as ws_helpers
from voicebot.web.helpers.auth import accepts_html as _accepts_html
from voicebot.web.helpers.auth import basic_auth_ok as _basic_auth_ok_base
from voicebot.web.helpers.auth import viewer_id_from_request as _viewer_id_from_request
from voicebot.web.helpers.auth import ws_auth_header as _ws_auth_header
from voicebot.web.helpers.host_actions import (
    build_host_action_tool_result as _build_host_action_tool_result,
    capture_screenshot_tool_def as _capture_screenshot_tool_def,
    copy_screenshot_to_user_dir as _copy_screenshot_to_user_dir,
    create_host_action as _create_host_action,
    execute_host_action as _execute_host_action,
    execute_host_action_and_update as _execute_host_action_and_update,
    execute_host_action_and_update_async as _execute_host_action_and_update_async,
    finalize_host_action_run as _finalize_host_action_run,
    host_action_payload as _host_action_payload,
    host_action_requires_approval as _host_action_requires_approval,
    maybe_copy_screenshot_from_command as _maybe_copy_screenshot_from_command,
    parse_host_action_args as _parse_host_action_args,
    prepare_screenshot_target as _prepare_screenshot_target,
    screencapture_command as _screencapture_command,
    screenshot_base_dir as _screenshot_base_dir,
    summarize_image_file as _summarize_image_file,
    summarize_screenshot as _summarize_screenshot,
    summarize_screenshot_tool_def as _summarize_screenshot_tool_def,
    tool_error_message as _tool_error_message,
    user_screenshot_dir as _user_screenshot_dir,
)
from voicebot.web.routers import register_all
from voicebot.web.state import AppState
from voicebot.web.ws import public_chat_ws as public_chat_ws_module
from voicebot.web.ws import public_tools as public_tools_module
from voicebot.web.ws import talk_stream as talk_stream_module
from voicebot.web.ws import talk_tool_integrations as talk_tool_integrations_module
from voicebot.web.ws import talk_tools as talk_tools_module
from voicebot.web.ws import talk_ws as talk_ws_module


# Backwards-compatible alias used in legacy tool handlers.
run_python_postprocess = run_python_postprocessor


def create_app() -> FastAPI:
    logger = logging.getLogger("voicebot.web")
    start = time.monotonic()
    logger.info("create_app: start")
    t0 = time.monotonic()
    settings = Settings()
    logger.info("create_app: settings loaded (%.2fs)", time.monotonic() - t0)
    t0 = time.monotonic()
    engine = make_engine(settings.db_url)
    logger.info("create_app: db engine ready (%.2fs)", time.monotonic() - t0)
    t0 = time.monotonic()
    init_db(engine)
    logger.info("create_app: init_db done (%.2fs)", time.monotonic() - t0)

    try:
        with Session(engine) as session:
            bot_helpers.get_or_create_system_bot(session)
            bot_helpers.get_or_create_showcase_bot(session)
        logger.info("create_app: system bots ensured (%.2fs)", time.monotonic() - t0)
    except Exception:
        logger.exception("create_app: failed to ensure system bots")

    t0 = time.monotonic()
    app = FastAPI(title="GravexStudio")
    logger.info("create_app: FastAPI init done (%.2fs)", time.monotonic() - t0)

    data_agent_kickoff_locks: dict[UUID, asyncio.Lock] = {}
    openai_models_cache: dict[str, Any] = {"ts": 0.0, "models": []}
    openrouter_models_cache: dict[str, Any] = {"ts": 0.0, "models": [], "pricing": {}}
    group_ws_clients: dict[str, set[WebSocket]] = {}
    group_ws_lock = asyncio.Lock()

    download_base_url = (getattr(settings, "download_base_url", "") or "127.0.0.1:8000").strip()

    def _download_url_for_token(token: str) -> str:
        return settings_helpers.download_url_for_token(download_base_url, token)

    basic_user = (settings.basic_auth_user or "").strip()
    basic_pass = (settings.basic_auth_pass or "").strip()
    basic_auth_enabled = bool(basic_user and basic_pass)

    _basic_auth_ok = partial(
        _basic_auth_ok_base,
        basic_auth_enabled=basic_auth_enabled,
        basic_user=basic_user,
        basic_pass=basic_pass,
    )

    @app.middleware("http")
    async def _basic_auth_middleware(request: Request, call_next):  # type: ignore[override]
        if not basic_auth_enabled or request.method == "OPTIONS":
            return await call_next(request)
        if _basic_auth_ok(request.headers.get("authorization", "")):
            return await call_next(request)
        return Response(status_code=401, headers={"WWW-Authenticate": "Basic"})

    cors_raw = (os.environ.get("VOICEBOT_CORS_ORIGINS") or "").strip()
    cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()] if cors_raw else []
    cors_origin_regex: str | None = None
    if not cors_origins:
        cors_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ]
        cors_origin_regex = r"^https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?$"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ui_options = {
        "openai_models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-5.2",
            "gpt-5.2-chat-latest",
            "gpt-5.2-pro",
            "gpt-5.1",
            "gpt-5.1-chat-latest",
            "gpt-5.1-mini",
            "gpt-5.1-nano",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex-mini",
            "gpt-5.1-codex",
            "gpt-5-codex",
            "o4-mini",
            "gpt-5",
            "gpt-5-chat-latest",
            "gpt-5-mini",
            "gpt-5-nano",
        ],
        "openai_asr_models": ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"],
        "openai_tts_models": ["gpt-4o-mini-tts", "gpt-4o-realtime-preview-tts"],
        "openai_tts_voices": ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"],
        "languages": [
            "auto",
            "ar",
            "bg",
            "ca",
            "cs",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "fa",
            "fi",
            "fr",
            "he",
            "hi",
            "hr",
            "hu",
            "id",
            "it",
            "ja",
            "ko",
            "lt",
            "lv",
            "nb",
            "nl",
            "pl",
            "pt",
            "ro",
            "ru",
            "sk",
            "sl",
            "sr",
            "sv",
            "th",
            "tr",
            "uk",
            "vi",
            "zh",
        ],
    }

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    ui_dir_env = (os.environ.get("VOICEBOT_UI_DIR") or "").strip()
    ui_dir = Path(ui_dir_env) if ui_dir_env else (Path(__file__).parent / "ui")
    ui_index = ui_dir / "index.html"

    async def _group_ws_broadcast(conversation_id: UUID, payload: dict) -> None:
        key = str(conversation_id)
        async with group_ws_lock:
            clients = list(group_ws_clients.get(key, set()))
        if not clients:
            return
        dead: list[WebSocket] = []
        data = json.dumps(payload, ensure_ascii=False)
        for ws in clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        if dead:
            async with group_ws_lock:
                cur = group_ws_clients.get(key, set())
                for ws in dead:
                    cur.discard(ws)
                if cur:
                    group_ws_clients[key] = cur
                else:
                    group_ws_clients.pop(key, None)

    app.state.igx_state = AppState(
        settings=settings,
        engine=engine,
        logger=logger,
        data_agent_kickoff_locks=data_agent_kickoff_locks,
        download_base_url=download_base_url,
        basic_user=basic_user,
        basic_pass=basic_pass,
        basic_auth_enabled=basic_auth_enabled,
        ui_options=ui_options,
        ui_dir=ui_dir,
        ui_index=ui_index,
        openai_models_cache=openai_models_cache,
        openrouter_models_cache=openrouter_models_cache,
    )

    ctx = SimpleNamespace(**globals(), **locals())

    ctx._build_history = partial(history_helpers.build_history, ctx)
    ctx._build_history_budgeted = partial(history_helpers.build_history_budgeted, ctx)
    ctx._build_history_budgeted_threadsafe = partial(history_helpers.build_history_budgeted_threadsafe, ctx)
    ctx._build_history_budgeted_async = partial(history_helpers.build_history_budgeted_async, ctx)
    ctx._get_conversation_meta = partial(history_helpers.get_conversation_meta, ctx)
    ctx._extract_group_mentions = partial(history_helpers.extract_group_mentions, ctx)
    ctx._run_group_bot_turn = partial(history_helpers.run_group_bot_turn, ctx)
    ctx._schedule_group_bots = partial(history_helpers.schedule_group_bots, ctx)

    ctx._data_agent_meta = data_agent_helpers.data_agent_meta
    ctx._data_agent_container_info = partial(data_agent_helpers.data_agent_container_info, ctx)
    ctx._ensure_data_agent_container = partial(data_agent_helpers.ensure_data_agent_container, ctx)
    ctx._build_data_agent_conversation_context = partial(data_agent_helpers.build_data_agent_conversation_context, ctx)
    ctx._initialize_data_agent_workspace = partial(data_agent_helpers.initialize_data_agent_workspace, ctx)
    ctx._kickoff_data_agent_container_if_enabled = partial(data_agent_helpers.kickoff_data_agent_container_if_enabled, ctx)
    ctx._run_data_agent_tool_persist = partial(data_agent_helpers.run_data_agent_tool_persist, ctx)

    ctx._apply_response_mapper = partial(integration_exec_helpers.apply_response_mapper, ctx)
    ctx._render_with_meta = partial(integration_exec_helpers.render_with_meta, ctx)
    ctx._render_static_reply = partial(integration_exec_helpers.render_static_reply, ctx)
    ctx._coerce_json_object = integration_exec_helpers.coerce_json_object
    ctx._render_templates_in_obj = partial(integration_exec_helpers.render_templates_in_obj, ctx)
    ctx._parse_query_params = partial(integration_exec_helpers.parse_query_params, ctx)
    ctx._parse_fields_required = integration_exec_helpers.parse_fields_required
    ctx._build_response_mapper_from_fields = integration_exec_helpers.build_response_mapper_from_fields
    ctx._execute_http_request_tool = partial(integration_exec_helpers.execute_http_request_tool, ctx)
    ctx._execute_integration_http = partial(integration_exec_helpers.execute_integration_http, ctx)

    ctx._ws_send_json = partial(ws_helpers.ws_send_json, ctx)
    ctx._aiter_from_blocking_iterator = partial(ws_helpers.aiter_from_blocking_iterator, ctx)
    ctx._stream_llm_reply = partial(ws_helpers.stream_llm_reply, ctx)
    ctx._record_llm_debug_payload = partial(ws_helpers.record_llm_debug_payload, ctx)
    ctx._emit_llm_debug_payload = partial(ws_helpers.emit_llm_debug_payload, ctx)
    ctx._public_send_done = partial(ws_helpers.public_send_done, ctx)
    ctx._public_send_interim = partial(ws_helpers.public_send_interim, ctx)
    ctx._public_send_greeting = partial(ws_helpers.public_send_greeting, ctx)
    ctx._NullWebSocket = ws_helpers.NullWebSocket

    ctx._normalize_llm_provider = llm_helpers.normalize_llm_provider
    ctx._provider_display_name = llm_helpers.provider_display_name
    ctx._get_openai_api_key = partial(llm_helpers.get_openai_api_key, settings=settings)
    ctx._get_openai_api_key_for_bot = partial(llm_helpers.get_openai_api_key_for_bot, settings=settings)
    ctx._get_openrouter_api_key = partial(llm_helpers.get_openrouter_api_key, settings=settings)
    ctx._get_openrouter_api_key_for_bot = partial(llm_helpers.get_openrouter_api_key_for_bot, settings=settings)
    ctx._llm_provider_for_bot = llm_helpers.llm_provider_for_bot
    ctx._get_llm_api_key_for_bot = partial(llm_helpers.get_llm_api_key_for_bot, settings=settings)
    ctx._build_llm_client = llm_helpers.build_llm_client
    ctx._require_llm_client = partial(llm_helpers.require_llm_client, settings=settings)
    ctx._get_openai_pricing = llm_helpers.get_openai_pricing

    def _refresh_openrouter_models_cache(session: Session) -> None:
        llm_helpers.refresh_openrouter_models_cache(
            session,
            cache=openrouter_models_cache,
            get_openrouter_api_key_fn=ctx._get_openrouter_api_key,
        )

    def _get_openrouter_pricing(session: Session) -> dict[str, ModelPrice]:
        return llm_helpers.get_openrouter_pricing(
            session,
            cache=openrouter_models_cache,
            get_openrouter_api_key_fn=ctx._get_openrouter_api_key,
        )

    def _get_model_price(session: Session, *, provider: str, model: str) -> Optional[ModelPrice]:
        return llm_helpers.get_model_price(
            session,
            provider=provider,
            model=model,
            cache=openrouter_models_cache,
            get_openrouter_api_key_fn=ctx._get_openrouter_api_key,
        )

    def _estimate_llm_cost_for_turn(
        *,
        session: Session,
        bot: Bot,
        provider: str,
        history: list[Message],
        assistant_text: str,
    ) -> tuple[int, int, float]:
        return llm_helpers.estimate_llm_cost_for_turn(
            session=session,
            bot=bot,
            provider=provider,
            history=history,
            assistant_text=assistant_text,
            get_model_price_fn=_get_model_price,
        )

    ctx._refresh_openrouter_models_cache = _refresh_openrouter_models_cache
    ctx._get_openrouter_pricing = _get_openrouter_pricing
    ctx._get_model_price = _get_model_price
    ctx._estimate_llm_cost_for_turn = _estimate_llm_cost_for_turn
    ctx._make_start_message_instruction = llm_helpers.make_start_message_instruction
    ctx._init_conversation_and_greet = conversation_init_helpers.init_conversation_and_greet

    ctx._get_openai_api_key_global = llm_keys_helpers.get_openai_api_key_global

    ctx._get_or_create_system_bot = bot_helpers.get_or_create_system_bot
    ctx._get_or_create_showcase_bot = bot_helpers.get_or_create_showcase_bot

    ctx._bot_to_dict = lambda bot: bot_helpers.bot_to_dict(
        bot,
        disabled_tool_names_fn=ctx._disabled_tool_names,
        llm_provider_for_bot_fn=ctx._llm_provider_for_bot,
    )
    ctx._tool_to_dict = bot_helpers.tool_to_dict

    ctx._group_message_payload = group_helpers.group_message_payload
    ctx._group_conversation_payload = group_helpers.group_conversation_payload
    ctx._group_bots_from_conv = group_helpers.group_bots_from_conv
    ctx._group_bot_aliases = group_helpers.group_bot_aliases
    ctx._ensure_group_individual_conversations = group_helpers.ensure_group_individual_conversations
    ctx._mirror_group_message = group_helpers.mirror_group_message
    ctx._assert_bot_in_conversation = group_helpers.assert_bot_in_conversation
    ctx._format_group_message_prefix = group_helpers.format_group_message_prefix
    ctx._sanitize_group_reply = group_helpers.sanitize_group_reply
    ctx._reset_conversation_state = group_helpers.reset_conversation_state
    ctx._slugify = group_helpers.slugify

    ctx._sanitize_upload_path = files_helpers.sanitize_upload_path
    ctx._is_path_within_root = files_helpers.is_path_within_root
    ctx._should_hide_data_agent_path = files_helpers.should_hide_workspace_path
    ctx._workspace_dir_for_conversation = files_helpers.workspace_dir_for_conversation
    ctx._resolve_data_agent_target = files_helpers.resolve_data_agent_target
    ctx._conversation_files_payload = files_helpers.conversation_files_payload

    ctx._data_agent_workspace_dir_for_conversation = files_helpers.data_agent_workspace_dir_for_conversation

    ctx._basic_auth_ok = _basic_auth_ok
    ctx._accepts_html = _accepts_html
    ctx._viewer_id_from_request = _viewer_id_from_request
    ctx._download_url_for_token = _download_url_for_token
    ctx._url_quote = _url_quote

    def _require_crypto():
        try:
            return get_crypto_box(settings.secret_key)
        except CryptoError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    ctx.require_crypto = _require_crypto

    ctx._origin_allowed = public_access_helpers.origin_allowed
    ctx._bot_allowed = public_access_helpers.bot_allowed
    ctx._require_public_conversation_access = public_access_helpers.require_public_conversation_access
    ctx._conversation_messages_payload = public_access_helpers.conversation_messages_payload
    ctx._render_conversation_html = public_access_helpers.render_conversation_html

    ctx._host_action_payload = _host_action_payload
    ctx._create_host_action = _create_host_action
    ctx._host_action_requires_approval = _host_action_requires_approval
    ctx._build_host_action_tool_result = _build_host_action_tool_result
    ctx._execute_host_action_and_update = _execute_host_action_and_update
    ctx._execute_host_action_and_update_async = _execute_host_action_and_update_async
    ctx._parse_host_action_args = _parse_host_action_args
    ctx._prepare_screenshot_target = _prepare_screenshot_target
    ctx._screencapture_command = _screencapture_command
    ctx._summarize_image_file = _summarize_image_file
    ctx._summarize_screenshot = _summarize_screenshot
    ctx._tool_error_message = _tool_error_message

    ctx._ndjson = audio_helpers.ndjson
    ctx._wav_bytes = audio_helpers.wav_bytes
    ctx._decode_wav_bytes_to_pcm16_16k = audio_helpers.decode_wav_bytes_to_pcm16_16k
    ctx._get_asr = audio_helpers.get_asr
    ctx._get_openai_tts_handle = audio_helpers.get_openai_tts_handle
    ctx._get_tts_synth_fn = audio_helpers.get_tts_synth_fn
    ctx._estimate_wav_seconds = audio_helpers.estimate_wav_seconds
    ctx._iter_tts_chunks = audio_helpers.iter_tts_chunks

    ctx._disabled_tool_names = tools_helpers.disabled_tool_names
    ctx._system_tools_defs = tools_helpers.system_tools_defs
    ctx._system_tools_public_list = tools_helpers.system_tools_public_list
    ctx._integration_tool_def = tools_helpers.integration_tool_def
    ctx._build_tools_for_bot = tools_helpers.build_tools_for_bot

    ctx._normalize_git_provider = git_helpers.normalize_git_provider
    ctx._get_git_token_plaintext = git_helpers.get_git_token_plaintext
    ctx._parse_auth_json = git_helpers.parse_auth_json
    ctx._git_auth_mode = git_helpers.git_auth_mode
    ctx._merge_git_token_auth = git_helpers.merge_git_token_auth
    ctx._validate_github_token = git_helpers.validate_github_token

    ctx._get_app_setting = settings_helpers.get_app_setting
    ctx._set_app_setting = settings_helpers.set_app_setting
    ctx._mask_headers_json = settings_helpers.mask_headers_json
    ctx._headers_configured = settings_helpers.headers_configured
    ctx._mask_secret = settings_helpers.mask_secret
    ctx._read_key_from_env_file = settings_helpers.read_key_from_env_file

    ctx._parse_required_args_json = integration_utils_helpers.parse_required_args_json
    ctx._parse_parameters_schema_json = integration_utils_helpers.parse_parameters_schema_json
    ctx._missing_required_args = integration_utils_helpers.missing_required_args
    ctx._apply_schema_defaults = integration_utils_helpers.apply_schema_defaults
    ctx._extract_required_tool_args = integration_utils_helpers.extract_required_tool_args
    ctx._safe_json_list = integration_utils_helpers.safe_json_list
    ctx._normalize_headers_for_json = integration_utils_helpers.normalize_headers_for_json
    ctx._normalize_content_type_header_value = integration_utils_helpers.normalize_content_type_header_value
    ctx._get_json_path = integration_utils_helpers.get_json_path
    ctx._set_json_path = integration_utils_helpers.set_json_path
    ctx._http_error_response = integration_utils_helpers.http_error_response
    ctx._integration_error_user_message = integration_utils_helpers.integration_error_user_message
    ctx._should_followup_llm_for_tool = integration_utils_helpers.should_followup_llm_for_tool
    ctx._parse_follow_up_flag = integration_utils_helpers.parse_follow_up_flag

    ctx._group_ws_broadcast = _group_ws_broadcast

    talk_stream_module.bind_ctx(ctx)
    talk_tool_integrations_module.bind_ctx(ctx)
    talk_tools_module.bind_ctx(ctx)
    public_tools_module.bind_ctx(ctx)
    conversation_init_helpers.bind_ctx(ctx)
    talk_ws_module.bind_ctx(ctx)
    public_chat_ws_module.bind_ctx(ctx)
    ctx.talk_ws = talk_ws_module.talk_ws
    ctx.public_chat_ws = public_chat_ws_module.public_chat_ws

    register_all(app, ctx)
    if ui_dir.exists():
        class SpaStaticFiles(StaticFiles):
            def __init__(self, *args, index_file: Path, **kwargs):
                super().__init__(*args, **kwargs)
                self._index_file = index_file

            async def get_response(self, path: str, scope):
                response = await super().get_response(path, scope)
                if response.status_code != 404:
                    return response
                try:
                    headers = Headers(scope=scope)
                    accept = headers.get("accept") or ""
                except Exception:
                    accept = ""
                if self._index_file.exists() and _accepts_html(accept) and "." not in path:
                    return FileResponse(str(self._index_file))
                return response

        app.mount("/", SpaStaticFiles(directory=str(ui_dir), html=True, index_file=ui_index), name="studio-ui")
    logger.info("create_app: done (%.2fs)", time.monotonic() - start)
    return app
