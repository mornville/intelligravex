from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import logging
import os
import queue
import re
import secrets
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Tuple
from uuid import UUID

import httpx

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session

from voicebot.config import Settings
from voicebot.crypto import CryptoError, get_crypto_box
from voicebot.db import init_db, make_engine
from voicebot.asr.whisper_asr import WhisperASR
from voicebot.llm.codex_http_agent import run_codex_http_agent_one_shot, run_codex_http_agent_one_shot_from_paths
from voicebot.llm.codex_http_agent import run_codex_export_from_paths
from voicebot.llm.codex_saved_runs import append_saved_run_index, find_saved_run
from voicebot.downloads import create_download_token, is_allowed_download_path, load_download_token
from voicebot.llm.openai_llm import Message, OpenAILLM, ToolCall
from voicebot.models import Bot
from voicebot.store import (
    create_bot,
    create_conversation,
    create_key,
    create_client_key,
    add_message,
    add_message_with_metrics,
    update_conversation_metrics,
    merge_conversation_metadata,
    decrypt_openai_key,
    delete_bot,
    delete_client_key,
    delete_key,
    get_bot,
    get_client_key,
    list_bots,
    bots_aggregate_metrics,
    list_client_keys,
    list_keys,
    get_or_create_conversation_by_external_id,
    get_conversation,
    list_conversations,
    count_conversations,
    list_messages,
    update_bot,
    list_integration_tools,
    create_integration_tool,
    update_integration_tool,
    delete_integration_tool,
    get_integration_tool,
    get_integration_tool_by_name,
    verify_client_key,
)
from voicebot.tts.xtts import XTTSv2
from voicebot.tts.openai_tts import OpenAITTS
from voicebot.utils.tokens import ModelPrice, estimate_cost_usd, estimate_messages_tokens, estimate_text_tokens
from voicebot.utils.python_postprocess import run_python_postprocessor
from voicebot.tools.set_metadata import set_metadata_tool_def, set_variable_tool_def
from voicebot.tools.web_search import web_search as run_web_search, web_search_tool_def
from voicebot.tools.data_agent import give_command_to_data_agent_tool_def
from voicebot.tools.recall_http_response import recall_http_response_tool_def
from voicebot.tools.export_http_response import export_http_response_tool_def
from voicebot.models import IntegrationTool
from voicebot.utils.template import eval_template_value, render_jinja_template, render_template, safe_json_loads
from voicebot.data_agent.docker_runner import (
    DEFAULT_DATA_AGENT_SYSTEM_PROMPT,
    default_workspace_dir_for_conversation,
    ensure_conversation_container,
    run_data_agent,
)


def _mask_secret(value: str, *, keep_start: int = 10, keep_end: int = 6) -> str:
    v = value or ""
    if len(v) <= keep_start + keep_end + 3:
        return "***"
    return f"{v[:keep_start]}...{v[-keep_end:]}"


def _mask_headers_json(headers_json: str) -> str:
    try:
        obj = json.loads(headers_json or "{}")
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, str):
            out[k] = v
            continue
        if k.lower() == "authorization":
            vv = v.strip()
            if vv.lower().startswith("bearer "):
                token = vv.split(" ", 1)[1].strip()
                out[k] = f"Bearer {_mask_secret(token)}"
            else:
                out[k] = _mask_secret(vv)
        else:
            out[k] = v
    try:
        return json.dumps(out, ensure_ascii=False)
    except Exception:
        return ""


def _headers_configured(headers_json: str) -> bool:
    try:
        obj = json.loads(headers_json or "{}")
    except Exception:
        return bool((headers_json or "").strip())
    if not isinstance(obj, dict):
        return bool((headers_json or "").strip())
    return any(k for k, v in obj.items() if str(k).strip() and (v is not None and str(v).strip()))


def _get_json_path(obj: Any, path: str) -> Any:
    """
    Very small dotted-path getter for dict/list JSON structures.

    Supports:
    - "a.b.c" for dict keys
    - "items.0.name" for list indices
    """
    cur: Any = obj
    p = (path or "").strip()
    if not p:
        return cur
    for raw in p.split("."):
        k = raw.strip()
        if not k:
            continue
        if isinstance(cur, dict):
            if k not in cur:
                return None
            cur = cur.get(k)
            continue
        if isinstance(cur, list):
            try:
                idx = int(k)
            except Exception:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
            continue
        return None
    return cur


def _set_json_path(obj: Any, path: str, value: Any) -> bool:
    """
    Set a dotted path inside a dict/list JSON structure. Returns True if set.
    Only supports paths that traverse existing containers.
    """
    p = (path or "").strip()
    if not p:
        return False
    parts = [x.strip() for x in p.split(".") if x.strip()]
    if not parts:
        return False
    cur: Any = obj
    for part in parts[:-1]:
        if isinstance(cur, dict):
            if part not in cur:
                return False
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except Exception:
                return False
            if idx < 0 or idx >= len(cur):
                return False
            cur = cur[idx]
        else:
            return False
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
        return True
    if isinstance(cur, list):
        try:
            idx = int(last)
        except Exception:
            return False
        if idx < 0 or idx >= len(cur):
            return False
        cur[idx] = value
        return True
    return False


def _apply_schema_defaults(schema: Any, value: Any) -> Any:
    """
    Best-effort JSON Schema defaults application.

    Supports:
    - object properties with "default"
    - nested objects/arrays

    Does not attempt to resolve oneOf/allOf/anyOf, refs, etc.
    """
    if not isinstance(schema, dict):
        return value
    t = schema.get("type")
    if t in (None, "object") and isinstance(value, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            return value
        out = dict(value)
        for k, sub in props.items():
            if not isinstance(k, str) or not k:
                continue
            if k not in out and isinstance(sub, dict) and "default" in sub:
                out[k] = sub.get("default")
            if k in out and isinstance(sub, dict):
                out[k] = _apply_schema_defaults(sub, out.get(k))
        return out
    if t == "array" and isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            return [_apply_schema_defaults(items, x) for x in value]
        return value
    return value


def _http_error_response(*, url: str, status_code: int | None, body: str | None, message: str | None) -> dict:
    out: dict[str, Any] = {"url": url}
    if status_code is not None:
        out["status_code"] = int(status_code)
    if message:
        out["message"] = str(message)
    if body:
        out["body"] = str(body)[:1200]
    return {"__http_error__": out}


_TEMPLATE_VAR_RE = re.compile(r"{{\s*([^}]+?)\s*}}")


def _extract_required_tool_args(tool: IntegrationTool) -> list[str]:
    """
    Best-effort: infer required tool args from {{args.*}} / {{params.*}} occurrences
    in URL/body/headers templates.
    """

    def scan(text: str) -> set[str]:
        if not text or "{{" not in text:
            return set()
        found: set[str] = set()
        for m in _TEMPLATE_VAR_RE.finditer(text):
            expr = (m.group(1) or "").strip()
            for prefix in ("args.", "params."):
                if not expr.startswith(prefix):
                    continue
                rest = expr[len(prefix) :].strip()
                # First segment up to '.' or '['
                key = ""
                for ch in rest:
                    if ch in ".[":
                        break
                    key += ch
                key = key.strip()
                if key:
                    found.add(key)
        return found

    keys: set[str] = set()
    keys |= scan(tool.url or "")
    keys |= scan(tool.request_body_template or "")
    keys |= scan(tool.headers_template_json or "")
    return sorted(keys)


def _parse_required_args_json(raw: str) -> list[str]:
    """
    Parse IntegrationTool required args.

    Storage is typically a JSON list (e.g. '["sql","user_id"]'), but older data may be:
    - a JSON string (e.g. '"sql"')
    - a raw comma-separated string (e.g. 'sql, user_id')
    """
    raw_s = (raw or "").strip()
    obj: Any = None
    if raw_s:
        try:
            obj = json.loads(raw_s)
        except Exception:
            obj = raw_s
    else:
        obj = []

    vals: list[str] = []
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, str):
                vals.append(v)
    elif isinstance(obj, str):
        # Allow CSV/newline formats for backwards-compat.
        vals.extend([x for x in re.split(r"[,\n]+", obj) if x is not None])
    else:
        vals = []

    out: list[str] = []
    for v in vals:
        s = str(v).strip()
        if s:
            out.append(s)
    # stable unique
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _parse_parameters_schema_json(raw: str) -> dict[str, Any] | None:
    """
    Parses a JSON-schema object for IntegrationTool.args.

    Expected: an object schema (dict) usable as the schema for the tool-call `args` field.
    """
    if not (raw or "").strip():
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    # Best-effort validation; keep permissive to support forward-compatible schemas.
    if obj.get("type") not in (None, "object"):
        return None
    return obj


def _missing_required_args(required: list[str], args: dict) -> list[str]:
    missing: list[str] = []
    for k in required:
        if k not in args:
            missing.append(k)
            continue
        v = args.get(k)
        if v is None:
            missing.append(k)
        elif isinstance(v, str) and not v.strip():
            missing.append(k)
    return missing


def _normalize_content_type_header_value(v: str) -> str:
    """
    Normalize common JSON content-types so upstream APIs that do strict matching
    (incorrectly) still parse JSON request bodies.

    Example: "Application/json" -> "application/json"
    """
    raw = (v or "").strip()
    if not raw:
        return raw
    parts = raw.split(";", 1)
    mime = parts[0].strip().lower()
    if mime != "application/json":
        return raw
    if len(parts) == 1:
        return "application/json"
    rest = parts[1].strip()
    return "application/json" + (f"; {rest}" if rest else "")


def _normalize_headers_for_json(headers: dict[str, str]) -> dict[str, str]:
    # httpx treats header names case-insensitively; normalize any provided Content-Type value.
    for k in list(headers.keys()):
        if k.lower() == "content-type":
            headers[k] = _normalize_content_type_header_value(headers.get(k) or "")
            break
    return headers


def _integration_error_user_message(*, tool_name: str, err: dict) -> str:
    sc = err.get("status_code")
    msg = (err.get("message") or "error").strip()
    body = (err.get("body") or "").strip()
    if sc == 401:
        return (
            f"The integration '{tool_name}' failed with HTTP 401 (Unauthorized). "
            "Please update the integration Authorization token and try again. "
            "What would you like to do next?"
        )
    if sc == 400:
        hint = "Bad Request"
        if body:
            hint = body
        return (
            f"The integration '{tool_name}' failed with HTTP 400 (Bad Request). "
            f"Details: {hint}. "
            "Do you want me to try a different SQL query?"
        )
    return (
        f"The integration '{tool_name}' failed with HTTP {sc} ({msg}). "
        "What would you like to do next?"
    )


def _should_followup_llm_for_tool(*, tool: IntegrationTool | None, static_rendered: str) -> bool:
    if not tool:
        return False
    # If the tool has no static template, or it rendered empty, do a follow-up LLM call
    # using the tool result stored in history.
    if not (tool.static_reply_template or "").strip():
        return True
    return not static_rendered.strip()


class ChatRequest(BaseModel):
    text: str
    speak: bool = True


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


class BotCreateRequest(BaseModel):
    name: str
    openai_model: str = "o4-mini"
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
    system_prompt: str
    language: str = "en"
    tts_language: str = "en"
    tts_vendor: str = "xtts_local"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    speaker_id: Optional[str] = None
    speaker_wav: Optional[str] = None
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    openai_tts_speed: float = 1.0
    openai_key_id: Optional[UUID] = None
    tts_split_sentences: bool = False
    tts_chunk_min_chars: int = 20
    tts_chunk_max_chars: int = 120
    start_message_mode: str = "llm"
    start_message_text: str = ""


class BotUpdateRequest(BaseModel):
    name: Optional[str] = None
    openai_model: Optional[str] = None
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
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    tts_language: Optional[str] = None
    tts_vendor: Optional[str] = None
    whisper_model: Optional[str] = None
    whisper_device: Optional[str] = None
    xtts_model: Optional[str] = None
    speaker_id: Optional[str] = None
    speaker_wav: Optional[str] = None
    openai_tts_model: Optional[str] = None
    openai_tts_voice: Optional[str] = None
    openai_tts_speed: Optional[float] = None
    openai_key_id: Optional[UUID] = None
    tts_split_sentences: Optional[bool] = None
    tts_chunk_min_chars: Optional[int] = None
    tts_chunk_max_chars: Optional[int] = None
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


def create_app() -> FastAPI:
    settings = Settings()
    engine = make_engine(settings.db_url)
    init_db(engine)

    app = FastAPI(title="Intelligravex VoiceBot Studio")
    logger = logging.getLogger("voicebot.web")
    data_agent_kickoff_locks: dict[UUID, asyncio.Lock] = {}

    download_base_url = (getattr(settings, "download_base_url", "") or "127.0.0.1:8000").strip()

    def _download_url_for_token(token: str) -> str:
        """
        Builds an absolute download URL for /api/downloads/{token}.

        Configure via VOICEBOT_DOWNLOAD_BASE_URL (supports full URL or host[:port]).
        """
        base = (download_base_url or "").strip()
        if not base:
            return f"/api/downloads/{token}"
        if not (base.startswith("http://") or base.startswith("https://")):
            base = "http://" + base
        base = base.rstrip("/")
        return f"{base}/api/downloads/{token}"

    cors_raw = (os.environ.get("VOICEBOT_CORS_ORIGINS") or "").strip()
    cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()] if cors_raw else []
    cors_origin_regex: str | None = None
    if not cors_origins:
        # Dev-friendly defaults; override via VOICEBOT_CORS_ORIGINS for production.
        cors_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
            "http://ashutosh-jha-macbook-pro:3001",
        ]
        # Also allow other dev ports on common local hostnames.
        cors_origin_regex = r"^https?://(localhost|127\.0\.0\.1|ashutosh-jha-macbook-pro)(:\d+)?$"
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
            # GPT-5 family (see OpenAI docs for naming; used by Responses API)
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
            "gpt-5.1",
            "gpt-5.1-chat-latest",
        ],
        "whisper_models": ["tiny", "base", "small", "medium", "large"],
        "whisper_devices": ["auto", "cpu", "mps", "cuda"],
        "languages": ["auto", "en", "hi", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl"],
        "xtts_models": ["tts_models/multilingual/multi-dataset/xtts_v2"],
        "openai_tts_models": ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
        "openai_tts_voices": [
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "fable",
            "nova",
            "onyx",
            "sage",
            "shimmer",
            "verse",
            # Extra voices observed in some SDK/model versions.
            "marin",
            "cedar",
        ],
    }

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Best-effort: keep the model dropdown up-to-date by periodically fetching available models
    # from the OpenAI API, if a key is configured in the environment.
    openai_models_cache: dict[str, Any] = {"ts": 0.0, "models": []}

    def get_session() -> Generator[Session, None, None]:
        with Session(engine) as s:
            yield s

    def require_crypto():
        try:
            return get_crypto_box(settings.secret_key)
        except CryptoError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/")
    def root() -> dict:
        return {"ok": True, "api_base": "/api", "public_widget_js": "/public/widget.js", "docs": "/docs"}

    def _ndjson(obj: dict) -> bytes:
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    def _wav_bytes(audio, sample_rate: int) -> bytes:
        import io

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, samplerate=sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _decode_wav_bytes_to_pcm16_16k(wav_bytes: bytes) -> bytes:
        import io

        import numpy as np
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if isinstance(data, np.ndarray) and data.ndim > 1:
            data = data[:, 0]
        audio_f32 = np.asarray(data, dtype=np.float32)
        if sr != 16000:
            # Simple linear resample to 16k (good enough for short test turns).
            ratio = 16000.0 / float(sr)
            n_out = int(round(len(audio_f32) * ratio))
            if n_out <= 0:
                return b""
            x_old = np.linspace(0.0, 1.0, num=len(audio_f32), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            audio_f32 = np.interp(x_new, x_old, audio_f32).astype(np.float32)
        audio_i16 = np.clip(audio_f32, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
        return audio_i16.tobytes()

    @lru_cache(maxsize=8)
    def _get_asr(model_name: str, device: str, language: str) -> WhisperASR:
        lang = None if (language or "").lower() in ("", "auto") else language
        return WhisperASR(model_name=model_name, device=device, language=lang)

    @lru_cache(maxsize=8)
    def _get_tts_meta(model_name: str) -> dict:
        try:
            from voicebot.compat.torch_mps import ensure_torch_mps_compat

            ensure_torch_mps_compat()
        except Exception:
            pass
        try:
            from TTS.api import TTS  # type: ignore
        except Exception:
            return {"speakers": [], "languages": []}

        tts = TTS(model_name=model_name, gpu=False, progress_bar=False)
        speakers = list(getattr(tts, "speakers", None) or [])
        languages = list(getattr(tts, "languages", None) or [])
        return {"speakers": speakers, "languages": languages}

    @lru_cache(maxsize=8)
    def _get_tts_handle(
        model_name: str,
        speaker_wav: Optional[str],
        speaker_id: Optional[str],
        use_gpu: bool,
        split_sentences: bool,
        language: str,
    ) -> Tuple[XTTSv2, threading.Lock]:
        return (
            XTTSv2(
                model_name=model_name,
                speaker_wav=speaker_wav,
                speaker_id=speaker_id,
                use_gpu=use_gpu,
                split_sentences=split_sentences,
                language=language,
            ),
            threading.Lock(),
        )

    def _tts_synthesize(tts_handle: Tuple[XTTSv2, threading.Lock], text: str):
        tts, lock = tts_handle
        with lock:
            return tts.synthesize(text)

    @lru_cache(maxsize=16)
    def _get_openai_tts_handle(
        api_key: str,
        model: str,
        voice: str,
        speed: float,
    ) -> tuple[OpenAITTS, threading.Lock]:
        return (
            OpenAITTS(api_key=api_key, model=model, voice=voice, speed=speed),
            threading.Lock(),
        )

    def _get_tts_synth_fn(bot: Bot, api_key: Optional[str]) -> Callable[[str], tuple[bytes, int]]:
        """
        Returns a thread-safe (per-handle lock) wav synthesizer for the bot's configured TTS vendor.
        """
        vendor = (getattr(bot, "tts_vendor", None) or "xtts_local").strip().lower()

        if vendor == "openai_tts":
            if not api_key:
                raise RuntimeError("No OpenAI API key configured for OpenAI TTS.")
            model = (getattr(bot, "openai_tts_model", None) or "gpt-4o-mini-tts").strip()
            voice = (getattr(bot, "openai_tts_voice", None) or "alloy").strip()
            speed_raw = getattr(bot, "openai_tts_speed", None)
            try:
                speed = float(speed_raw) if speed_raw is not None else 1.0
            except Exception:
                speed = 1.0

            tts, lock = _get_openai_tts_handle(api_key, model, voice, speed)

            def synth(text: str) -> tuple[bytes, int]:
                with lock:
                    wav = tts.synthesize_wav_bytes(text)
                return wav, OpenAITTS.DEFAULT_SAMPLE_RATE

            return synth

        # Default: local XTTS
        tts_handle = _get_tts_handle(
            bot.xtts_model,
            bot.speaker_wav,
            bot.speaker_id,
            True,
            bot.tts_split_sentences,
            bot.tts_language,
        )

        def synth(text: str) -> tuple[bytes, int]:
            a = _tts_synthesize(tts_handle, text)
            return _wav_bytes(a.audio, a.sample_rate), a.sample_rate

        return synth

    def _build_history(session: Session, bot: Bot, conversation_id: Optional[UUID]) -> list[Message]:
        def _system_prompt_with_runtime(*, prompt: str) -> str:
            ts = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            return f"Current Date Time(UTC): {ts}\n\n{prompt}"

        messages: list[Message] = [Message(role="system", content=_system_prompt_with_runtime(prompt=bot.system_prompt))]
        if not conversation_id:
            return messages
        conv = get_conversation(session, conversation_id)
        if conv.bot_id != bot.id:
            raise HTTPException(status_code=400, detail="Conversation does not belong to bot")
        meta = safe_json_loads(conv.metadata_json or "{}") or {}
        ctx = {"meta": meta}
        # Render system prompt with metadata variables (if any)
        messages = [
            Message(
                role="system",
                content=_system_prompt_with_runtime(prompt=render_template(bot.system_prompt, ctx=ctx)),
            )
        ]
        if meta:
            messages.append(Message(role="system", content=f"Conversation metadata (JSON): {json.dumps(meta, ensure_ascii=False)}"))
        for m in list_messages(session, conversation_id=conversation_id):
            if m.role in ("user", "assistant"):
                messages.append(Message(role=m.role, content=render_template(m.content, ctx=ctx)))
            elif m.role == "tool":
                # Store tool calls/results as system breadcrumbs to prevent repeated calls.
                try:
                    obj = json.loads(m.content or "")
                    if isinstance(obj, dict) and obj.get("tool") == "debug_llm_request":
                        continue
                except Exception:
                    pass
                messages.append(Message(role="system", content=render_template(f"Tool event: {m.content}", ctx=ctx)))
        return messages

    def _build_history_budgeted(
        *,
        session: Session,
        bot: Bot,
        conversation_id: Optional[UUID],
        api_key: Optional[str],
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> list[Message]:
        """
        Builds an LLM history capped to a token budget by using:
        - rolling summary stored in conversation metadata
        - a sliding window of the most recent N user turns (configurable per bot)

        If summarization runs, status_cb("summarizing") is invoked (UI-only).
        """

        HISTORY_TOKEN_BUDGET = 400000
        SUMMARY_BATCH_MIN_MESSAGES = 8

        def _system_prompt_with_runtime(*, prompt: str) -> str:
            ts = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            return f"Current Date Time(UTC): {ts}\n\n{prompt}"

        if not conversation_id:
            return [Message(role="system", content=_system_prompt_with_runtime(prompt=bot.system_prompt))]

        conv = get_conversation(session, conversation_id)
        if conv.bot_id != bot.id:
            raise HTTPException(status_code=400, detail="Conversation does not belong to bot")

        meta = safe_json_loads(conv.metadata_json or "{}") or {}
        if not isinstance(meta, dict):
            meta = {}

        memory = meta.get("memory") if isinstance(meta.get("memory"), dict) else {}
        if not isinstance(memory, dict):
            memory = {}
        memory_summary = str(memory.get("summary") or "").strip()
        pinned_facts = str(memory.get("pinned_facts") or "").strip()
        last_summarized_id = str(memory.get("last_summarized_message_id") or "").strip()

        ctx = {"meta": meta}
        system_prompt = _system_prompt_with_runtime(prompt=render_template(bot.system_prompt, ctx=ctx))

        db_msgs = list_messages(session, conversation_id=conversation_id)
        # Determine sliding window start index by last N user turns.
        try:
            n_turns = int(getattr(bot, "history_window_turns", 16) or 16)
        except Exception:
            n_turns = 16
        if n_turns < 1:
            n_turns = 1
        if n_turns > 64:
            n_turns = 64

        user_indices = [i for i, m in enumerate(db_msgs) if m.role == "user"]
        if len(user_indices) > n_turns:
            start_idx = user_indices[-n_turns]
        else:
            start_idx = 0

        def _format_for_summary(m) -> str:
            role = m.role
            content = (m.content or "").strip()
            if role == "tool":
                # Avoid huge tool payloads; keep a short breadcrumb.
                content = content[:2000]
            return f"{role.upper()}: {content}"

        # Update rolling summary for messages that will be dropped from the prompt.
        old_msgs = db_msgs[:start_idx]
        new_old_msgs = old_msgs
        if last_summarized_id:
            # Keep only messages after last summarized id.
            found = False
            tmp = []
            for m in old_msgs:
                if found:
                    tmp.append(m)
                elif str(getattr(m, "id", "") or "") == last_summarized_id:
                    found = True
            new_old_msgs = tmp if found else old_msgs

        should_summarize = False
        if old_msgs and (not memory_summary):
            should_summarize = True
        elif len(new_old_msgs) >= SUMMARY_BATCH_MIN_MESSAGES:
            should_summarize = True

        if should_summarize and api_key:
            if status_cb:
                status_cb("summarizing")
            chunk = "\n".join(_format_for_summary(m) for m in new_old_msgs)
            # Keep summarizer input bounded.
            chunk = chunk[:24000]

            summary_model = (getattr(bot, "summary_model", "") or "gpt-5-nano").strip() or "gpt-5-nano"
            summarizer = OpenAILLM(model=summary_model, api_key=api_key)
            summary_prompt = (
                "You are a conversation summarizer.\n"
                "Return STRICT JSON with keys: summary, pinned_facts, open_tasks.\n"
                "- summary: concise running summary (<= 1200 words)\n"
                "- pinned_facts: stable facts/preferences (<= 400 words)\n"
                "- open_tasks: short list (<= 12 items)\n"
                "Do not include any extra keys.\n"
            )
            prior = memory_summary or ""
            summarizer_input = (
                f"PRIOR_SUMMARY:\n{prior}\n\n"
                f"NEW_MESSAGES_TO_ABSORB:\n{chunk}\n"
            )
            text = summarizer.complete_text(
                messages=[
                    Message(role="system", content=summary_prompt),
                    Message(role="user", content=summarizer_input),
                ]
            )
            new_summary = ""
            new_pinned = ""
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    new_summary = str(obj.get("summary") or "").strip()
                    new_pinned = str(obj.get("pinned_facts") or "").strip()
            except Exception:
                # Fail closed: keep existing summary.
                new_summary = memory_summary
                new_pinned = pinned_facts

            if new_summary:
                patch = {
                    "memory.summary": new_summary,
                    "memory.pinned_facts": new_pinned,
                    "memory.last_summarized_message_id": str(getattr(new_old_msgs[-1], "id", "")) if new_old_msgs else last_summarized_id,
                    "memory.updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
                meta = merge_conversation_metadata(session, conversation_id=conversation_id, patch=patch)
                if isinstance(meta, dict):
                    memory = meta.get("memory") if isinstance(meta.get("memory"), dict) else {}
                    if isinstance(memory, dict):
                        memory_summary = str(memory.get("summary") or "").strip()
                        pinned_facts = str(memory.get("pinned_facts") or "").strip()

        # Build prompt messages: system prompt + summary + recent window.
        messages: list[Message] = [Message(role="system", content=system_prompt)]
        if memory_summary:
            messages.append(Message(role="system", content=f"Conversation summary:\n{memory_summary}"))
        if pinned_facts:
            messages.append(Message(role="system", content=f"Pinned facts:\n{pinned_facts}"))

        for m in db_msgs[start_idx:]:
            if m.role in ("user", "assistant"):
                messages.append(Message(role=m.role, content=render_template(m.content, ctx={"meta": meta})))
            elif m.role == "tool":
                try:
                    obj = json.loads(m.content or "")
                    if isinstance(obj, dict) and obj.get("tool") == "debug_llm_request":
                        continue
                except Exception:
                    pass
                # Keep full tool breadcrumbs. (Integration tools already store filtered results; truncation can
                # hide items and confuse follow-up questions.)
                messages.append(
                    Message(
                        role="system",
                        content=render_template(f"Tool event: {m.content or ''}", ctx={"meta": meta}),
                    )
                )

        # If still over budget, trim oldest messages (keeping system + last user turn).
        try:
            while estimate_messages_tokens(messages, bot.openai_model) > HISTORY_TOKEN_BUDGET and len(messages) > 4:
                # Drop the oldest non-system message after the initial system+summary blocks.
                del messages[3]
        except Exception:
            pass
        return messages

    def _get_conversation_meta(session: Session, *, conversation_id: UUID) -> dict:
        conv = get_conversation(session, conversation_id)
        meta = safe_json_loads(conv.metadata_json or "{}") or {}
        return meta if isinstance(meta, dict) else {}

    def _get_openai_api_key_for_bot(session: Session, *, bot: Bot) -> str:
        # Prefer the per-bot stored key, fall back to env.
        try:
            crypto = require_crypto()
        except Exception:
            crypto = None
        key = ""
        if crypto is not None:
            try:
                key = decrypt_openai_key(session, crypto=crypto, bot=bot) or ""
            except Exception:
                key = ""
        if not key:
            key = os.environ.get("OPENAI_API_KEY") or ""
        return (key or "").strip()

    def _data_agent_meta(meta: dict) -> dict:
        da = meta.get("data_agent")
        return da if isinstance(da, dict) else {}

    def _ensure_data_agent_container(
        session: Session, *, bot: Bot, conversation_id: UUID, meta_current: dict
    ) -> tuple[str, str, str]:
        """
        Ensures the per-conversation Data Agent runtime exists.

        Returns: (container_id, session_id, workspace_dir).
        """
        da = _data_agent_meta(meta_current)
        workspace_dir = str(da.get("workspace_dir") or "").strip() or default_workspace_dir_for_conversation(conversation_id)
        container_id = str(da.get("container_id") or "").strip()
        session_id = str(da.get("session_id") or "").strip()

        api_key = _get_openai_api_key_for_bot(session, bot=bot)
        if not api_key:
            raise RuntimeError("No OpenAI API key configured for this bot (needed for Data Agent).")

        if not container_id:
            container_id = ensure_conversation_container(
                conversation_id=conversation_id,
                workspace_dir=workspace_dir,
                openai_api_key=api_key,
            )
            meta_current = merge_conversation_metadata(
                session,
                conversation_id=conversation_id,
                patch={
                    "data_agent.container_id": container_id,
                    "data_agent.workspace_dir": workspace_dir,
                    "data_agent.session_id": session_id,
                },
            )
        return container_id, session_id, workspace_dir

    def _build_data_agent_conversation_context(session: Session, *, bot: Bot, conversation_id: UUID, meta: dict) -> dict[str, Any]:
        # Keep this small: reuse our existing summary (if any) + last N messages.
        summary = ""
        try:
            mem = meta.get("memory")
            if isinstance(mem, dict):
                summary = str(mem.get("summary") or "").strip()
        except Exception:
            summary = ""

        msgs = list_messages(session, conversation_id=conversation_id)
        n_turns = int(getattr(bot, "history_window_turns", 16) or 16)
        max_msgs = max(8, min(96, n_turns * 2))
        tail = msgs[-max_msgs:] if len(msgs) > max_msgs else msgs
        history = [{"role": m.role, "content": m.content} for m in tail]
        return {"summary": summary, "messages": history}

    async def _kickoff_data_agent_container_if_enabled(*, bot_id: UUID, conversation_id: UUID) -> None:
        """
        Best-effort: start (and optionally prewarm) the per-conversation Data Agent runtime at conversation start.

        NOTE: This uses Docker locally. For Kubernetes, this should be replaced with a Pod/Job-based runner.
        """
        lock = data_agent_kickoff_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            data_agent_kickoff_locks[conversation_id] = lock

        async with lock:
            try:
                with Session(engine) as session:
                    bot = get_bot(session, bot_id)
                    if not bool(getattr(bot, "enable_data_agent", False)):
                        return

                    prewarm = bool(getattr(bot, "data_agent_prewarm_on_start", False))
                    meta = _get_conversation_meta(session, conversation_id=conversation_id)
                    da = _data_agent_meta(meta)
                    if prewarm and (bool(da.get("ready", False)) or bool(da.get("prewarm_in_progress", False))):
                        return
                    if (not prewarm) and str(da.get("container_id") or "").strip():
                        return

                    api_key = _get_openai_api_key_for_bot(session, bot=bot)
                    if not api_key:
                        logger.warning("Data Agent kickoff: missing OpenAI key conv=%s bot=%s", conversation_id, bot_id)
                        merge_conversation_metadata(
                            session,
                            conversation_id=conversation_id,
                            patch={
                                "data_agent.init_error": "No OpenAI API key configured for this bot.",
                                "data_agent.ready": False,
                            },
                        )
                        return

                    workspace_dir = (
                        str(da.get("workspace_dir") or "").strip()
                        or default_workspace_dir_for_conversation(conversation_id)
                    )
                    container_id = str(da.get("container_id") or "").strip()
                    session_id = str(da.get("session_id") or "").strip()

                if not container_id:
                    logger.info(
                        "Data Agent kickoff: starting container conv=%s workspace=%s",
                        conversation_id,
                        workspace_dir,
                    )
                    container_id = await asyncio.to_thread(
                        ensure_conversation_container,
                        conversation_id=conversation_id,
                        workspace_dir=workspace_dir,
                        openai_api_key=api_key,
                    )
                    with Session(engine) as session:
                        merge_conversation_metadata(
                            session,
                            conversation_id=conversation_id,
                            patch={
                                "data_agent.container_id": container_id,
                                "data_agent.workspace_dir": workspace_dir,
                                "data_agent.session_id": session_id,
                            },
                        )

                if not prewarm:
                    return

                logger.info(
                    "Data Agent prewarm: begin conv=%s container_id=%s session_id=%s",
                    conversation_id,
                    container_id,
                    session_id or "",
                )
                with Session(engine) as session:
                    bot = get_bot(session, bot_id)
                    meta = _get_conversation_meta(session, conversation_id=conversation_id)
                    da = _data_agent_meta(meta)
                    if bool(da.get("ready", False)) or bool(da.get("prewarm_in_progress", False)):
                        return

                    merge_conversation_metadata(
                        session,
                        conversation_id=conversation_id,
                        patch={
                            "data_agent.ready": False,
                            "data_agent.init_error": "",
                            "data_agent.prewarm_in_progress": True,
                            "data_agent.prewarm_started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        },
                    )

                    ctx = _build_data_agent_conversation_context(
                        session,
                        bot=bot,
                        conversation_id=conversation_id,
                        meta=meta,
                    )
                    api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
                    auth_json = getattr(bot, "data_agent_auth_json", "") or "{}"
                    sys_prompt = (
                        (getattr(bot, "data_agent_system_prompt", "") or "").strip()
                        or DEFAULT_DATA_AGENT_SYSTEM_PROMPT
                    )

                init_task = (
                    "INIT / PREWARM:\n"
                    "- Open and read: api_spec.json, auth.json, conversation_context.json.\n"
                    "- Do NOT call external APIs.\n"
                    "- Output ok=true and result_text='READY'."
                )
                try:
                    res = await asyncio.to_thread(
                        run_data_agent,
                        conversation_id=conversation_id,
                        container_id=container_id,
                        session_id=session_id,
                        workspace_dir=workspace_dir,
                        api_spec_text=api_spec_text,
                        auth_json=auth_json,
                        system_prompt=sys_prompt,
                        conversation_context=ctx,
                        what_to_do=init_task,
                        timeout_s=180.0,
                    )
                    with Session(engine) as session:
                        merge_conversation_metadata(
                            session,
                            conversation_id=conversation_id,
                            patch={
                                "data_agent.prewarm_in_progress": False,
                                "data_agent.prewarm_finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "data_agent.ready": bool(res.ok),
                                "data_agent.init_error": str(res.error or ""),
                                "data_agent.session_id": str(res.session_id or ""),
                                "data_agent.container_id": str(res.container_id or container_id),
                                "data_agent.workspace_dir": workspace_dir,
                            },
                        )
                    logger.info(
                        "Data Agent prewarm: done conv=%s ok=%s ready=%s session_id=%s error=%s",
                        conversation_id,
                        bool(res.ok),
                        bool(res.ok),
                        str(res.session_id or ""),
                        str(res.error or ""),
                    )
                except Exception as exc:
                    with Session(engine) as session:
                        merge_conversation_metadata(
                            session,
                            conversation_id=conversation_id,
                            patch={
                                "data_agent.prewarm_in_progress": False,
                                "data_agent.prewarm_finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "data_agent.ready": False,
                                "data_agent.init_error": str(exc),
                            },
                        )
                    logger.info("Data Agent prewarm: failed conv=%s error=%s", conversation_id, str(exc))

            except Exception:
                logger.exception("Data Agent kickoff failed conv=%s bot=%s", conversation_id, bot_id)
                return

    def _set_metadata_tool_def() -> dict:
        return set_metadata_tool_def()

    def _set_variable_tool_def() -> dict:
        return set_variable_tool_def()

    def _web_search_tool_def() -> dict:
        return web_search_tool_def()

    def _recall_http_response_tool_def() -> dict:
        return recall_http_response_tool_def()

    def _export_http_response_tool_def() -> dict:
        return export_http_response_tool_def()

    def _disabled_tool_names(bot: Bot) -> set[str]:
        raw = (getattr(bot, "disabled_tools_json", "") or "[]").strip() or "[]"
        try:
            obj = json.loads(raw)
        except Exception:
            obj = []
        out: set[str] = set()
        if isinstance(obj, list):
            for x in obj:
                s = str(x or "").strip()
                if s:
                    out.add(s)
        # Never allow disabling set_metadata; conversations depend on it.
        out.discard("set_metadata")
        out.discard("set_variable")
        return out

    def _system_tools_defs(*, bot: Bot) -> list[dict[str, Any]]:
        # Tools that are always available for every bot (plus optional per-bot tools).
        #
        # Note: `set_variable` is kept as a runtime alias for backwards-compat, but we only expose
        # `set_metadata` to the model to avoid duplicate tools that do the same thing.
        tools = [
            _set_metadata_tool_def(),
            _web_search_tool_def(),
            _recall_http_response_tool_def(),
            _export_http_response_tool_def(),
        ]
        if bool(getattr(bot, "enable_data_agent", False)):
            tools.append(give_command_to_data_agent_tool_def())
        return tools

    def _system_tools_public_list(*, bot: Bot, disabled: set[str]) -> list[dict[str, Any]]:
        # UI-friendly list of built-in tools (do not include full JSON Schema).
        out: list[dict[str, Any]] = []
        for d in _system_tools_defs(bot=bot):
            name = str(d.get("name") or "")
            if not name:
                continue
            can_disable = name not in ("set_metadata", "set_variable")
            out.append(
                {
                    "name": name,
                    "description": str(d.get("description") or ""),
                    "enabled": name not in disabled,
                    "can_disable": can_disable,
                }
            )
        return [x for x in out if x.get("name")]

    def _integration_tool_def(t: IntegrationTool) -> dict[str, Any]:
        required_args = _parse_required_args_json(getattr(t, "args_required_json", "[]"))
        explicit_schema = _parse_parameters_schema_json(getattr(t, "parameters_schema_json", ""))
        if explicit_schema:
            args_schema = explicit_schema
        else:
            args_schema = {
                "type": "object",
                "properties": {k: {"type": "string"} for k in required_args},
                "required": required_args,
                "additionalProperties": True,
            }

        def _append_required_args_to_schema(schema: dict[str, Any], required: list[str]) -> dict[str, Any]:
            if not required:
                return schema
            props = schema.get("properties")
            if not isinstance(props, dict) and schema.get("type") in (None, "object"):
                props = {}
            if isinstance(props, dict):
                merged = dict(schema)
                merged["type"] = "object"
                merged_props = dict(props)
                for k in required:
                    merged_props.setdefault(k, {"type": "string"})
                merged["properties"] = merged_props
                req = merged.get("required")
                if not isinstance(req, list):
                    req = []
                for k in required:
                    if k not in req:
                        req.append(k)
                merged["required"] = req
                if "additionalProperties" not in merged:
                    merged["additionalProperties"] = True
                return merged
            return {
                "allOf": [
                    schema,
                    {
                        "type": "object",
                        "properties": {k: {"type": "string"} for k in required},
                        "required": required,
                        "additionalProperties": True,
                    },
                ]
            }

        # Always enforce required args (even if a custom args schema is provided).
        args_schema = _append_required_args_to_schema(args_schema, required_args)

        use_codex_response = bool(getattr(t, "use_codex_response", False))
        if use_codex_response:
            # Minimal schema extension: append intent fields so the executor model can
            # reliably understand what data is being fetched and why.
            intent_schema = {
                "type": "object",
                "properties": {
                    "fields_required": {
                        "type": "string",
                        "description": "Fields required from the HTTP response to craft the final user-facing response.",
                    },
                    "why_api_was_called": {
                        "type": "string",
                        "description": "User intent or business reason for calling this API.",
                    },
                    # Backwards-compat: accept older keys if already present in existing tool calls.
                    "what_to_search_for": {"type": "string"},
                    "why_to_search_for": {"type": "string"},
                },
                "required": ["fields_required", "why_api_was_called"],
                "additionalProperties": True,
            }

            args_schema = _append_required_args_to_schema(args_schema, list(intent_schema["required"]))
            props = args_schema.get("properties")
            if isinstance(props, dict):
                merged = dict(args_schema)
                merged_props = dict(props)
                for k, v in intent_schema["properties"].items():
                    merged_props.setdefault(k, v)
                merged["properties"] = merged_props
                args_schema = merged
            else:
                args_schema = {"allOf": [args_schema, intent_schema]}

        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "wait_reply": {
                    "type": "string",
                    "description": "Short filler message to say while the tool runs (not used for set_metadata).",
                },
                "args": {
                    "description": "Arguments used to call the integration (required).",
                    **args_schema,
                },
            },
            "additionalProperties": True,
        }
        if use_codex_response:
            schema["required"] = ["args", "wait_reply"]
        else:
            schema["properties"]["next_reply"] = {
                "type": "string",
                "description": (
                    "What the assistant should say next (no second LLM call). "
                    "Variables like {{.firstName}} are allowed."
                ),
            }
            schema["required"] = ["args", "next_reply", "wait_reply"]

        # Pagination: allow the model to optionally request fewer items than the API page size.
        try:
            pag = safe_json_loads(getattr(t, "pagination_json", "") or "") or None
        except Exception:
            pag = None
        if isinstance(pag, dict) and str(pag.get("items_path") or "").strip():
            max_items_cap = pag.get("max_items_cap")
            try:
                max_items_cap_i = int(max_items_cap) if max_items_cap is not None else 5000
            except Exception:
                max_items_cap_i = 5000
            if max_items_cap_i < 1:
                max_items_cap_i = 5000
            if max_items_cap_i > 50000:
                max_items_cap_i = 50000
            props = schema.get("properties")
            if isinstance(props, dict):
                args_props = props.get("args")
                if isinstance(args_props, dict):
                    # args_props is a schema object; add `max_items` as an optional arg.
                    args_props.setdefault("properties", {})
                    if isinstance(args_props.get("properties"), dict):
                        args_props["properties"].setdefault(
                            "max_items",
                            {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": max_items_cap_i,
                                "description": (
                                    f"Optional: stop pagination after collecting this many items (max {max_items_cap_i}). "
                                    "Backend will fetch multiple pages if needed."
                                ),
                            },
                        )
        return {
            "type": "function",
            "name": t.name,
            "description": (t.description or "").strip()
            + " This tool calls an external HTTP API and maps selected response fields into conversation metadata. "
            + (
                "Return your spoken/text response in next_reply (you can use metadata variables like {{.firstName}})."
                if not use_codex_response
                else "If Codex mode is enabled, the backend runs a Codex agent to generate a result string, and the main chat model will rephrase it."
            ),
            "parameters": schema,
            "strict": False,
        }

    def _build_tools_for_bot(session: Session, bot_id: UUID) -> list[dict[str, Any]]:
        bot = get_bot(session, bot_id)
        disabled = _disabled_tool_names(bot)
        tools: list[dict[str, Any]] = [
            d for d in _system_tools_defs(bot=bot) if str(d.get("name") or "") not in disabled
        ]
        for t in list_integration_tools(session, bot_id=bot_id):
            if not bool(getattr(t, "enabled", True)):
                continue
            if str(getattr(t, "name", "") or "").strip() in disabled:
                continue
            try:
                tools.append(_integration_tool_def(t))
            except Exception:
                continue
        return tools

    @lru_cache(maxsize=1)
    def _get_openai_pricing() -> dict[str, ModelPrice]:
        raw = os.environ.get("OPENAI_PRICING_JSON") or ""
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except Exception:
            return {}
        out: dict[str, ModelPrice] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                try:
                    input_per_1m = v.get("input_per_1m")
                    output_per_1m = v.get("output_per_1m")
                    if input_per_1m is None or output_per_1m is None:
                        continue
                    out[k] = ModelPrice(input_per_1m=float(input_per_1m), output_per_1m=float(output_per_1m))
                except Exception:
                    continue
        return out

    def _estimate_llm_cost_for_turn(*, bot: Bot, history: list[Message], assistant_text: str) -> tuple[int, int, float]:
        prompt_tokens = estimate_messages_tokens(history, bot.openai_model)
        output_tokens = estimate_text_tokens(assistant_text, bot.openai_model)
        price = _get_openai_pricing().get(bot.openai_model)
        cost = estimate_cost_usd(model_price=price, input_tokens=prompt_tokens, output_tokens=output_tokens)
        return prompt_tokens, output_tokens, cost

    def _make_start_message_instruction(bot: Bot) -> str:
        # Keep this short and safe; system_prompt should drive tone/language.
        return (
            "Generate a short opening message to start a voice conversation. "
            "Keep it concise and end with a question."
        )

    async def _init_conversation_and_greet(
        *,
        bot_id: UUID,
        speak: bool,
        test_flag: bool,
        ws: WebSocket,
        req_id: str,
        debug: bool,
    ) -> UUID:
        init_start = time.time()
        # Create conversation + store first assistant message.
        with Session(engine) as session:
            bot = get_bot(session, bot_id)
            conv = create_conversation(session, bot_id=bot.id, test_flag=test_flag)
            conv_id = conv.id

            greeting_text = (bot.start_message_text or "").strip()
            llm_ttfb_ms: Optional[int] = None
            llm_total_ms: Optional[int] = None
            input_tokens_est: Optional[int] = None
            output_tokens_est: Optional[int] = None
            cost_usd_est: Optional[float] = None
            sent_greeting_delta = False
            api_key: Optional[str] = None

            needs_openai_key = not (bot.start_message_mode == "static" and greeting_text) or (
                speak and (bot.tts_vendor or "xtts_local").strip().lower() == "openai_tts"
            )
            if needs_openai_key:
                if bot.openai_key_id:
                    crypto = require_crypto()
                    api_key = decrypt_openai_key(session, crypto=crypto, bot=bot)
                else:
                    api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot.")

            if bot.start_message_mode == "static" and greeting_text:
                # Static greeting (no LLM).
                pass
            else:
                llm = OpenAILLM(model=bot.openai_model, api_key=api_key)
                ts = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
                sys_prompt = f"Current Date Time(UTC): {ts}\n\n{render_template(bot.system_prompt, ctx={'meta': {}})}"
                msgs = [
                    Message(role="system", content=sys_prompt),
                    Message(role="user", content=_make_start_message_instruction(bot)),
                ]
                if debug:
                    await _emit_llm_debug_payload(
                        ws=ws,
                        req_id=req_id,
                        conversation_id=conv_id,
                        phase="greeting_llm",
                        payload=llm.build_request_payload(messages=msgs, stream=True),
                    )
                t0 = time.time()
                first = None
                parts: list[str] = []
                for d in llm.stream_text(messages=msgs):
                    if first is None:
                        first = time.time()
                    parts.append(d)
                    await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
                    sent_greeting_delta = True
                t1 = time.time()
                greeting_text = "".join(parts).strip()
                if first is not None:
                    llm_ttfb_ms = int(round((first - t0) * 1000.0))
                llm_total_ms = int(round((t1 - t0) * 1000.0))

                # Estimate cost for this greeting turn.
                input_tokens_est = estimate_messages_tokens(msgs, bot.openai_model)
                output_tokens_est = estimate_text_tokens(greeting_text, bot.openai_model)
                price = _get_openai_pricing().get(bot.openai_model)
                cost_usd_est = estimate_cost_usd(
                    model_price=price, input_tokens=input_tokens_est, output_tokens=output_tokens_est
                )

            if not greeting_text:
                greeting_text = "Hi! How can I help you today?"

            # If this was static (or LLM produced no streamed deltas), still send text to UI.
            if not sent_greeting_delta:
                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": greeting_text})

            # Store assistant greeting as first message.
            add_message_with_metrics(
                session,
                conversation_id=conv_id,
                role="assistant",
                content=greeting_text,
                input_tokens_est=input_tokens_est,
                output_tokens_est=output_tokens_est,
                cost_usd_est=cost_usd_est,
                llm_ttfb_ms=llm_ttfb_ms,
                llm_total_ms=llm_total_ms,
            )
            if input_tokens_est is not None or output_tokens_est is not None or cost_usd_est is not None:
                update_conversation_metrics(
                    session,
                    conversation_id=conv_id,
                    add_input_tokens_est=int(input_tokens_est or 0),
                    add_output_tokens_est=int(output_tokens_est or 0),
                    add_cost_usd_est=float(cost_usd_est or 0.0),
                    last_asr_ms=None,
                    last_llm_ttfb_ms=llm_ttfb_ms,
                    last_llm_total_ms=llm_total_ms,
                    last_tts_first_audio_ms=None,
                    last_total_ms=None,
                )

            if speak:
                tts_synth = await asyncio.to_thread(_get_tts_synth_fn, bot, api_key)
                # Synthesize whole greeting as one chunk for now.
                wav, sr = await asyncio.to_thread(tts_synth, greeting_text)
                await _ws_send_json(
                    ws,
                    {
                        "type": "audio_wav",
                        "req_id": req_id,
                        "wav_base64": base64.b64encode(wav).decode(),
                        "sr": sr,
                    },
                )

        timings: dict[str, int] = {"total": int(round((time.time() - init_start) * 1000.0))}
        if llm_ttfb_ms is not None:
            timings["llm_ttfb"] = llm_ttfb_ms
        if llm_total_ms is not None:
            timings["llm_total"] = llm_total_ms
        await _ws_send_json(ws, {"type": "metrics", "req_id": req_id, "timings_ms": timings})
        return conv_id

    def _apply_response_mapper(
        *,
        mapper_json: str,
        response_json: Any,
        meta: dict,
        tool_args: dict,
    ) -> dict:
        mapper = safe_json_loads(mapper_json or "{}") or {}
        if not isinstance(mapper, dict):
            return {}
        out: dict = {}
        ctx = {"response": response_json, "meta": meta, "args": tool_args, "params": tool_args}
        for k, tmpl in mapper.items():
            if not isinstance(k, str):
                continue
            if isinstance(tmpl, (dict, list)):
                out[k] = tmpl
                continue
            if tmpl is None:
                out[k] = None
                continue
            out[k] = eval_template_value(str(tmpl), ctx=ctx)
        return out

    def _render_with_meta(text: str, meta: dict) -> str:
        return render_template(text, ctx={"meta": meta})

    def _render_static_reply(
        *,
        template_text: str,
        meta: dict,
        response_json: Any,
        tool_args: dict,
    ) -> str:
        return render_jinja_template(
            template_text,
            ctx={"meta": meta, "response": response_json, "args": tool_args, "params": tool_args},
        )

    def _execute_integration_http(
        *,
        tool: IntegrationTool,
        meta: dict,
        tool_args: dict,
    ) -> dict:
        pagination_raw = (getattr(tool, "pagination_json", "") or "").strip()
        pagination_cfg: dict[str, Any] | None = None
        pagination_cfg_error: str | None = None
        if pagination_raw:
            try:
                obj = json.loads(pagination_raw)
                if isinstance(obj, dict):
                    pagination_cfg = obj
                else:
                    pagination_cfg_error = "pagination_json must be a JSON object."
            except Exception as exc:
                pagination_cfg_error = f"invalid pagination_json: {exc}"

        # Apply JSON Schema defaults to tool args (helps URL/body templates that reference args.page/args.limit).
        args0 = dict(tool_args or {})
        try:
            schema_obj = json.loads(getattr(tool, "parameters_schema_json", "") or "null")
        except Exception:
            schema_obj = None
        if isinstance(schema_obj, dict):
            args0 = _apply_schema_defaults(schema_obj, args0)
        tool_args = args0

        required_args = _parse_required_args_json(getattr(tool, "args_required_json", "[]"))
        missing = _missing_required_args(required_args, tool_args or {})
        if bool(getattr(tool, "use_codex_response", False)):
            args0 = tool_args or {}
            # Prefer new keys, but accept old ones if present.
            if not str(args0.get("fields_required") or "").strip() and not str(args0.get("what_to_search_for") or "").strip():
                missing.append("fields_required")
            if not str(args0.get("why_api_was_called") or "").strip() and not str(args0.get("why_to_search_for") or "").strip():
                missing.append("why_api_was_called")
        if missing:
            return {
                "__tool_args_error__": {
                    "missing": sorted(set(missing)),
                    "message": f"Missing required tool args: {', '.join(sorted(set(missing)))}",
                }
            }

        def _single_request(*, loop_args: dict[str, Any]) -> tuple[Any, str]:
            # Render URL/body templates using current metadata + tool args.
            # Include env so integrations can reference server-provided secrets without storing them in metadata.
            ctx = {"meta": meta, "args": loop_args, "params": loop_args, "env": dict(os.environ)}
            url = render_template(tool.url, ctx=ctx)
            method = (tool.method or "GET").upper()

            headers_obj: dict[str, str] = {}
            headers_template = tool.headers_template_json or ""
            if headers_template.strip():
                rendered_headers = render_template(headers_template, ctx=ctx)
                try:
                    h = json.loads(rendered_headers)
                    if isinstance(h, dict):
                        for k, v in h.items():
                            if isinstance(k, str) and isinstance(v, str) and k.strip():
                                headers_obj[k] = v
                except Exception:
                    headers_obj = {}
            headers_obj = _normalize_headers_for_json(headers_obj)

            body_template = tool.request_body_template or ""
            body_obj = None
            if body_template.strip():
                rendered_body = render_template(body_template, ctx=ctx)
                try:
                    body_obj = json.loads(rendered_body)
                except Exception:
                    # If body isn't valid JSON, send as raw string.
                    body_obj = rendered_body

            loop_request_args = dict(loop_args)
            # Internal intent keys (LLM-only); never forward to the upstream HTTP API.
            for k in ("fields_required", "why_api_was_called", "what_to_search_for", "why_to_search_for", "max_items"):
                loop_request_args.pop(k, None)

            timeout = httpx.Timeout(60.0, connect=20.0)
            try:
                with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                    if method == "GET":
                        resp = client.request(method, url, headers=headers_obj or None)
                    else:
                        # Prefer JSON for objects/lists; otherwise raw data.
                        if isinstance(body_obj, (dict, list)):
                            # If pagination is enabled and the body is a dict, the caller may have inserted
                            # page/limit keys into loop_args and referenced them in the template; we keep
                            # the rendered body as the source of truth.
                            resp = client.request(method, url, json=body_obj, headers=headers_obj or None)
                        elif body_obj is None:
                            # Most APIs expect an object for JSON bodies. Send {} instead of null when args are empty.
                            resp = client.request(
                                method, url, json=(loop_request_args or {}), headers=headers_obj or None
                            )
                        else:
                            resp = client.request(method, url, content=str(body_obj), headers=headers_obj or None)
                if resp.status_code >= 400:
                    return (
                        _http_error_response(
                            url=str(resp.request.url),
                            status_code=resp.status_code,
                            body=(resp.text or None),
                            message=resp.reason_phrase,
                        ),
                        url,
                    )
                try:
                    return resp.json(), url
                except Exception:
                    return {"raw": resp.text}, url
            except httpx.RequestError as exc:
                return _http_error_response(url=url, status_code=None, body=None, message=str(exc)), url

        # If pagination is not configured, do a single request.
        if not isinstance(pagination_cfg, dict) or not pagination_cfg:
            resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
            if pagination_cfg_error and isinstance(resp_json, dict) and "__http_error__" not in resp_json:
                resp_json["__igx_pagination__"] = {"error": pagination_cfg_error}
            return resp_json

        mode = str(pagination_cfg.get("mode") or "page_limit").strip()
        if mode not in ("page_limit", "offset_limit"):
            resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
            if isinstance(resp_json, dict) and "__http_error__" not in resp_json:
                resp_json["__igx_pagination__"] = {"error": f"unsupported pagination mode: {mode}"}
            return resp_json

        items_path = str(pagination_cfg.get("items_path") or "").strip()
        if not items_path:
            resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
            if isinstance(resp_json, dict) and "__http_error__" not in resp_json:
                resp_json["__igx_pagination__"] = {"error": "pagination_json missing items_path."}
            return resp_json

        page_arg = str(pagination_cfg.get("page_arg") or "page").strip() or "page"
        limit_arg = str(pagination_cfg.get("limit_arg") or "limit").strip() or "limit"
        offset_arg = str(pagination_cfg.get("offset_arg") or "offset").strip() or "offset"
        max_pages = int(pagination_cfg.get("max_pages") or 5)
        if max_pages < 1:
            max_pages = 1
        if max_pages > 50:
            max_pages = 50

        # max_items is optional; if missing, we still cap total work via max_pages.
        max_items_cap = int(pagination_cfg.get("max_items_cap") or 5000)
        if max_items_cap < 1:
            max_items_cap = 5000
        if max_items_cap > 50000:
            max_items_cap = 50000

        requested_max_items = tool_args.get("max_items")
        if requested_max_items is None:
            requested_max_items = pagination_cfg.get("max_items_default")
        try:
            max_items = int(requested_max_items) if requested_max_items is not None else None
        except Exception:
            max_items = None
        if max_items is not None:
            if max_items < 1:
                max_items = None
            else:
                max_items = min(max_items, max_items_cap)

        def _read_int(v: Any, default: int) -> int:
            try:
                x = int(v)
                return x
            except Exception:
                return default

        limit_val = _read_int(tool_args.get(limit_arg), int(pagination_cfg.get("limit_default") or 100))
        if limit_val < 1:
            limit_val = 100

        start_page = _read_int(tool_args.get(page_arg), 1)
        if start_page < 1:
            start_page = 1

        start_offset = _read_int(tool_args.get(offset_arg), 0)
        if start_offset < 0:
            start_offset = 0

        base_resp: Any = None
        aggregated: list[Any] = []
        fetched = 0

        for i in range(max_pages):
            loop_args = dict(tool_args or {})
            loop_args[limit_arg] = limit_val
            if mode == "page_limit":
                loop_args[page_arg] = start_page + i
            else:
                loop_args[offset_arg] = start_offset + (i * limit_val)

            resp_json, _url = _single_request(loop_args=loop_args)
            # Pass-through errors / non-JSON bodies immediately.
            if isinstance(resp_json, dict) and resp_json.get("__http_error__"):
                return resp_json

            if base_resp is None:
                # Shallow copy so we can mutate the items list in-place without affecting downstream.
                base_resp = resp_json if not isinstance(resp_json, dict) else dict(resp_json)

            page_items = _get_json_path(resp_json, items_path)
            if not isinstance(page_items, list):
                # If we can't find a list at items_path, return first page + diagnostics.
                if isinstance(base_resp, dict):
                    base_resp["__igx_pagination__"] = {
                        "mode": mode,
                        "items_path": items_path,
                        "limit": limit_val,
                        "pages_fetched": fetched,
                        "items_returned": len(aggregated),
                        "max_items": max_items,
                        "max_pages": max_pages,
                        "error": f"items_path not a list: {items_path}",
                    }
                return base_resp

            fetched += 1
            aggregated.extend(page_items)

            if max_items is not None and len(aggregated) >= max_items:
                aggregated = aggregated[:max_items]
                break

            # Stop if the API returned fewer than the page size (common "last page" signal).
            if len(page_items) < limit_val:
                break

        if base_resp is None:
            resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
            return resp_json

        # Replace the items list with the aggregated list (if possible).
        if not _set_json_path(base_resp, items_path, aggregated):
            # If we can't set, just return the first page response.
            if isinstance(base_resp, dict):
                base_resp["__igx_pagination__"] = {
                    "mode": mode,
                    "items_path": items_path,
                    "limit": limit_val,
                    "pages_fetched": fetched,
                    "items_returned": len(aggregated),
                    "max_items": max_items,
                    "max_pages": max_pages,
                    "error": f"failed to set items_path: {items_path}",
                }
            return base_resp

        # Minimal pagination diagnostics (kept out of the merged items list; caller can surface it in tool_result).
        if isinstance(base_resp, dict):
            base_resp["__igx_pagination__"] = {
                "mode": mode,
                "items_path": items_path,
                "limit": limit_val,
                "pages_fetched": fetched,
                "items_returned": len(aggregated),
                "max_items": max_items,
                "max_pages": max_pages,
            }

        return base_resp

    async def _ws_send_json(ws: WebSocket, obj: dict) -> None:
        await ws.send_text(json.dumps(obj, ensure_ascii=False))

    async def _stream_llm_reply(
        *,
        ws: WebSocket,
        req_id: str,
        llm: OpenAILLM,
        messages: list[Message],
    ) -> tuple[str, Optional[int], int]:
        t0 = time.time()
        first: Optional[float] = None
        parts: list[str] = []
        for d in llm.stream_text(messages=messages):
            if first is None:
                first = time.time()
            if d:
                parts.append(d)
                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
        t1 = time.time()
        text = "".join(parts).strip()
        ttfb_ms = int(round((first - t0) * 1000.0)) if first is not None else None
        total_ms = int(round((t1 - t0) * 1000.0))
        return text, ttfb_ms, total_ms

    def _estimate_wav_seconds(wav_bytes: bytes, sr: int) -> float:
        # Best-effort WAV duration extraction to avoid interrupting ongoing speech.
        # If parsing fails, fall back to a heuristic based on byte size.
        try:
            if len(wav_bytes) < 44 or sr <= 0:
                raise ValueError("bad wav")
            if wav_bytes[0:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
                raise ValueError("not wav")
            # Parse chunks.
            i = 12
            channels = 1
            bits_per_sample = 16
            data_size = None
            while i + 8 <= len(wav_bytes):
                cid = wav_bytes[i : i + 4]
                size = int.from_bytes(wav_bytes[i + 4 : i + 8], "little", signed=False)
                i += 8
                if cid == b"fmt " and i + 16 <= len(wav_bytes):
                    channels = int.from_bytes(wav_bytes[i + 2 : i + 4], "little", signed=False) or 1
                    bits_per_sample = int.from_bytes(wav_bytes[i + 14 : i + 16], "little", signed=False) or 16
                if cid == b"data":
                    data_size = size
                    break
                i += size + (size % 2)
            if data_size is None:
                data_size = max(0, len(wav_bytes) - 44)
            bytes_per_frame = max(1, int(channels) * max(1, int(bits_per_sample) // 8))
            frames = float(data_size) / float(bytes_per_frame)
            return max(0.0, frames / float(sr))
        except Exception:
            return max(0.5, min(12.0, float(len(wav_bytes)) / float(max(1, sr * 2))))

    def _record_llm_debug_payload(
        *,
        conversation_id: UUID,
        payload: dict[str, Any],
        phase: str,
    ) -> None:
        try:
            with Session(engine) as session:
                add_message_with_metrics(
                    session,
                    conversation_id=conversation_id,
                    role="tool",
                    content=json.dumps(
                        {"tool": "debug_llm_request", "arguments": {"phase": phase, "payload": payload}},
                        ensure_ascii=False,
                    ),
                )
        except Exception:
            # Debugging must never break the conversation.
            return

    async def _emit_llm_debug_payload(
        *,
        ws: WebSocket,
        req_id: str,
        conversation_id: UUID,
        payload: dict[str, Any],
        phase: str,
    ) -> None:
        # Best-effort: send to UI and persist to DB.
        try:
            await _ws_send_json(
                ws,
                {
                    "type": "tool_call",
                    "req_id": req_id,
                    "name": "debug_llm_request",
                    "arguments_json": json.dumps({"phase": phase, "payload": payload}, ensure_ascii=False),
                },
            )
        except Exception:
            pass
        _record_llm_debug_payload(conversation_id=conversation_id, payload=payload, phase=phase)

    @app.websocket("/ws/bots/{bot_id}/talk")
    async def talk_ws(bot_id: UUID, ws: WebSocket) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        await ws.accept()
        loop = asyncio.get_running_loop()

        def status(req_id: str, stage: str) -> None:
            asyncio.run_coroutine_threadsafe(
                _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": stage}), loop
            )

        active_req_id: Optional[str] = None
        audio_buf = bytearray()
        conv_id: Optional[UUID] = None
        speak = True
        test_flag = True
        debug_mode = False
        stop_ts: Optional[float] = None
        accepting_audio = False

        try:
            while True:
                msg = await ws.receive()
                if "text" in msg and msg["text"] is not None:
                    try:
                        payload = json.loads(msg["text"])
                    except Exception:
                        await _ws_send_json(ws, {"type": "error", "error": "Invalid JSON"})
                        continue

                    msg_type = payload.get("type")
                    req_id = str(payload.get("req_id") or "")
                    if msg_type == "init":
                        if not req_id:
                            await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                            continue
                        if active_req_id is not None:
                            await _ws_send_json(
                                ws,
                                {
                                    "type": "error",
                                    "req_id": req_id,
                                    "error": "Another request is already in progress",
                                },
                            )
                            continue
                        active_req_id = req_id
                        speak = bool(payload.get("speak", True))
                        test_flag = bool(payload.get("test_flag", True))
                        debug_mode = bool(payload.get("debug", False))
                        accepting_audio = False
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "init"})
                        try:
                            conv_id = await _init_conversation_and_greet(
                                bot_id=bot_id,
                                speak=speak,
                                test_flag=test_flag,
                                ws=ws,
                                req_id=req_id,
                                debug=debug_mode,
                            )
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue
                        await _ws_send_json(
                            ws,
                            {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)},
                        )
                        await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        accepting_audio = False
                        continue

                    if msg_type == "start":
                        if not req_id:
                            await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                            continue
                        if active_req_id is not None:
                            await _ws_send_json(
                                ws,
                                {
                                    "type": "error",
                                    "req_id": req_id,
                                    "error": "Another request is already in progress",
                                },
                            )
                            continue

                        active_req_id = req_id
                        audio_buf = bytearray()
                        debug_mode = bool(payload.get("debug", False))
                        speak = bool(payload.get("speak", True))
                        test_flag = bool(payload.get("test_flag", True))
                        accepting_audio = True

                        conversation_id_str = str(payload.get("conversation_id") or "").strip()

                        try:
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                if conversation_id_str:
                                    conv_id = UUID(conversation_id_str)
                                    conv = get_conversation(session, conv_id)
                                    if conv.bot_id != bot.id:
                                        raise HTTPException(
                                            status_code=400, detail="Conversation does not belong to bot"
                                        )
                                else:
                                    conv = create_conversation(session, bot_id=bot.id, test_flag=test_flag)
                                    conv_id = conv.id
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            active_req_id = None
                            conv_id = None
                            continue

                        await _ws_send_json(
                            ws,
                            {
                                "type": "conversation",
                                "req_id": req_id,
                                "conversation_id": str(conv_id),
                                "id": str(conv_id),
                            },
                        )
                        asyncio.create_task(
                            _kickoff_data_agent_container_if_enabled(bot_id=bot_id, conversation_id=conv_id)
                        )
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "recording"})

                    elif msg_type == "chat":
                        # Text-only chat turn (for when Speak is disabled).
                        if not req_id:
                            await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                            continue
                        if active_req_id is not None:
                            await _ws_send_json(
                                ws,
                                {
                                    "type": "error",
                                    "req_id": req_id,
                                    "error": "Another request is already in progress",
                                },
                            )
                            continue

                        active_req_id = req_id
                        speak = bool(payload.get("speak", True))
                        test_flag = bool(payload.get("test_flag", True))
                        debug_mode = bool(payload.get("debug", False))
                        user_text = str(payload.get("text") or "").strip()
                        conversation_id_str = str(payload.get("conversation_id") or "").strip()
                        accepting_audio = False
                        if not user_text:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": "Empty text"})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            continue

                        try:
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                if conversation_id_str:
                                    conv_id = UUID(conversation_id_str)
                                    conv = get_conversation(session, conv_id)
                                    if conv.bot_id != bot.id:
                                        raise HTTPException(status_code=400, detail="Conversation does not belong to bot")
                                else:
                                    conv = create_conversation(session, bot_id=bot.id, test_flag=test_flag)
                                    conv_id = conv.id
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue

                        await _ws_send_json(
                            ws,
                            {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)},
                        )
                        asyncio.create_task(_kickoff_data_agent_container_if_enabled(bot_id=bot_id, conversation_id=conv_id))

                        try:
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                if bot.openai_key_id:
                                    crypto = require_crypto()
                                    api_key = decrypt_openai_key(session, crypto=crypto, bot=bot)
                                else:
                                    api_key = os.environ.get("OPENAI_API_KEY")
                                if not api_key:
                                    raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot.")

                                add_message_with_metrics(
                                    session,
                                    conversation_id=conv_id,
                                    role="user",
                                    content=user_text,
                                )
                                loop = asyncio.get_running_loop()

                                def _status_cb(stage: str) -> None:
                                    asyncio.run_coroutine_threadsafe(
                                        _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": stage}), loop
                                    )

                                history = _build_history_budgeted(
                                    session=session,
                                    bot=bot,
                                    conversation_id=conv_id,
                                    api_key=api_key,
                                    status_cb=_status_cb,
                                )
                                tools_defs = _build_tools_for_bot(session, bot.id)
                                llm = OpenAILLM(model=bot.openai_model, api_key=api_key)
                                tts_synth = await asyncio.to_thread(_get_tts_synth_fn, bot, api_key)
                                if debug_mode:
                                    await _emit_llm_debug_payload(
                                        ws=ws,
                                        req_id=req_id,
                                        conversation_id=conv_id,
                                        phase="chat_llm",
                                        payload=llm.build_request_payload(
                                            messages=history, tools=tools_defs, stream=True
                                        ),
                                    )
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                        llm_start_ts = time.time()

                        delta_q_client: "queue.Queue[Optional[str]]" = queue.Queue()
                        delta_q_tts: "queue.Queue[Optional[str]]" = queue.Queue()
                        audio_q: "queue.Queue[Optional[tuple[bytes, int]]]" = queue.Queue()
                        tool_calls: list[ToolCall] = []
                        error_q: "queue.Queue[Optional[str]]" = queue.Queue()
                        full_text_parts: list[str] = []
                        metrics_lock = threading.Lock()
                        first_token_ts: Optional[float] = None
                        tts_start_ts: Optional[float] = None
                        first_audio_ts: Optional[float] = None

                        def llm_thread() -> None:
                            try:
                                for ev in llm.stream_text_or_tool(messages=history, tools=tools_defs):
                                    if isinstance(ev, ToolCall):
                                        tool_calls.append(ev)
                                        continue
                                    d = ev
                                    full_text_parts.append(d)
                                    delta_q_client.put(d)
                                    if speak:
                                        delta_q_tts.put(d)
                            except Exception as exc:
                                error_q.put(str(exc))
                            finally:
                                delta_q_client.put(None)
                                if speak:
                                    delta_q_tts.put(None)

                        def tts_thread() -> None:
                            nonlocal tts_start_ts
                            if not speak:
                                audio_q.put(None)
                                return
                            try:
                                parts: list[str] = []
                                while True:
                                    d = delta_q_tts.get()
                                    if d is None:
                                        break
                                    if d:
                                        parts.append(d)
                                text_to_speak = "".join(parts).strip()
                                if text_to_speak:
                                    with metrics_lock:
                                        if tts_start_ts is None:
                                            tts_start_ts = time.time()
                                    status(req_id, "tts")
                                    wav, sr = tts_synth(text_to_speak)
                                    audio_q.put((wav, sr))
                            except Exception as exc:
                                error_q.put(f"TTS failed: {exc}")
                            finally:
                                audio_q.put(None)

                        t1 = threading.Thread(target=llm_thread, daemon=True)
                        t2 = threading.Thread(target=tts_thread, daemon=True)
                        t1.start()
                        t2.start()

                        open_deltas = True
                        open_audio = True
                        while open_deltas or open_audio:
                            try:
                                err = error_q.get_nowait()
                                if err:
                                    await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": err})
                                    open_deltas = False
                                    open_audio = False
                                    break
                            except queue.Empty:
                                pass

                            try:
                                d = delta_q_client.get_nowait()
                                if d is None:
                                    open_deltas = False
                                else:
                                    if first_token_ts is None:
                                        first_token_ts = time.time()
                                    await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
                            except queue.Empty:
                                pass

                            if speak:
                                try:
                                    item = audio_q.get_nowait()
                                    if item is None:
                                        open_audio = False
                                    else:
                                        wav, sr = item
                                        if first_audio_ts is None:
                                            first_audio_ts = time.time()
                                        await _ws_send_json(
                                            ws,
                                            {
                                                "type": "audio_wav",
                                                "req_id": req_id,
                                                "wav_base64": base64.b64encode(wav).decode(),
                                                "sr": sr,
                                            },
                                        )
                                except queue.Empty:
                                    pass
                            else:
                                open_audio = False

                            if (open_deltas or open_audio) and first_token_ts is None:
                                time.sleep(0.01)
                            else:
                                time.sleep(0.005)

                        t1.join()
                        t2.join()

                        llm_end_ts = time.time()
                        final_text = "".join(full_text_parts).strip()

                        timings: dict[str, int] = {"total": int(round((llm_end_ts - llm_start_ts) * 1000.0))}
                        if first_token_ts is not None:
                            timings["llm_ttfb"] = int(round((first_token_ts - llm_start_ts) * 1000.0))
                        elif tool_calls and tool_calls[0].first_event_ts is not None:
                            timings["llm_ttfb"] = int(round((tool_calls[0].first_event_ts - llm_start_ts) * 1000.0))
                        timings["llm_total"] = int(round((llm_end_ts - llm_start_ts) * 1000.0))
                        if first_audio_ts is not None and tts_start_ts is not None:
                            timings["tts_first_audio"] = int(round((first_audio_ts - tts_start_ts) * 1000.0))

                        await _ws_send_json(ws, {"type": "metrics", "req_id": req_id, "timings_ms": timings})

                        if tool_calls and conv_id is not None:
                            rendered_reply = ""
                            tool_error: Optional[str] = None
                            needs_followup_llm = False
                            tool_failed = False
                            followup_streamed = False
                            followup_persisted = False
                            tts_busy_until: float = 0.0

                            async def _send_interim(text: str, *, kind: str) -> None:
                                nonlocal tts_busy_until
                                t = (text or "").strip()
                                if not t:
                                    return
                                await _ws_send_json(
                                    ws,
                                    {"type": "interim", "req_id": req_id, "kind": kind, "text": t},
                                )
                                if not speak:
                                    return
                                now = time.time()
                                if now < tts_busy_until:
                                    await asyncio.sleep(tts_busy_until - now)
                                status(req_id, "tts")
                                try:
                                    wav, sr = await asyncio.to_thread(tts_synth, t)
                                    tts_busy_until = time.time() + _estimate_wav_seconds(wav, sr) + 0.15
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "audio_wav",
                                            "req_id": req_id,
                                            "wav_base64": base64.b64encode(wav).decode(),
                                            "sr": sr,
                                        },
                                    )
                                except Exception:
                                    return
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                meta_current = _get_conversation_meta(session, conversation_id=conv_id)
                                disabled_tools = _disabled_tool_names(bot)

                                for tc in tool_calls:
                                    tool_name = tc.name
                                    if tool_name == "set_variable":
                                        tool_name = "set_metadata"

                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "tool_call",
                                            "req_id": req_id,
                                            "name": tool_name,
                                            "arguments_json": tc.arguments_json,
                                        },
                                    )
                                    try:
                                        tool_args = json.loads(tc.arguments_json or "{}")
                                        if not isinstance(tool_args, dict):
                                            raise ValueError("tool args must be an object")
                                    except Exception as exc:
                                        tool_error = str(exc)
                                        break

                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="tool",
                                        content=json.dumps({"tool": tool_name, "arguments": tool_args}, ensure_ascii=False),
                                    )

                                    next_reply = str(tool_args.get("next_reply") or "").strip()
                                    wait_reply = str(tool_args.get("wait_reply") or "").strip() or "Working on it…"
                                    raw_args = tool_args.get("args")
                                    if isinstance(raw_args, dict):
                                        patch = dict(raw_args)
                                    else:
                                        patch = dict(tool_args)
                                        patch.pop("next_reply", None)
                                        patch.pop("wait_reply", None)
                                        patch.pop("args", None)

                                    tool_cfg: IntegrationTool | None = None
                                    response_json: Any | None = None

                                    if tool_name in disabled_tools:
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": f"Tool '{tool_name}' is disabled for this bot."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        rendered_reply = ""
                                    elif tool_name == "set_metadata":
                                        new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=patch)
                                        tool_result = {"ok": True, "updated": patch, "metadata": new_meta}
                                    elif tool_name == "web_search":
                                        scrapingbee_key = (settings.scrapingbee_api_key or os.environ.get("SCRAPINGBEE_API_KEY") or "").strip()
                                        search_term = str(patch.get("search_term") or patch.get("query") or "").strip()
                                        vector_queries = str(
                                            patch.get("vector_search_queries") or patch.get("vector_searcg_queries") or ""
                                        ).strip()
                                        why = str(patch.get("why") or patch.get("reason") or "").strip()
                                        top_k_arg = patch.get("top_k")
                                        max_results_arg = patch.get("max_results")
                                        try:
                                            top_k_val = int(top_k_arg) if top_k_arg is not None else None
                                        except Exception:
                                            top_k_val = None
                                        try:
                                            max_results_val = int(max_results_arg) if max_results_arg is not None else None
                                        except Exception:
                                            max_results_val = None
                                        progress_q: "queue.Queue[str]" = queue.Queue()

                                        def _progress(s: str) -> None:
                                            try:
                                                progress_q.put_nowait(str(s))
                                            except Exception:
                                                return
                                        try:
                                            ws_model = (getattr(bot, "web_search_model", "") or bot.openai_model).strip()
                                            task = asyncio.create_task(
                                                asyncio.to_thread(
                                                    run_web_search,
                                                    search_term=search_term,
                                                    vector_search_queries=vector_queries,
                                                    why=why,
                                                    openai_api_key=api_key or "",
                                                    scrapingbee_api_key=scrapingbee_key,
                                                    model=ws_model,
                                                    progress_fn=_progress,
                                                    top_k=top_k_val,
                                                    max_results=max_results_val,
                                                )
                                            )
                                            if wait_reply:
                                                await _send_interim(wait_reply, kind="wait")
                                            last_wait = time.time()
                                            while not task.done():
                                                # Drain progress updates (best-effort).
                                                try:
                                                    while True:
                                                        p = progress_q.get_nowait()
                                                        if p:
                                                            await _send_interim(p, kind="progress")
                                                except queue.Empty:
                                                    pass
                                                if wait_reply and (time.time() - last_wait) >= 7.0:
                                                    await _send_interim(wait_reply, kind="wait")
                                                    last_wait = time.time()
                                                await asyncio.sleep(0.2)
                                            summary_text = await task
                                            tool_result = str(summary_text or "").strip()
                                        except Exception as exc:
                                            tool_result = f"WEB_SEARCH_ERROR: {exc}"
                                            tool_failed = True
                                        # If the tool finishes while the bot is still speaking, wait before continuing.
                                        if speak:
                                            now = time.time()
                                            if now < tts_busy_until:
                                                await asyncio.sleep(tts_busy_until - now)
                                        if tool_failed or not next_reply:
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                    elif tool_name == "give_command_to_data_agent":
                                        if not bool(getattr(bot, "enable_data_agent", False)):
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Data Agent is disabled for this bot."},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            what_to_do = str(patch.get("what_to_do") or "").strip()
                                            if not what_to_do:
                                                tool_result = {
                                                    "ok": False,
                                                    "error": {"message": "Missing required tool arg: what_to_do"},
                                                }
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                # Ensure the per-conversation runtime exists (Docker) and run Codex CLI.
                                                try:
                                                    logger.info(
                                                        "Data Agent tool: start conv=%s bot=%s what_to_do=%s",
                                                        conv_id,
                                                        bot_id,
                                                        (what_to_do[:200] + "…") if len(what_to_do) > 200 else what_to_do,
                                                    )
                                                    da = _data_agent_meta(meta_current)
                                                    workspace_dir = (
                                                        str(da.get("workspace_dir") or "").strip()
                                                        or default_workspace_dir_for_conversation(conv_id)
                                                    )
                                                    container_id = str(da.get("container_id") or "").strip()
                                                    session_id = str(da.get("session_id") or "").strip()

                                                    api_key = _get_openai_api_key_for_bot(session, bot=bot)
                                                    if not api_key:
                                                        raise RuntimeError(
                                                            "No OpenAI API key configured for this bot (needed for Data Agent)."
                                                        )

                                                    # Ensure the container exists and is running even if metadata has a stale id.
                                                    ensured_container_id = await asyncio.to_thread(
                                                        ensure_conversation_container,
                                                        conversation_id=conv_id,
                                                        workspace_dir=workspace_dir,
                                                        openai_api_key=api_key,
                                                    )
                                                    if ensured_container_id and ensured_container_id != container_id:
                                                        container_id = ensured_container_id
                                                        meta_current = merge_conversation_metadata(
                                                            session,
                                                            conversation_id=conv_id,
                                                            patch={
                                                                "data_agent.container_id": container_id,
                                                                "data_agent.workspace_dir": workspace_dir,
                                                            },
                                                        )

                                                    ctx = _build_data_agent_conversation_context(
                                                        session,
                                                        bot=bot,
                                                        conversation_id=conv_id,
                                                        meta=meta_current,
                                                    )
                                                    api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
                                                    auth_json = getattr(bot, "data_agent_auth_json", "") or "{}"
                                                    sys_prompt = (
                                                        (getattr(bot, "data_agent_system_prompt", "") or "").strip()
                                                        or DEFAULT_DATA_AGENT_SYSTEM_PROMPT
                                                    )

                                                    task = asyncio.create_task(
                                                        asyncio.to_thread(
                                                            run_data_agent,
                                                            conversation_id=conv_id,
                                                            container_id=container_id,
                                                            session_id=session_id,
                                                            workspace_dir=workspace_dir,
                                                            api_spec_text=api_spec_text,
                                                            auth_json=auth_json,
                                                            system_prompt=sys_prompt,
                                                            conversation_context=ctx,
                                                            what_to_do=what_to_do,
                                                        )
                                                    )
                                                    if wait_reply:
                                                        await _send_interim(wait_reply, kind="wait")
                                                    last_wait = time.time()
                                                    while not task.done():
                                                        if wait_reply and (time.time() - last_wait) >= 10.0:
                                                            await _send_interim(wait_reply, kind="wait")
                                                            last_wait = time.time()
                                                        await asyncio.sleep(0.2)
                                                    da_res = await task

                                                    if da_res.session_id and da_res.session_id != session_id:
                                                        meta_current = merge_conversation_metadata(
                                                            session,
                                                            conversation_id=conv_id,
                                                            patch={"data_agent.session_id": da_res.session_id},
                                                        )
                                                    logger.info(
                                                        "Data Agent tool: done conv=%s ok=%s container_id=%s session_id=%s output_file=%s error=%s",
                                                        conv_id,
                                                        bool(da_res.ok),
                                                        da_res.container_id,
                                                        da_res.session_id,
                                                        da_res.output_file,
                                                        da_res.error,
                                                    )
                                                    tool_result = {
                                                        "ok": bool(da_res.ok),
                                                        "result_text": da_res.result_text,
                                                        "data_agent_container_id": da_res.container_id,
                                                        "data_agent_session_id": da_res.session_id,
                                                        "data_agent_output_file": da_res.output_file,
                                                        "data_agent_debug_file": da_res.debug_file,
                                                        "error": da_res.error,
                                                    }
                                                    tool_failed = not bool(da_res.ok)
                                                    if (
                                                        bool(getattr(bot, "data_agent_return_result_directly", False))
                                                        and bool(da_res.ok)
                                                        and str(da_res.result_text or "").strip()
                                                    ):
                                                        needs_followup_llm = False
                                                        rendered_reply = str(da_res.result_text or "").strip()
                                                    else:
                                                        needs_followup_llm = True
                                                        rendered_reply = ""
                                                except Exception as exc:
                                                    logger.exception("Data Agent tool failed conv=%s bot=%s", conv_id, bot_id)
                                                    tool_result = {
                                                        "ok": False,
                                                        "error": {"message": str(exc)},
                                                    }
                                                    tool_failed = True
                                                    needs_followup_llm = True
                                                    rendered_reply = ""
                                    elif tool_name == "recall_http_response":
                                        source_tool_name = str(patch.get("source_tool_name") or "").strip()
                                        source_req_id = str(patch.get("source_req_id") or "").strip()
                                        fields_required = str(patch.get("fields_required") or "").strip()
                                        why_api_was_called = str(patch.get("why_api_was_called") or "").strip()

                                        missing_keys = [
                                            k
                                            for k in ("source_tool_name", "fields_required", "why_api_was_called")
                                            if not str(patch.get(k) or "").strip()
                                        ]
                                        if missing_keys:
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Missing required tool args.", "missing": missing_keys},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            ev = find_saved_run(
                                                conversation_id=str(conv_id),
                                                source_tool_name=source_tool_name,
                                                source_req_id=(source_req_id or None),
                                            )
                                            if not ev:
                                                tool_result = {
                                                    "ok": False,
                                                    "error": {
                                                        "message": f"No saved response found for tool '{source_tool_name}' in this conversation.",
                                                        "tool_name": source_tool_name,
                                                        "req_id": source_req_id or None,
                                                    },
                                                }
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                saved_input = str(ev.get("input_json_path") or "").strip()
                                                saved_schema = str(ev.get("schema_json_path") or "").strip()
                                                if not saved_input and str(ev.get("output_dir") or "").strip():
                                                    saved_input = os.path.join(str(ev.get("output_dir") or ""), "input_response.json")
                                                if not saved_schema and str(ev.get("output_dir") or "").strip():
                                                    saved_schema = os.path.join(str(ev.get("output_dir") or ""), "input_schema.json")

                                                source_tool_cfg = get_integration_tool_by_name(
                                                    session, bot_id=bot.id, name=source_tool_name
                                                )
                                                codex_model = (
                                                    (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                    or "gpt-5.1-codex-mini"
                                                )
                                                progress_q: "queue.Queue[str]" = queue.Queue()

                                                def _progress(s: str) -> None:
                                                    try:
                                                        progress_q.put_nowait(str(s))
                                                    except Exception:
                                                        return

                                                agent_task = asyncio.create_task(
                                                    asyncio.to_thread(
                                                        run_codex_http_agent_one_shot_from_paths,
                                                        api_key=api_key or "",
                                                        model=codex_model,
                                                        input_json_path=saved_input,
                                                        input_schema_json_path=saved_schema or None,
                                                        fields_required=fields_required,
                                                        why_api_was_called=why_api_was_called,
                                                        conversation_id=str(conv_id) if conv_id is not None else None,
                                                        req_id=req_id,
                                                        tool_codex_prompt=getattr(source_tool_cfg, "codex_prompt", "") or "",
                                                        progress_fn=_progress,
                                                    )
                                                )
                                                if wait_reply:
                                                    await _send_interim(wait_reply, kind="wait")
                                                last_wait = time.time()
                                                last_progress = last_wait
                                                wait_interval_s = 15.0
                                                while not agent_task.done():
                                                    try:
                                                        while True:
                                                            p = progress_q.get_nowait()
                                                            if p:
                                                                await _send_interim(p, kind="progress")
                                                                last_progress = time.time()
                                                    except queue.Empty:
                                                        pass
                                                    now = time.time()
                                                    if (
                                                        wait_reply
                                                        and (now - last_wait) >= wait_interval_s
                                                        and (now - last_progress) >= wait_interval_s
                                                    ):
                                                        await _send_interim(wait_reply, kind="wait")
                                                        last_wait = now
                                                    await asyncio.sleep(0.2)

                                                tool_result = {
                                                    "ok": True,
                                                    "recall_source_tool": source_tool_name,
                                                    "recall_source_req_id": str(ev.get("req_id") or "") or None,
                                                }
                                                try:
                                                    agent_res = await agent_task
                                                    tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                                    tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                                    tool_result["codex_output_file"] = getattr(agent_res, "result_text_path", "")
                                                    tool_result["codex_debug_file"] = getattr(agent_res, "debug_json_path", "")
                                                    tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                                    tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                                    tool_result["codex_continue_reason"] = getattr(agent_res, "continue_reason", "")
                                                    tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                                    err = getattr(agent_res, "error", None)
                                                    if err:
                                                        tool_result["codex_error"] = str(err)
                                                except Exception as exc:
                                                    tool_result["codex_ok"] = False
                                                    tool_result["codex_error"] = str(exc)

                                                try:
                                                    append_saved_run_index(
                                                        conversation_id=str(conv_id),
                                                        event={
                                                            "kind": "recall",
                                                            "tool_name": source_tool_name,
                                                            "req_id": req_id,
                                                            "source_req_id": str(ev.get("req_id") or "") or None,
                                                            "input_json_path": saved_input,
                                                            "schema_json_path": saved_schema,
                                                            "fields_required": fields_required,
                                                            "why_api_was_called": why_api_was_called,
                                                            "codex_output_dir": tool_result.get("codex_output_dir"),
                                                            "codex_ok": tool_result.get("codex_ok"),
                                                        },
                                                    )
                                                except Exception:
                                                    pass

                                                needs_followup_llm = True
                                                rendered_reply = ""
                                    elif tool_name == "export_http_response":
                                        source_tool_name = str(patch.get("source_tool_name") or "").strip()
                                        source_req_id = str(patch.get("source_req_id") or "").strip()
                                        export_request = str(patch.get("export_request") or "").strip()
                                        output_format = str(patch.get("output_format") or "csv").strip().lower()
                                        file_name_hint = str(patch.get("file_name_hint") or "").strip()

                                        missing_keys = [
                                            k
                                            for k in ("source_tool_name", "export_request")
                                            if not str(patch.get(k) or "").strip()
                                        ]
                                        if missing_keys:
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Missing required tool args.", "missing": missing_keys},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            ev = find_saved_run(
                                                conversation_id=str(conv_id),
                                                source_tool_name=source_tool_name,
                                                source_req_id=(source_req_id or None),
                                            )
                                            if not ev:
                                                tool_result = {
                                                    "ok": False,
                                                    "error": {
                                                        "message": f"No saved response found for tool '{source_tool_name}' in this conversation.",
                                                        "tool_name": source_tool_name,
                                                        "req_id": source_req_id or None,
                                                    },
                                                }
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                saved_input = str(ev.get("input_json_path") or "").strip()
                                                saved_schema = str(ev.get("schema_json_path") or "").strip()
                                                if not saved_input and str(ev.get("output_dir") or "").strip():
                                                    saved_input = os.path.join(str(ev.get("output_dir") or ""), "input_response.json")
                                                if not saved_schema and str(ev.get("output_dir") or "").strip():
                                                    saved_schema = os.path.join(str(ev.get("output_dir") or ""), "input_schema.json")

                                                codex_model = (
                                                    (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                    or "gpt-5.1-codex-mini"
                                                )
                                                progress_q: "queue.Queue[str]" = queue.Queue()

                                                def _progress(s: str) -> None:
                                                    try:
                                                        progress_q.put_nowait(str(s))
                                                    except Exception:
                                                        return

                                                agent_task = asyncio.create_task(
                                                    asyncio.to_thread(
                                                        run_codex_export_from_paths,
                                                        api_key=api_key or "",
                                                        model=codex_model,
                                                        input_json_path=saved_input,
                                                        input_schema_json_path=saved_schema or None,
                                                        export_request=export_request,
                                                        output_format=output_format,
                                                        conversation_id=str(conv_id) if conv_id is not None else None,
                                                        req_id=req_id,
                                                        progress_fn=_progress,
                                                    )
                                                )
                                                if wait_reply:
                                                    await _send_interim(wait_reply, kind="wait")
                                                last_wait = time.time()
                                                last_progress = last_wait
                                                wait_interval_s = 15.0
                                                while not agent_task.done():
                                                    try:
                                                        while True:
                                                            p = progress_q.get_nowait()
                                                            if p:
                                                                await _send_interim(p, kind="progress")
                                                                last_progress = time.time()
                                                    except queue.Empty:
                                                        pass
                                                    now = time.time()
                                                    if (
                                                        wait_reply
                                                        and (now - last_wait) >= wait_interval_s
                                                        and (now - last_progress) >= wait_interval_s
                                                    ):
                                                        await _send_interim(wait_reply, kind="wait")
                                                        last_wait = now
                                                    await asyncio.sleep(0.2)

                                                tool_result = {
                                                    "ok": True,
                                                    "export_ok": False,
                                                    "export_format": output_format,
                                                    "export_source_tool": source_tool_name,
                                                    "export_source_req_id": str(ev.get("req_id") or "") or None,
                                                }
                                                try:
                                                    exp = await agent_task
                                                    tool_result["export_ok"] = bool(getattr(exp, "ok", False))
                                                    tool_result["export_output_dir"] = getattr(exp, "output_dir", "")
                                                    tool_result["export_debug_file"] = getattr(exp, "debug_json_path", "")
                                                    tool_result["export_file_path"] = getattr(exp, "export_file_path", "")
                                                    tool_result["export_stop_reason"] = getattr(exp, "stop_reason", "")
                                                    err = getattr(exp, "error", None)
                                                    if err:
                                                        tool_result["export_error"] = str(err)
                                                except Exception as exc:
                                                    tool_result["export_ok"] = False
                                                    tool_result["export_error"] = str(exc)

                                                export_path = str(tool_result.get("export_file_path") or "").strip()
                                                if tool_result.get("export_ok") and export_path and os.path.exists(export_path):
                                                    if not is_allowed_download_path(export_path):
                                                        tool_result["export_ok"] = False
                                                        tool_result["export_error"] = "Export file path not allowed for download."
                                                    else:
                                                        base_name = file_name_hint or os.path.basename(export_path)
                                                        if not base_name:
                                                            base_name = os.path.basename(export_path)
                                                        mime = (
                                                            "text/csv"
                                                            if output_format == "csv"
                                                            else "application/json"
                                                        )
                                                        token = create_download_token(
                                                            file_path=export_path,
                                                            filename=base_name,
                                                            mime_type=mime,
                                                            conversation_id=str(conv_id),
                                                            metadata={
                                                                "source_tool_name": source_tool_name,
                                                                "source_req_id": str(ev.get("req_id") or "") or None,
                                                            },
                                                        )
                                                        tool_result["download_token"] = token
                                                        tool_result["download_url"] = _download_url_for_token(token)
                                                        try:
                                                            tool_result["size_bytes"] = int(os.path.getsize(export_path))
                                                        except Exception:
                                                            pass

                                                try:
                                                    append_saved_run_index(
                                                        conversation_id=str(conv_id),
                                                        event={
                                                            "kind": "export",
                                                            "tool_name": source_tool_name,
                                                            "req_id": req_id,
                                                            "source_req_id": str(ev.get("req_id") or "") or None,
                                                            "input_json_path": saved_input,
                                                            "schema_json_path": saved_schema,
                                                            "export_format": output_format,
                                                            "export_request": export_request[:2000],
                                                            "export_file_path": tool_result.get("export_file_path"),
                                                            "download_token": tool_result.get("download_token"),
                                                        },
                                                    )
                                                except Exception:
                                                    pass

                                                needs_followup_llm = True
                                                rendered_reply = ""
                                    elif tool_name == "export_http_response":
                                        source_tool_name = str(patch.get("source_tool_name") or "").strip()
                                        source_req_id = str(patch.get("source_req_id") or "").strip()
                                        export_request = str(patch.get("export_request") or "").strip()
                                        output_format = str(patch.get("output_format") or "csv").strip().lower()
                                        file_name_hint = str(patch.get("file_name_hint") or "").strip()

                                        missing_keys = [
                                            k
                                            for k in ("source_tool_name", "export_request")
                                            if not str(patch.get(k) or "").strip()
                                        ]
                                        if missing_keys:
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Missing required tool args.", "missing": missing_keys},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            ev = find_saved_run(
                                                conversation_id=str(conv_id),
                                                source_tool_name=source_tool_name,
                                                source_req_id=(source_req_id or None),
                                            )
                                            if not ev:
                                                tool_result = {
                                                    "ok": False,
                                                    "error": {
                                                        "message": f"No saved response found for tool '{source_tool_name}' in this conversation.",
                                                        "tool_name": source_tool_name,
                                                        "req_id": source_req_id or None,
                                                    },
                                                }
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                saved_input = str(ev.get("input_json_path") or "").strip()
                                                saved_schema = str(ev.get("schema_json_path") or "").strip()
                                                if not saved_input and str(ev.get("output_dir") or "").strip():
                                                    saved_input = os.path.join(str(ev.get("output_dir") or ""), "input_response.json")
                                                if not saved_schema and str(ev.get("output_dir") or "").strip():
                                                    saved_schema = os.path.join(str(ev.get("output_dir") or ""), "input_schema.json")

                                                codex_model = (
                                                    (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                    or "gpt-5.1-codex-mini"
                                                )
                                                progress_q: "queue.Queue[str]" = queue.Queue()

                                                def _progress(s: str) -> None:
                                                    try:
                                                        progress_q.put_nowait(str(s))
                                                    except Exception:
                                                        return

                                                agent_task = asyncio.create_task(
                                                    asyncio.to_thread(
                                                        run_codex_export_from_paths,
                                                        api_key=api_key or "",
                                                        model=codex_model,
                                                        input_json_path=saved_input,
                                                        input_schema_json_path=saved_schema or None,
                                                        export_request=export_request,
                                                        output_format=output_format,
                                                        conversation_id=str(conv_id) if conv_id is not None else None,
                                                        req_id=req_id,
                                                        progress_fn=_progress,
                                                    )
                                                )
                                                if wait_reply:
                                                    await _send_interim(wait_reply, kind="wait")
                                                last_wait = time.time()
                                                last_progress = last_wait
                                                wait_interval_s = 15.0
                                                while not agent_task.done():
                                                    try:
                                                        while True:
                                                            p = progress_q.get_nowait()
                                                            if p:
                                                                await _send_interim(p, kind="progress")
                                                                last_progress = time.time()
                                                    except queue.Empty:
                                                        pass
                                                    now = time.time()
                                                    if (
                                                        wait_reply
                                                        and (now - last_wait) >= wait_interval_s
                                                        and (now - last_progress) >= wait_interval_s
                                                    ):
                                                        await _send_interim(wait_reply, kind="wait")
                                                        last_wait = now
                                                    await asyncio.sleep(0.2)

                                                tool_result = {
                                                    "ok": True,
                                                    "export_ok": False,
                                                    "export_format": output_format,
                                                    "export_source_tool": source_tool_name,
                                                    "export_source_req_id": str(ev.get("req_id") or "") or None,
                                                }
                                                try:
                                                    exp = await agent_task
                                                    tool_result["export_ok"] = bool(getattr(exp, "ok", False))
                                                    tool_result["export_output_dir"] = getattr(exp, "output_dir", "")
                                                    tool_result["export_debug_file"] = getattr(exp, "debug_json_path", "")
                                                    tool_result["export_file_path"] = getattr(exp, "export_file_path", "")
                                                    tool_result["export_stop_reason"] = getattr(exp, "stop_reason", "")
                                                    err = getattr(exp, "error", None)
                                                    if err:
                                                        tool_result["export_error"] = str(err)
                                                except Exception as exc:
                                                    tool_result["export_ok"] = False
                                                    tool_result["export_error"] = str(exc)

                                                export_path = str(tool_result.get("export_file_path") or "").strip()
                                                if tool_result.get("export_ok") and export_path and os.path.exists(export_path):
                                                    if not is_allowed_download_path(export_path):
                                                        tool_result["export_ok"] = False
                                                        tool_result["export_error"] = "Export file path not allowed for download."
                                                    else:
                                                        base_name = file_name_hint or os.path.basename(export_path)
                                                        if not base_name:
                                                            base_name = os.path.basename(export_path)
                                                        mime = (
                                                            "text/csv"
                                                            if output_format == "csv"
                                                            else "application/json"
                                                        )
                                                        token = create_download_token(
                                                            file_path=export_path,
                                                            filename=base_name,
                                                            mime_type=mime,
                                                            conversation_id=str(conv_id),
                                                            metadata={
                                                                "source_tool_name": source_tool_name,
                                                                "source_req_id": str(ev.get("req_id") or "") or None,
                                                            },
                                                        )
                                                        tool_result["download_token"] = token
                                                        tool_result["download_url"] = _download_url_for_token(token)
                                                        try:
                                                            tool_result["size_bytes"] = int(os.path.getsize(export_path))
                                                        except Exception:
                                                            pass

                                                try:
                                                    append_saved_run_index(
                                                        conversation_id=str(conv_id),
                                                        event={
                                                            "kind": "export",
                                                            "tool_name": source_tool_name,
                                                            "req_id": req_id,
                                                            "source_req_id": str(ev.get("req_id") or "") or None,
                                                            "input_json_path": saved_input,
                                                            "schema_json_path": saved_schema,
                                                            "export_format": output_format,
                                                            "export_request": export_request[:2000],
                                                            "export_file_path": tool_result.get("export_file_path"),
                                                            "download_token": tool_result.get("download_token"),
                                                        },
                                                    )
                                                except Exception:
                                                    pass

                                                needs_followup_llm = True
                                                rendered_reply = ""
                                    else:
                                        tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                        if not tool_cfg:
                                            raise RuntimeError(f"Unknown tool: {tool_name}")
                                        if not bool(getattr(tool_cfg, "enabled", True)):
                                            response_json = {
                                                "__tool_args_error__": {
                                                    "missing": [],
                                                    "message": f"Tool '{tool_name}' is disabled for this bot.",
                                                }
                                            }
                                        else:
                                            task = asyncio.create_task(
                                                asyncio.to_thread(
                                                    _execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                                                )
                                            )
                                            if wait_reply:
                                                await _send_interim(wait_reply, kind="wait")
                                            while True:
                                                try:
                                                    response_json = await asyncio.wait_for(asyncio.shield(task), timeout=60.0)
                                                    break
                                                except asyncio.TimeoutError:
                                                    if wait_reply:
                                                        await _send_interim(wait_reply, kind="wait")
                                                    continue
                                        if speak:
                                            now = time.time()
                                            if now < tts_busy_until:
                                                await asyncio.sleep(tts_busy_until - now)
                                        if isinstance(response_json, dict) and "__tool_args_error__" in response_json:
                                            err = response_json["__tool_args_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        elif isinstance(response_json, dict) and "__http_error__" in response_json:
                                            err = response_json["__http_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            pagination_info = None
                                            if isinstance(response_json, dict) and "__igx_pagination__" in response_json:
                                                pagination_info = response_json.pop("__igx_pagination__", None)
                                            if bool(getattr(tool_cfg, "use_codex_response", False)):
                                                # Avoid bloating LLM-visible conversation metadata in Codex mode.
                                                tool_result = {"ok": True}
                                                new_meta = meta_current
                                            else:
                                                mapped = _apply_response_mapper(
                                                    mapper_json=tool_cfg.response_mapper_json,
                                                    response_json=response_json,
                                                    meta=meta_current,
                                                    tool_args=patch,
                                                )
                                                new_meta = merge_conversation_metadata(
                                                    session, conversation_id=conv_id, patch=mapped
                                                )
                                                meta_current = new_meta
                                                tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}
                                            if pagination_info:
                                                tool_result["pagination"] = pagination_info

                                            # Optional Codex HTTP agent (post-process the raw response JSON).
                                            # Static reply (if configured) takes priority.
                                            static_preview = ""
                                            if (tool_cfg.static_reply_template or "").strip():
                                                try:
                                                    static_preview = _render_static_reply(
                                                        template_text=tool_cfg.static_reply_template,
                                                        meta=new_meta or meta_current,
                                                        response_json=response_json,
                                                        tool_args=patch,
                                                    ).strip()
                                                except Exception:
                                                    static_preview = ""
                                            if (not static_preview) and bool(getattr(tool_cfg, "use_codex_response", False)):
                                                fields_required = str(patch.get("fields_required") or "").strip()
                                                what_to_search_for = str(patch.get("what_to_search_for") or "").strip()
                                                if not fields_required:
                                                    fields_required = what_to_search_for
                                                why_api_was_called = str(patch.get("why_api_was_called") or "").strip()
                                                if not why_api_was_called:
                                                    why_api_was_called = str(patch.get("why_to_search_for") or "").strip()
                                                if not fields_required or not why_api_was_called:
                                                    tool_result["codex_ok"] = False
                                                    tool_result["codex_error"] = "Missing fields_required / why_api_was_called."
                                                else:
                                                    fields_required_for_codex = fields_required
                                                    if what_to_search_for:
                                                        fields_required_for_codex = (
                                                            f"{fields_required_for_codex}\n\n(what_to_search_for) {what_to_search_for}"
                                                        )
                                                    did_postprocess = False
                                                    postprocess_python = str(
                                                        getattr(tool_cfg, "postprocess_python", "") or ""
                                                    ).strip()
                                                    if postprocess_python:
                                                        py_task = asyncio.create_task(
                                                            asyncio.to_thread(
                                                                run_python_postprocessor,
                                                                python_code=postprocess_python,
                                                                payload={
                                                                    "response_json": response_json,
                                                                    "meta": new_meta or meta_current,
                                                                    "args": patch,
                                                                    "fields_required": fields_required,
                                                                    "why_api_was_called": why_api_was_called,
                                                                },
                                                                timeout_s=60,
                                                            )
                                                        )
                                                        if wait_reply:
                                                            await _send_interim(wait_reply, kind="wait")
                                                        last_wait = time.time()
                                                        wait_interval_s = 15.0
                                                        while not py_task.done():
                                                            now = time.time()
                                                            if wait_reply and (now - last_wait) >= wait_interval_s:
                                                                await _send_interim(wait_reply, kind="wait")
                                                                last_wait = now
                                                            await asyncio.sleep(0.2)
                                                        try:
                                                            py_res = await py_task
                                                            tool_result["python_ok"] = bool(getattr(py_res, "ok", False))
                                                            tool_result["python_duration_ms"] = int(
                                                                getattr(py_res, "duration_ms", 0) or 0
                                                            )
                                                            if getattr(py_res, "error", None):
                                                                tool_result["python_error"] = str(getattr(py_res, "error"))
                                                            if getattr(py_res, "stderr", None):
                                                                tool_result["python_stderr"] = str(getattr(py_res, "stderr"))
                                                            if py_res.ok:
                                                                did_postprocess = True
                                                                tool_result["postprocess_mode"] = "python"
                                                                tool_result["codex_ok"] = True
                                                                tool_result["codex_result_text"] = str(
                                                                    getattr(py_res, "result_text", "") or ""
                                                                )
                                                                mp = getattr(py_res, "metadata_patch", None)
                                                                if isinstance(mp, dict) and mp:
                                                                    try:
                                                                        meta_current = merge_conversation_metadata(
                                                                            session,
                                                                            conversation_id=conv_id,
                                                                            patch=mp,
                                                                        )
                                                                        tool_result["python_metadata_patch"] = mp
                                                                    except Exception:
                                                                        pass
                                                                try:
                                                                    append_saved_run_index(
                                                                        conversation_id=str(conv_id),
                                                                        event={
                                                                            "kind": "integration_python_postprocess",
                                                                            "tool_name": tool_name,
                                                                            "req_id": req_id,
                                                                            "python_ok": tool_result.get("python_ok"),
                                                                            "python_duration_ms": tool_result.get(
                                                                                "python_duration_ms"
                                                                            ),
                                                                        },
                                                                    )
                                                                except Exception:
                                                                    pass
                                                        except Exception as exc:
                                                            tool_result["python_ok"] = False
                                                            tool_result["python_error"] = str(exc)

                                                    if not did_postprocess:
                                                        codex_model = (
                                                            (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                            or "gpt-5.1-codex-mini"
                                                        )
                                                        progress_q: "queue.Queue[str]" = queue.Queue()

                                                        def _progress(s: str) -> None:
                                                            try:
                                                                progress_q.put_nowait(str(s))
                                                            except Exception:
                                                                return

                                                        agent_task = asyncio.create_task(
                                                            asyncio.to_thread(
                                                                run_codex_http_agent_one_shot,
                                                                api_key=api_key or "",
                                                                model=codex_model,
                                                                response_json=response_json,
                                                                fields_required=fields_required_for_codex,
                                                                why_api_was_called=why_api_was_called,
                                                                response_schema_json=getattr(tool_cfg, "response_schema_json", "")
                                                                or "",
                                                                conversation_id=str(conv_id) if conv_id is not None else None,
                                                                req_id=req_id,
                                                                tool_codex_prompt=getattr(tool_cfg, "codex_prompt", "") or "",
                                                                progress_fn=_progress,
                                                            )
                                                        )
                                                        if wait_reply:
                                                            await _send_interim(wait_reply, kind="wait")
                                                        last_wait = time.time()
                                                        last_progress = last_wait
                                                        wait_interval_s = 15.0
                                                        while not agent_task.done():
                                                            try:
                                                                while True:
                                                                    p = progress_q.get_nowait()
                                                                    if p:
                                                                        await _send_interim(p, kind="progress")
                                                                        last_progress = time.time()
                                                            except queue.Empty:
                                                                pass
                                                            now = time.time()
                                                            if (
                                                                wait_reply
                                                                and (now - last_wait) >= wait_interval_s
                                                                and (now - last_progress) >= wait_interval_s
                                                            ):
                                                                await _send_interim(wait_reply, kind="wait")
                                                                last_wait = now
                                                            await asyncio.sleep(0.2)
                                                        try:
                                                            agent_res = await agent_task
                                                            tool_result["postprocess_mode"] = "codex"
                                                            tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                                            tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                                            tool_result["codex_output_file"] = getattr(
                                                                agent_res, "result_text_path", ""
                                                            )
                                                            tool_result["codex_debug_file"] = getattr(
                                                                agent_res, "debug_json_path", ""
                                                            )
                                                            tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                                            tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                                            tool_result["codex_continue_reason"] = getattr(
                                                                agent_res, "continue_reason", ""
                                                            )
                                                            tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                                            saved_input_json_path = getattr(agent_res, "input_json_path", "")
                                                            saved_schema_json_path = getattr(agent_res, "schema_json_path", "")
                                                            err = getattr(agent_res, "error", None)
                                                            if err:
                                                                tool_result["codex_error"] = str(err)
                                                        except Exception as exc:
                                                            tool_result["codex_ok"] = False
                                                            tool_result["codex_error"] = str(exc)
                                                            saved_input_json_path = ""
                                                            saved_schema_json_path = ""

                                                        try:
                                                            append_saved_run_index(
                                                                conversation_id=str(conv_id),
                                                                event={
                                                                    "kind": "integration_http",
                                                                    "tool_name": tool_name,
                                                                    "req_id": req_id,
                                                                    "input_json_path": saved_input_json_path,
                                                                    "schema_json_path": saved_schema_json_path,
                                                                    "fields_required": fields_required,
                                                                    "why_api_was_called": why_api_was_called,
                                                                    "codex_output_dir": tool_result.get("codex_output_dir"),
                                                                    "codex_ok": tool_result.get("codex_ok"),
                                                                },
                                                            )
                                                        except Exception:
                                                            pass
                                                if speak:
                                                    now = time.time()
                                                    if now < tts_busy_until:
                                                        await asyncio.sleep(tts_busy_until - now)

                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="tool",
                                        content=json.dumps({"tool": tool_name, "result": tool_result}, ensure_ascii=False),
                                    )
                                    if isinstance(tool_result, dict):
                                        meta_current = tool_result.get("metadata") or meta_current

                                    await _ws_send_json(
                                        ws,
                                        {"type": "tool_result", "req_id": req_id, "name": tool_name, "result": tool_result},
                                    )

                                    if tool_error:
                                        break
                                    if tool_failed:
                                        break

                                    candidate = ""
                                    if tool_name != "set_metadata" and tool_cfg:
                                        static_text = ""
                                        if (tool_cfg.static_reply_template or "").strip():
                                            static_text = _render_static_reply(
                                                template_text=tool_cfg.static_reply_template,
                                                meta=meta_current,
                                                response_json=response_json,
                                                tool_args=patch,
                                            ).strip()
                                        if static_text:
                                            needs_followup_llm = False
                                            rendered_reply = static_text
                                        else:
                                            if bool(getattr(tool_cfg, "use_codex_response", False)):
                                                if bool(getattr(tool_cfg, "return_result_directly", False)) and isinstance(
                                                    tool_result, dict
                                                ):
                                                    direct = str(tool_result.get("codex_result_text") or "").strip()
                                                    if direct:
                                                        needs_followup_llm = False
                                                        rendered_reply = direct
                                                    else:
                                                        needs_followup_llm = True
                                                        rendered_reply = ""
                                                else:
                                                    needs_followup_llm = True
                                                    rendered_reply = ""
                                            else:
                                                needs_followup_llm = _should_followup_llm_for_tool(
                                                    tool=tool_cfg, static_rendered=static_text
                                                )
                                                candidate = _render_with_meta(next_reply, meta_current).strip()
                                                if candidate:
                                                    rendered_reply = candidate
                                                    needs_followup_llm = False
                                                else:
                                                    rendered_reply = ""
                                    else:
                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                        rendered_reply = candidate or rendered_reply

                            # If static reply is missing/empty for an integration tool, ask the LLM again
                            # with tool call + tool result already persisted in history.
                            if needs_followup_llm and not tool_error and conv_id is not None:
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                                with Session(engine) as session:
                                    followup_bot = get_bot(session, bot_id)
                                    followup_history = _build_history_budgeted(
                                        session=session,
                                        bot=followup_bot,
                                        conversation_id=conv_id,
                                        api_key=api_key,
                                        status_cb=None,
                                    )
                                followup_history.append(
                                    Message(
                                        role="system",
                                        content=(
                                            ("The previous tool call failed. " if tool_failed else "")
                                            + "Using the latest tool result(s) above, write the next assistant reply. "
                                            "If the tool result contains codex_result_text, rephrase it for the user and do not mention file paths. "
                                            "Do not call any tools."
                                        ),
                                    )
                                )
                                followup_model = followup_bot.openai_model
                                follow_llm = OpenAILLM(model=followup_model, api_key=api_key)
                                if debug_mode:
                                    await _emit_llm_debug_payload(
                                        ws=ws,
                                        req_id=req_id,
                                        conversation_id=conv_id,
                                        phase="tool_followup_llm",
                                        payload=follow_llm.build_request_payload(messages=followup_history, stream=True),
                                    )
                                text2, ttfb2, total2 = await _stream_llm_reply(
                                    ws=ws, req_id=req_id, llm=follow_llm, messages=followup_history
                                )
                                rendered_reply = text2.strip()
                                if rendered_reply:
                                    followup_streamed = True
                                    in_tok = int(estimate_messages_tokens(followup_history, followup_model) or 0)
                                    out_tok = int(estimate_text_tokens(rendered_reply, followup_model) or 0)
                                    price = _get_openai_pricing().get(followup_model)
                                    cost = float(
                                        estimate_cost_usd(
                                            model_price=price,
                                            input_tokens=in_tok,
                                            output_tokens=out_tok,
                                        )
                                        or 0.0
                                    )
                                    with Session(engine) as session:
                                        add_message_with_metrics(
                                            session,
                                            conversation_id=conv_id,
                                            role="assistant",
                                            content=rendered_reply,
                                            input_tokens_est=in_tok or None,
                                            output_tokens_est=out_tok or None,
                                            cost_usd_est=cost or None,
                                            llm_ttfb_ms=ttfb2,
                                            llm_total_ms=total2,
                                            total_ms=total2,
                                        )
                                        update_conversation_metrics(
                                            session,
                                            conversation_id=conv_id,
                                            add_input_tokens_est=in_tok,
                                            add_output_tokens_est=out_tok,
                                            add_cost_usd_est=cost,
                                            last_asr_ms=None,
                                            last_llm_ttfb_ms=ttfb2,
                                            last_llm_total_ms=total2,
                                            last_tts_first_audio_ms=None,
                                            last_total_ms=total2,
                                        )
                                    followup_persisted = True
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "metrics",
                                            "req_id": req_id,
                                            "timings_ms": {"llm_ttfb": ttfb2, "llm_total": total2, "total": total2},
                                        },
                                    )

                            if tool_error:
                                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": tool_error})

                            if rendered_reply:
                                if not followup_streamed:
                                    await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": rendered_reply})
                                try:
                                    if not followup_persisted:
                                        in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                                            bot=bot, history=history, assistant_text=rendered_reply
                                        )
                                        with Session(engine) as session:
                                            add_message_with_metrics(
                                                session,
                                                conversation_id=conv_id,
                                                role="assistant",
                                                content=rendered_reply,
                                                input_tokens_est=in_tok,
                                                output_tokens_est=out_tok,
                                                cost_usd_est=cost,
                                                llm_ttfb_ms=timings.get("llm_ttfb"),
                                                llm_total_ms=timings.get("llm_total"),
                                                total_ms=timings.get("total"),
                                            )
                                            update_conversation_metrics(
                                                session,
                                                conversation_id=conv_id,
                                                add_input_tokens_est=in_tok,
                                                add_output_tokens_est=out_tok,
                                                add_cost_usd_est=cost,
                                                last_asr_ms=None,
                                                last_llm_ttfb_ms=timings.get("llm_ttfb"),
                                                last_llm_total_ms=timings.get("llm_total"),
                                                last_tts_first_audio_ms=None,
                                                last_total_ms=timings.get("total"),
                                            )
                                except Exception:
                                    pass

                                if speak:
                                    status(req_id, "tts")
                                    wav, sr = await asyncio.to_thread(tts_synth, rendered_reply)
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "audio_wav",
                                            "req_id": req_id,
                                            "wav_base64": base64.b64encode(wav).decode(),
                                            "sr": sr,
                                        },
                                    )

                            await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": rendered_reply})
                        else:
                            # Store assistant response.
                            try:
                                in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                                    bot=bot, history=history, assistant_text=final_text
                                )
                                with Session(engine) as session:
                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="assistant",
                                        content=final_text,
                                        input_tokens_est=in_tok,
                                        output_tokens_est=out_tok,
                                        cost_usd_est=cost,
                                        llm_ttfb_ms=timings.get("llm_ttfb"),
                                        llm_total_ms=timings.get("llm_total"),
                                        tts_first_audio_ms=timings.get("tts_first_audio"),
                                        total_ms=timings.get("total"),
                                    )
                                    update_conversation_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        add_input_tokens_est=in_tok,
                                        add_output_tokens_est=out_tok,
                                        add_cost_usd_est=cost,
                                        last_asr_ms=None,
                                        last_llm_ttfb_ms=timings.get("llm_ttfb"),
                                        last_llm_total_ms=timings.get("llm_total"),
                                        last_tts_first_audio_ms=timings.get("tts_first_audio"),
                                        last_total_ms=timings.get("total"),
                                    )
                            except Exception:
                                pass

                            await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": final_text})

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        conv_id = None
                        accepting_audio = False
                        continue

                    elif msg_type == "stop":
                        if not req_id or active_req_id != req_id:
                            await _ws_send_json(
                                ws, {"type": "error", "req_id": req_id or None, "error": "Unknown req_id"}
                            )
                            continue
                        accepting_audio = False
                        if not conv_id:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": "No conversation"})
                            active_req_id = None
                            continue

                        stop_ts = time.time()
                        if not audio_buf:
                            await _ws_send_json(ws, {"type": "asr", "req_id": req_id, "text": ""})
                            await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "asr"})

                        asr_start_ts = time.time()
                        try:
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                if bot.openai_key_id:
                                    crypto = require_crypto()
                                    api_key = decrypt_openai_key(session, crypto=crypto, bot=bot)
                                else:
                                    api_key = os.environ.get("OPENAI_API_KEY")
                                if not api_key:
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "error",
                                            "req_id": req_id,
                                            "error": "No OpenAI key configured for this bot.",
                                        },
                                    )
                                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                                    active_req_id = None
                                    conv_id = None
                                    continue

                                pcm16 = bytes(audio_buf)

                                asr = await asyncio.to_thread(
                                    _get_asr(bot.whisper_model, bot.whisper_device, bot.language).transcribe_pcm16,
                                    pcm16=pcm16,
                                    sample_rate=16000,
                                )
                                asr_end_ts = time.time()

                            user_text = (asr.text or "").strip()
                            await _ws_send_json(ws, {"type": "asr", "req_id": req_id, "text": user_text})
                            if not user_text:
                                await _ws_send_json(
                                    ws,
                                    {
                                        "type": "metrics",
                                        "req_id": req_id,
                                        "timings_ms": {
                                            "asr": int(round((asr_end_ts - asr_start_ts) * 1000.0)),
                                            "total": int(
                                                round((time.time() - (stop_ts or asr_start_ts)) * 1000.0)
                                            ),
                                        },
                                    },
                                )
                                await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                                active_req_id = None
                                conv_id = None
                                continue

                            add_message_with_metrics(
                                session,
                                conversation_id=conv_id,
                                role="user",
                                content=user_text,
                                asr_ms=int(round((asr_end_ts - asr_start_ts) * 1000.0)),
                            )
                            loop = asyncio.get_running_loop()

                            def _status_cb(stage: str) -> None:
                                asyncio.run_coroutine_threadsafe(
                                    _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": stage}), loop
                                )

                            history = _build_history_budgeted(
                                session=session,
                                bot=bot,
                                conversation_id=conv_id,
                                api_key=api_key,
                                status_cb=_status_cb,
                            )
                            tools_defs = _build_tools_for_bot(session, bot.id)

                            llm = OpenAILLM(model=bot.openai_model, api_key=api_key)
                            tts_synth = await asyncio.to_thread(_get_tts_synth_fn, bot, api_key)
                            if debug_mode:
                                await _emit_llm_debug_payload(
                                    ws=ws,
                                    req_id=req_id,
                                    conversation_id=conv_id,
                                    phase="asr_turn_llm",
                                    payload=llm.build_request_payload(messages=history, tools=tools_defs, stream=True),
                                )
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                        llm_start_ts = time.time()

                        # Stream LLM deltas + TTS audio from background threads.
                        delta_q_client: "queue.Queue[Optional[str]]" = queue.Queue()
                        delta_q_tts: "queue.Queue[Optional[str]]" = queue.Queue()
                        audio_q: "queue.Queue[Optional[tuple[bytes, int]]]" = queue.Queue()
                        tool_calls: list[ToolCall] = []
                        error_q: "queue.Queue[Optional[str]]" = queue.Queue()
                        full_text_parts: list[str] = []
                        metrics_lock = threading.Lock()
                        first_token_ts: Optional[float] = None
                        tts_start_ts: Optional[float] = None
                        first_audio_ts: Optional[float] = None

                        def llm_thread() -> None:
                            try:
                                for ev in llm.stream_text_or_tool(messages=history, tools=tools_defs):
                                    if isinstance(ev, ToolCall):
                                        tool_calls.append(ev)
                                        continue
                                    d = ev
                                    full_text_parts.append(d)
                                    delta_q_client.put(d)
                                    if speak:
                                        delta_q_tts.put(d)
                            except Exception as exc:
                                error_q.put(str(exc))
                            finally:
                                delta_q_client.put(None)
                                if speak:
                                    delta_q_tts.put(None)

                        def tts_thread() -> None:
                            nonlocal tts_start_ts
                            if not speak:
                                audio_q.put(None)
                                return
                            try:
                                parts: list[str] = []
                                while True:
                                    d = delta_q_tts.get()
                                    if d is None:
                                        break
                                    if d:
                                        parts.append(d)
                                text_to_speak = "".join(parts).strip()
                                if text_to_speak:
                                    with metrics_lock:
                                        if tts_start_ts is None:
                                            tts_start_ts = time.time()
                                    status(req_id, "tts")
                                    wav, sr = tts_synth(text_to_speak)
                                    audio_q.put((wav, sr))
                            except Exception as exc:
                                error_q.put(f"TTS failed: {exc}")
                            finally:
                                audio_q.put(None)

                        t1 = threading.Thread(target=llm_thread, daemon=True)
                        t2 = threading.Thread(target=tts_thread, daemon=True)
                        t1.start()
                        t2.start()

                        # Pump the queues to the websocket.
                        open_deltas = True
                        open_audio = True
                        while open_deltas or open_audio:
                            try:
                                err = error_q.get_nowait()
                                if err:
                                    await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": err})
                                    open_deltas = False
                                    open_audio = False
                                    break
                            except queue.Empty:
                                pass

                            sent_any = False
                            try:
                                d = delta_q_client.get_nowait()
                                if d is None:
                                    open_deltas = False
                                else:
                                    if first_token_ts is None:
                                        first_token_ts = time.time()
                                    await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
                                sent_any = True
                            except queue.Empty:
                                pass

                            if speak:
                                try:
                                    item = audio_q.get_nowait()
                                    if item is None:
                                        open_audio = False
                                    else:
                                        wav, sr = item
                                        if first_audio_ts is None:
                                            first_audio_ts = time.time()
                                        await _ws_send_json(
                                            ws,
                                            {
                                                "type": "audio_wav",
                                                "req_id": req_id,
                                                "wav_base64": base64.b64encode(wav).decode(),
                                                "sr": sr,
                                            },
                                        )
                                    sent_any = True
                                except queue.Empty:
                                    pass
                            else:
                                open_audio = False

                            if not sent_any:
                                await asyncio.sleep(0.01)

                        t1.join()
                        t2.join()
                        llm_end_ts = time.time()

                        final_text = "".join(full_text_parts).strip()

                        timings: dict[str, int] = {
                            "asr": int(round((asr_end_ts - asr_start_ts) * 1000.0)),
                            "llm_total": int(round((llm_end_ts - llm_start_ts) * 1000.0)),
                            "total": int(round((time.time() - (stop_ts or llm_start_ts)) * 1000.0)),
                        }
                        if first_token_ts is not None:
                            timings["llm_ttfb"] = int(round((first_token_ts - llm_start_ts) * 1000.0))
                        elif tool_calls and tool_calls[0].first_event_ts is not None:
                            timings["llm_ttfb"] = int(round((tool_calls[0].first_event_ts - llm_start_ts) * 1000.0))
                        if speak and tts_start_ts is not None and first_audio_ts is not None:
                            timings["tts_first_audio"] = int(round((first_audio_ts - tts_start_ts) * 1000.0))
                            timings["tts_from_llm_start"] = int(round((first_audio_ts - llm_start_ts) * 1000.0))

                        # Persist aggregates + last latencies (best-effort).
                        try:
                            in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                                bot=bot, history=history, assistant_text=final_text
                            )
                            with Session(engine) as session:
                                if final_text and conv_id and not tool_calls:
                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="assistant",
                                        content=final_text,
                                        input_tokens_est=in_tok,
                                        output_tokens_est=out_tok,
                                        cost_usd_est=cost,
                                        asr_ms=timings.get("asr"),
                                        llm_ttfb_ms=timings.get("llm_ttfb"),
                                        llm_total_ms=timings.get("llm_total"),
                                        tts_first_audio_ms=timings.get("tts_first_audio"),
                                        total_ms=timings.get("total"),
                                    )
                                    update_conversation_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        add_input_tokens_est=in_tok,
                                        add_output_tokens_est=out_tok,
                                        add_cost_usd_est=cost,
                                        last_asr_ms=timings.get("asr"),
                                        last_llm_ttfb_ms=timings.get("llm_ttfb"),
                                        last_llm_total_ms=timings.get("llm_total"),
                                        last_tts_first_audio_ms=timings.get("tts_first_audio"),
                                        last_total_ms=timings.get("total"),
                                    )
                        except Exception:
                            pass

                        await _ws_send_json(ws, {"type": "metrics", "req_id": req_id, "timings_ms": timings})
                        if tool_calls and conv_id is not None:
                            rendered_reply = ""
                            tool_error: Optional[str] = None
                            needs_followup_llm = False
                            tool_failed = False
                            followup_streamed = False
                            followup_persisted = False
                            tts_busy_until: float = 0.0

                            async def _send_interim(text: str, *, kind: str) -> None:
                                nonlocal tts_busy_until
                                t = (text or "").strip()
                                if not t:
                                    return
                                await _ws_send_json(
                                    ws,
                                    {"type": "interim", "req_id": req_id, "kind": kind, "text": t},
                                )
                                if not speak:
                                    return
                                now = time.time()
                                if now < tts_busy_until:
                                    await asyncio.sleep(tts_busy_until - now)
                                status(req_id, "tts")
                                try:
                                    wav, sr = await asyncio.to_thread(tts_synth, t)
                                    tts_busy_until = time.time() + _estimate_wav_seconds(wav, sr) + 0.15
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "audio_wav",
                                            "req_id": req_id,
                                            "wav_base64": base64.b64encode(wav).decode(),
                                            "sr": sr,
                                        },
                                    )
                                except Exception:
                                    return

                            with Session(engine) as session:
                                bot2 = get_bot(session, bot_id)
                                meta_current = _get_conversation_meta(session, conversation_id=conv_id)
                                disabled_tools = _disabled_tool_names(bot2)

                                for tc in tool_calls:
                                    tool_name = tc.name
                                    if tool_name == "set_variable":
                                        tool_name = "set_metadata"

                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "tool_call",
                                            "req_id": req_id,
                                            "name": tool_name,
                                            "arguments_json": tc.arguments_json,
                                        },
                                    )
                                    try:
                                        tool_args = json.loads(tc.arguments_json or "{}")
                                        if not isinstance(tool_args, dict):
                                            raise ValueError("tool args must be an object")
                                    except Exception as exc:
                                        tool_error = str(exc)
                                        break

                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="tool",
                                        content=json.dumps({"tool": tool_name, "arguments": tool_args}, ensure_ascii=False),
                                    )

                                    next_reply = str(tool_args.get("next_reply") or "").strip()
                                    wait_reply = str(tool_args.get("wait_reply") or "").strip() or "Working on it…"
                                    raw_args = tool_args.get("args")
                                    if isinstance(raw_args, dict):
                                        patch = dict(raw_args)
                                    else:
                                        patch = dict(tool_args)
                                        patch.pop("next_reply", None)
                                        patch.pop("wait_reply", None)
                                        patch.pop("args", None)

                                    tool_cfg: IntegrationTool | None = None
                                    response_json: Any | None = None

                                    if tool_name in disabled_tools:
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": f"Tool '{tool_name}' is disabled for this bot."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        rendered_reply = ""
                                    elif tool_name == "set_metadata":
                                        new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=patch)
                                        tool_result = {"ok": True, "updated": patch, "metadata": new_meta}
                                    elif tool_name == "web_search":
                                        scrapingbee_key = (settings.scrapingbee_api_key or os.environ.get("SCRAPINGBEE_API_KEY") or "").strip()
                                        search_term = str(patch.get("search_term") or patch.get("query") or "").strip()
                                        vector_queries = str(
                                            patch.get("vector_search_queries") or patch.get("vector_searcg_queries") or ""
                                        ).strip()
                                        why = str(patch.get("why") or patch.get("reason") or "").strip()
                                        top_k_arg = patch.get("top_k")
                                        max_results_arg = patch.get("max_results")
                                        try:
                                            top_k_val = int(top_k_arg) if top_k_arg is not None else None
                                        except Exception:
                                            top_k_val = None
                                        try:
                                            max_results_val = int(max_results_arg) if max_results_arg is not None else None
                                        except Exception:
                                            max_results_val = None
                                        progress_q: "queue.Queue[str]" = queue.Queue()

                                        def _progress(s: str) -> None:
                                            try:
                                                progress_q.put_nowait(str(s))
                                            except Exception:
                                                return
                                        try:
                                            ws_model = (getattr(bot, "web_search_model", "") or bot.openai_model).strip()
                                            task = asyncio.create_task(
                                                asyncio.to_thread(
                                                    run_web_search,
                                                    search_term=search_term,
                                                    vector_search_queries=vector_queries,
                                                    why=why,
                                                    openai_api_key=api_key or "",
                                                    scrapingbee_api_key=scrapingbee_key,
                                                    model=ws_model,
                                                    progress_fn=_progress,
                                                    top_k=top_k_val,
                                                    max_results=max_results_val,
                                                )
                                            )
                                            if wait_reply:
                                                await _send_interim(wait_reply, kind="wait")
                                            last_wait = time.time()
                                            while not task.done():
                                                try:
                                                    while True:
                                                        p = progress_q.get_nowait()
                                                        if p:
                                                            await _send_interim(p, kind="progress")
                                                except queue.Empty:
                                                    pass
                                                if wait_reply and (time.time() - last_wait) >= 7.0:
                                                    await _send_interim(wait_reply, kind="wait")
                                                    last_wait = time.time()
                                                await asyncio.sleep(0.2)
                                            summary_text = await task
                                            tool_result = str(summary_text or "").strip()
                                        except Exception as exc:
                                            tool_result = f"WEB_SEARCH_ERROR: {exc}"
                                            tool_failed = True
                                        if speak:
                                            now = time.time()
                                            if now < tts_busy_until:
                                                await asyncio.sleep(tts_busy_until - now)
                                        if tool_failed or not next_reply:
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                    elif tool_name == "give_command_to_data_agent":
                                        if not bool(getattr(bot2, "enable_data_agent", False)):
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Data Agent is disabled for this bot."},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            what_to_do = str(patch.get("what_to_do") or "").strip()
                                            if not what_to_do:
                                                tool_result = {
                                                    "ok": False,
                                                    "error": {"message": "Missing required tool arg: what_to_do"},
                                                }
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                try:
                                                    logger.info(
                                                        "Data Agent tool: start conv=%s bot=%s what_to_do=%s",
                                                        conv_id,
                                                        bot_id,
                                                        (what_to_do[:200] + "…") if len(what_to_do) > 200 else what_to_do,
                                                    )
                                                    da = _data_agent_meta(meta_current)
                                                    workspace_dir = (
                                                        str(da.get("workspace_dir") or "").strip()
                                                        or default_workspace_dir_for_conversation(conv_id)
                                                    )
                                                    container_id = str(da.get("container_id") or "").strip()
                                                    session_id = str(da.get("session_id") or "").strip()

                                                    api_key = _get_openai_api_key_for_bot(session, bot=bot2)
                                                    if not api_key:
                                                        raise RuntimeError(
                                                            "No OpenAI API key configured for this bot (needed for Data Agent)."
                                                        )

                                                    if not container_id:
                                                        container_id = await asyncio.to_thread(
                                                            ensure_conversation_container,
                                                            conversation_id=conv_id,
                                                            workspace_dir=workspace_dir,
                                                            openai_api_key=api_key,
                                                        )
                                                        meta_current = merge_conversation_metadata(
                                                            session,
                                                            conversation_id=conv_id,
                                                            patch={
                                                                "data_agent.container_id": container_id,
                                                                "data_agent.workspace_dir": workspace_dir,
                                                            },
                                                        )

                                                    ctx = _build_data_agent_conversation_context(
                                                        session,
                                                        bot=bot2,
                                                        conversation_id=conv_id,
                                                        meta=meta_current,
                                                    )
                                                    api_spec_text = getattr(bot2, "data_agent_api_spec_text", "") or ""
                                                    auth_json = getattr(bot2, "data_agent_auth_json", "") or "{}"
                                                    sys_prompt = (
                                                        (getattr(bot2, "data_agent_system_prompt", "") or "").strip()
                                                        or DEFAULT_DATA_AGENT_SYSTEM_PROMPT
                                                    )

                                                    task = asyncio.create_task(
                                                        asyncio.to_thread(
                                                            run_data_agent,
                                                            conversation_id=conv_id,
                                                            container_id=container_id,
                                                            session_id=session_id,
                                                            workspace_dir=workspace_dir,
                                                            api_spec_text=api_spec_text,
                                                            auth_json=auth_json,
                                                            system_prompt=sys_prompt,
                                                            conversation_context=ctx,
                                                            what_to_do=what_to_do,
                                                        )
                                                    )
                                                    if wait_reply:
                                                        await _send_interim(wait_reply, kind="wait")
                                                    last_wait = time.time()
                                                    while not task.done():
                                                        if wait_reply and (time.time() - last_wait) >= 10.0:
                                                            await _send_interim(wait_reply, kind="wait")
                                                            last_wait = time.time()
                                                        await asyncio.sleep(0.2)
                                                    da_res = await task

                                                    if da_res.session_id and da_res.session_id != session_id:
                                                        meta_current = merge_conversation_metadata(
                                                            session,
                                                            conversation_id=conv_id,
                                                            patch={"data_agent.session_id": da_res.session_id},
                                                        )
                                                    logger.info(
                                                        "Data Agent tool: done conv=%s ok=%s container_id=%s session_id=%s output_file=%s error=%s",
                                                        conv_id,
                                                        bool(da_res.ok),
                                                        da_res.container_id,
                                                        da_res.session_id,
                                                        da_res.output_file,
                                                        da_res.error,
                                                    )
                                                    tool_result = {
                                                        "ok": bool(da_res.ok),
                                                        "result_text": da_res.result_text,
                                                        "data_agent_container_id": da_res.container_id,
                                                        "data_agent_session_id": da_res.session_id,
                                                        "data_agent_output_file": da_res.output_file,
                                                        "data_agent_debug_file": da_res.debug_file,
                                                        "error": da_res.error,
                                                    }
                                                    tool_failed = not bool(da_res.ok)
                                                    if (
                                                        bool(getattr(bot2, "data_agent_return_result_directly", False))
                                                        and bool(da_res.ok)
                                                        and str(da_res.result_text or "").strip()
                                                    ):
                                                        needs_followup_llm = False
                                                        rendered_reply = str(da_res.result_text or "").strip()
                                                    else:
                                                        needs_followup_llm = True
                                                        rendered_reply = ""
                                                except Exception as exc:
                                                    logger.exception("Data Agent tool failed conv=%s bot=%s", conv_id, bot_id)
                                                    tool_result = {
                                                        "ok": False,
                                                        "error": {"message": str(exc)},
                                                    }
                                                    tool_failed = True
                                                    needs_followup_llm = True
                                                    rendered_reply = ""
                                    elif tool_name == "recall_http_response":
                                        source_tool_name = str(patch.get("source_tool_name") or "").strip()
                                        source_req_id = str(patch.get("source_req_id") or "").strip()
                                        fields_required = str(patch.get("fields_required") or "").strip()
                                        why_api_was_called = str(patch.get("why_api_was_called") or "").strip()

                                        missing_keys = [
                                            k
                                            for k in ("source_tool_name", "fields_required", "why_api_was_called")
                                            if not str(patch.get(k) or "").strip()
                                        ]
                                        if missing_keys:
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Missing required tool args.", "missing": missing_keys},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            ev = find_saved_run(
                                                conversation_id=str(conv_id),
                                                source_tool_name=source_tool_name,
                                                source_req_id=(source_req_id or None),
                                            )
                                            if not ev:
                                                tool_result = {
                                                    "ok": False,
                                                    "error": {
                                                        "message": f"No saved response found for tool '{source_tool_name}' in this conversation.",
                                                        "tool_name": source_tool_name,
                                                        "req_id": source_req_id or None,
                                                    },
                                                }
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                saved_input = str(ev.get("input_json_path") or "").strip()
                                                saved_schema = str(ev.get("schema_json_path") or "").strip()
                                                if not saved_input and str(ev.get("output_dir") or "").strip():
                                                    saved_input = os.path.join(str(ev.get("output_dir") or ""), "input_response.json")
                                                if not saved_schema and str(ev.get("output_dir") or "").strip():
                                                    saved_schema = os.path.join(str(ev.get("output_dir") or ""), "input_schema.json")

                                                source_tool_cfg = get_integration_tool_by_name(
                                                    session, bot_id=bot.id, name=source_tool_name
                                                )
                                                codex_model = (
                                                    (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                    or "gpt-5.1-codex-mini"
                                                )
                                                progress_q: "queue.Queue[str]" = queue.Queue()

                                                def _progress(s: str) -> None:
                                                    try:
                                                        progress_q.put_nowait(str(s))
                                                    except Exception:
                                                        return

                                                agent_task = asyncio.create_task(
                                                    asyncio.to_thread(
                                                        run_codex_http_agent_one_shot_from_paths,
                                                        api_key=api_key or "",
                                                        model=codex_model,
                                                        input_json_path=saved_input,
                                                        input_schema_json_path=saved_schema or None,
                                                        fields_required=fields_required,
                                                        why_api_was_called=why_api_was_called,
                                                        conversation_id=str(conv_id) if conv_id is not None else None,
                                                        req_id=req_id,
                                                        tool_codex_prompt=getattr(source_tool_cfg, "codex_prompt", "") or "",
                                                        progress_fn=_progress,
                                                    )
                                                )
                                                if wait_reply:
                                                    await _send_interim(wait_reply, kind="wait")
                                                last_wait = time.time()
                                                last_progress = last_wait
                                                wait_interval_s = 15.0
                                                while not agent_task.done():
                                                    try:
                                                        while True:
                                                            p = progress_q.get_nowait()
                                                            if p:
                                                                await _send_interim(p, kind="progress")
                                                                last_progress = time.time()
                                                    except queue.Empty:
                                                        pass
                                                    now = time.time()
                                                    if (
                                                        wait_reply
                                                        and (now - last_wait) >= wait_interval_s
                                                        and (now - last_progress) >= wait_interval_s
                                                    ):
                                                        await _send_interim(wait_reply, kind="wait")
                                                        last_wait = now
                                                    await asyncio.sleep(0.2)

                                                tool_result = {
                                                    "ok": True,
                                                    "recall_source_tool": source_tool_name,
                                                    "recall_source_req_id": str(ev.get("req_id") or "") or None,
                                                }
                                                try:
                                                    agent_res = await agent_task
                                                    tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                                    tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                                    tool_result["codex_output_file"] = getattr(agent_res, "result_text_path", "")
                                                    tool_result["codex_debug_file"] = getattr(agent_res, "debug_json_path", "")
                                                    tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                                    tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                                    tool_result["codex_continue_reason"] = getattr(agent_res, "continue_reason", "")
                                                    tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                                    err = getattr(agent_res, "error", None)
                                                    if err:
                                                        tool_result["codex_error"] = str(err)
                                                except Exception as exc:
                                                    tool_result["codex_ok"] = False
                                                    tool_result["codex_error"] = str(exc)

                                                try:
                                                    append_saved_run_index(
                                                        conversation_id=str(conv_id),
                                                        event={
                                                            "kind": "recall",
                                                            "tool_name": source_tool_name,
                                                            "req_id": req_id,
                                                            "source_req_id": str(ev.get("req_id") or "") or None,
                                                            "input_json_path": saved_input,
                                                            "schema_json_path": saved_schema,
                                                            "fields_required": fields_required,
                                                            "why_api_was_called": why_api_was_called,
                                                            "codex_output_dir": tool_result.get("codex_output_dir"),
                                                            "codex_ok": tool_result.get("codex_ok"),
                                                        },
                                                    )
                                                except Exception:
                                                    pass

                                                needs_followup_llm = True
                                                rendered_reply = ""
                                    else:
                                        tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                        if not tool_cfg:
                                            raise RuntimeError(f"Unknown tool: {tool_name}")
                                        if not bool(getattr(tool_cfg, "enabled", True)):
                                            response_json = {
                                                "__tool_args_error__": {
                                                    "missing": [],
                                                    "message": f"Tool '{tool_name}' is disabled for this bot.",
                                                }
                                            }
                                        else:
                                            task = asyncio.create_task(
                                                asyncio.to_thread(
                                                    _execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                                                )
                                            )
                                            if wait_reply:
                                                await _send_interim(wait_reply, kind="wait")
                                            while True:
                                                try:
                                                    response_json = await asyncio.wait_for(asyncio.shield(task), timeout=60.0)
                                                    break
                                                except asyncio.TimeoutError:
                                                    if wait_reply:
                                                        await _send_interim(wait_reply, kind="wait")
                                                    continue
                                        if speak:
                                            now = time.time()
                                            if now < tts_busy_until:
                                                await asyncio.sleep(tts_busy_until - now)
                                        if isinstance(response_json, dict) and "__tool_args_error__" in response_json:
                                            err = response_json["__tool_args_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        elif isinstance(response_json, dict) and "__http_error__" in response_json:
                                            err = response_json["__http_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            pagination_info = None
                                            if isinstance(response_json, dict) and "__igx_pagination__" in response_json:
                                                pagination_info = response_json.pop("__igx_pagination__", None)
                                            if bool(getattr(tool_cfg, "use_codex_response", False)):
                                                # Avoid bloating LLM-visible conversation metadata in Codex mode.
                                                tool_result = {"ok": True}
                                                new_meta = meta_current
                                            else:
                                                mapped = _apply_response_mapper(
                                                    mapper_json=tool_cfg.response_mapper_json,
                                                    response_json=response_json,
                                                    meta=meta_current,
                                                    tool_args=patch,
                                                )
                                                new_meta = merge_conversation_metadata(
                                                    session, conversation_id=conv_id, patch=mapped
                                                )
                                                meta_current = new_meta
                                                tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}
                                            if pagination_info:
                                                tool_result["pagination"] = pagination_info

                                            # Optional Codex HTTP agent (post-process the raw response JSON).
                                            # Static reply (if configured) takes priority.
                                            static_preview = ""
                                            if (tool_cfg.static_reply_template or "").strip():
                                                try:
                                                    static_preview = _render_static_reply(
                                                        template_text=tool_cfg.static_reply_template,
                                                        meta=new_meta or meta_current,
                                                        response_json=response_json,
                                                        tool_args=patch,
                                                    ).strip()
                                                except Exception:
                                                    static_preview = ""
                                            if (not static_preview) and bool(getattr(tool_cfg, "use_codex_response", False)):
                                                fields_required = str(patch.get("fields_required") or "").strip()
                                                what_to_search_for = str(patch.get("what_to_search_for") or "").strip()
                                                if not fields_required:
                                                    fields_required = what_to_search_for
                                                why_api_was_called = str(patch.get("why_api_was_called") or "").strip()
                                                if not why_api_was_called:
                                                    why_api_was_called = str(patch.get("why_to_search_for") or "").strip()
                                                if not fields_required or not why_api_was_called:
                                                    tool_result["codex_ok"] = False
                                                    tool_result["codex_error"] = "Missing fields_required / why_api_was_called."
                                                else:
                                                    fields_required_for_codex = fields_required
                                                    if what_to_search_for:
                                                        fields_required_for_codex = (
                                                            f"{fields_required_for_codex}\n\n(what_to_search_for) {what_to_search_for}"
                                                        )
                                                    did_postprocess = False
                                                    postprocess_python = str(
                                                        getattr(tool_cfg, "postprocess_python", "") or ""
                                                    ).strip()
                                                    if postprocess_python:
                                                        py_task = asyncio.create_task(
                                                            asyncio.to_thread(
                                                                run_python_postprocessor,
                                                                python_code=postprocess_python,
                                                                payload={
                                                                    "response_json": response_json,
                                                                    "meta": new_meta or meta_current,
                                                                    "args": patch,
                                                                    "fields_required": fields_required,
                                                                    "why_api_was_called": why_api_was_called,
                                                                },
                                                                timeout_s=60,
                                                            )
                                                        )
                                                        if wait_reply:
                                                            await _public_send_interim(
                                                                ws, req_id=req_id, kind="wait", text=wait_reply
                                                            )
                                                        last_wait = time.time()
                                                        wait_interval_s = 15.0
                                                        while not py_task.done():
                                                            now = time.time()
                                                            if wait_reply and (now - last_wait) >= wait_interval_s:
                                                                await _public_send_interim(
                                                                    ws, req_id=req_id, kind="wait", text=wait_reply
                                                                )
                                                                last_wait = now
                                                            await asyncio.sleep(0.2)
                                                        try:
                                                            py_res = await py_task
                                                            tool_result["python_ok"] = bool(getattr(py_res, "ok", False))
                                                            tool_result["python_duration_ms"] = int(
                                                                getattr(py_res, "duration_ms", 0) or 0
                                                            )
                                                            if getattr(py_res, "error", None):
                                                                tool_result["python_error"] = str(getattr(py_res, "error"))
                                                            if getattr(py_res, "stderr", None):
                                                                tool_result["python_stderr"] = str(getattr(py_res, "stderr"))
                                                            if py_res.ok:
                                                                did_postprocess = True
                                                                tool_result["postprocess_mode"] = "python"
                                                                tool_result["codex_ok"] = True
                                                                tool_result["codex_result_text"] = str(
                                                                    getattr(py_res, "result_text", "") or ""
                                                                )
                                                                mp = getattr(py_res, "metadata_patch", None)
                                                                if isinstance(mp, dict) and mp:
                                                                    try:
                                                                        meta_current = merge_conversation_metadata(
                                                                            session,
                                                                            conversation_id=conv_id,
                                                                            patch=mp,
                                                                        )
                                                                        tool_result["python_metadata_patch"] = mp
                                                                    except Exception:
                                                                        pass
                                                                try:
                                                                    append_saved_run_index(
                                                                        conversation_id=str(conv_id),
                                                                        event={
                                                                            "kind": "integration_python_postprocess",
                                                                            "tool_name": tool_name,
                                                                            "req_id": req_id,
                                                                            "python_ok": tool_result.get("python_ok"),
                                                                            "python_duration_ms": tool_result.get(
                                                                                "python_duration_ms"
                                                                            ),
                                                                        },
                                                                    )
                                                                except Exception:
                                                                    pass
                                                        except Exception as exc:
                                                            tool_result["python_ok"] = False
                                                            tool_result["python_error"] = str(exc)

                                                    if not did_postprocess:
                                                        codex_model = (
                                                            (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                            or "gpt-5.1-codex-mini"
                                                        )
                                                        progress_q: "queue.Queue[str]" = queue.Queue()

                                                        def _progress(s: str) -> None:
                                                            try:
                                                                progress_q.put_nowait(str(s))
                                                            except Exception:
                                                                return

                                                        agent_task = asyncio.create_task(
                                                            asyncio.to_thread(
                                                                run_codex_http_agent_one_shot,
                                                                api_key=api_key or "",
                                                                model=codex_model,
                                                                response_json=response_json,
                                                                fields_required=fields_required_for_codex,
                                                                why_api_was_called=why_api_was_called,
                                                                response_schema_json=getattr(tool_cfg, "response_schema_json", "")
                                                                or "",
                                                                conversation_id=str(conv_id) if conv_id is not None else None,
                                                                req_id=req_id,
                                                                tool_codex_prompt=getattr(tool_cfg, "codex_prompt", "") or "",
                                                                progress_fn=_progress,
                                                            )
                                                        )
                                                        if wait_reply:
                                                            await _public_send_interim(
                                                                ws, req_id=req_id, kind="wait", text=wait_reply
                                                            )
                                                        last_wait = time.time()
                                                        last_progress = last_wait
                                                        wait_interval_s = 15.0
                                                        while not agent_task.done():
                                                            try:
                                                                while True:
                                                                    p = progress_q.get_nowait()
                                                                    if p:
                                                                        await _public_send_interim(
                                                                            ws,
                                                                            req_id=req_id,
                                                                            kind="progress",
                                                                            text=p,
                                                                        )
                                                                        last_progress = time.time()
                                                            except queue.Empty:
                                                                pass
                                                            now = time.time()
                                                            if (
                                                                wait_reply
                                                                and (now - last_wait) >= wait_interval_s
                                                                and (now - last_progress) >= wait_interval_s
                                                            ):
                                                                await _public_send_interim(
                                                                    ws, req_id=req_id, kind="wait", text=wait_reply
                                                                )
                                                                last_wait = now
                                                            await asyncio.sleep(0.2)
                                                        try:
                                                            agent_res = await agent_task
                                                            tool_result["postprocess_mode"] = "codex"
                                                            tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                                            tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                                            tool_result["codex_output_file"] = getattr(agent_res, "result_text_path", "")
                                                            tool_result["codex_debug_file"] = getattr(agent_res, "debug_json_path", "")
                                                            tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                                            tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                                            tool_result["codex_continue_reason"] = getattr(
                                                                agent_res, "continue_reason", ""
                                                            )
                                                            tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                                            saved_input_json_path = getattr(agent_res, "input_json_path", "")
                                                            saved_schema_json_path = getattr(agent_res, "schema_json_path", "")
                                                            err = getattr(agent_res, "error", None)
                                                            if err:
                                                                tool_result["codex_error"] = str(err)
                                                        except Exception as exc:
                                                            tool_result["codex_ok"] = False
                                                            tool_result["codex_error"] = str(exc)
                                                            saved_input_json_path = ""
                                                            saved_schema_json_path = ""

                                                        try:
                                                            append_saved_run_index(
                                                                conversation_id=str(conv_id),
                                                                event={
                                                                    "kind": "integration_http",
                                                                    "tool_name": tool_name,
                                                                    "req_id": req_id,
                                                                    "input_json_path": saved_input_json_path,
                                                                    "schema_json_path": saved_schema_json_path,
                                                                    "fields_required": fields_required,
                                                                    "why_api_was_called": why_api_was_called,
                                                                    "codex_output_dir": tool_result.get("codex_output_dir"),
                                                                    "codex_ok": tool_result.get("codex_ok"),
                                                                },
                                                            )
                                                        except Exception:
                                                            pass

                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="tool",
                                        content=json.dumps({"tool": tool_name, "result": tool_result}, ensure_ascii=False),
                                    )
                                    if isinstance(tool_result, dict):
                                        meta_current = tool_result.get("metadata") or meta_current

                                    await _ws_send_json(
                                        ws,
                                        {"type": "tool_result", "req_id": req_id, "name": tool_name, "result": tool_result},
                                    )

                                    if tool_error:
                                        break
                                    if tool_failed:
                                        break

                                    if tool_name != "set_metadata" and tool_cfg:
                                        static_text = ""
                                        if (tool_cfg.static_reply_template or "").strip():
                                            static_text = _render_static_reply(
                                                template_text=tool_cfg.static_reply_template,
                                                meta=meta_current,
                                                response_json=response_json,
                                                tool_args=patch,
                                            ).strip()
                                        if static_text:
                                            needs_followup_llm = False
                                            rendered_reply = static_text
                                        else:
                                            needs_followup_llm = _should_followup_llm_for_tool(
                                                tool=tool_cfg, static_rendered=static_text
                                            )
                                            rendered_reply = ""
                                    else:
                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                        rendered_reply = candidate or rendered_reply

                            # If static reply is missing/empty for an integration tool, ask the LLM again
                            # with tool call + tool result already persisted in history.
                            if needs_followup_llm and not tool_error and conv_id is not None:
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                                with Session(engine) as session:
                                    bot2 = get_bot(session, bot_id)
                                    followup_history = _build_history_budgeted(
                                        session=session,
                                        bot=bot2,
                                        conversation_id=conv_id,
                                        api_key=api_key,
                                        status_cb=None,
                                    )
                                    followup_history.append(
                                        Message(
                                            role="system",
                                            content=(
                                                "The previous tool call failed. "
                                                if tool_failed
                                                else ""
                                            )
                                            + "Using the latest tool result(s) above, write the next assistant reply. "
                                            "If the tool result contains codex_result_text, rephrase it for the user and do not mention file paths. "
                                            "Do not call any tools.",
                                        )
                                    )
                                follow_llm = OpenAILLM(model=bot.openai_model, api_key=api_key)
                                if debug_mode:
                                    await _emit_llm_debug_payload(
                                        ws=ws,
                                        req_id=req_id,
                                        conversation_id=conv_id,
                                        phase="tool_followup_llm",
                                        payload=follow_llm.build_request_payload(
                                            messages=followup_history, stream=True
                                        ),
                                    )
                                text2, ttfb2, total2 = await _stream_llm_reply(
                                    ws=ws, req_id=req_id, llm=follow_llm, messages=followup_history
                                )
                                rendered_reply = text2.strip()
                                if rendered_reply:
                                    followup_streamed = True
                                    in_tok = int(estimate_messages_tokens(followup_history, bot.openai_model) or 0)
                                    out_tok = int(estimate_text_tokens(rendered_reply, bot.openai_model) or 0)
                                    price = _get_openai_pricing().get(bot.openai_model)
                                    cost = float(
                                        estimate_cost_usd(model_price=price, input_tokens=in_tok, output_tokens=out_tok) or 0.0
                                    )
                                    with Session(engine) as session:
                                        add_message_with_metrics(
                                            session,
                                            conversation_id=conv_id,
                                            role="assistant",
                                            content=rendered_reply,
                                            input_tokens_est=in_tok or None,
                                            output_tokens_est=out_tok or None,
                                            cost_usd_est=cost or None,
                                            asr_ms=timings.get("asr"),
                                            llm_ttfb_ms=ttfb2,
                                            llm_total_ms=total2,
                                            total_ms=total2,
                                        )
                                        update_conversation_metrics(
                                            session,
                                            conversation_id=conv_id,
                                            add_input_tokens_est=in_tok,
                                            add_output_tokens_est=out_tok,
                                            add_cost_usd_est=cost,
                                            last_asr_ms=timings.get("asr"),
                                            last_llm_ttfb_ms=ttfb2,
                                            last_llm_total_ms=total2,
                                            last_tts_first_audio_ms=None,
                                            last_total_ms=total2,
                                        )
                                    followup_persisted = True
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "metrics",
                                            "req_id": req_id,
                                            "timings_ms": {"llm_ttfb": ttfb2, "llm_total": total2, "total": total2},
                                        },
                                    )

                            if tool_error:
                                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": tool_error})

                            if rendered_reply:
                                if not followup_streamed:
                                    await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": rendered_reply})
                                try:
                                    if not followup_persisted:
                                        in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                                            bot=bot, history=history, assistant_text=rendered_reply
                                        )
                                        with Session(engine) as session:
                                            add_message_with_metrics(
                                                session,
                                                conversation_id=conv_id,
                                                role="assistant",
                                                content=rendered_reply,
                                                input_tokens_est=in_tok,
                                                output_tokens_est=out_tok,
                                                cost_usd_est=cost,
                                                asr_ms=timings.get("asr"),
                                                llm_ttfb_ms=timings.get("llm_ttfb"),
                                                llm_total_ms=timings.get("llm_total"),
                                                total_ms=timings.get("total"),
                                            )
                                            update_conversation_metrics(
                                                session,
                                                conversation_id=conv_id,
                                                add_input_tokens_est=in_tok,
                                                add_output_tokens_est=out_tok,
                                                add_cost_usd_est=cost,
                                                last_asr_ms=timings.get("asr"),
                                                last_llm_ttfb_ms=timings.get("llm_ttfb"),
                                                last_llm_total_ms=timings.get("llm_total"),
                                                last_tts_first_audio_ms=None,
                                                last_total_ms=timings.get("total"),
                                            )
                                except Exception:
                                    pass

                                if speak:
                                    status(req_id, "tts")
                                    wav, sr = await asyncio.to_thread(tts_synth, rendered_reply)
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "audio_wav",
                                            "req_id": req_id,
                                            "wav_base64": base64.b64encode(wav).decode(),
                                            "sr": sr,
                                        },
                                    )

                            await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": rendered_reply})
                        else:
                            await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": final_text})

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})

                        active_req_id = None
                        conv_id = None
                        audio_buf = bytearray()
                        accepting_audio = False

                    else:
                        await _ws_send_json(
                            ws,
                            {"type": "error", "req_id": req_id or None, "error": f"Unknown message type: {msg_type}"},
                        )

                elif "bytes" in msg and msg["bytes"] is not None:
                    # Be tolerant to stray/late audio frames (browser worklet flush, etc.)
                    if active_req_id is None or not accepting_audio:
                        continue
                    audio_buf.extend(msg["bytes"])
                else:
                    # ignore
                    pass

        except WebSocketDisconnect:
            return
        except RuntimeError:
            # Starlette can raise RuntimeError if receive() is called after disconnect was already processed.
            return

    def _parse_allowed_bot_ids(k) -> set[str]:
        try:
            ids = json.loads(getattr(k, "allowed_bot_ids_json", "") or "[]")
            if not isinstance(ids, list):
                return set()
            return {str(x) for x in ids if isinstance(x, str) and str(x).strip()}
        except Exception:
            return set()

    def _origin_allowed(k, origin: Optional[str]) -> bool:
        allowed = (getattr(k, "allowed_origins", "") or "").strip()
        if not allowed:
            return True
        origin_val = (origin or "").strip()
        allowset = {o.strip() for o in allowed.split(",") if o.strip()}
        return origin_val in allowset

    def _bot_allowed(k, bot_id: UUID) -> bool:
        allowset = _parse_allowed_bot_ids(k)
        if not allowset:
            return True
        return str(bot_id) in allowset

    async def _public_send_done(ws: WebSocket, *, req_id: str, text: str, metrics: dict) -> None:
        await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": text, "metrics": metrics})

    async def _public_send_interim(ws: WebSocket, *, req_id: str, kind: str, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        await _ws_send_json(ws, {"type": "interim", "req_id": req_id, "kind": kind, "text": t})

    async def _public_send_greeting(
        *,
        ws: WebSocket,
        req_id: str,
        bot: Bot,
        conv_id: UUID,
        api_key: str,
    ) -> tuple[str, dict]:
        greeting_text = (bot.start_message_text or "").strip()
        llm_ttfb_ms: Optional[int] = None
        llm_total_ms: Optional[int] = None
        input_tokens_est: int = 0
        output_tokens_est: int = 0
        cost_usd_est: float = 0.0
        sent_delta = False

        if bot.start_message_mode == "static" and greeting_text:
            pass
        else:
            llm = OpenAILLM(model=bot.openai_model, api_key=api_key)
            msgs = [
                Message(role="system", content=render_template(bot.system_prompt, ctx={"meta": {}})),
                Message(role="user", content=_make_start_message_instruction(bot)),
            ]
            t0 = time.time()
            first = None
            parts: list[str] = []
            for d in llm.stream_text(messages=msgs):
                if first is None:
                    first = time.time()
                parts.append(d)
                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
                sent_delta = True
            t1 = time.time()
            greeting_text = "".join(parts).strip() or greeting_text
            if first is not None:
                llm_ttfb_ms = int(round((first - t0) * 1000.0))
            llm_total_ms = int(round((t1 - t0) * 1000.0))

            input_tokens_est = int(estimate_messages_tokens(msgs, bot.openai_model) or 0)
            output_tokens_est = int(estimate_text_tokens(greeting_text, bot.openai_model) or 0)
            price = _get_openai_pricing().get(bot.openai_model)
            cost_usd_est = float(
                estimate_cost_usd(model_price=price, input_tokens=input_tokens_est, output_tokens=output_tokens_est) or 0.0
            )

        if not greeting_text:
            greeting_text = "Hi! How can I help you today?"

        if not sent_delta:
            await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": greeting_text})

        with Session(engine) as session:
            add_message_with_metrics(
                session,
                conversation_id=conv_id,
                role="assistant",
                content=greeting_text,
                input_tokens_est=input_tokens_est or None,
                output_tokens_est=output_tokens_est or None,
                cost_usd_est=cost_usd_est or None,
                llm_ttfb_ms=llm_ttfb_ms,
                llm_total_ms=llm_total_ms,
            )
            update_conversation_metrics(
                session,
                conversation_id=conv_id,
                add_input_tokens_est=input_tokens_est,
                add_output_tokens_est=output_tokens_est,
                add_cost_usd_est=cost_usd_est,
                last_asr_ms=None,
                last_llm_ttfb_ms=llm_ttfb_ms,
                last_llm_total_ms=llm_total_ms,
                last_tts_first_audio_ms=None,
                last_total_ms=None,
            )

        metrics = {
            "model": bot.openai_model,
            "input_tokens_est": input_tokens_est,
            "output_tokens_est": output_tokens_est,
            "cost_usd_est": cost_usd_est,
            "llm_ttfb_ms": llm_ttfb_ms,
            "llm_total_ms": llm_total_ms,
        }
        return greeting_text, metrics

    @app.websocket("/public/v1/ws/bots/{bot_id}/chat")
    async def public_chat_ws(bot_id: UUID, ws: WebSocket) -> None:
        await ws.accept()

        key_secret = (ws.query_params.get("key") or "").strip()
        external_id = (ws.query_params.get("user_conversation_id") or "").strip()
        if not key_secret or not external_id:
            await _ws_send_json(ws, {"type": "error", "error": "Missing key or user_conversation_id"})
            await ws.close(code=4400)
            return

        origin = ws.headers.get("origin")
        conv_id: Optional[UUID] = None
        with Session(engine) as session:
            ck = verify_client_key(session, secret=key_secret)
            if not ck:
                await _ws_send_json(ws, {"type": "error", "error": "Invalid client key"})
                await ws.close(code=4401)
                return
            if not _origin_allowed(ck, origin):
                await _ws_send_json(ws, {"type": "error", "error": "Origin not allowed"})
                await ws.close(code=4403)
                return
            if not _bot_allowed(ck, bot_id):
                await _ws_send_json(ws, {"type": "error", "error": "Bot not allowed for this key"})
                await ws.close(code=4403)
                return
            # Create (or load) the conversation immediately on connect so we can prewarm the Data Agent
            # as soon as the conversation exists (before the first user message).
            try:
                bot = get_bot(session, bot_id)
                conv = get_or_create_conversation_by_external_id(
                    session,
                    bot_id=bot.id,
                    test_flag=False,
                    client_key_id=ck.id,
                    external_id=external_id,
                )
                conv_id = conv.id
            except Exception:
                conv_id = None

        if conv_id is not None:
            asyncio.create_task(_kickoff_data_agent_container_if_enabled(bot_id=bot_id, conversation_id=conv_id))

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    payload = json.loads(raw)
                except Exception:
                    await _ws_send_json(ws, {"type": "error", "error": "Invalid JSON"})
                    continue

                msg_type = str(payload.get("type") or "")
                req_id = str(payload.get("req_id") or "")
                if not req_id:
                    await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                    continue

                if msg_type == "start":
                    with Session(engine) as session:
                        bot = get_bot(session, bot_id)
                        ck = verify_client_key(session, secret=key_secret)
                        if not ck:
                            raise HTTPException(status_code=401, detail="Invalid client key")
                        conv = get_or_create_conversation_by_external_id(
                            session, bot_id=bot.id, test_flag=False, client_key_id=ck.id, external_id=external_id
                        )
                        conv_id = conv.id

                        await _ws_send_json(
                            ws,
                            {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)},
                        )
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})

                        if len(list_messages(session, conversation_id=conv_id)) == 0:
                            need_llm = not (
                                bot.start_message_mode == "static" and (bot.start_message_text or "").strip()
                            )
                            api_key = ""
                            if need_llm:
                                if bot.openai_key_id:
                                    crypto = require_crypto()
                                    api_key = decrypt_openai_key(session, crypto=crypto, bot=bot) or ""
                                else:
                                    api_key = os.environ.get("OPENAI_API_KEY") or ""
                                if not api_key:
                                    raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot.")
                            text, metrics = await _public_send_greeting(
                                ws=ws, req_id=req_id, bot=bot, conv_id=conv_id, api_key=api_key
                            )
                            await _public_send_done(ws, req_id=req_id, text=text, metrics=metrics)
                        else:
                            await _public_send_done(ws, req_id=req_id, text="", metrics={})

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                    continue

                if msg_type == "chat":
                    user_text = str(payload.get("text") or "").strip()
                    if not user_text:
                        await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": "Empty text"})
                        continue

                    with Session(engine) as session:
                        bot = get_bot(session, bot_id)
                        ck = verify_client_key(session, secret=key_secret)
                        if not ck:
                            raise HTTPException(status_code=401, detail="Invalid client key")
                        conv = get_or_create_conversation_by_external_id(
                            session, bot_id=bot.id, test_flag=False, client_key_id=ck.id, external_id=external_id
                        )
                        conv_id = conv.id
                        await _ws_send_json(
                            ws, {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)}
                        )

                        if bot.openai_key_id:
                            crypto = require_crypto()
                            api_key = decrypt_openai_key(session, crypto=crypto, bot=bot)
                        else:
                            api_key = os.environ.get("OPENAI_API_KEY")
                        if not api_key:
                            raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot.")

                        add_message_with_metrics(session, conversation_id=conv_id, role="user", content=user_text)
                        loop = asyncio.get_running_loop()

                        def _status_cb(stage: str) -> None:
                            asyncio.run_coroutine_threadsafe(
                                _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": stage}), loop
                            )

                        history = _build_history_budgeted(
                            session=session,
                            bot=bot,
                            conversation_id=conv_id,
                            api_key=api_key,
                            status_cb=_status_cb,
                        )
                        tools_defs = _build_tools_for_bot(session, bot.id)
                        llm = OpenAILLM(model=bot.openai_model, api_key=api_key)

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                        t0 = time.time()
                        first_token_ts: Optional[float] = None
                        full_text_parts: list[str] = []
                        tool_calls: list[ToolCall] = []

                        for ev in llm.stream_text_or_tool(messages=history, tools=tools_defs):
                            if isinstance(ev, ToolCall):
                                tool_calls.append(ev)
                                continue
                            d = str(ev)
                            if d:
                                if first_token_ts is None:
                                    first_token_ts = time.time()
                                full_text_parts.append(d)
                                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})

                        llm_end_ts = time.time()
                        rendered_reply = "".join(full_text_parts).strip()

                        llm_ttfb_ms: Optional[int] = None
                        if first_token_ts is not None:
                            llm_ttfb_ms = int(round((first_token_ts - t0) * 1000.0))
                        elif tool_calls and tool_calls[0].first_event_ts is not None:
                            llm_ttfb_ms = int(round((tool_calls[0].first_event_ts - t0) * 1000.0))
                        llm_total_ms = int(round((llm_end_ts - t0) * 1000.0))

                        if tool_calls:
                            meta_current = _get_conversation_meta(session, conversation_id=conv_id)
                            disabled_tools = _disabled_tool_names(bot)
                            final = ""
                            needs_followup_llm = False
                            tool_failed = False
                            followup_streamed = False

                            for tc in tool_calls:
                                tool_name = tc.name
                                if tool_name == "set_variable":
                                    tool_name = "set_metadata"

                                tool_args = json.loads(tc.arguments_json or "{}")
                                if not isinstance(tool_args, dict):
                                    tool_args = {}

                                add_message_with_metrics(
                                    session,
                                    conversation_id=conv_id,
                                    role="tool",
                                    content=json.dumps({"tool": tool_name, "arguments": tool_args}, ensure_ascii=False),
                                )

                                next_reply = str(tool_args.get("next_reply") or "").strip()
                                wait_reply = str(tool_args.get("wait_reply") or "").strip()
                                raw_args = tool_args.get("args")
                                if isinstance(raw_args, dict):
                                    patch = dict(raw_args)
                                else:
                                    patch = dict(tool_args)
                                    patch.pop("next_reply", None)
                                    patch.pop("wait_reply", None)
                                    patch.pop("args", None)

                                tool_cfg: IntegrationTool | None = None
                                response_json: Any | None = None
                                if tool_name in disabled_tools:
                                    tool_result = {
                                        "ok": False,
                                        "error": {"message": f"Tool '{tool_name}' is disabled for this bot."},
                                    }
                                    tool_failed = True
                                    needs_followup_llm = True
                                    final = ""
                                elif tool_name == "set_metadata":
                                    new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=patch)
                                    tool_result = {"ok": True, "updated": patch, "metadata": new_meta}
                                elif tool_name == "web_search":
                                    scrapingbee_key = (
                                        settings.scrapingbee_api_key or os.environ.get("SCRAPINGBEE_API_KEY") or ""
                                    ).strip()
                                    search_term = str(patch.get("search_term") or patch.get("query") or "").strip()
                                    vector_queries = str(
                                        patch.get("vector_search_queries") or patch.get("vector_searcg_queries") or ""
                                    ).strip()
                                    why = str(patch.get("why") or patch.get("reason") or "").strip()
                                    top_k_arg = patch.get("top_k")
                                    max_results_arg = patch.get("max_results")
                                    try:
                                        top_k_val = int(top_k_arg) if top_k_arg is not None else None
                                    except Exception:
                                        top_k_val = None
                                    try:
                                        max_results_val = int(max_results_arg) if max_results_arg is not None else None
                                    except Exception:
                                        max_results_val = None

                                    progress_q: "queue.Queue[str]" = queue.Queue()

                                    def _progress(s: str) -> None:
                                        try:
                                            progress_q.put_nowait(str(s))
                                        except Exception:
                                            return

                                    try:
                                        ws_model = (getattr(bot, "web_search_model", "") or bot.openai_model).strip()
                                        task = asyncio.create_task(
                                            asyncio.to_thread(
                                                run_web_search,
                                                search_term=search_term,
                                                vector_search_queries=vector_queries,
                                                why=why,
                                                openai_api_key=api_key or "",
                                                scrapingbee_api_key=scrapingbee_key,
                                                model=ws_model,
                                                progress_fn=_progress,
                                                top_k=top_k_val,
                                                max_results=max_results_val,
                                            )
                                        )
                                        if wait_reply:
                                            await _public_send_interim(ws, req_id=req_id, kind="wait", text=wait_reply)
                                        last_wait = time.time()
                                        while not task.done():
                                            try:
                                                while True:
                                                    p = progress_q.get_nowait()
                                                    if p:
                                                        await _public_send_interim(
                                                            ws, req_id=req_id, kind="progress", text=p
                                                        )
                                            except queue.Empty:
                                                pass
                                            if wait_reply and (time.time() - last_wait) >= 7.0:
                                                await _public_send_interim(ws, req_id=req_id, kind="wait", text=wait_reply)
                                                last_wait = time.time()
                                            await asyncio.sleep(0.2)
                                        summary_text = await task
                                        tool_result = str(summary_text or "").strip()
                                    except Exception as exc:
                                        tool_result = f"WEB_SEARCH_ERROR: {exc}"
                                        tool_failed = True
                                    if tool_failed or not next_reply:
                                        needs_followup_llm = True
                                        final = ""
                                elif tool_name == "give_command_to_data_agent":
                                    if not bool(getattr(bot, "enable_data_agent", False)):
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": "Data Agent is disabled for this bot."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        what_to_do = str(patch.get("what_to_do") or "").strip()
                                        if not what_to_do:
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Missing required tool arg: what_to_do"},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            final = ""
                                        else:
                                            try:
                                                logger.info(
                                                    "Data Agent tool: start conv=%s bot=%s what_to_do=%s",
                                                    conv_id,
                                                    bot_id,
                                                    (what_to_do[:200] + "…") if len(what_to_do) > 200 else what_to_do,
                                                )
                                                da = _data_agent_meta(meta_current)
                                                workspace_dir = (
                                                    str(da.get("workspace_dir") or "").strip()
                                                    or default_workspace_dir_for_conversation(conv_id)
                                                )
                                                container_id = str(da.get("container_id") or "").strip()
                                                session_id = str(da.get("session_id") or "").strip()

                                                if not container_id:
                                                    api_key = _get_openai_api_key_for_bot(session, bot=bot)
                                                    if not api_key:
                                                        raise RuntimeError(
                                                            "No OpenAI API key configured for this bot (needed for Data Agent)."
                                                        )
                                                    container_id = await asyncio.to_thread(
                                                        ensure_conversation_container,
                                                        conversation_id=conv_id,
                                                        workspace_dir=workspace_dir,
                                                        openai_api_key=api_key,
                                                    )
                                                    meta_current = merge_conversation_metadata(
                                                        session,
                                                        conversation_id=conv_id,
                                                        patch={
                                                            "data_agent.container_id": container_id,
                                                            "data_agent.workspace_dir": workspace_dir,
                                                        },
                                                    )

                                                ctx = _build_data_agent_conversation_context(
                                                    session,
                                                    bot=bot,
                                                    conversation_id=conv_id,
                                                    meta=meta_current,
                                                )
                                                api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
                                                auth_json = getattr(bot, "data_agent_auth_json", "") or "{}"
                                                sys_prompt = (
                                                    (getattr(bot, "data_agent_system_prompt", "") or "").strip()
                                                    or DEFAULT_DATA_AGENT_SYSTEM_PROMPT
                                                )

                                                task = asyncio.create_task(
                                                    asyncio.to_thread(
                                                        run_data_agent,
                                                        conversation_id=conv_id,
                                                        container_id=container_id,
                                                        session_id=session_id,
                                                        workspace_dir=workspace_dir,
                                                        api_spec_text=api_spec_text,
                                                        auth_json=auth_json,
                                                        system_prompt=sys_prompt,
                                                        conversation_context=ctx,
                                                        what_to_do=what_to_do,
                                                    )
                                                )
                                                if wait_reply:
                                                    await _public_send_interim(
                                                        ws, req_id=req_id, kind="wait", text=wait_reply
                                                    )
                                                last_wait = time.time()
                                                while not task.done():
                                                    if wait_reply and (time.time() - last_wait) >= 10.0:
                                                        await _public_send_interim(
                                                            ws, req_id=req_id, kind="wait", text=wait_reply
                                                        )
                                                        last_wait = time.time()
                                                    await asyncio.sleep(0.2)
                                                da_res = await task

                                                if da_res.session_id and da_res.session_id != session_id:
                                                    meta_current = merge_conversation_metadata(
                                                        session,
                                                        conversation_id=conv_id,
                                                        patch={"data_agent.session_id": da_res.session_id},
                                                    )
                                                logger.info(
                                                    "Data Agent tool: done conv=%s ok=%s container_id=%s session_id=%s output_file=%s error=%s",
                                                    conv_id,
                                                    bool(da_res.ok),
                                                    da_res.container_id,
                                                    da_res.session_id,
                                                    da_res.output_file,
                                                    da_res.error,
                                                )
                                                tool_result = {
                                                    "ok": bool(da_res.ok),
                                                    "result_text": da_res.result_text,
                                                    "data_agent_container_id": da_res.container_id,
                                                    "data_agent_session_id": da_res.session_id,
                                                    "data_agent_output_file": da_res.output_file,
                                                    "data_agent_debug_file": da_res.debug_file,
                                                    "error": da_res.error,
                                                }
                                                tool_failed = not bool(da_res.ok)
                                                if (
                                                    bool(getattr(bot, "data_agent_return_result_directly", False))
                                                    and bool(da_res.ok)
                                                    and str(da_res.result_text or "").strip()
                                                ):
                                                    needs_followup_llm = False
                                                    final = str(da_res.result_text or "").strip()
                                                else:
                                                    needs_followup_llm = True
                                                    final = ""
                                            except Exception as exc:
                                                logger.exception("Data Agent tool failed conv=%s bot=%s", conv_id, bot_id)
                                                tool_result = {"ok": False, "error": {"message": str(exc)}}
                                                tool_failed = True
                                                needs_followup_llm = True
                                                final = ""
                                elif tool_name == "recall_http_response":
                                    source_tool_name = str(patch.get("source_tool_name") or "").strip()
                                    source_req_id = str(patch.get("source_req_id") or "").strip()
                                    fields_required = str(patch.get("fields_required") or "").strip()
                                    why_api_was_called = str(patch.get("why_api_was_called") or "").strip()

                                    missing_keys = [
                                        k
                                        for k in ("source_tool_name", "fields_required", "why_api_was_called")
                                        if not str(patch.get(k) or "").strip()
                                    ]
                                    if missing_keys:
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": "Missing required tool args.", "missing": missing_keys},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        ev = find_saved_run(
                                            conversation_id=str(conv_id),
                                            source_tool_name=source_tool_name,
                                            source_req_id=(source_req_id or None),
                                        )
                                        if not ev:
                                            tool_result = {
                                                "ok": False,
                                                "error": {
                                                    "message": f"No saved response found for tool '{source_tool_name}' in this conversation.",
                                                    "tool_name": source_tool_name,
                                                    "req_id": source_req_id or None,
                                                },
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            final = ""
                                        else:
                                            saved_input = str(ev.get("input_json_path") or "").strip()
                                            saved_schema = str(ev.get("schema_json_path") or "").strip()
                                            if not saved_input and str(ev.get("output_dir") or "").strip():
                                                saved_input = os.path.join(str(ev.get("output_dir") or ""), "input_response.json")
                                            if not saved_schema and str(ev.get("output_dir") or "").strip():
                                                saved_schema = os.path.join(str(ev.get("output_dir") or ""), "input_schema.json")

                                            source_tool_cfg = get_integration_tool_by_name(
                                                session, bot_id=bot.id, name=source_tool_name
                                            )
                                            codex_model = (
                                                (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                or "gpt-5.1-codex-mini"
                                            )
                                            progress_q: "queue.Queue[str]" = queue.Queue()

                                            def _progress(s: str) -> None:
                                                try:
                                                    progress_q.put_nowait(str(s))
                                                except Exception:
                                                    return

                                            agent_task = asyncio.create_task(
                                                asyncio.to_thread(
                                                    run_codex_http_agent_one_shot_from_paths,
                                                    api_key=api_key or "",
                                                    model=codex_model,
                                                    input_json_path=saved_input,
                                                    input_schema_json_path=saved_schema or None,
                                                    fields_required=fields_required,
                                                    why_api_was_called=why_api_was_called,
                                                    conversation_id=str(conv_id) if conv_id is not None else None,
                                                    req_id=req_id,
                                                    tool_codex_prompt=getattr(source_tool_cfg, "codex_prompt", "") or "",
                                                    progress_fn=_progress,
                                                )
                                            )
                                            if wait_reply:
                                                await _public_send_interim(ws, req_id=req_id, kind="wait", text=wait_reply)
                                            last_wait = time.time()
                                            last_progress = last_wait
                                            wait_interval_s = 15.0
                                            while not agent_task.done():
                                                try:
                                                    while True:
                                                        p = progress_q.get_nowait()
                                                        if p:
                                                            await _public_send_interim(
                                                                ws, req_id=req_id, kind="progress", text=p
                                                            )
                                                            last_progress = time.time()
                                                except queue.Empty:
                                                    pass
                                                now = time.time()
                                                if (
                                                    wait_reply
                                                    and (now - last_wait) >= wait_interval_s
                                                    and (now - last_progress) >= wait_interval_s
                                                ):
                                                    await _public_send_interim(
                                                        ws, req_id=req_id, kind="wait", text=wait_reply
                                                    )
                                                    last_wait = now
                                                await asyncio.sleep(0.2)

                                            tool_result = {
                                                "ok": True,
                                                "recall_source_tool": source_tool_name,
                                                "recall_source_req_id": str(ev.get("req_id") or "") or None,
                                            }
                                            try:
                                                agent_res = await agent_task
                                                tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                                tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                                tool_result["codex_output_file"] = getattr(agent_res, "result_text_path", "")
                                                tool_result["codex_debug_file"] = getattr(agent_res, "debug_json_path", "")
                                                tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                                tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                                tool_result["codex_continue_reason"] = getattr(agent_res, "continue_reason", "")
                                                tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                                err = getattr(agent_res, "error", None)
                                                if err:
                                                    tool_result["codex_error"] = str(err)
                                            except Exception as exc:
                                                tool_result["codex_ok"] = False
                                                tool_result["codex_error"] = str(exc)

                                            try:
                                                append_saved_run_index(
                                                    conversation_id=str(conv_id),
                                                    event={
                                                        "kind": "recall",
                                                        "tool_name": source_tool_name,
                                                        "req_id": req_id,
                                                        "source_req_id": str(ev.get("req_id") or "") or None,
                                                        "input_json_path": saved_input,
                                                        "schema_json_path": saved_schema,
                                                        "fields_required": fields_required,
                                                        "why_api_was_called": why_api_was_called,
                                                        "codex_output_dir": tool_result.get("codex_output_dir"),
                                                        "codex_ok": tool_result.get("codex_ok"),
                                                    },
                                                )
                                            except Exception:
                                                pass

                                            needs_followup_llm = True
                                            final = ""
                                elif tool_name == "export_http_response":
                                    source_tool_name = str(patch.get("source_tool_name") or "").strip()
                                    source_req_id = str(patch.get("source_req_id") or "").strip()
                                    export_request = str(patch.get("export_request") or "").strip()
                                    output_format = str(patch.get("output_format") or "csv").strip().lower()
                                    file_name_hint = str(patch.get("file_name_hint") or "").strip()

                                    missing_keys = [
                                        k for k in ("source_tool_name", "export_request") if not str(patch.get(k) or "").strip()
                                    ]
                                    if missing_keys:
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": "Missing required tool args.", "missing": missing_keys},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        ev = find_saved_run(
                                            conversation_id=str(conv_id),
                                            source_tool_name=source_tool_name,
                                            source_req_id=(source_req_id or None),
                                        )
                                        if not ev:
                                            tool_result = {
                                                "ok": False,
                                                "error": {
                                                    "message": f"No saved response found for tool '{source_tool_name}' in this conversation.",
                                                    "tool_name": source_tool_name,
                                                    "req_id": source_req_id or None,
                                                },
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            final = ""
                                        else:
                                            saved_input = str(ev.get("input_json_path") or "").strip()
                                            saved_schema = str(ev.get("schema_json_path") or "").strip()
                                            if not saved_input and str(ev.get("output_dir") or "").strip():
                                                saved_input = os.path.join(str(ev.get("output_dir") or ""), "input_response.json")
                                            if not saved_schema and str(ev.get("output_dir") or "").strip():
                                                saved_schema = os.path.join(str(ev.get("output_dir") or ""), "input_schema.json")

                                            codex_model = (
                                                (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                or "gpt-5.1-codex-mini"
                                            )
                                            progress_q: "queue.Queue[str]" = queue.Queue()

                                            def _progress(s: str) -> None:
                                                try:
                                                    progress_q.put_nowait(str(s))
                                                except Exception:
                                                    return

                                            agent_task = asyncio.create_task(
                                                asyncio.to_thread(
                                                    run_codex_export_from_paths,
                                                    api_key=api_key or "",
                                                    model=codex_model,
                                                    input_json_path=saved_input,
                                                    input_schema_json_path=saved_schema or None,
                                                    export_request=export_request,
                                                    output_format=output_format,
                                                    conversation_id=str(conv_id) if conv_id is not None else None,
                                                    req_id=req_id,
                                                    progress_fn=_progress,
                                                )
                                            )
                                            if wait_reply:
                                                await _public_send_interim(ws, req_id=req_id, kind="wait", text=wait_reply)
                                            last_wait = time.time()
                                            last_progress = last_wait
                                            wait_interval_s = 15.0
                                            while not agent_task.done():
                                                try:
                                                    while True:
                                                        p = progress_q.get_nowait()
                                                        if p:
                                                            await _public_send_interim(
                                                                ws, req_id=req_id, kind="progress", text=p
                                                            )
                                                            last_progress = time.time()
                                                except queue.Empty:
                                                    pass
                                                now = time.time()
                                                if (
                                                    wait_reply
                                                    and (now - last_wait) >= wait_interval_s
                                                    and (now - last_progress) >= wait_interval_s
                                                ):
                                                    await _public_send_interim(
                                                        ws, req_id=req_id, kind="wait", text=wait_reply
                                                    )
                                                    last_wait = now
                                                await asyncio.sleep(0.2)

                                            tool_result = {
                                                "ok": True,
                                                "export_ok": False,
                                                "export_format": output_format,
                                                "export_source_tool": source_tool_name,
                                                "export_source_req_id": str(ev.get("req_id") or "") or None,
                                            }
                                            try:
                                                exp = await agent_task
                                                tool_result["export_ok"] = bool(getattr(exp, "ok", False))
                                                tool_result["export_output_dir"] = getattr(exp, "output_dir", "")
                                                tool_result["export_debug_file"] = getattr(exp, "debug_json_path", "")
                                                tool_result["export_file_path"] = getattr(exp, "export_file_path", "")
                                                tool_result["export_stop_reason"] = getattr(exp, "stop_reason", "")
                                                err = getattr(exp, "error", None)
                                                if err:
                                                    tool_result["export_error"] = str(err)
                                            except Exception as exc:
                                                tool_result["export_ok"] = False
                                                tool_result["export_error"] = str(exc)

                                            export_path = str(tool_result.get("export_file_path") or "").strip()
                                            if tool_result.get("export_ok") and export_path and os.path.exists(export_path):
                                                if not is_allowed_download_path(export_path):
                                                    tool_result["export_ok"] = False
                                                    tool_result["export_error"] = "Export file path not allowed for download."
                                                else:
                                                    base_name = file_name_hint or os.path.basename(export_path)
                                                    if not base_name:
                                                        base_name = os.path.basename(export_path)
                                                    mime = "text/csv" if output_format == "csv" else "application/json"
                                                    token = create_download_token(
                                                        file_path=export_path,
                                                        filename=base_name,
                                                        mime_type=mime,
                                                        conversation_id=str(conv_id),
                                                        metadata={
                                                            "source_tool_name": source_tool_name,
                                                            "source_req_id": str(ev.get("req_id") or "") or None,
                                                        },
                                                    )
                                                    tool_result["download_token"] = token
                                                    tool_result["download_url"] = _download_url_for_token(token)
                                                    try:
                                                        tool_result["size_bytes"] = int(os.path.getsize(export_path))
                                                    except Exception:
                                                        pass

                                            try:
                                                append_saved_run_index(
                                                    conversation_id=str(conv_id),
                                                    event={
                                                        "kind": "export",
                                                        "tool_name": source_tool_name,
                                                        "req_id": req_id,
                                                        "source_req_id": str(ev.get("req_id") or "") or None,
                                                        "input_json_path": saved_input,
                                                        "schema_json_path": saved_schema,
                                                        "export_format": output_format,
                                                        "export_request": export_request[:2000],
                                                        "export_file_path": tool_result.get("export_file_path"),
                                                        "download_token": tool_result.get("download_token"),
                                                    },
                                                )
                                            except Exception:
                                                pass

                                            needs_followup_llm = True
                                            final = ""
                                else:
                                    tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                    if not tool_cfg:
                                        raise RuntimeError(f"Unknown tool: {tool_name}")
                                    if not bool(getattr(tool_cfg, "enabled", True)):
                                        response_json = {
                                            "__tool_args_error__": {
                                                "missing": [],
                                                "message": f"Tool '{tool_name}' is disabled for this bot.",
                                            }
                                        }
                                    else:
                                        task = asyncio.create_task(
                                            asyncio.to_thread(
                                                _execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                                            )
                                        )
                                        if wait_reply:
                                            await _public_send_interim(ws, req_id=req_id, kind="wait", text=wait_reply)
                                        while True:
                                            try:
                                                response_json = await asyncio.wait_for(asyncio.shield(task), timeout=60.0)
                                                break
                                            except asyncio.TimeoutError:
                                                if wait_reply:
                                                    await _public_send_interim(
                                                        ws, req_id=req_id, kind="wait", text=wait_reply
                                                    )
                                                continue
                                    if isinstance(response_json, dict) and "__tool_args_error__" in response_json:
                                        err = response_json["__tool_args_error__"] or {}
                                        tool_result = {"ok": False, "error": err}
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    elif isinstance(response_json, dict) and "__http_error__" in response_json:
                                        err = response_json["__http_error__"] or {}
                                        tool_result = {"ok": False, "error": err}
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        pagination_info = None
                                        if isinstance(response_json, dict) and "__igx_pagination__" in response_json:
                                            pagination_info = response_json.pop("__igx_pagination__", None)
                                        if bool(getattr(tool_cfg, "use_codex_response", False)):
                                            # Avoid bloating LLM-visible conversation metadata in Codex mode.
                                            tool_result = {"ok": True}
                                            new_meta = meta_current
                                        else:
                                            mapped = _apply_response_mapper(
                                                mapper_json=tool_cfg.response_mapper_json,
                                                response_json=response_json,
                                                meta=meta_current,
                                                tool_args=patch,
                                            )
                                            new_meta = merge_conversation_metadata(
                                                session, conversation_id=conv_id, patch=mapped
                                            )
                                            meta_current = new_meta
                                            tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}
                                        if pagination_info:
                                            tool_result["pagination"] = pagination_info

                                        # Optional Codex HTTP agent (post-process the raw response JSON).
                                        # Static reply (if configured) takes priority.
                                        static_preview = ""
                                        if (tool_cfg.static_reply_template or "").strip():
                                            try:
                                                static_preview = _render_static_reply(
                                                    template_text=tool_cfg.static_reply_template,
                                                    meta=new_meta or meta_current,
                                                    response_json=response_json,
                                                    tool_args=patch,
                                                ).strip()
                                            except Exception:
                                                static_preview = ""
                                        if (not static_preview) and bool(getattr(tool_cfg, "use_codex_response", False)):
                                            fields_required = str(patch.get("fields_required") or "").strip()
                                            what_to_search_for = str(patch.get("what_to_search_for") or "").strip()
                                            if not fields_required:
                                                fields_required = what_to_search_for
                                            why_api_was_called = str(patch.get("why_api_was_called") or "").strip()
                                            if not why_api_was_called:
                                                why_api_was_called = str(patch.get("why_to_search_for") or "").strip()
                                            if not fields_required or not why_api_was_called:
                                                tool_result["codex_ok"] = False
                                                tool_result["codex_error"] = "Missing fields_required / why_api_was_called."
                                            else:
                                                fields_required_for_codex = fields_required
                                                if what_to_search_for:
                                                    fields_required_for_codex = (
                                                        f"{fields_required_for_codex}\n\n(what_to_search_for) {what_to_search_for}"
                                                    )
                                                did_postprocess = False
                                                postprocess_python = str(getattr(tool_cfg, "postprocess_python", "") or "").strip()
                                                if postprocess_python:
                                                    py_task = asyncio.create_task(
                                                        asyncio.to_thread(
                                                            run_python_postprocessor,
                                                            python_code=postprocess_python,
                                                            payload={
                                                                "response_json": response_json,
                                                                "meta": new_meta or meta_current,
                                                                "args": patch,
                                                                "fields_required": fields_required,
                                                                "why_api_was_called": why_api_was_called,
                                                            },
                                                            timeout_s=60,
                                                        )
                                                    )
                                                    if wait_reply:
                                                        await _public_send_interim(
                                                            ws, req_id=req_id, kind="wait", text=wait_reply
                                                        )
                                                    last_wait = time.time()
                                                    wait_interval_s = 15.0
                                                    while not py_task.done():
                                                        now = time.time()
                                                        if wait_reply and (now - last_wait) >= wait_interval_s:
                                                            await _public_send_interim(
                                                                ws, req_id=req_id, kind="wait", text=wait_reply
                                                            )
                                                            last_wait = now
                                                        await asyncio.sleep(0.2)
                                                    try:
                                                        py_res = await py_task
                                                        tool_result["python_ok"] = bool(getattr(py_res, "ok", False))
                                                        tool_result["python_duration_ms"] = int(
                                                            getattr(py_res, "duration_ms", 0) or 0
                                                        )
                                                        if getattr(py_res, "error", None):
                                                            tool_result["python_error"] = str(getattr(py_res, "error"))
                                                        if getattr(py_res, "stderr", None):
                                                            tool_result["python_stderr"] = str(getattr(py_res, "stderr"))
                                                        if py_res.ok:
                                                            did_postprocess = True
                                                            tool_result["postprocess_mode"] = "python"
                                                            tool_result["codex_ok"] = True
                                                            tool_result["codex_result_text"] = str(
                                                                getattr(py_res, "result_text", "") or ""
                                                            )
                                                            mp = getattr(py_res, "metadata_patch", None)
                                                            if isinstance(mp, dict) and mp:
                                                                try:
                                                                    meta_current = merge_conversation_metadata(
                                                                        session,
                                                                        conversation_id=conv_id,
                                                                        patch=mp,
                                                                    )
                                                                    tool_result["python_metadata_patch"] = mp
                                                                except Exception:
                                                                    pass
                                                            try:
                                                                append_saved_run_index(
                                                                    conversation_id=str(conv_id),
                                                                    event={
                                                                        "kind": "integration_python_postprocess",
                                                                        "tool_name": tool_name,
                                                                        "req_id": req_id,
                                                                        "python_ok": tool_result.get("python_ok"),
                                                                        "python_duration_ms": tool_result.get(
                                                                            "python_duration_ms"
                                                                        ),
                                                                    },
                                                                )
                                                            except Exception:
                                                                pass
                                                    except Exception as exc:
                                                        tool_result["python_ok"] = False
                                                        tool_result["python_error"] = str(exc)

                                                if not did_postprocess:
                                                    codex_model = (
                                                        (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                        or "gpt-5.1-codex-mini"
                                                    )
                                                    progress_q: "queue.Queue[str]" = queue.Queue()

                                                    def _progress(s: str) -> None:
                                                        try:
                                                            progress_q.put_nowait(str(s))
                                                        except Exception:
                                                            return

                                                    agent_task = asyncio.create_task(
                                                        asyncio.to_thread(
                                                            run_codex_http_agent_one_shot,
                                                            api_key=api_key or "",
                                                            model=codex_model,
                                                            response_json=response_json,
                                                            fields_required=fields_required_for_codex,
                                                            why_api_was_called=why_api_was_called,
                                                            response_schema_json=getattr(tool_cfg, "response_schema_json", "")
                                                            or "",
                                                            conversation_id=str(conv_id) if conv_id is not None else None,
                                                            req_id=req_id,
                                                            tool_codex_prompt=getattr(tool_cfg, "codex_prompt", "") or "",
                                                            progress_fn=_progress,
                                                        )
                                                    )
                                                    if wait_reply:
                                                        await _public_send_interim(
                                                            ws, req_id=req_id, kind="wait", text=wait_reply
                                                        )
                                                    last_wait = time.time()
                                                    last_progress = last_wait
                                                    wait_interval_s = 15.0
                                                    while not agent_task.done():
                                                        try:
                                                            while True:
                                                                p = progress_q.get_nowait()
                                                                if p:
                                                                    await _public_send_interim(
                                                                        ws, req_id=req_id, kind="progress", text=p
                                                                    )
                                                                    last_progress = time.time()
                                                        except queue.Empty:
                                                            pass
                                                        now = time.time()
                                                        if (
                                                            wait_reply
                                                            and (now - last_wait) >= wait_interval_s
                                                            and (now - last_progress) >= wait_interval_s
                                                        ):
                                                            await _public_send_interim(
                                                                ws, req_id=req_id, kind="wait", text=wait_reply
                                                            )
                                                            last_wait = now
                                                        await asyncio.sleep(0.2)
                                                    try:
                                                        agent_res = await agent_task
                                                        tool_result["postprocess_mode"] = "codex"
                                                        tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                                        tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                                        tool_result["codex_output_file"] = getattr(
                                                            agent_res, "result_text_path", ""
                                                        )
                                                        tool_result["codex_debug_file"] = getattr(agent_res, "debug_json_path", "")
                                                        tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                                        tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                                        tool_result["codex_continue_reason"] = getattr(
                                                            agent_res, "continue_reason", ""
                                                        )
                                                        tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                                        saved_input_json_path = getattr(agent_res, "input_json_path", "")
                                                        saved_schema_json_path = getattr(agent_res, "schema_json_path", "")
                                                        err = getattr(agent_res, "error", None)
                                                        if err:
                                                            tool_result["codex_error"] = str(err)
                                                    except Exception as exc:
                                                        tool_result["codex_ok"] = False
                                                        tool_result["codex_error"] = str(exc)
                                                        saved_input_json_path = ""
                                                        saved_schema_json_path = ""

                                                    try:
                                                        append_saved_run_index(
                                                            conversation_id=str(conv_id),
                                                            event={
                                                                "kind": "integration_http",
                                                                "tool_name": tool_name,
                                                                "req_id": req_id,
                                                                "input_json_path": saved_input_json_path,
                                                                "schema_json_path": saved_schema_json_path,
                                                                "fields_required": fields_required,
                                                                "why_api_was_called": why_api_was_called,
                                                                "codex_output_dir": tool_result.get("codex_output_dir"),
                                                                "codex_ok": tool_result.get("codex_ok"),
                                                            },
                                                        )
                                                    except Exception:
                                                        pass

                                add_message_with_metrics(
                                    session,
                                    conversation_id=conv_id,
                                    role="tool",
                                    content=json.dumps({"tool": tool_name, "result": tool_result}, ensure_ascii=False),
                                )
                                if isinstance(tool_result, dict):
                                    meta_current = tool_result.get("metadata") or meta_current

                                if tool_failed:
                                    break

                                candidate = ""
                                if tool_name != "set_metadata" and tool_cfg:
                                    static_text = ""
                                    if (tool_cfg.static_reply_template or "").strip():
                                        static_text = _render_static_reply(
                                            template_text=tool_cfg.static_reply_template,
                                            meta=meta_current,
                                            response_json=response_json,
                                            tool_args=patch,
                                        ).strip()
                                    if static_text:
                                        needs_followup_llm = False
                                        final = static_text
                                    else:
                                        if bool(getattr(tool_cfg, "use_codex_response", False)):
                                            if bool(getattr(tool_cfg, "return_result_directly", False)) and isinstance(
                                                tool_result, dict
                                            ):
                                                direct = str(tool_result.get("codex_result_text") or "").strip()
                                                if direct:
                                                    needs_followup_llm = False
                                                    final = direct
                                                else:
                                                    needs_followup_llm = True
                                                    final = ""
                                            else:
                                                needs_followup_llm = True
                                                final = ""
                                        else:
                                            needs_followup_llm = _should_followup_llm_for_tool(
                                                tool=tool_cfg, static_rendered=static_text
                                            )
                                            candidate = _render_with_meta(next_reply, meta_current).strip()
                                            if candidate:
                                                final = candidate
                                                needs_followup_llm = False
                                            else:
                                                final = ""
                                else:
                                    candidate = _render_with_meta(next_reply, meta_current).strip()
                                    final = candidate or final

                            if needs_followup_llm:
                                followup_history = _build_history_budgeted(
                                    session=session,
                                    bot=bot,
                                    conversation_id=conv_id,
                                    api_key=api_key,
                                    status_cb=None,
                                )
                                followup_history.append(
                                    Message(
                                        role="system",
                                        content=(
                                            ("The previous tool call failed. " if tool_failed else "")
                                            + "Using the latest tool result(s) above, write the next assistant reply. "
                                            "If the tool result contains codex_result_text, rephrase it for the user and do not mention file paths. "
                                            "Do not call any tools."
                                        ),
                                    )
                                )
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                                text2, ttfb2, total2 = await _stream_llm_reply(
                                    ws=ws, req_id=req_id, llm=llm, messages=followup_history
                                )
                                rendered_reply = text2.strip()
                                followup_streamed = True
                                llm_ttfb_ms = ttfb2
                                llm_total_ms = total2
                            else:
                                rendered_reply = final

                            if rendered_reply and not followup_streamed:
                                await _ws_send_json(
                                    ws, {"type": "text_delta", "req_id": req_id, "delta": rendered_reply}
                                )

                        in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                            bot=bot, history=history, assistant_text=rendered_reply
                        )
                        add_message_with_metrics(
                            session,
                            conversation_id=conv_id,
                            role="assistant",
                            content=rendered_reply,
                            input_tokens_est=in_tok,
                            output_tokens_est=out_tok,
                            cost_usd_est=cost,
                            llm_ttfb_ms=llm_ttfb_ms,
                            llm_total_ms=llm_total_ms,
                            total_ms=llm_total_ms,
                        )
                        update_conversation_metrics(
                            session,
                            conversation_id=conv_id,
                            add_input_tokens_est=in_tok,
                            add_output_tokens_est=out_tok,
                            add_cost_usd_est=cost,
                            last_asr_ms=None,
                            last_llm_ttfb_ms=llm_ttfb_ms,
                            last_llm_total_ms=llm_total_ms,
                            last_tts_first_audio_ms=None,
                            last_total_ms=llm_total_ms,
                        )

                        metrics = {
                            "model": bot.openai_model,
                            "input_tokens_est": in_tok,
                            "output_tokens_est": out_tok,
                            "cost_usd_est": cost,
                            "llm_ttfb_ms": llm_ttfb_ms,
                            "llm_total_ms": llm_total_ms,
                        }
                        await _public_send_done(ws, req_id=req_id, text=rendered_reply, metrics=metrics)
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                    continue

                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": f"Unknown message type: {msg_type}"})

        except WebSocketDisconnect:
            return
        except RuntimeError:
            return

    @app.get("/api/tts/meta")
    def tts_meta(model_name: str) -> dict:
        return _get_tts_meta(model_name)

    @app.get("/public/widget.js")
    def public_widget_js() -> Response:
        p = Path(__file__).parent / "static" / "embed-widget.js"
        if not p.exists():
            raise HTTPException(status_code=404, detail="widget.js not found")
        return Response(content=p.read_text("utf-8"), media_type="application/javascript")

    @app.get("/api/options")
    def api_options() -> dict:
        pricing = _get_openai_pricing()
        dynamic_models: list[str] = []
        try:
            # Only fetch occasionally to avoid slow UI loads.
            now = time.time()
            if (now - float(openai_models_cache.get("ts") or 0.0)) > 3600.0:
                openai_models_cache["ts"] = now
                openai_models_cache["models"] = []
                api_key = (os.environ.get("OPENAI_API_KEY") or settings.openai_api_key or "").strip()
                if api_key:
                    try:
                        from openai import OpenAI  # type: ignore

                        client = OpenAI(api_key=api_key)
                        resp = client.models.list()
                        data = getattr(resp, "data", None) or []
                        ids: list[str] = []
                        for m in data:
                            mid = getattr(m, "id", None)
                            if not isinstance(mid, str) or not mid.strip():
                                continue
                            mid = mid.strip()
                            # Keep LLM-ish models; drop embeddings/audio/moderation/image models.
                            if not (mid.startswith("gpt-") or mid.startswith("o")):
                                continue
                            if mid.startswith(("tts-", "whisper-", "text-embedding-", "omni-moderation", "gpt-4o-mini-tts")):
                                continue
                            ids.append(mid)
                        openai_models_cache["models"] = sorted(set(ids))
                    except Exception:
                        # Ignore fetch failures; fall back to the curated list.
                        openai_models_cache["models"] = []
            dynamic_models = list(openai_models_cache.get("models") or [])
        except Exception:
            dynamic_models = []

        openai_models = sorted(set(ui_options.get("openai_models", []) + list(pricing.keys()) + dynamic_models))
        return {
            "openai_models": openai_models,
            "openai_pricing": {k: {"input_per_1m": v.input_per_1m, "output_per_1m": v.output_per_1m} for k, v in pricing.items()},
            "whisper_models": ui_options.get("whisper_models", []),
            "whisper_devices": ui_options.get("whisper_devices", []),
            "languages": ui_options.get("languages", []),
            "xtts_models": ui_options.get("xtts_models", []),
            "openai_tts_models": ui_options.get("openai_tts_models", []),
            "openai_tts_voices": ui_options.get("openai_tts_voices", []),
            "start_message_modes": ["llm", "static"],
            "asr_vendors": ["whisper_local"],
            "tts_vendors": ["xtts_local", "openai_tts"],
            "http_methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],
        }

    @app.get("/api/downloads/{token}")
    def download_file(token: str) -> FileResponse:
        """
        Download a previously exported file by token.

        NOTE: This uses an unguessable token and a strict allowlist of temp roots.
        TODO(cleanup): add TTL + authentication/authorization as needed for production.
        """
        obj = load_download_token(token=token)
        if not obj:
            raise HTTPException(status_code=404, detail="Download token not found")
        file_path = str(obj.get("file_path") or "").strip()
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        if not is_allowed_download_path(file_path):
            raise HTTPException(status_code=403, detail="File path not allowed")
        filename = str(obj.get("filename") or os.path.basename(file_path) or "download").strip()
        mime_type = str(obj.get("mime_type") or "application/octet-stream").strip()
        return FileResponse(path=file_path, media_type=mime_type, filename=filename)

    def _bot_to_dict(bot: Bot) -> dict:
        disabled = _disabled_tool_names(bot)
        return {
            "id": str(bot.id),
            "name": bot.name,
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
            "disabled_tools": sorted(disabled),
            "openai_key_id": str(bot.openai_key_id) if bot.openai_key_id else None,
            "system_prompt": bot.system_prompt,
            "language": bot.language,
            "tts_language": bot.tts_language,
            "tts_vendor": bot.tts_vendor,
            "whisper_model": bot.whisper_model,
            "whisper_device": bot.whisper_device,
            "xtts_model": bot.xtts_model,
            "speaker_id": bot.speaker_id,
            "speaker_wav": bot.speaker_wav,
            "openai_tts_model": bot.openai_tts_model,
            "openai_tts_voice": bot.openai_tts_voice,
            "openai_tts_speed": float(bot.openai_tts_speed),
            "tts_split_sentences": bool(bot.tts_split_sentences),
            "tts_chunk_min_chars": int(bot.tts_chunk_min_chars),
            "tts_chunk_max_chars": int(bot.tts_chunk_max_chars),
            "start_message_mode": bot.start_message_mode,
            "start_message_text": bot.start_message_text,
            "created_at": bot.created_at.isoformat(),
            "updated_at": bot.updated_at.isoformat(),
        }

    def _tool_to_dict(t: IntegrationTool) -> dict:
        return {
            "id": str(t.id),
            "bot_id": str(t.bot_id),
            "name": t.name,
            "description": t.description,
            "url": t.url,
            "method": t.method,
            "enabled": bool(getattr(t, "enabled", True)),
            "args_required": _parse_required_args_json(getattr(t, "args_required_json", "[]")),
            # Never expose secret headers (write-only). Return masked preview for UI.
            "headers_template_json": "{}",
            "headers_template_json_masked": _mask_headers_json(t.headers_template_json),
            "headers_configured": _headers_configured(t.headers_template_json),
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

    @app.get("/api/bots")
    def api_list_bots(session: Session = Depends(get_session)) -> dict:
        bots = list_bots(session)
        stats = bots_aggregate_metrics(session)
        items = []
        for b in bots:
            d = _bot_to_dict(b)
            d["stats"] = stats.get(
                b.id,
                {
                    "conversations": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "avg_llm_ttfb_ms": None,
                    "avg_llm_total_ms": None,
                    "avg_total_ms": None,
                },
            )
            items.append(d)
        return {"items": items}

    @app.post("/api/bots")
    def api_create_bot(payload: BotCreateRequest, session: Session = Depends(get_session)) -> dict:
        bot = Bot(
            name=payload.name,
            openai_model=payload.openai_model,
            web_search_model=(payload.web_search_model or payload.openai_model).strip() or payload.openai_model,
            codex_model=(payload.codex_model or "gpt-5.1-codex-mini").strip() or "gpt-5.1-codex-mini",
            summary_model=(payload.summary_model or "gpt-5-nano").strip() or "gpt-5-nano",
            history_window_turns=int(payload.history_window_turns or 16),
            enable_data_agent=bool(getattr(payload, "enable_data_agent", False)),
            data_agent_api_spec_text=(payload.data_agent_api_spec_text or ""),
            data_agent_auth_json=(payload.data_agent_auth_json or "{}"),
            data_agent_system_prompt=(payload.data_agent_system_prompt or ""),
            data_agent_return_result_directly=bool(getattr(payload, "data_agent_return_result_directly", False)),
            data_agent_prewarm_on_start=bool(getattr(payload, "data_agent_prewarm_on_start", False)),
            system_prompt=payload.system_prompt,
            language=payload.language,
            tts_language=payload.tts_language,
            tts_vendor=(payload.tts_vendor or "xtts_local").strip() or "xtts_local",
            whisper_model=payload.whisper_model,
            whisper_device=payload.whisper_device,
            xtts_model=payload.xtts_model,
            speaker_id=(payload.speaker_id or "").strip() or None,
            speaker_wav=(payload.speaker_wav or "").strip() or None,
            openai_tts_model=(payload.openai_tts_model or "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts",
            openai_tts_voice=(payload.openai_tts_voice or "alloy").strip() or "alloy",
            openai_tts_speed=float(payload.openai_tts_speed or 1.0),
            openai_key_id=payload.openai_key_id,
            tts_split_sentences=bool(payload.tts_split_sentences),
            tts_chunk_min_chars=int(payload.tts_chunk_min_chars),
            tts_chunk_max_chars=int(payload.tts_chunk_max_chars),
            start_message_mode=(payload.start_message_mode or "llm").strip() or "llm",
            start_message_text=payload.start_message_text or "",
        )
        create_bot(session, bot)
        return _bot_to_dict(bot)

    @app.get("/api/bots/{bot_id}")
    def api_get_bot(bot_id: UUID, session: Session = Depends(get_session)) -> dict:
        bot = get_bot(session, bot_id)
        return _bot_to_dict(bot)

    @app.put("/api/bots/{bot_id}")
    def api_update_bot(bot_id: UUID, payload: BotUpdateRequest, session: Session = Depends(get_session)) -> dict:
        patch = {}
        for k, v in payload.model_dump(exclude_unset=True).items():
            if k in ("speaker_id", "speaker_wav"):
                patch[k] = (v or "").strip() or None
            elif k in (
                "tts_vendor",
                "openai_tts_model",
                "openai_tts_voice",
                "web_search_model",
                "codex_model",
                "summary_model",
            ):
                patch[k] = (v or "").strip()
            elif k == "openai_tts_speed":
                patch[k] = float(v) if v is not None else 1.0
            elif k == "history_window_turns":
                try:
                    n = int(v) if v is not None else 16
                except Exception:
                    n = 16
                if n < 1:
                    n = 1
                if n > 64:
                    n = 64
                patch[k] = n
            elif k == "disabled_tools":
                vals = v or []
                if not isinstance(vals, list):
                    vals = []
                cleaned: list[str] = []
                for x in vals:
                    s = str(x or "").strip()
                    if not s:
                        continue
                    if s in ("set_metadata", "set_variable"):
                        continue
                    if s not in cleaned:
                        cleaned.append(s)
                patch["disabled_tools_json"] = json.dumps(cleaned, ensure_ascii=False)
            else:
                patch[k] = v
        bot = update_bot(session, bot_id, patch)
        return _bot_to_dict(bot)

    @app.delete("/api/bots/{bot_id}")
    def api_delete_bot(bot_id: UUID, session: Session = Depends(get_session)) -> dict:
        delete_bot(session, bot_id)
        return {"ok": True}

    @app.get("/api/bots/{bot_id}/tools")
    def api_list_tools(bot_id: UUID, session: Session = Depends(get_session)) -> dict:
        bot = get_bot(session, bot_id)
        tools = list_integration_tools(session, bot_id=bot_id)
        disabled = _disabled_tool_names(bot)
        return {
            "items": [_tool_to_dict(t) for t in tools],
            "system_tools": _system_tools_public_list(bot=bot, disabled=disabled),
            "disabled_tools": sorted(disabled),
        }

    @app.post("/api/bots/{bot_id}/tools")
    def api_create_tool(bot_id: UUID, payload: IntegrationToolCreateRequest, session: Session = Depends(get_session)) -> dict:
        _ = get_bot(session, bot_id)
        name = (payload.name or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Tool name is required")
        if get_integration_tool_by_name(session, bot_id=bot_id, name=name):
            raise HTTPException(status_code=400, detail="Tool name already exists for this bot")
        tool = IntegrationTool(
            bot_id=bot_id,
            name=name,
            description=payload.description or "",
            url=payload.url,
            method=(payload.method or "GET").upper(),
            use_codex_response=bool(payload.use_codex_response),
            enabled=bool(payload.enabled),
            args_required_json=json.dumps(payload.args_required or [], ensure_ascii=False),
            headers_template_json=payload.headers_template_json or "{}",
            request_body_template=payload.request_body_template or "{}",
            parameters_schema_json=payload.parameters_schema_json or "",
            response_schema_json=payload.response_schema_json or "",
            codex_prompt=(payload.codex_prompt or ""),
            postprocess_python=(payload.postprocess_python or ""),
            return_result_directly=bool(payload.return_result_directly),
            response_mapper_json=payload.response_mapper_json or "{}",
            pagination_json=payload.pagination_json or "",
            static_reply_template=payload.static_reply_template or "",
        )
        create_integration_tool(session, tool)
        return _tool_to_dict(tool)

    @app.put("/api/bots/{bot_id}/tools/{tool_id}")
    def api_update_tool(
        bot_id: UUID, tool_id: UUID, payload: IntegrationToolUpdateRequest, session: Session = Depends(get_session)
    ) -> dict:
        _ = get_bot(session, bot_id)
        tool = get_integration_tool(session, tool_id)
        if tool.bot_id != bot_id:
            raise HTTPException(status_code=400, detail="Tool does not belong to bot")
        patch = payload.model_dump(exclude_unset=True)
        if "name" in patch:
            name = str(patch["name"] or "").strip()
            if not name:
                raise HTTPException(status_code=400, detail="Tool name is required")
            existing = get_integration_tool_by_name(session, bot_id=bot_id, name=name)
            if existing and existing.id != tool_id:
                raise HTTPException(status_code=400, detail="Tool name already exists for this bot")
            patch["name"] = name
        if "method" in patch and patch["method"] is not None:
            patch["method"] = str(patch["method"]).upper()
        if "args_required" in patch:
            patch["args_required_json"] = json.dumps(patch.pop("args_required") or [], ensure_ascii=False)
        if "headers_template_json" in patch:
            patch["headers_template_json"] = patch["headers_template_json"] or "{}"
        if "parameters_schema_json" in patch:
            patch["parameters_schema_json"] = patch["parameters_schema_json"] or ""
        if "response_schema_json" in patch:
            patch["response_schema_json"] = patch["response_schema_json"] or ""
        if "codex_prompt" in patch:
            patch["codex_prompt"] = patch["codex_prompt"] or ""
        if "postprocess_python" in patch:
            patch["postprocess_python"] = patch["postprocess_python"] or ""
        if "return_result_directly" in patch and patch["return_result_directly"] is not None:
            patch["return_result_directly"] = bool(patch["return_result_directly"])
        if "pagination_json" in patch:
            patch["pagination_json"] = patch["pagination_json"] or ""
        tool = update_integration_tool(session, tool_id, patch)
        return _tool_to_dict(tool)

    @app.delete("/api/bots/{bot_id}/tools/{tool_id}")
    def api_delete_tool(bot_id: UUID, tool_id: UUID, session: Session = Depends(get_session)) -> dict:
        _ = get_bot(session, bot_id)
        tool = get_integration_tool(session, tool_id)
        if tool.bot_id != bot_id:
            raise HTTPException(status_code=400, detail="Tool does not belong to bot")
        delete_integration_tool(session, tool_id)
        return {"ok": True}

    @app.get("/api/keys")
    def api_list_keys(provider: Optional[str] = None, session: Session = Depends(get_session)) -> dict:
        keys = list_keys(session, provider=provider)
        return {
            "items": [
                {
                    "id": str(k.id),
                    "provider": k.provider,
                    "name": k.name,
                    "hint": k.hint,
                    "created_at": k.created_at.isoformat(),
                }
                for k in keys
            ]
        }

    @app.post("/api/keys")
    def api_create_key(payload: ApiKeyCreateRequest, session: Session = Depends(get_session)) -> dict:
        provider = (payload.provider or "").strip() or "openai"
        if provider != "openai":
            raise HTTPException(status_code=400, detail="Only provider=openai is supported right now.")
        crypto = require_crypto()
        k = create_key(session, crypto=crypto, provider=provider, name=payload.name, secret=payload.secret)
        return {
            "id": str(k.id),
            "provider": k.provider,
            "name": k.name,
            "hint": k.hint,
            "created_at": k.created_at.isoformat(),
        }

    @app.delete("/api/keys/{key_id}")
    def api_delete_key(key_id: UUID, session: Session = Depends(get_session)) -> dict:
        # Prevent deleting a key that's still referenced by a bot.
        bots = list_bots(session)
        if any(b.openai_key_id == key_id for b in bots):
            raise HTTPException(status_code=400, detail="Key is in use by one or more bots. Remove it from bots first.")
        delete_key(session, key_id)
        return {"ok": True}

    @app.get("/api/client-keys")
    def api_list_client_keys(session: Session = Depends(get_session)) -> dict:
        items = []
        for k in list_client_keys(session):
            try:
                allowed_bot_ids = json.loads(k.allowed_bot_ids_json or "[]")
                if not isinstance(allowed_bot_ids, list):
                    allowed_bot_ids = []
            except Exception:
                allowed_bot_ids = []
            items.append(
                {
                    "id": str(k.id),
                    "name": k.name,
                    "hint": k.hint,
                    "allowed_origins": k.allowed_origins,
                    "allowed_bot_ids": [str(x) for x in allowed_bot_ids if isinstance(x, str)],
                    "created_at": k.created_at.isoformat(),
                }
            )
        return {"items": items}

    @app.post("/api/client-keys")
    def api_create_client_key(payload: ClientKeyCreateRequest, session: Session = Depends(get_session)) -> dict:
        name = (payload.name or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name is required")
        secret = (payload.secret or "").strip()
        generated = False
        if not secret:
            generated = True
            secret = "igx_" + secrets.token_urlsafe(24)
        allowed_bot_ids = [str(x) for x in (payload.allowed_bot_ids or []) if str(x).strip()]
        k = create_client_key(
            session,
            name=name,
            secret=secret,
            allowed_origins=(payload.allowed_origins or "").strip(),
            allowed_bot_ids=allowed_bot_ids,
        )
        out = {
            "id": str(k.id),
            "name": k.name,
            "hint": k.hint,
            "allowed_origins": k.allowed_origins,
            "allowed_bot_ids": allowed_bot_ids,
            "created_at": k.created_at.isoformat(),
        }
        if generated:
            out["secret"] = secret
        return out

    @app.delete("/api/client-keys/{key_id}")
    def api_delete_client_key(key_id: UUID, session: Session = Depends(get_session)) -> dict:
        _ = get_client_key(session, key_id)
        delete_client_key(session, key_id)
        return {"ok": True}

    @app.get("/api/conversations")
    def api_list_conversations(
        page: int = 1,
        page_size: int = 50,
        bot_id: Optional[UUID] = None,
        test_flag: Optional[bool] = None,
        session: Session = Depends(get_session),
    ) -> dict:
        page = max(1, int(page))
        page_size = min(200, max(10, int(page_size)))
        offset = (page - 1) * page_size
        total = count_conversations(session, bot_id=bot_id, test_flag=test_flag)
        convs = list_conversations(session, bot_id=bot_id, test_flag=test_flag, limit=page_size, offset=offset)
        bots_by_id = {b.id: b for b in list_bots(session)}
        items = []
        for c in convs:
            b = bots_by_id.get(c.bot_id)
            items.append(
                {
                    "id": str(c.id),
                    "bot_id": str(c.bot_id),
                    "bot_name": b.name if b else None,
                    "test_flag": bool(c.test_flag),
                    "metadata_json": c.metadata_json or "{}",
                    "llm_input_tokens_est": int(c.llm_input_tokens_est or 0),
                    "llm_output_tokens_est": int(c.llm_output_tokens_est or 0),
                    "cost_usd_est": float(c.cost_usd_est or 0.0),
                    "last_asr_ms": c.last_asr_ms,
                    "last_llm_ttfb_ms": c.last_llm_ttfb_ms,
                    "last_llm_total_ms": c.last_llm_total_ms,
                    "last_tts_first_audio_ms": c.last_tts_first_audio_ms,
                    "last_total_ms": c.last_total_ms,
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                }
            )
        return {"items": items, "page": page, "page_size": page_size, "total": total}

    @app.get("/api/conversations/{conversation_id}")
    def api_conversation_detail(conversation_id: UUID, session: Session = Depends(get_session)) -> dict:
        conv = get_conversation(session, conversation_id)
        bot = get_bot(session, conv.bot_id)
        msgs_raw = list_messages(session, conversation_id=conversation_id)

        def _safe_json_loads(s: str) -> dict | None:
            try:
                obj = json.loads(s)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None

        messages: list[dict] = []
        for m in msgs_raw:
            tool_obj = _safe_json_loads(m.content) if m.role == "tool" else None
            tool_name = tool_obj.get("tool") if tool_obj and isinstance(tool_obj.get("tool"), str) else None
            tool_kind = None
            if tool_obj:
                if "arguments" in tool_obj:
                    tool_kind = "call"
                elif "result" in tool_obj:
                    tool_kind = "result"
            messages.append(
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                    "tool": tool_obj,
                    "tool_name": tool_name,
                    "tool_kind": tool_kind,
                    "metrics": {
                        "in": m.input_tokens_est,
                        "out": m.output_tokens_est,
                        "cost": m.cost_usd_est,
                        "asr": m.asr_ms,
                        "llm1": m.llm_ttfb_ms,
                        "llm": m.llm_total_ms,
                        "tts1": m.tts_first_audio_ms,
                        "total": m.total_ms,
                    },
                }
            )

        return {
            "conversation": {
                "id": str(conv.id),
                "bot_id": str(conv.bot_id),
                "bot_name": bot.name,
                "test_flag": bool(conv.test_flag),
                "metadata_json": conv.metadata_json or "{}",
                "llm_input_tokens_est": int(conv.llm_input_tokens_est or 0),
                "llm_output_tokens_est": int(conv.llm_output_tokens_est or 0),
                "cost_usd_est": float(conv.cost_usd_est or 0.0),
                "last_asr_ms": conv.last_asr_ms,
                "last_llm_ttfb_ms": conv.last_llm_ttfb_ms,
                "last_llm_total_ms": conv.last_llm_total_ms,
                "last_tts_first_audio_ms": conv.last_tts_first_audio_ms,
                "last_total_ms": conv.last_total_ms,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            },
            "bot": _bot_to_dict(bot),
            "messages": messages,
        }

    @app.post("/api/bots/{bot_id}/chat/stream")
    def chat_stream(
        bot_id: UUID,
        payload: ChatRequest = Body(...),
        session: Session = Depends(get_session),
    ) -> StreamingResponse:
        bot = get_bot(session, bot_id)
        if bot.openai_key_id:
            crypto = require_crypto()
            api_key = decrypt_openai_key(session, crypto=crypto, bot=bot)
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot.")

        llm = OpenAILLM(model=bot.openai_model, api_key=api_key)
        tts_synth = _get_tts_synth_fn(bot, api_key)
        def gen() -> Generator[bytes, None, None]:
            text = (payload.text or "").strip()
            speak = bool(payload.speak)
            if not text:
                yield _ndjson({"type": "error", "error": "Empty text"})
                return

            messages = [Message(role="system", content=bot.system_prompt), Message(role="user", content=text)]

            delta_q_client: "queue.Queue[Optional[str]]" = queue.Queue()
            delta_q_tts: "queue.Queue[Optional[str]]" = queue.Queue()
            audio_q: "queue.Queue[Optional[tuple[bytes, int]]]" = queue.Queue()
            full_text_parts: list[str] = []
            error_q: "queue.Queue[Optional[str]]" = queue.Queue()

            def llm_thread() -> None:
                try:
                    for delta in llm.stream_text(messages=messages):
                        full_text_parts.append(delta)
                        delta_q_client.put(delta)
                        if speak:
                            delta_q_tts.put(delta)
                except Exception as exc:
                    error_q.put(str(exc))
                finally:
                    delta_q_client.put(None)
                    if speak:
                        delta_q_tts.put(None)

            def tts_thread() -> None:
                if not speak:
                    audio_q.put(None)
                    return
                try:
                    parts: list[str] = []
                    while True:
                        d = delta_q_tts.get()
                        if d is None:
                            break
                        if d:
                            parts.append(d)
                    text_to_speak = "".join(parts).strip()
                    if text_to_speak:
                        wav, sr = tts_synth(text_to_speak)
                        audio_q.put((wav, sr))
                finally:
                    audio_q.put(None)

            t1 = threading.Thread(target=llm_thread, daemon=True)
            t2 = threading.Thread(target=tts_thread, daemon=True)
            t1.start()
            t2.start()

            open_deltas = True
            open_audio = True
            last_heartbeat = time.time()

            while open_deltas or open_audio:
                sent = False
                try:
                    err = error_q.get_nowait()
                    if err:
                        yield _ndjson({"type": "error", "error": err})
                        open_deltas = False
                        open_audio = False
                        break
                except queue.Empty:
                    pass
                try:
                    d = delta_q_client.get_nowait()
                    if d is None:
                        open_deltas = False
                    else:
                        yield _ndjson({"type": "text_delta", "delta": d})
                    sent = True
                except queue.Empty:
                    pass

                if speak:
                    try:
                        item = audio_q.get_nowait()
                        if item is None:
                            open_audio = False
                        else:
                            wav, sr = item
                            yield _ndjson(
                                {"type": "audio_wav", "wav_base64": base64.b64encode(wav).decode(), "sr": sr}
                            )
                        sent = True
                    except queue.Empty:
                        pass
                else:
                    open_audio = False

                if not sent:
                    if time.time() - last_heartbeat > 10:
                        yield _ndjson({"type": "ping"})
                        last_heartbeat = time.time()
                    time.sleep(0.01)

            t1.join()
            t2.join()
            yield _ndjson({"type": "done", "text": "".join(full_text_parts).strip()})

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/api/bots/{bot_id}/talk/stream")
    def talk_stream(
        bot_id: UUID,
        audio: UploadFile = File(...),
        conversation_id: str = Form(""),
        test_flag: bool = Form(True),
        speak: bool = Form(True),
        session: Session = Depends(get_session),
    ) -> StreamingResponse:
        bot = get_bot(session, bot_id)
        if bot.openai_key_id:
            crypto = require_crypto()
            api_key = decrypt_openai_key(session, crypto=crypto, bot=bot)
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot.")

        conv_id: Optional[UUID] = UUID(conversation_id) if conversation_id.strip() else None
        if conv_id is None:
            conv = create_conversation(session, bot_id=bot.id, test_flag=bool(test_flag))
            conv_id = conv.id

        wav_bytes = audio.file.read()
        pcm16 = _decode_wav_bytes_to_pcm16_16k(wav_bytes)
        if not pcm16:
            raise HTTPException(status_code=400, detail="Empty audio")

        asr = _get_asr(bot.whisper_model, bot.whisper_device, bot.language).transcribe_pcm16(
            pcm16=pcm16, sample_rate=16000
        )
        user_text = asr.text.strip()
        if not user_text:
            # Return the conversation id so UI can keep the session even if no speech recognized.
            def empty_gen() -> Generator[bytes, None, None]:
                yield _ndjson({"type": "conversation", "id": str(conv_id)})
                yield _ndjson({"type": "asr", "text": ""})
                yield _ndjson({"type": "done", "text": ""})

            return StreamingResponse(empty_gen(), media_type="application/x-ndjson")

        add_message(session, conversation_id=conv_id, role="user", content=user_text)

        llm = OpenAILLM(model=bot.openai_model, api_key=api_key)
        tts_synth = _get_tts_synth_fn(bot, api_key)

        def gen() -> Generator[bytes, None, None]:
            yield _ndjson({"type": "conversation", "id": str(conv_id)})
            yield _ndjson({"type": "asr", "text": user_text})

            history = _build_history_budgeted(
                session=session,
                bot=bot,
                conversation_id=conv_id,
                api_key=api_key,
                status_cb=None,
            )

            delta_q_client: "queue.Queue[Optional[str]]" = queue.Queue()
            delta_q_tts: "queue.Queue[Optional[str]]" = queue.Queue()
            audio_q: "queue.Queue[Optional[tuple[bytes, int]]]" = queue.Queue()
            full_text_parts: list[str] = []
            error_q: "queue.Queue[Optional[str]]" = queue.Queue()

            def llm_thread() -> None:
                try:
                    for delta in llm.stream_text(messages=history):
                        full_text_parts.append(delta)
                        delta_q_client.put(delta)
                        if speak:
                            delta_q_tts.put(delta)
                except Exception as exc:
                    error_q.put(str(exc))
                finally:
                    delta_q_client.put(None)
                    if speak:
                        delta_q_tts.put(None)

            def tts_thread() -> None:
                if not speak:
                    audio_q.put(None)
                    return
                try:
                    parts: list[str] = []
                    while True:
                        d = delta_q_tts.get()
                        if d is None:
                            break
                        if d:
                            parts.append(d)
                    text_to_speak = "".join(parts).strip()
                    if text_to_speak:
                        wav, sr = tts_synth(text_to_speak)
                        audio_q.put((wav, sr))
                finally:
                    audio_q.put(None)

            t1 = threading.Thread(target=llm_thread, daemon=True)
            t2 = threading.Thread(target=tts_thread, daemon=True)
            t1.start()
            t2.start()

            open_deltas = True
            open_audio = True
            last_heartbeat = time.time()

            while open_deltas or open_audio:
                sent = False
                try:
                    err = error_q.get_nowait()
                    if err:
                        yield _ndjson({"type": "error", "error": err})
                        open_deltas = False
                        open_audio = False
                        break
                except queue.Empty:
                    pass
                try:
                    d = delta_q_client.get_nowait()
                    if d is None:
                        open_deltas = False
                    else:
                        yield _ndjson({"type": "text_delta", "delta": d})
                    sent = True
                except queue.Empty:
                    pass

                if speak:
                    try:
                        item = audio_q.get_nowait()
                        if item is None:
                            open_audio = False
                        else:
                            wav, sr = item
                            yield _ndjson(
                                {"type": "audio_wav", "wav_base64": base64.b64encode(wav).decode(), "sr": sr}
                            )
                        sent = True
                    except queue.Empty:
                        pass
                else:
                    open_audio = False

                if not sent:
                    if time.time() - last_heartbeat > 10:
                        yield _ndjson({"type": "ping"})
                        last_heartbeat = time.time()
                    time.sleep(0.01)

            t1.join()
            t2.join()
            final_text = "".join(full_text_parts).strip()
            if final_text:
                add_message(session, conversation_id=conv_id, role="assistant", content=final_text)
            yield _ndjson({"type": "done", "text": final_text})

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/api/bots/{bot_id}/chat")
    def chat_once(
        bot_id: UUID,
        payload: ChatRequest = Body(...),
        session: Session = Depends(get_session),
    ) -> dict:
        bot = get_bot(session, bot_id)
        if bot.openai_key_id:
            crypto = require_crypto()
            api_key = decrypt_openai_key(session, crypto=crypto, bot=bot)
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot.")

        llm = OpenAILLM(model=bot.openai_model, api_key=api_key)
        text = (payload.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text")

        messages = [Message(role="system", content=bot.system_prompt), Message(role="user", content=text)]
        out_text = llm.complete_text(messages=messages)

        if not payload.speak:
            return {"text": out_text}

        tts_synth = _get_tts_synth_fn(bot, api_key)
        wav, sr = tts_synth(out_text)
        return {"text": out_text, "audio_wav_base64": base64.b64encode(wav).decode(), "sr": sr}

    return app
