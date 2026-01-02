from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
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
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session

from voicebot.config import Settings
from voicebot.crypto import CryptoError, get_crypto_box
from voicebot.db import init_db, make_engine
from voicebot.asr.whisper_asr import WhisperASR
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
from voicebot.utils.text import SentenceChunker
from voicebot.utils.tokens import ModelPrice, estimate_cost_usd, estimate_messages_tokens, estimate_text_tokens
from voicebot.tools.set_metadata import set_metadata_tool_def, set_variable_tool_def
from voicebot.tools.web_search import web_search as run_web_search, web_search_tool_def
from voicebot.models import IntegrationTool
from voicebot.utils.template import eval_template_value, render_jinja_template, render_template, safe_json_loads


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
    try:
        obj = json.loads(raw or "[]")
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: list[str] = []
    for v in obj:
        if isinstance(v, str):
            s = v.strip()
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


class IntegrationToolCreateRequest(BaseModel):
    name: str
    description: str = ""
    url: str
    method: str = "GET"
    args_required: list[str] = []
    headers_template_json: str = "{}"
    request_body_template: str = "{}"
    parameters_schema_json: str = ""
    response_mapper_json: str = "{}"
    static_reply_template: str = ""


class IntegrationToolUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    method: Optional[str] = None
    args_required: Optional[list[str]] = None
    headers_template_json: Optional[str] = None
    request_body_template: Optional[str] = None
    parameters_schema_json: Optional[str] = None
    response_mapper_json: Optional[str] = None
    static_reply_template: Optional[str] = None


def create_app() -> FastAPI:
    settings = Settings()
    engine = make_engine(settings.db_url)
    init_db(engine)

    app = FastAPI(title="Intelligravex VoiceBot Studio")
    cors_raw = (os.environ.get("VOICEBOT_CORS_ORIGINS") or "").strip()
    cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()] if cors_raw else []
    if not cors_origins:
        cors_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
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

    def _get_conversation_meta(session: Session, *, conversation_id: UUID) -> dict:
        conv = get_conversation(session, conversation_id)
        meta = safe_json_loads(conv.metadata_json or "{}") or {}
        return meta if isinstance(meta, dict) else {}

    def _set_metadata_tool_def() -> dict:
        return set_metadata_tool_def()

    def _set_variable_tool_def() -> dict:
        return set_variable_tool_def()

    def _web_search_tool_def() -> dict:
        return web_search_tool_def()

    def _system_tools_defs() -> list[dict[str, Any]]:
        # Tools that are always available for every bot.
        #
        # Note: `set_variable` is kept as a runtime alias for backwards-compat, but we only expose
        # `set_metadata` to the model to avoid duplicate tools that do the same thing.
        return [_set_metadata_tool_def(), _web_search_tool_def()]

    def _system_tools_public_list() -> list[dict[str, Any]]:
        # UI-friendly list of built-in tools (do not include full JSON Schema).
        out: list[dict[str, Any]] = []
        for d in _system_tools_defs():
            out.append(
                {
                    "name": str(d.get("name") or ""),
                    "description": str(d.get("description") or ""),
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
        schema = {
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
                "next_reply": {
                    "type": "string",
                    "description": "What the assistant should say next (no second LLM call). Variables like {{.firstName}} are allowed.",
                },
            },
            "required": ["args", "next_reply"],
            "additionalProperties": True,
        }
        return {
            "type": "function",
            "name": t.name,
            "description": (t.description or "").strip()
            + " This tool calls an external HTTP API and maps selected response fields into conversation metadata. "
            + "Return your spoken/text response in next_reply (you can use metadata variables like {{.firstName}}).",
            "parameters": schema,
            "strict": False,
        }

    def _build_tools_for_bot(session: Session, bot_id: UUID) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = list(_system_tools_defs())
        for t in list_integration_tools(session, bot_id=bot_id):
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
        # Render URL/body templates using current metadata + tool args.
        ctx = {"meta": meta, "args": tool_args, "params": tool_args}
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

        body_template = tool.request_body_template or ""
        body_obj = None
        if body_template.strip():
            rendered_body = render_template(body_template, ctx=ctx)
            try:
                body_obj = json.loads(rendered_body)
            except Exception:
                # If body isn't valid JSON, send as raw string.
                body_obj = rendered_body

        timeout = httpx.Timeout(20.0, connect=10.0)
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                if method == "GET":
                    resp = client.request(method, url, params=tool_args or None, headers=headers_obj or None)
                else:
                    # Prefer JSON for objects/lists; otherwise raw data.
                    if isinstance(body_obj, (dict, list)):
                        resp = client.request(method, url, json=body_obj, headers=headers_obj or None)
                    elif body_obj is None:
                        # Most APIs expect an object for JSON bodies. Send {} instead of null when args are empty.
                        resp = client.request(method, url, json=(tool_args or {}), headers=headers_obj or None)
                    else:
                        resp = client.request(method, url, content=str(body_obj), headers=headers_obj or None)
            if resp.status_code >= 400:
                return _http_error_response(
                    url=str(resp.request.url),
                    status_code=resp.status_code,
                    body=(resp.text or None),
                    message=resp.reason_phrase,
                )
            try:
                return resp.json()
            except Exception:
                return {"raw": resp.text}
        except httpx.RequestError as exc:
            return _http_error_response(url=url, status_code=None, body=None, message=str(exc))

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
                            {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)},
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
                                history = _build_history(session, bot, conv_id)
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
                                local_chunker = SentenceChunker(
                                    min_chars=bot.tts_chunk_min_chars, max_chars=bot.tts_chunk_max_chars
                                )
                                did_send_any = False
                                while True:
                                    d = delta_q_tts.get()
                                    if d is None:
                                        break
                                    for chunk in local_chunker.push(d):
                                        with metrics_lock:
                                            if tts_start_ts is None:
                                                tts_start_ts = time.time()
                                        if not did_send_any:
                                            status(req_id, "tts")
                                            did_send_any = True
                                        wav, sr = tts_synth(chunk)
                                        audio_q.put((wav, sr))
                                tail = local_chunker.flush()
                                if tail:
                                    with metrics_lock:
                                        if tts_start_ts is None:
                                            tts_start_ts = time.time()
                                    if not did_send_any:
                                        status(req_id, "tts")
                                        did_send_any = True
                                    wav, sr = tts_synth(tail)
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

                                    if tool_name == "set_metadata":
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
                                    else:
                                        tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                        if not tool_cfg:
                                            raise RuntimeError(f"Unknown tool: {tool_name}")
                                        task = asyncio.create_task(
                                            asyncio.to_thread(
                                                _execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                                            )
                                        )
                                        if wait_reply:
                                            await _send_interim(wait_reply, kind="wait")
                                        while True:
                                            try:
                                                response_json = await asyncio.wait_for(task, timeout=7.0)
                                                break
                                            except asyncio.TimeoutError:
                                                if wait_reply:
                                                    await _send_interim(wait_reply, kind="wait")
                                                continue
                                        if speak:
                                            now = time.time()
                                            if now < tts_busy_until:
                                                await asyncio.sleep(tts_busy_until - now)
                                        if isinstance(response_json, dict) and "__http_error__" in response_json:
                                            err = response_json["__http_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            mapped = _apply_response_mapper(
                                                mapper_json=tool_cfg.response_mapper_json,
                                                response_json=response_json,
                                                meta=meta_current,
                                                tool_args=patch,
                                            )
                                            new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=mapped)
                                            tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}

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
                                    bot = get_bot(session, bot_id)
                                    followup_history = _build_history(session, bot, conv_id)
                                    followup_history.append(
                                        Message(
                                            role="system",
                                            content=(
                                                "The previous tool call failed. "
                                                if tool_failed
                                                else ""
                                            )
                                            + "Using the latest tool result(s) above, write the next assistant reply. Do not call any tools.",
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
                                    cost = float(estimate_cost_usd(model_price=price, input_tokens=in_tok, output_tokens=out_tok) or 0.0)
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
                                        {"type": "metrics", "req_id": req_id, "timings_ms": {"llm_ttfb": ttfb2, "llm_total": total2, "total": total2}},
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
                            history = _build_history(session, bot, conv_id)
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
                                local_chunker = SentenceChunker(
                                    min_chars=bot.tts_chunk_min_chars, max_chars=bot.tts_chunk_max_chars
                                )
                                did_send_any = False
                                while True:
                                    d = delta_q_tts.get()
                                    if d is None:
                                        break
                                    for chunk in local_chunker.push(d):
                                        with metrics_lock:
                                            if tts_start_ts is None:
                                                tts_start_ts = time.time()
                                        if not did_send_any:
                                            status(req_id, "tts")
                                            did_send_any = True
                                        wav, sr = tts_synth(chunk)
                                        audio_q.put((wav, sr))
                                tail = local_chunker.flush()
                                if tail:
                                    with metrics_lock:
                                        if tts_start_ts is None:
                                            tts_start_ts = time.time()
                                    if not did_send_any:
                                        status(req_id, "tts")
                                        did_send_any = True
                                    wav, sr = tts_synth(tail)
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
                                meta_current = _get_conversation_meta(session, conversation_id=conv_id)

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

                                    if tool_name == "set_metadata":
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
                                    else:
                                        tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                        if not tool_cfg:
                                            raise RuntimeError(f"Unknown tool: {tool_name}")
                                        task = asyncio.create_task(
                                            asyncio.to_thread(
                                                _execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                                            )
                                        )
                                        if wait_reply:
                                            await _send_interim(wait_reply, kind="wait")
                                        while True:
                                            try:
                                                response_json = await asyncio.wait_for(task, timeout=7.0)
                                                break
                                            except asyncio.TimeoutError:
                                                if wait_reply:
                                                    await _send_interim(wait_reply, kind="wait")
                                                continue
                                        if speak:
                                            now = time.time()
                                            if now < tts_busy_until:
                                                await asyncio.sleep(tts_busy_until - now)
                                        if isinstance(response_json, dict) and "__http_error__" in response_json:
                                            err = response_json["__http_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            mapped = _apply_response_mapper(
                                                mapper_json=tool_cfg.response_mapper_json,
                                                response_json=response_json,
                                                meta=meta_current,
                                                tool_args=patch,
                                            )
                                            new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=mapped)
                                            tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}

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
                                    followup_history = _build_history(session, bot2, conv_id)
                                    followup_history.append(
                                        Message(
                                            role="system",
                                            content=(
                                                "The previous tool call failed. "
                                                if tool_failed
                                                else ""
                                            )
                                            + "Using the latest tool result(s) above, write the next assistant reply. Do not call any tools.",
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

        conv_id: Optional[UUID] = None

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
                        history = _build_history(session, bot, conv_id)
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
                                if tool_name == "set_metadata":
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
                                else:
                                    tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                    if not tool_cfg:
                                        raise RuntimeError(f"Unknown tool: {tool_name}")
                                    task = asyncio.create_task(
                                        asyncio.to_thread(
                                            _execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                                        )
                                    )
                                    if wait_reply:
                                        await _public_send_interim(ws, req_id=req_id, kind="wait", text=wait_reply)
                                    while True:
                                        try:
                                            response_json = await asyncio.wait_for(task, timeout=7.0)
                                            break
                                        except asyncio.TimeoutError:
                                            if wait_reply:
                                                await _public_send_interim(ws, req_id=req_id, kind="wait", text=wait_reply)
                                            continue
                                    if isinstance(response_json, dict) and "__http_error__" in response_json:
                                        err = response_json["__http_error__"] or {}
                                        tool_result = {"ok": False, "error": err}
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        mapped = _apply_response_mapper(
                                            mapper_json=tool_cfg.response_mapper_json,
                                            response_json=response_json,
                                            meta=meta_current,
                                            tool_args=patch,
                                        )
                                        new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=mapped)
                                        tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}

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
                                        needs_followup_llm = _should_followup_llm_for_tool(
                                            tool=tool_cfg, static_rendered=static_text
                                        )
                                        final = ""
                                else:
                                    candidate = _render_with_meta(next_reply, meta_current).strip()
                                    final = candidate or final

                            if needs_followup_llm:
                                followup_history = _build_history(session, bot, conv_id)
                                followup_history.append(
                                    Message(
                                        role="system",
                                        content=(
                                            "The previous tool call failed. "
                                            if tool_failed
                                            else ""
                                        )
                                        + "Using the latest tool result(s) above, write the next assistant reply. Do not call any tools.",
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
        openai_models = sorted(set(ui_options.get("openai_models", []) + list(pricing.keys())))
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

    def _bot_to_dict(bot: Bot) -> dict:
        return {
            "id": str(bot.id),
            "name": bot.name,
            "openai_model": bot.openai_model,
            "web_search_model": getattr(bot, "web_search_model", bot.openai_model),
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
            "args_required": _parse_required_args_json(getattr(t, "args_required_json", "[]")),
            # Never expose secret headers (write-only). Return masked preview for UI.
            "headers_template_json": "{}",
            "headers_template_json_masked": _mask_headers_json(t.headers_template_json),
            "headers_configured": _headers_configured(t.headers_template_json),
            "request_body_template": t.request_body_template,
            "parameters_schema_json": t.parameters_schema_json,
            "response_mapper_json": t.response_mapper_json,
            "static_reply_template": t.static_reply_template,
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
            elif k in ("tts_vendor", "openai_tts_model", "openai_tts_voice", "web_search_model"):
                patch[k] = (v or "").strip()
            elif k == "openai_tts_speed":
                patch[k] = float(v) if v is not None else 1.0
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
        _ = get_bot(session, bot_id)
        tools = list_integration_tools(session, bot_id=bot_id)
        return {"items": [_tool_to_dict(t) for t in tools], "system_tools": _system_tools_public_list()}

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
            args_required_json=json.dumps(payload.args_required or [], ensure_ascii=False),
            headers_template_json=payload.headers_template_json or "{}",
            request_body_template=payload.request_body_template or "{}",
            parameters_schema_json=payload.parameters_schema_json or "",
            response_mapper_json=payload.response_mapper_json or "{}",
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
                    local_chunker = SentenceChunker(
                        min_chars=bot.tts_chunk_min_chars, max_chars=bot.tts_chunk_max_chars
                    )
                    while True:
                        d = delta_q_tts.get()
                        if d is None:
                            break
                        for chunk in local_chunker.push(d):
                            wav, sr = tts_synth(chunk)
                            audio_q.put((wav, sr))
                    tail = local_chunker.flush()
                    if tail:
                        wav, sr = tts_synth(tail)
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

            history = _build_history(session, bot, conv_id)

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
                    local_chunker = SentenceChunker(
                        min_chars=bot.tts_chunk_min_chars, max_chars=bot.tts_chunk_max_chars
                    )
                    while True:
                        d = delta_q_tts.get()
                        if d is None:
                            break
                        for chunk in local_chunker.push(d):
                            wav, sr = tts_synth(chunk)
                            audio_q.put((wav, sr))
                    tail = local_chunker.flush()
                    if tail:
                        wav, sr = tts_synth(tail)
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
