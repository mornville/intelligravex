from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import secrets
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Generator, Optional, Tuple
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
from voicebot.utils.text import SentenceChunker
from voicebot.utils.tokens import ModelPrice, estimate_cost_usd, estimate_messages_tokens, estimate_text_tokens
from voicebot.tools.set_metadata import set_metadata_tool_def, set_variable_tool_def
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
    openai_model: str = "gpt-4o"
    system_prompt: str
    language: str = "en"
    tts_language: str = "en"
    whisper_model: str = "small"
    whisper_device: str = "auto"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    speaker_id: Optional[str] = None
    speaker_wav: Optional[str] = None
    openai_key_id: Optional[UUID] = None
    tts_split_sentences: bool = False
    tts_chunk_min_chars: int = 20
    tts_chunk_max_chars: int = 120
    start_message_mode: str = "llm"
    start_message_text: str = ""


class BotUpdateRequest(BaseModel):
    name: Optional[str] = None
    openai_model: Optional[str] = None
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    tts_language: Optional[str] = None
    whisper_model: Optional[str] = None
    whisper_device: Optional[str] = None
    xtts_model: Optional[str] = None
    speaker_id: Optional[str] = None
    speaker_wav: Optional[str] = None
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
    headers_template_json: str = "{}"
    request_body_template: str = "{}"
    response_mapper_json: str = "{}"
    static_reply_template: str = ""


class IntegrationToolUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    method: Optional[str] = None
    headers_template_json: Optional[str] = None
    request_body_template: Optional[str] = None
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

    def _build_history(session: Session, bot: Bot, conversation_id: Optional[UUID]) -> list[Message]:
        messages: list[Message] = [Message(role="system", content=bot.system_prompt)]
        if not conversation_id:
            return messages
        conv = get_conversation(session, conversation_id)
        if conv.bot_id != bot.id:
            raise HTTPException(status_code=400, detail="Conversation does not belong to bot")
        meta = safe_json_loads(conv.metadata_json or "{}") or {}
        ctx = {"meta": meta}
        # Render system prompt with metadata variables (if any)
        messages = [Message(role="system", content=render_template(bot.system_prompt, ctx=ctx))]
        if meta:
            messages.append(Message(role="system", content=f"Conversation metadata (JSON): {json.dumps(meta, ensure_ascii=False)}"))
        for m in list_messages(session, conversation_id=conversation_id):
            if m.role in ("user", "assistant"):
                messages.append(Message(role=m.role, content=render_template(m.content, ctx=ctx)))
            elif m.role == "tool":
                # Store tool calls/results as system breadcrumbs to prevent repeated calls.
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

    def _integration_tool_def(t: IntegrationTool) -> dict[str, Any]:
        schema = {
            "type": "object",
            "properties": {
                "next_reply": {
                    "type": "string",
                    "description": "What the assistant should say next (no second LLM call). Variables like {{.firstName}} are allowed.",
                }
            },
            "required": ["next_reply"],
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
        tools: list[dict[str, Any]] = [_set_metadata_tool_def(), _set_variable_tool_def()]
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
                    out[k] = ModelPrice(
                        input_per_1m=float(v.get("input_per_1m")),
                        output_per_1m=float(v.get("output_per_1m")),
                    )
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

            if bot.start_message_mode == "static" and greeting_text:
                # Static greeting (no LLM).
                pass
            else:
                if bot.openai_key_id:
                    crypto = require_crypto()
                    api_key = decrypt_openai_key(session, crypto=crypto, bot=bot)
                else:
                    api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot.")
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
                tts_handle = _get_tts_handle(
                    bot.xtts_model,
                    bot.speaker_wav,
                    bot.speaker_id,
                    True,
                    bot.tts_split_sentences,
                    bot.tts_language,
                )
                # Synthesize whole greeting as one chunk for now.
                a = await asyncio.to_thread(_tts_synthesize, tts_handle, greeting_text)
                await _ws_send_json(
                    ws,
                    {
                        "type": "audio_wav",
                        "req_id": req_id,
                        "wav_base64": base64.b64encode(_wav_bytes(a.audio, a.sample_rate)).decode(),
                        "sr": a.sample_rate,
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
                        resp = client.request(method, url, json=tool_args or None, headers=headers_obj or None)
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

    @app.websocket("/ws/bots/{bot_id}/talk")
    async def talk_ws(bot_id: UUID, ws: WebSocket) -> None:
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
                        accepting_audio = False
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "init"})
                        try:
                            conv_id = await _init_conversation_and_greet(
                                bot_id=bot_id, speak=speak, test_flag=test_flag, ws=ws, req_id=req_id
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
                                tts_handle = await asyncio.to_thread(
                                    _get_tts_handle,
                                    bot.xtts_model,
                                    bot.speaker_wav,
                                    bot.speaker_id,
                                    True,
                                    bot.tts_split_sentences,
                                    bot.tts_language,
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
                                        a = _tts_synthesize(tts_handle, chunk)
                                        if not did_send_any:
                                            status(req_id, "tts")
                                            did_send_any = True
                                        audio_q.put((_wav_bytes(a.audio, a.sample_rate), a.sample_rate))
                                tail = local_chunker.flush()
                                if tail:
                                    with metrics_lock:
                                        if tts_start_ts is None:
                                            tts_start_ts = time.time()
                                    a = _tts_synthesize(tts_handle, tail)
                                    if not did_send_any:
                                        status(req_id, "tts")
                                        did_send_any = True
                                    audio_q.put((_wav_bytes(a.audio, a.sample_rate), a.sample_rate))
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
                                    patch = dict(tool_args)
                                    patch.pop("next_reply", None)

                                    tool_cfg: IntegrationTool | None = None
                                    response_json: Any | None = None

                                    if tool_name == "set_metadata":
                                        new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=patch)
                                        tool_result = {"ok": True, "updated": patch, "metadata": new_meta}
                                    else:
                                        tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                        if not tool_cfg:
                                            raise RuntimeError(f"Unknown tool: {tool_name}")
                                        response_json = _execute_integration_http(
                                            tool=tool_cfg, meta=meta_current, tool_args=patch
                                        )
                                        if isinstance(response_json, dict) and "__http_error__" in response_json:
                                            err = response_json["__http_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            sc = err.get("status_code")
                                            tool_error = (
                                                f"{tool_name}: HTTP {sc} Unauthorized"
                                                if sc == 401
                                                else f"{tool_name}: HTTP {sc or 'error'}"
                                            )
                                            rendered_reply = (
                                                f"The integration '{tool_name}' failed with HTTP {sc} "
                                                f"({err.get('message') or 'error'}). "
                                                "Please update the integration Authorization token and try again. "
                                                "What would you like to do next?"
                                            )
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
                                    meta_current = tool_result.get("metadata") or meta_current

                                    await _ws_send_json(
                                        ws,
                                        {"type": "tool_result", "req_id": req_id, "name": tool_name, "result": tool_result},
                                    )

                                    if tool_error:
                                        break

                                    candidate = ""
                                    if tool_name != "set_metadata" and tool_cfg and (tool_cfg.static_reply_template or "").strip():
                                        candidate = _render_static_reply(
                                            template_text=tool_cfg.static_reply_template,
                                            meta=meta_current,
                                            response_json=response_json,
                                            tool_args=patch,
                                        ).strip()
                                    if not candidate:
                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                    rendered_reply = candidate or rendered_reply

                            if tool_error:
                                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": tool_error})

                            if rendered_reply:
                                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": rendered_reply})
                                try:
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
                                    a = await asyncio.to_thread(_tts_synthesize, tts_handle, rendered_reply)
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "audio_wav",
                                            "req_id": req_id,
                                            "wav_base64": base64.b64encode(_wav_bytes(a.audio, a.sample_rate)).decode(),
                                            "sr": a.sample_rate,
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
                            tts_handle = await asyncio.to_thread(
                                _get_tts_handle,
                                bot.xtts_model,
                                bot.speaker_wav,
                                bot.speaker_id,
                                True,
                                bot.tts_split_sentences,
                                bot.tts_language,
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
                                        a = _tts_synthesize(tts_handle, chunk)
                                        if not did_send_any:
                                            status(req_id, "tts")
                                            did_send_any = True
                                        audio_q.put((_wav_bytes(a.audio, a.sample_rate), a.sample_rate))
                                tail = local_chunker.flush()
                                if tail:
                                    with metrics_lock:
                                        if tts_start_ts is None:
                                            tts_start_ts = time.time()
                                    a = _tts_synthesize(tts_handle, tail)
                                    if not did_send_any:
                                        status(req_id, "tts")
                                        did_send_any = True
                                    audio_q.put((_wav_bytes(a.audio, a.sample_rate), a.sample_rate))
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
                                    patch = dict(tool_args)
                                    patch.pop("next_reply", None)

                                    tool_cfg: IntegrationTool | None = None
                                    response_json: Any | None = None

                                    if tool_name == "set_metadata":
                                        new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=patch)
                                        tool_result = {"ok": True, "updated": patch, "metadata": new_meta}
                                    else:
                                        tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                        if not tool_cfg:
                                            raise RuntimeError(f"Unknown tool: {tool_name}")
                                        response_json = _execute_integration_http(
                                            tool=tool_cfg, meta=meta_current, tool_args=patch
                                        )
                                        if isinstance(response_json, dict) and "__http_error__" in response_json:
                                            err = response_json["__http_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            sc = err.get("status_code")
                                            tool_error = (
                                                f"{tool_name}: HTTP {sc} Unauthorized"
                                                if sc == 401
                                                else f"{tool_name}: HTTP {sc or 'error'}"
                                            )
                                            rendered_reply = (
                                                f"The integration '{tool_name}' failed with HTTP {sc} "
                                                f"({err.get('message') or 'error'}). "
                                                "Please update the integration Authorization token and try again. "
                                                "What would you like to do next?"
                                            )
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
                                    meta_current = tool_result.get("metadata") or meta_current

                                    await _ws_send_json(
                                        ws,
                                        {"type": "tool_result", "req_id": req_id, "name": tool_name, "result": tool_result},
                                    )

                                    if tool_error:
                                        break

                                    candidate = ""
                                    if tool_name != "set_metadata" and tool_cfg and (tool_cfg.static_reply_template or "").strip():
                                        candidate = _render_static_reply(
                                            template_text=tool_cfg.static_reply_template,
                                            meta=meta_current,
                                            response_json=response_json,
                                            tool_args=patch,
                                        ).strip()
                                    if not candidate:
                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                    rendered_reply = candidate or rendered_reply

                            if tool_error:
                                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": tool_error})

                            if rendered_reply:
                                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": rendered_reply})
                                try:
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
                                    a = await asyncio.to_thread(_tts_synthesize, tts_handle, rendered_reply)
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "audio_wav",
                                            "req_id": req_id,
                                            "wav_base64": base64.b64encode(_wav_bytes(a.audio, a.sample_rate)).decode(),
                                            "sr": a.sample_rate,
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
                                patch = dict(tool_args)
                                patch.pop("next_reply", None)

                                tool_cfg: IntegrationTool | None = None
                                response_json: Any | None = None
                                if tool_name == "set_metadata":
                                    new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=patch)
                                    tool_result = {"ok": True, "updated": patch, "metadata": new_meta}
                                else:
                                    tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                    if not tool_cfg:
                                        raise RuntimeError(f"Unknown tool: {tool_name}")
                                    response_json = _execute_integration_http(
                                        tool=tool_cfg, meta=meta_current, tool_args=patch
                                    )
                                    if isinstance(response_json, dict) and "__http_error__" in response_json:
                                        err = response_json["__http_error__"] or {}
                                        tool_result = {"ok": False, "error": err}
                                        sc = err.get("status_code")
                                        final = (
                                            f"The integration '{tool_name}' failed with HTTP {sc} "
                                            f"({err.get('message') or 'error'}). "
                                            "Please update the integration Authorization token and try again."
                                        )
                                        await _ws_send_json(
                                            ws,
                                            {"type": "error", "req_id": req_id, "error": f"{tool_name}: HTTP {sc}"},
                                        )
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
                                meta_current = tool_result.get("metadata") or meta_current

                                if not tool_result.get("ok", True):
                                    break

                                candidate = ""
                                if tool_name != "set_metadata" and tool_cfg and (tool_cfg.static_reply_template or "").strip():
                                    candidate = _render_static_reply(
                                        template_text=tool_cfg.static_reply_template,
                                        meta=meta_current,
                                        response_json=response_json,
                                        tool_args=patch,
                                    ).strip()
                                if not candidate:
                                    candidate = _render_with_meta(next_reply, meta_current).strip()
                                final = candidate or final

                            rendered_reply = final
                            if rendered_reply:
                                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": rendered_reply})

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
            "start_message_modes": ["llm", "static"],
            "asr_vendors": ["whisper_local"],
            "tts_vendors": ["xtts_local"],
            "http_methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],
        }

    def _bot_to_dict(bot: Bot) -> dict:
        return {
            "id": str(bot.id),
            "name": bot.name,
            "openai_model": bot.openai_model,
            "openai_key_id": str(bot.openai_key_id) if bot.openai_key_id else None,
            "system_prompt": bot.system_prompt,
            "language": bot.language,
            "tts_language": bot.tts_language,
            "whisper_model": bot.whisper_model,
            "whisper_device": bot.whisper_device,
            "xtts_model": bot.xtts_model,
            "speaker_id": bot.speaker_id,
            "speaker_wav": bot.speaker_wav,
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
            # Never expose secret headers (write-only). Return masked preview for UI.
            "headers_template_json": "{}",
            "headers_template_json_masked": _mask_headers_json(t.headers_template_json),
            "headers_configured": _headers_configured(t.headers_template_json),
            "request_body_template": t.request_body_template,
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
            system_prompt=payload.system_prompt,
            language=payload.language,
            tts_language=payload.tts_language,
            whisper_model=payload.whisper_model,
            whisper_device=payload.whisper_device,
            xtts_model=payload.xtts_model,
            speaker_id=(payload.speaker_id or "").strip() or None,
            speaker_wav=(payload.speaker_wav or "").strip() or None,
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
        return {"items": [_tool_to_dict(t) for t in tools]}

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
            headers_template_json=payload.headers_template_json or "{}",
            request_body_template=payload.request_body_template or "{}",
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
        if "headers_template_json" in patch:
            patch["headers_template_json"] = patch["headers_template_json"] or "{}"
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
        tts_handle = _get_tts_handle(
            bot.xtts_model,
            bot.speaker_wav,
            bot.speaker_id,
            True,
            bot.tts_split_sentences,
            bot.tts_language,
        )
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
                            audio = _tts_synthesize(tts_handle, chunk)
                            audio_q.put((_wav_bytes(audio.audio, audio.sample_rate), audio.sample_rate))
                    tail = local_chunker.flush()
                    if tail:
                        audio = _tts_synthesize(tts_handle, tail)
                        audio_q.put((_wav_bytes(audio.audio, audio.sample_rate), audio.sample_rate))
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
        tts_handle = _get_tts_handle(
            bot.xtts_model,
            bot.speaker_wav,
            bot.speaker_id,
            True,
            bot.tts_split_sentences,
            bot.tts_language,
        )

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
                            a = _tts_synthesize(tts_handle, chunk)
                            audio_q.put((_wav_bytes(a.audio, a.sample_rate), a.sample_rate))
                    tail = local_chunker.flush()
                    if tail:
                        a = _tts_synthesize(tts_handle, tail)
                        audio_q.put((_wav_bytes(a.audio, a.sample_rate), a.sample_rate))
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

        tts_handle = _get_tts_handle(
            bot.xtts_model,
            bot.speaker_wav,
            bot.speaker_id,
            True,
            bot.tts_split_sentences,
            bot.tts_language,
        )
        audio = _tts_synthesize(tts_handle, out_text)
        wav = _wav_bytes(audio.audio, audio.sample_rate)
        return {"text": out_text, "audio_wav_base64": base64.b64encode(wav).decode(), "sr": audio.sample_rate}

    return app
