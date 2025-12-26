from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Generator, Optional, Tuple
from uuid import UUID

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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
    add_message,
    add_message_with_metrics,
    update_conversation_metrics,
    merge_conversation_metadata,
    decrypt_openai_key,
    delete_bot,
    delete_key,
    get_bot,
    list_bots,
    list_keys,
    get_conversation,
    list_conversations,
    count_conversations,
    list_messages,
    update_bot,
)
from voicebot.tts.xtts import XTTSv2
from voicebot.utils.text import SentenceChunker
from voicebot.utils.tokens import ModelPrice, estimate_cost_usd, estimate_messages_tokens, estimate_text_tokens
from voicebot.tools.set_metadata import set_metadata_tool_def


class ChatRequest(BaseModel):
    text: str
    speak: bool = True


class TalkResponseEvent(BaseModel):
    type: str


def create_app() -> FastAPI:
    settings = Settings()
    engine = make_engine(settings.db_url)
    init_db(engine)

    app = FastAPI(title="Intelligravex VoiceBot Studio")

    templates_dir = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))

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

    @app.get("/", response_class=HTMLResponse)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/bots")

    @app.get("/bots", response_class=HTMLResponse)
    def bots_index(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
        bots = list_bots(session)
        keys = list_keys(session, provider="openai")
        return templates.TemplateResponse(
            "bots.html",
            {"request": request, "bots": bots, "keys": keys},
        )

    @app.get("/conversations", response_class=HTMLResponse)
    def conversations_index(
        request: Request,
        page: int = 1,
        page_size: int = 50,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        page = max(1, int(page))
        page_size = min(200, max(10, int(page_size)))
        offset = (page - 1) * page_size
        total = count_conversations(session)
        convs = list_conversations(session, limit=page_size, offset=offset)
        bots = {b.id: b for b in list_bots(session)}
        return templates.TemplateResponse(
            "conversations.html",
            {
                "request": request,
                "conversations": convs,
                "bots_by_id": bots,
                "page": page,
                "page_size": page_size,
                "total": total,
            },
        )

    @app.get("/bots/{bot_id}/conversations", response_class=HTMLResponse)
    def bot_conversations(
        bot_id: UUID,
        request: Request,
        page: int = 1,
        page_size: int = 50,
        session: Session = Depends(get_session),
    ) -> HTMLResponse:
        bot = get_bot(session, bot_id)
        page = max(1, int(page))
        page_size = min(200, max(10, int(page_size)))
        offset = (page - 1) * page_size
        total = count_conversations(session, bot_id=bot_id)
        convs = list_conversations(session, bot_id=bot_id, limit=page_size, offset=offset)
        return templates.TemplateResponse(
            "conversations.html",
            {
                "request": request,
                "conversations": convs,
                "bots_by_id": {bot.id: bot},
                "bot": bot,
                "page": page,
                "page_size": page_size,
                "total": total,
            },
        )

    @app.get("/conversations/{conversation_id}", response_class=HTMLResponse)
    def conversation_detail(
        conversation_id: UUID, request: Request, session: Session = Depends(get_session)
    ) -> HTMLResponse:
        conv = get_conversation(session, conversation_id)
        bot = get_bot(session, conv.bot_id)
        msgs = list_messages(session, conversation_id=conversation_id)
        return templates.TemplateResponse(
            "conversation_detail.html",
            {"request": request, "conversation": conv, "bot": bot, "messages": msgs},
        )

    @app.get("/bots/new", response_class=HTMLResponse)
    def bots_new(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
        keys = list_keys(session, provider="openai")
        return templates.TemplateResponse(
            "bot_new.html", {"request": request, "keys": keys, "options": ui_options}
        )

    @app.post("/bots")
    def bots_create(
        name: str = Form(...),
        openai_model: str = Form("gpt-4o"),
        system_prompt: str = Form(...),
        language: str = Form("en"),
        tts_language: str = Form("en"),
        whisper_model: str = Form("small"),
        whisper_device: str = Form("auto"),
        xtts_model: str = Form("tts_models/multilingual/multi-dataset/xtts_v2"),
        speaker_id: str = Form(""),
        speaker_wav: str = Form(""),
        openai_key_id: str = Form(""),
        tts_split_sentences: bool = Form(False),
        tts_chunk_min_chars: int = Form(20),
        tts_chunk_max_chars: int = Form(120),
        start_message_mode: str = Form("llm"),
        start_message_text: str = Form(""),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        bot = Bot(
            name=name,
            openai_model=openai_model,
            system_prompt=system_prompt,
            language=language,
            tts_language=tts_language,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            xtts_model=xtts_model,
            speaker_id=speaker_id.strip() or None,
            speaker_wav=speaker_wav.strip() or None,
            openai_key_id=UUID(openai_key_id) if openai_key_id.strip() else None,
            tts_split_sentences=tts_split_sentences,
            tts_chunk_min_chars=int(tts_chunk_min_chars),
            tts_chunk_max_chars=int(tts_chunk_max_chars),
            start_message_mode=(start_message_mode or "llm").strip() or "llm",
            start_message_text=start_message_text or "",
        )
        create_bot(session, bot)
        return RedirectResponse(url=f"/bots/{bot.id}", status_code=303)

    @app.get("/bots/{bot_id}", response_class=HTMLResponse)
    def bots_edit(bot_id: UUID, request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
        bot = get_bot(session, bot_id)
        keys = list_keys(session, provider="openai")
        return templates.TemplateResponse(
            "bot_edit.html", {"request": request, "bot": bot, "keys": keys, "options": ui_options}
        )

    @app.post("/bots/{bot_id}")
    def bots_update(
        bot_id: UUID,
        name: str = Form(...),
        openai_model: str = Form("gpt-4o"),
        system_prompt: str = Form(...),
        language: str = Form("en"),
        tts_language: str = Form("en"),
        whisper_model: str = Form("small"),
        whisper_device: str = Form("auto"),
        xtts_model: str = Form("tts_models/multilingual/multi-dataset/xtts_v2"),
        speaker_id: str = Form(""),
        speaker_wav: str = Form(""),
        openai_key_id: str = Form(""),
        tts_split_sentences: bool = Form(False),
        tts_chunk_min_chars: int = Form(20),
        tts_chunk_max_chars: int = Form(120),
        start_message_mode: str = Form("llm"),
        start_message_text: str = Form(""),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        update_bot(
            session,
            bot_id,
            {
                "name": name,
                "openai_model": openai_model,
                "system_prompt": system_prompt,
                "language": language,
                "tts_language": tts_language,
                "whisper_model": whisper_model,
                "whisper_device": whisper_device,
                "xtts_model": xtts_model,
                "speaker_id": speaker_id.strip() or None,
                "speaker_wav": speaker_wav.strip() or None,
                "openai_key_id": UUID(openai_key_id) if openai_key_id.strip() else None,
                "tts_split_sentences": tts_split_sentences,
                "tts_chunk_min_chars": int(tts_chunk_min_chars),
                "tts_chunk_max_chars": int(tts_chunk_max_chars),
                "start_message_mode": (start_message_mode or "llm").strip() or "llm",
                "start_message_text": start_message_text or "",
            },
        )
        return RedirectResponse(url=f"/bots/{bot_id}", status_code=303)

    @app.post("/bots/{bot_id}/delete")
    def bots_delete(bot_id: UUID, session: Session = Depends(get_session)) -> RedirectResponse:
        delete_bot(session, bot_id)
        return RedirectResponse(url="/bots", status_code=303)

    @app.get("/keys", response_class=HTMLResponse)
    def keys_index(request: Request, session: Session = Depends(get_session)) -> HTMLResponse:
        keys = list_keys(session)
        return templates.TemplateResponse(
            "keys.html", {"request": request, "keys": keys, "secret_configured": bool(settings.secret_key)}
        )

    @app.get("/keys/new", response_class=HTMLResponse)
    def keys_new(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "key_new.html", {"request": request, "secret_configured": bool(settings.secret_key)}
        )

    @app.post("/keys")
    def keys_create(
        provider: str = Form("openai"),
        name: str = Form(...),
        secret: str = Form(...),
        session: Session = Depends(get_session),
    ) -> RedirectResponse:
        crypto = require_crypto()
        create_key(session, crypto=crypto, provider=provider, name=name, secret=secret)
        return RedirectResponse(url="/keys", status_code=303)

    @app.post("/keys/{key_id}/delete")
    def keys_delete(key_id: UUID, session: Session = Depends(get_session)) -> RedirectResponse:
        delete_key(session, key_id)
        return RedirectResponse(url="/keys", status_code=303)

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
        if conv.metadata_json and conv.metadata_json.strip() and conv.metadata_json.strip() != "{}":
            messages.append(Message(role="system", content=f"Conversation metadata (JSON): {conv.metadata_json}"))
        for m in list_messages(session, conversation_id=conversation_id):
            if m.role in ("user", "assistant"):
                messages.append(Message(role=m.role, content=m.content))
            elif m.role == "tool":
                # Store tool calls/results as system breadcrumbs to prevent repeated calls.
                messages.append(Message(role="system", content=f"Tool event: {m.content}"))
        return messages

    def _set_metadata_tool_def() -> dict:
        return set_metadata_tool_def()

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

            if bot.openai_key_id:
                crypto = require_crypto()
                api_key = decrypt_openai_key(session, crypto=crypto, bot=bot)
            else:
                api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot.")

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
                llm = OpenAILLM(model=bot.openai_model, api_key=api_key)
                msgs = [
                    Message(role="system", content=bot.system_prompt),
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
                        await _ws_send_json(ws, {"type": "conversation", "req_id": req_id, "id": str(conv_id)})
                        await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
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

                        await _ws_send_json(ws, {"type": "conversation", "req_id": req_id, "id": str(conv_id)})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "recording"})

                    elif msg_type == "stop":
                        if not req_id or active_req_id != req_id:
                            await _ws_send_json(
                                ws, {"type": "error", "req_id": req_id or None, "error": "Unknown req_id"}
                            )
                            continue
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
                        tool_q: "queue.Queue[Optional[ToolCall]]" = queue.Queue()
                        error_q: "queue.Queue[Optional[str]]" = queue.Queue()
                        full_text_parts: list[str] = []
                        metrics_lock = threading.Lock()
                        first_token_ts: Optional[float] = None
                        tts_start_ts: Optional[float] = None
                        first_audio_ts: Optional[float] = None

                        def llm_thread() -> None:
                            try:
                                tools = [_set_metadata_tool_def()]
                                for ev in llm.stream_text_or_tool(messages=history, tools=tools):
                                    if isinstance(ev, ToolCall):
                                        tool_q.put(ev)
                                        break
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
                                tool_q.put(None)

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
                        tool_call: Optional[ToolCall] = None
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
                                tc = tool_q.get_nowait()
                                if tc is not None:
                                    tool_call = tc
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
                        if speak and tts_start_ts is not None and first_audio_ts is not None:
                            timings["tts_first_audio"] = int(round((first_audio_ts - tts_start_ts) * 1000.0))
                            timings["tts_from_llm_start"] = int(round((first_audio_ts - llm_start_ts) * 1000.0))

                        # Persist aggregates + last latencies (best-effort).
                        try:
                            in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                                bot=bot, history=history, assistant_text=final_text
                            )
                            with Session(engine) as session:
                                if final_text and conv_id and tool_call is None:
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

                        if tool_call is not None and conv_id is not None:
                            # Apply tool locally and respond with `next_reply` (no second LLM call).
                            await _ws_send_json(
                                ws,
                                {
                                    "type": "tool_call",
                                    "req_id": req_id,
                                    "name": tool_call.name,
                                    "arguments_json": tool_call.arguments_json,
                                },
                            )
                            try:
                                tool_args = json.loads(tool_call.arguments_json or "{}")
                                if not isinstance(tool_args, dict):
                                    raise ValueError("tool args must be an object")
                            except Exception as exc:
                                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                                await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                                active_req_id = None
                                conv_id = None
                                audio_buf = bytearray()
                                continue

                            next_reply = str(tool_args.get("next_reply") or "").strip()
                            meta_patch = dict(tool_args)
                            meta_patch.pop("next_reply", None)

                            with Session(engine) as session:
                                # Store tool call + tool result in message history.
                                add_message_with_metrics(
                                    session,
                                    conversation_id=conv_id,
                                    role="tool",
                                    content=json.dumps(
                                        {"tool": tool_call.name, "arguments": tool_args}, ensure_ascii=False
                                    ),
                                )
                                new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=meta_patch)
                                tool_result = {"ok": True, "updated": meta_patch, "metadata": new_meta}
                                add_message_with_metrics(
                                    session,
                                    conversation_id=conv_id,
                                    role="tool",
                                    content=json.dumps({"tool": tool_call.name, "result": tool_result}, ensure_ascii=False),
                                )

                            await _ws_send_json(
                                ws,
                                {
                                    "type": "tool_result",
                                    "req_id": req_id,
                                    "name": tool_call.name,
                                    "result": tool_result,
                                },
                            )

                            if next_reply:
                                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": next_reply})
                                # Store assistant response and update aggregates.
                                try:
                                    in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                                        bot=bot, history=history, assistant_text=next_reply
                                    )
                                    with Session(engine) as session:
                                        add_message_with_metrics(
                                            session,
                                            conversation_id=conv_id,
                                            role="assistant",
                                            content=next_reply,
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
                                    a = _tts_synthesize(
                                        tts_handle,
                                        next_reply,
                                        speaker_wav=bot.speaker_wav,
                                        speaker_id=bot.speaker_id,
                                        language=bot.tts_language,
                                        split_sentences=bot.tts_split_sentences,
                                    )
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "audio_wav",
                                            "req_id": req_id,
                                            "wav_base64": base64.b64encode(_wav_bytes(a.audio, a.sample_rate)).decode(),
                                            "sr": a.sample_rate,
                                        },
                                    )

                            await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": next_reply})
                        else:
                            await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": final_text})

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})

                        active_req_id = None
                        conv_id = None
                        audio_buf = bytearray()

                    else:
                        await _ws_send_json(
                            ws,
                            {"type": "error", "req_id": req_id or None, "error": f"Unknown message type: {msg_type}"},
                        )

                elif "bytes" in msg and msg["bytes"] is not None:
                    if active_req_id is None:
                        await _ws_send_json(ws, {"type": "error", "error": "Send start before audio"})
                        continue
                    audio_buf.extend(msg["bytes"])
                else:
                    # ignore
                    pass

        except WebSocketDisconnect:
            return

    @app.get("/api/tts/meta")
    def tts_meta(model_name: str) -> dict:
        return _get_tts_meta(model_name)

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
