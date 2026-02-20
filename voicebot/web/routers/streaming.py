from __future__ import annotations
from uuid import UUID

from typing import Generator, Optional

from fastapi import APIRouter, Body, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from ..schemas import ChatRequest


def register(app, ctx) -> None:
    router = APIRouter()

    @router.post("/api/bots/{bot_id}/chat/stream")
    def chat_stream(
        bot_id: UUID,
        payload: ChatRequest = Body(...),
        session: Session = Depends(ctx.get_session),
    ) -> StreamingResponse:
        bot = ctx.get_bot(session, bot_id)
        _provider, llm_api_key, llm = ctx._require_llm_client(session, bot=bot)
        speak = bool(payload.speak)
        openai_api_key: Optional[str] = None
        if speak:
            openai_api_key = ctx._get_openai_api_key_for_bot(session, bot=bot)
            if not openai_api_key:
                raise ctx.HTTPException(status_code=400, detail="No OpenAI key configured for this bot (needed for TTS).")

        def gen() -> Generator[bytes, None, None]:
            text = (payload.text or "").strip()
            if not text:
                yield ctx._ndjson({"type": "error", "error": "Empty text"})
                return

            messages = [ctx.Message(role="system", content=bot.system_prompt), ctx.Message(role="user", content=text)]

            delta_q_client: "ctx.queue.Queue[Optional[str]]" = ctx.queue.Queue()
            delta_q_tts: "ctx.queue.Queue[Optional[str]]" = ctx.queue.Queue()
            audio_q: "ctx.queue.Queue[Optional[tuple[bytes, int]]]" = ctx.queue.Queue()
            full_text_parts: list[str] = []
            error_q: "ctx.queue.Queue[Optional[str]]" = ctx.queue.Queue()

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
                    tts_synth = ctx._get_tts_synth_fn(bot, openai_api_key)
                    for text_to_speak in ctx._iter_tts_chunks(delta_q_tts):
                        if not text_to_speak:
                            continue
                        wav, sr = tts_synth(text_to_speak)
                        audio_q.put((wav, sr))
                finally:
                    audio_q.put(None)

            t1 = ctx.threading.Thread(target=llm_thread, daemon=True)
            t2 = ctx.threading.Thread(target=tts_thread, daemon=True)
            t1.start()
            t2.start()

            open_deltas = True
            open_audio = True
            last_heartbeat = ctx.time.time()

            while open_deltas or open_audio:
                sent = False
                try:
                    err = error_q.get_nowait()
                    if err:
                        yield ctx._ndjson({"type": "error", "error": err})
                        open_deltas = False
                        open_audio = False
                        break
                except ctx.queue.Empty:
                    pass
                try:
                    d = delta_q_client.get_nowait()
                    if d is None:
                        open_deltas = False
                    else:
                        yield ctx._ndjson({"type": "text_delta", "delta": d})
                    sent = True
                except ctx.queue.Empty:
                    pass

                if speak:
                    try:
                        item = audio_q.get_nowait()
                        if item is None:
                            open_audio = False
                        else:
                            wav, sr = item
                            yield ctx._ndjson(
                                {"type": "audio_wav", "wav_base64": ctx.base64.b64encode(wav).decode(), "sr": sr}
                            )
                        sent = True
                    except ctx.queue.Empty:
                        pass
                else:
                    open_audio = False

                if not sent:
                    if ctx.time.time() - last_heartbeat > 10:
                        yield ctx._ndjson({"type": "ping"})
                        last_heartbeat = ctx.time.time()
                    ctx.time.sleep(0.01)

            t1.join()
            t2.join()
            yield ctx._ndjson({"type": "done", "text": "".join(full_text_parts).strip()})

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @router.post("/api/bots/{bot_id}/talk/stream")
    def talk_stream(
        bot_id: UUID,
        audio: UploadFile = File(...),
        conversation_id: str = Form(""),
        test_flag: bool = Form(True),
        speak: bool = Form(True),
        session: Session = Depends(ctx.get_session),
    ) -> StreamingResponse:
        bot = ctx.get_bot(session, bot_id)
        openai_api_key = ctx._get_openai_api_key_for_bot(session, bot=bot)
        if not openai_api_key:
            raise ctx.HTTPException(status_code=400, detail="No OpenAI key configured for this bot.")
        _provider, llm_api_key, llm = ctx._require_llm_client(session, bot=bot)

        conv_id: Optional[UUID] = UUID(conversation_id) if conversation_id.strip() else None
        if conv_id is None:
            conv = ctx.create_conversation(session, bot_id=bot.id, test_flag=bool(test_flag))
            conv_id = conv.id

        wav_bytes = audio.file.read()
        pcm16 = ctx._decode_wav_bytes_to_pcm16_16k(wav_bytes)
        if not pcm16:
            raise ctx.HTTPException(status_code=400, detail="Empty audio")

        asr = ctx._get_asr(openai_api_key, bot.openai_asr_model, bot.language).transcribe_pcm16(
            pcm16=pcm16, sample_rate=16000
        )
        user_text = asr.text.strip()
        if not user_text:
            def empty_gen() -> Generator[bytes, None, None]:
                yield ctx._ndjson({"type": "conversation", "id": str(conv_id)})
                yield ctx._ndjson({"type": "asr", "text": ""})
                yield ctx._ndjson({"type": "done", "text": ""})

            return StreamingResponse(empty_gen(), media_type="application/x-ndjson")

        ctx.add_message(session, conversation_id=conv_id, role="user", content=user_text)

        def gen() -> Generator[bytes, None, None]:
            yield ctx._ndjson({"type": "conversation", "id": str(conv_id)})
            yield ctx._ndjson({"type": "asr", "text": user_text})

            history = ctx._build_history_budgeted(
                session=session,
                bot=bot,
                conversation_id=conv_id,
                llm_api_key=llm_api_key,
                status_cb=None,
            )

            delta_q_client: "ctx.queue.Queue[Optional[str]]" = ctx.queue.Queue()
            delta_q_tts: "ctx.queue.Queue[Optional[str]]" = ctx.queue.Queue()
            audio_q: "ctx.queue.Queue[Optional[tuple[bytes, int]]]" = ctx.queue.Queue()
            full_text_parts: list[str] = []
            error_q: "ctx.queue.Queue[Optional[str]]" = ctx.queue.Queue()

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
                    tts_synth = ctx._get_tts_synth_fn(bot, openai_api_key)
                    for text_to_speak in ctx._iter_tts_chunks(delta_q_tts):
                        if not text_to_speak:
                            continue
                        wav, sr = tts_synth(text_to_speak)
                        audio_q.put((wav, sr))
                finally:
                    audio_q.put(None)

            t1 = ctx.threading.Thread(target=llm_thread, daemon=True)
            t2 = ctx.threading.Thread(target=tts_thread, daemon=True)
            t1.start()
            t2.start()

            open_deltas = True
            open_audio = True
            last_heartbeat = ctx.time.time()

            while open_deltas or open_audio:
                sent = False
                try:
                    err = error_q.get_nowait()
                    if err:
                        yield ctx._ndjson({"type": "error", "error": err})
                        open_deltas = False
                        open_audio = False
                        break
                except ctx.queue.Empty:
                    pass
                try:
                    d = delta_q_client.get_nowait()
                    if d is None:
                        open_deltas = False
                    else:
                        yield ctx._ndjson({"type": "text_delta", "delta": d})
                    sent = True
                except ctx.queue.Empty:
                    pass

                if speak:
                    try:
                        item = audio_q.get_nowait()
                        if item is None:
                            open_audio = False
                        else:
                            wav, sr = item
                            yield ctx._ndjson(
                                {"type": "audio_wav", "wav_base64": ctx.base64.b64encode(wav).decode(), "sr": sr}
                            )
                        sent = True
                    except ctx.queue.Empty:
                        pass
                else:
                    open_audio = False

                if not sent:
                    if ctx.time.time() - last_heartbeat > 10:
                        yield ctx._ndjson({"type": "ping"})
                        last_heartbeat = ctx.time.time()
                    ctx.time.sleep(0.01)

            t1.join()
            t2.join()
            final_text = "".join(full_text_parts).strip()
            if final_text:
                ctx.add_message(session, conversation_id=conv_id, role="assistant", content=final_text)
            yield ctx._ndjson({"type": "done", "text": final_text})

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @router.post("/api/bots/{bot_id}/chat")
    def chat_once(
        bot_id: UUID,
        payload: ChatRequest = Body(...),
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        bot = ctx.get_bot(session, bot_id)
        _provider, _llm_api_key, llm = ctx._require_llm_client(session, bot=bot)
        text = (payload.text or "").strip()
        if not text:
            raise ctx.HTTPException(status_code=400, detail="Empty text")

        messages = [ctx.Message(role="system", content=bot.system_prompt), ctx.Message(role="user", content=text)]
        out_text = llm.complete_text(messages=messages)

        if not payload.speak:
            return {"text": out_text}

        openai_api_key = ctx._get_openai_api_key_for_bot(session, bot=bot)
        if not openai_api_key:
            raise ctx.HTTPException(status_code=400, detail="No OpenAI key configured for this bot (needed for TTS).")
        tts_synth = ctx._get_tts_synth_fn(bot, openai_api_key)
        wav, sr = tts_synth(out_text)
        return {"text": out_text, "audio_wav_base64": ctx.base64.b64encode(wav).decode(), "sr": sr}

    app.include_router(router)
