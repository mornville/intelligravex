from __future__ import annotations

import asyncio
import base64
import time
from typing import Optional
from uuid import UUID

from sqlmodel import Session

from voicebot.utils.prompt import system_prompt_with_runtime


def bind_ctx(ctx):
    globals().update(ctx.__dict__)


async def init_conversation_and_greet(
    *,
    bot_id: UUID,
    speak: bool,
    test_flag: bool,
    ws,
    req_id: str,
    debug: bool,
) -> UUID:
    init_start = time.time()
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
        llm_api_key: Optional[str] = None
        openai_api_key: Optional[str] = None
        provider = _llm_provider_for_bot(bot)

        needs_llm = not (bot.start_message_mode == "static" and greeting_text)
        if needs_llm:
            provider, llm_api_key, llm = _require_llm_client(session, bot=bot)
        else:
            llm = None
        if speak:
            openai_api_key = _get_openai_api_key_for_bot(session, bot=bot)
            if not openai_api_key:
                raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot (needed for TTS).")

        if bot.start_message_mode == "static" and greeting_text:
            pass
        else:
            if llm is None or (provider != "local" and not llm_api_key):
                raise HTTPException(
                    status_code=400,
                    detail=f"No {_provider_display_name(provider)} key configured for this bot.",
                )
            sys_prompt = system_prompt_with_runtime(
                render_template(bot.system_prompt, ctx={"meta": {}}),
                require_approval=bool(getattr(bot, "require_host_action_approval", False)),
                host_actions_enabled=bool(getattr(bot, "enable_host_actions", False)),
            )
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
            async for d in _aiter_from_blocking_iterator(lambda: llm.stream_text(messages=msgs)):
                d = str(d or "")
                if first is None:
                    first = time.time()
                if d:
                    parts.append(d)
                    await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
                    sent_greeting_delta = True
            t1 = time.time()
            greeting_text = "".join(parts).strip()
            if first is not None:
                llm_ttfb_ms = int(round((first - t0) * 1000.0))
            llm_total_ms = int(round((t1 - t0) * 1000.0))

            input_tokens_est, output_tokens_est, cost_usd_est = _estimate_llm_cost_for_turn(
                session=session,
                bot=bot,
                provider=provider,
                history=msgs,
                assistant_text=greeting_text,
            )

        if not greeting_text:
            greeting_text = "Hi! How can I help you today?"

        if not sent_greeting_delta:
            await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": greeting_text})

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
            tts_synth = await asyncio.to_thread(_get_tts_synth_fn, bot, openai_api_key)
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
