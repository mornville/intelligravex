from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional
from uuid import UUID

from sqlmodel import Session

from voicebot.llm.openai_llm import Message, OpenAILLM
from voicebot.models import Bot
from voicebot.utils.prompt import append_host_action_approval_notice


async def ws_send_json(ctx, ws, obj: dict) -> None:
    try:
        await ws.send_text(json.dumps(obj, ensure_ascii=False))
    except Exception:
        return


_ASYNC_STREAM_DONE = object()


async def aiter_from_blocking_iterator(ctx, iterator_fn):
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _runner() -> None:
        try:
            for item in iterator_fn():
                loop.call_soon_threadsafe(q.put_nowait, item)
        except BaseException as exc:
            loop.call_soon_threadsafe(q.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _ASYNC_STREAM_DONE)

    t = ctx.threading.Thread(target=_runner, daemon=True)
    t.start()

    while True:
        item = await q.get()
        if item is _ASYNC_STREAM_DONE:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


async def stream_llm_reply(
    ctx,
    *,
    ws,
    req_id: str,
    llm: OpenAILLM,
    messages: list[Message],
) -> tuple[str, Optional[int], int]:
    t0 = time.time()
    first: Optional[float] = None
    parts: list[str] = []
    async for d in aiter_from_blocking_iterator(ctx, lambda: llm.stream_text(messages=messages)):
        d = str(d or "")
        if first is None:
            first = time.time()
        if d:
            parts.append(d)
            await ws_send_json(ctx, ws, {"type": "text_delta", "req_id": req_id, "delta": d})
    t1 = time.time()
    text = "".join(parts).strip()
    ttfb_ms = int(round((first - t0) * 1000.0)) if first is not None else None
    total_ms = int(round((t1 - t0) * 1000.0))
    return text, ttfb_ms, total_ms


class NullWebSocket:
    async def send_text(self, _text: str) -> None:
        return None


def record_llm_debug_payload(ctx, *, conversation_id: UUID, payload: dict[str, Any], phase: str) -> None:
    try:
        with Session(ctx.engine) as session:
            ctx.add_message_with_metrics(
                session,
                conversation_id=conversation_id,
                role="tool",
                content=json.dumps(
                    {"tool": "debug_llm_request", "arguments": {"phase": phase, "payload": payload}},
                    ensure_ascii=False,
                ),
            )
    except Exception:
        return


async def emit_llm_debug_payload(
    ctx,
    *,
    ws,
    req_id: str,
    conversation_id: UUID,
    payload: dict[str, Any],
    phase: str,
) -> None:
    try:
        await ws_send_json(
            ctx,
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
    record_llm_debug_payload(ctx, conversation_id=conversation_id, payload=payload, phase=phase)


async def public_send_done(
    ctx,
    ws,
    *,
    req_id: str,
    text: str,
    metrics: dict,
    citations: Optional[list[dict]] = None,
) -> None:
    payload = {"type": "done", "req_id": req_id, "text": text, "metrics": metrics}
    if citations:
        payload["citations"] = citations
    await ws_send_json(ctx, ws, payload)


async def public_send_interim(ctx, ws, *, req_id: str, kind: str, text: str) -> None:
    t = (text or "").strip()
    if not t:
        return
    if kind == "wait":
        now = time.time()
        cache = getattr(public_send_interim, "_wait_cache", None)
        if cache is None:
            cache = {}
            setattr(public_send_interim, "_wait_cache", cache)
        key = f"{id(ws)}:{req_id}:{t}"
        last_ts = cache.get(key)
        if isinstance(last_ts, float) and (now - last_ts) < 45.0:
            return
        cache[key] = now
        if len(cache) > 1024:
            for k, ts in list(cache.items()):
                if not isinstance(ts, float) or (now - ts) > 300.0:
                    cache.pop(k, None)
    await ws_send_json(ctx, ws, {"type": "interim", "req_id": req_id, "kind": kind, "text": t})


async def public_send_greeting(
    ctx,
    *,
    ws,
    req_id: str,
    bot: Bot,
    conv_id: UUID,
    provider: str,
    llm_api_key: str,
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
        llm = ctx._build_llm_client(bot=bot, api_key=llm_api_key)
        sys_prompt = append_host_action_approval_notice(
            ctx.render_template(bot.system_prompt, ctx={"meta": {}}),
            require_approval=bool(getattr(bot, "require_host_action_approval", False)),
        )
        msgs = [
            Message(role="system", content=sys_prompt),
            Message(role="user", content=ctx._make_start_message_instruction(bot)),
        ]
        t0 = time.time()
        first = None
        parts: list[str] = []
        async for d in aiter_from_blocking_iterator(ctx, lambda: llm.stream_text(messages=msgs)):
            d = str(d or "")
            if first is None:
                first = time.time()
            if d:
                parts.append(d)
                await ws_send_json(ctx, ws, {"type": "text_delta", "req_id": req_id, "delta": d})
                sent_delta = True
        t1 = time.time()
        greeting_text = "".join(parts).strip() or greeting_text
        if first is not None:
            llm_ttfb_ms = int(round((first - t0) * 1000.0))
        llm_total_ms = int(round((t1 - t0) * 1000.0))

        input_tokens_est = int(ctx.estimate_messages_tokens(msgs, bot.openai_model) or 0)
        output_tokens_est = int(ctx.estimate_text_tokens(greeting_text, bot.openai_model) or 0)
        with Session(ctx.engine) as session:
            price = ctx._get_model_price(session, provider=provider, model=bot.openai_model)
        cost_usd_est = float(
            ctx.estimate_cost_usd(model_price=price, input_tokens=input_tokens_est, output_tokens=output_tokens_est) or 0.0
        )

    if not greeting_text:
        greeting_text = "Hi! How can I help you today?"

    if not sent_delta:
        await ws_send_json(ctx, ws, {"type": "text_delta", "req_id": req_id, "delta": greeting_text})

    with Session(ctx.engine) as session:
        ctx.add_message_with_metrics(
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
        ctx.update_conversation_metrics(
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
