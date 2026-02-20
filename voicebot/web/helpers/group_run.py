from __future__ import annotations

import asyncio
import re
import secrets
import time
from typing import Optional
from uuid import UUID

from sqlmodel import Session

from voicebot.llm.openai_llm import ToolCall, CitationEvent
from voicebot.models import Conversation
from voicebot.web.helpers.group_tool_calls import process_group_tool_calls
from voicebot.web.helpers.history_build import build_history_budgeted_async


def extract_group_mentions(ctx, text: str, conv: Conversation) -> list[UUID]:
    alias_map = ctx._group_bot_aliases(conv)
    if not alias_map:
        return []
    hits: list[UUID] = []
    for m in re.finditer(r"@([a-zA-Z0-9][a-zA-Z0-9_-]{0,48})", text or ""):
        token = m.group(1).lower()
        candidates = {token}
        compact = re.sub(r"[-_]+", "", token)
        if compact:
            candidates.add(compact)
        for cand in candidates:
            for bot_id in alias_map.get(cand, []):
                try:
                    bid = UUID(bot_id)
                except Exception:
                    continue
                if bid not in hits:
                    hits.append(bid)
    return hits


async def run_group_bot_turn(
    ctx,
    *,
    bot_id: UUID,
    conversation_id: UUID,
) -> str:
    req_id = f"group_{secrets.token_hex(6)}"
    with Session(ctx.engine) as session:
        bot = ctx.get_bot(session, bot_id)
        conv = ctx.get_conversation(session, conversation_id)
        provider, api_key, llm = ctx._require_llm_client(session, bot=bot)
        history = await build_history_budgeted_async(
            ctx,
            bot_id=bot.id,
            conversation_id=conversation_id,
            llm_api_key=api_key,
            status_cb=None,
        )
        tools_defs = ctx._build_tools_for_bot(session, bot.id)

    t0 = time.time()
    first_token_ts: Optional[float] = None
    tool_calls: list[ToolCall] = []
    full_text_parts: list[str] = []
    citations: list[dict] = []
    dispatch_targets: list[UUID] = []

    async for ev in ctx._aiter_from_blocking_iterator(lambda: llm.stream_text_or_tool(messages=history, tools=tools_defs)):
        if isinstance(ev, ToolCall):
            tool_calls.append(ev)
            continue
        if isinstance(ev, CitationEvent):
            citations.extend(ev.citations)
            continue
        d = str(ev)
        if d:
            if first_token_ts is None:
                first_token_ts = time.time()
            full_text_parts.append(d)

    llm_end_ts = time.time()
    rendered_reply = "".join(full_text_parts).strip()

    llm_ttfb_ms: Optional[int] = None
    if first_token_ts is not None:
        llm_ttfb_ms = int(round((first_token_ts - t0) * 1000.0))
    elif tool_calls and tool_calls[0].first_event_ts is not None:
        llm_ttfb_ms = int(round((tool_calls[0].first_event_ts - t0) * 1000.0))
    llm_total_ms = int(round((llm_end_ts - t0) * 1000.0))

    if tool_calls:
        rendered_reply, llm_ttfb_ms, llm_total_ms = await process_group_tool_calls(
            ctx,
            bot_id=bot_id,
            conversation_id=conversation_id,
            tool_calls=tool_calls,
            provider=provider,
            api_key=api_key,
            llm=llm,
            history=history,
            rendered_reply=rendered_reply,
            llm_ttfb_ms=llm_ttfb_ms,
            llm_total_ms=llm_total_ms,
        )

    if rendered_reply:
        with Session(ctx.engine) as session:
            conv = ctx.get_conversation(session, conversation_id)
        if bool(conv.is_group):
            rendered_reply = ctx._sanitize_group_reply(rendered_reply, conv, bot_id)

    payload = None
    raw_reply = rendered_reply or ""
    if re.fullmatch(r"\s*<no_reply>\s*", raw_reply, flags=re.IGNORECASE):
        rendered_reply = ""
        suppress_reply = True
    else:
        rendered_reply = re.sub(r"\s*<no_reply>\s*$", "", raw_reply, flags=re.IGNORECASE).strip()
        suppress_reply = False
    with Session(ctx.engine) as session:
        in_tok, out_tok, cost = ctx._estimate_llm_cost_for_turn(
            session=session,
            bot=bot,
            provider=provider,
            history=history,
            assistant_text=rendered_reply,
        )
        conv = ctx.get_conversation(session, conversation_id)
        assistant_msg = None
        if suppress_reply:
            if bool(conv.is_group):
                mapping = ctx._ensure_group_individual_conversations(session, conv)
                target_id = mapping.get(str(bot.id))
                if target_id:
                    ctx.add_message_with_metrics(
                        session,
                        conversation_id=UUID(target_id),
                        role="assistant",
                        content=rendered_reply,
                        sender_bot_id=bot.id,
                        sender_name=bot.name,
                        input_tokens_est=in_tok,
                        output_tokens_est=out_tok,
                        cost_usd_est=cost,
                        llm_ttfb_ms=llm_ttfb_ms,
                        llm_total_ms=llm_total_ms,
                        total_ms=llm_total_ms,
                        citations_json=ctx.json.dumps(citations, ensure_ascii=False),
                    )
        else:
            assistant_msg = ctx.add_message_with_metrics(
                session,
                conversation_id=conversation_id,
                role="assistant",
                content=rendered_reply,
                sender_bot_id=bot.id,
                sender_name=bot.name,
                input_tokens_est=in_tok,
                output_tokens_est=out_tok,
                cost_usd_est=cost,
                llm_ttfb_ms=llm_ttfb_ms,
                llm_total_ms=llm_total_ms,
                total_ms=llm_total_ms,
                citations_json=ctx.json.dumps(citations, ensure_ascii=False),
            )
            ctx._mirror_group_message(session, conv=conv, msg=assistant_msg)
        ctx.update_conversation_metrics(
            session,
            conversation_id=conversation_id,
            add_input_tokens_est=in_tok,
            add_output_tokens_est=out_tok,
            add_cost_usd_est=cost,
            last_asr_ms=None,
            last_llm_ttfb_ms=llm_ttfb_ms,
            last_llm_total_ms=llm_total_ms,
            last_tts_first_audio_ms=None,
            last_total_ms=llm_total_ms,
        )

        if assistant_msg is not None:
            payload = ctx._group_message_payload(assistant_msg)
        if assistant_msg is not None:
            dispatch_targets = extract_group_mentions(ctx, rendered_reply, conv)
            dispatch_targets = [bid for bid in dispatch_targets if str(bid) != str(bot.id)]

    if payload:
        await ctx._group_ws_broadcast(conversation_id, {"type": "message", "message": payload})
    if dispatch_targets:
        schedule_group_bots(ctx, conversation_id, dispatch_targets)

    return rendered_reply


def schedule_group_bots(ctx, conversation_id: UUID, targets: list[UUID]) -> None:
    if not targets:
        return
    with Session(ctx.engine) as session:
        conv = ctx.get_conversation(session, conversation_id)
        bot_map = {b["id"]: b for b in ctx._group_bots_from_conv(conv)}

    async def _run_target(target_id: UUID) -> None:
        binfo = bot_map.get(str(target_id), {})
        await ctx._group_ws_broadcast(
            conversation_id,
            {
                "type": "status",
                "bot_id": str(target_id),
                "bot_name": binfo.get("name") or "assistant",
                "state": "working",
            },
        )
        try:
            await run_group_bot_turn(ctx, bot_id=target_id, conversation_id=conversation_id)
        finally:
            await ctx._group_ws_broadcast(
                conversation_id,
                {
                    "type": "status",
                    "bot_id": str(target_id),
                    "bot_name": binfo.get("name") or "assistant",
                    "state": "idle",
                },
            )

    try:
        loop = asyncio.get_running_loop()
        for bid in targets:
            loop.create_task(_run_target(bid))
    except Exception:
        pass
