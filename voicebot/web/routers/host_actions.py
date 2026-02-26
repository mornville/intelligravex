from __future__ import annotations
from uuid import UUID

import time

from fastapi import APIRouter, Depends
from sqlmodel import Session

from voicebot.llm.openai_llm import Message


def register(app, ctx) -> None:
    router = APIRouter()

    @router.get("/api/conversations/{conversation_id}/host-actions")
    def api_conversation_host_actions(
        conversation_id: UUID,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        _ = ctx.get_conversation(session, conversation_id)
        stmt = (
            ctx.select(ctx.HostAction)
            .where(ctx.HostAction.conversation_id == conversation_id)
            .order_by(ctx.HostAction.created_at.desc())
        )
        items = list(session.exec(stmt))
        return {"items": [ctx._host_action_payload(a) for a in items]}

    @router.post("/api/host-actions/{action_id}/run")
    def api_run_host_action(
        action_id: UUID,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        action = session.get(ctx.HostAction, action_id)
        if not action:
            raise ctx.HTTPException(status_code=404, detail="Host action not found")
        conv = ctx.get_conversation(session, action.conversation_id)
        req_bot_id = action.requested_by_bot_id or conv.bot_id
        bot = ctx.get_bot(session, req_bot_id)
        if not bool(getattr(bot, "enable_host_actions", False)):
            raise ctx.HTTPException(status_code=400, detail="Host actions are disabled for this assistant.")
        if action.action_type == "run_shell" and not bool(getattr(bot, "enable_host_shell", False)):
            raise ctx.HTTPException(status_code=400, detail="Shell commands are disabled for this assistant.")

        tool_result = ctx._execute_host_action_and_update(session, action=action)
        tool_result_msg = ctx.add_message_with_metrics(
            session,
            conversation_id=action.conversation_id,
            role="tool",
            content=ctx.json.dumps({"tool": "request_host_action", "result": tool_result}, ensure_ascii=False),
            sender_bot_id=bot.id,
            sender_name=bot.name,
        )
        ctx._mirror_group_message(session, conv=conv, msg=tool_result_msg)
        try:
            ctx._conversation_ws_broadcast(
                action.conversation_id,
                {
                    "type": "tool_result",
                    "conversation_id": str(action.conversation_id),
                    "name": "request_host_action",
                    "result": tool_result,
                },
            )
        except Exception:
            pass

        rendered_reply = ""
        llm_ttfb_ms = None
        llm_total_ms = None
        provider = ctx._llm_provider_for_bot(bot)
        history = None
        try:
            provider, llm_api_key, llm = ctx._require_llm_client(session, bot=bot)
            history = ctx._build_history_budgeted(
                session=session,
                bot=bot,
                conversation_id=action.conversation_id,
                llm_api_key=llm_api_key,
                status_cb=None,
            )
            history.append(
                Message(
                    role="system",
                    content=(
                        ("The previous tool call failed. " if not bool(tool_result.get("ok", False)) else "")
                        + "Using the latest tool result(s) above, write the next assistant reply. "
                        "If the tool result contains codex_result_text, rephrase it for the user and do not mention file paths. "
                        "Do not call any tools."
                    ),
                )
            )
            t0 = time.time()
            first = None
            parts: list[str] = []
            for d in llm.stream_text(messages=history):
                if not d:
                    continue
                if first is None:
                    first = time.time()
                parts.append(str(d))
            rendered_reply = "".join(parts).strip()
            t1 = time.time()
            if first is not None:
                llm_ttfb_ms = int(round((first - t0) * 1000.0))
            llm_total_ms = int(round((t1 - t0) * 1000.0))
        except Exception:
            rendered_reply = ""

        if not rendered_reply:
            if isinstance(tool_result, dict) and not bool(tool_result.get("ok", False)):
                msg = ctx._tool_error_message(tool_result, fallback="Host action failed.")
                rendered_reply = f"Host action failed: {msg}"
            else:
                rendered_reply = "Done."

        if rendered_reply:
            try:
                if history is None:
                    history = ctx._build_history_budgeted(
                        session=session,
                        bot=bot,
                        conversation_id=action.conversation_id,
                        llm_api_key="",
                        status_cb=None,
                    )
                in_tok, out_tok, cost = ctx._estimate_llm_cost_for_turn(
                    session=session,
                    bot=bot,
                    provider=provider,
                    history=history,
                    assistant_text=rendered_reply,
                )
                reply_msg = ctx.add_message_with_metrics(
                    session,
                    conversation_id=action.conversation_id,
                    role="assistant",
                    content=rendered_reply,
                    input_tokens_est=in_tok,
                    output_tokens_est=out_tok,
                    cost_usd_est=cost,
                    llm_ttfb_ms=llm_ttfb_ms,
                    llm_total_ms=llm_total_ms,
                    total_ms=llm_total_ms,
                    sender_bot_id=bot.id,
                    sender_name=bot.name,
                )
                ctx._mirror_group_message(session, conv=conv, msg=reply_msg)
                ctx.update_conversation_metrics(
                    session,
                    conversation_id=action.conversation_id,
                    add_input_tokens_est=in_tok,
                    add_output_tokens_est=out_tok,
                    add_cost_usd_est=cost,
                    last_asr_ms=None,
                    last_llm_ttfb_ms=llm_ttfb_ms,
                    last_llm_total_ms=llm_total_ms,
                    last_tts_first_audio_ms=None,
                    last_total_ms=llm_total_ms,
                )
                try:
                    ctx._conversation_ws_broadcast(
                        action.conversation_id,
                        {"type": "conversation_update", "conversation_id": str(action.conversation_id)},
                    )
                except Exception:
                    pass
            except Exception:
                pass

        return ctx._host_action_payload(action)

    app.include_router(router)
