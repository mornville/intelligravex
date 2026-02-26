from __future__ import annotations

import asyncio
import datetime as dt
import json
from typing import Callable, Optional
from uuid import UUID

from sqlmodel import Session, delete

from voicebot.llm.openai_llm import Message
from voicebot.models import Bot, ConversationMessage
from voicebot.utils.prompt import system_prompt_with_runtime


def build_history(ctx, session: Session, bot: Bot, conversation_id: Optional[UUID]) -> list[Message]:
    require_approval = bool(getattr(bot, "require_host_action_approval", False))
    messages: list[Message] = [
        Message(role="system", content=system_prompt_with_runtime(bot.system_prompt, require_approval=require_approval))
    ]
    if not conversation_id:
        return messages
    conv = ctx.get_conversation(session, conversation_id)
    ctx._assert_bot_in_conversation(conv, bot.id)
    meta = ctx.safe_json_loads(conv.metadata_json or "{}") or {}
    ctx_obj = {"meta": meta}
    messages = [
        Message(
            role="system",
            content=system_prompt_with_runtime(
                ctx.render_template(bot.system_prompt, ctx=ctx_obj),
                require_approval=require_approval,
            ),
        )
    ]
    if meta:
        messages.append(
            Message(role="system", content=f"Conversation metadata (JSON): {json.dumps(meta, ensure_ascii=False)}")
        )
    if bool(conv.is_group):
        bots = ctx._group_bots_from_conv(conv)
        slugs = ", ".join(f"@{b['slug']}" for b in bots)
        is_default = str(conv.bot_id or "") == str(bot.id)
        if is_default:
            routing = (
                "Group routing: If the latest message includes @mentions, only respond when you are mentioned. "
                "If there are no @mentions, you are the default responder and should reply. "
                "If you have nothing to add, respond with <no_reply> (this will be hidden). "
                f"Available: {slugs}"
            )
        else:
            routing = (
                "Group routing: assistants should only respond when explicitly mentioned with @slug in the latest message. "
                "If you are not mentioned, respond with <no_reply> (this will be hidden). "
                f"Available: {slugs}"
            )
        messages.append(Message(role="system", content=routing))
    for m in ctx.list_messages(session, conversation_id=conversation_id):
        prefix = ctx._format_group_message_prefix(
            conv=conv,
            sender_bot_id=m.sender_bot_id,
            sender_name=m.sender_name,
            fallback_role=m.role,
        )
        if m.role in ("user", "assistant"):
            messages.append(Message(role=m.role, content=ctx.render_template(f"{prefix}{m.content}", ctx=ctx_obj)))
        elif m.role == "tool":
            try:
                obj = json.loads(m.content or "")
                if isinstance(obj, dict) and obj.get("tool") == "debug_llm_request":
                    continue
            except Exception:
                pass
            messages.append(
                Message(
                    role="system",
                    content=ctx.render_template(f"{prefix}Tool event: {m.content}", ctx=ctx_obj),
                )
            )
    return messages


def build_history_budgeted(
    ctx,
    *,
    session: Session,
    bot: Bot,
    conversation_id: Optional[UUID],
    llm_api_key: Optional[str],
    status_cb: Optional[Callable[[str], None]] = None,
) -> list[Message]:
    HISTORY_TOKEN_BUDGET = 400000
    SUMMARY_BATCH_MIN_MESSAGES = 8

    if not conversation_id:
        require_approval = bool(getattr(bot, "require_host_action_approval", False))
        return [
            Message(
                role="system",
                content=system_prompt_with_runtime(bot.system_prompt, require_approval=require_approval),
            )
        ]

    conv = ctx.get_conversation(session, conversation_id)
    ctx._assert_bot_in_conversation(conv, bot.id)

    meta = ctx.safe_json_loads(conv.metadata_json or "{}") or {}
    if not isinstance(meta, dict):
        meta = {}

    memory = meta.get("memory") if isinstance(meta.get("memory"), dict) else {}
    if not isinstance(memory, dict):
        memory = {}
    memory_summary = str(memory.get("summary") or "").strip()
    pinned_facts = str(memory.get("pinned_facts") or "").strip()
    last_summarized_id = str(memory.get("last_summarized_message_id") or "").strip()

    ctx_obj = {"meta": meta}
    require_approval = bool(getattr(bot, "require_host_action_approval", False))
    system_prompt = system_prompt_with_runtime(
        ctx.render_template(bot.system_prompt, ctx=ctx_obj),
        require_approval=require_approval,
    )

    db_msgs = ctx.list_messages(session, conversation_id=conversation_id)
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
            content = content[:2000]
        prefix = ctx._format_group_message_prefix(
            conv=conv,
            sender_bot_id=m.sender_bot_id,
            sender_name=m.sender_name,
            fallback_role=role,
        )
        return f"{role.upper()}: {prefix}{content}"

    old_msgs = db_msgs[:start_idx]
    new_old_msgs = old_msgs
    if last_summarized_id:
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

    if should_summarize and llm_api_key:
        if status_cb:
            status_cb("summarizing")
        chunk = "\n".join(_format_for_summary(m) for m in new_old_msgs)
        chunk = chunk[:24000]

        summary_model = (getattr(bot, "summary_model", "") or "gpt-5-nano").strip() or "gpt-5-nano"
        summarizer = ctx._build_llm_client(bot=bot, api_key=llm_api_key, model_override=summary_model)
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
            new_summary = memory_summary
            new_pinned = pinned_facts

        if new_summary:
            patch = {
                "memory.summary": new_summary,
                "memory.pinned_facts": new_pinned,
                "memory.last_summarized_message_id": str(getattr(new_old_msgs[-1], "id", "")) if new_old_msgs else last_summarized_id,
                "memory.updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            meta = ctx.merge_conversation_metadata(session, conversation_id=conversation_id, patch=patch)
            if isinstance(meta, dict):
                memory = meta.get("memory") if isinstance(meta.get("memory"), dict) else {}
                if isinstance(memory, dict):
                    memory_summary = str(memory.get("summary") or "").strip()
                    pinned_facts = str(memory.get("pinned_facts") or "").strip()
            if new_old_msgs:
                try:
                    ids = [m.id for m in new_old_msgs if getattr(m, "id", None)]
                    if ids:
                        session.exec(delete(ConversationMessage).where(ConversationMessage.id.in_(ids)))
                        session.commit()
                except Exception:
                    session.rollback()

    messages: list[Message] = [Message(role="system", content=system_prompt)]
    if bool(conv.is_group):
        bots = ctx._group_bots_from_conv(conv)
        slugs = ", ".join(f"@{b['slug']}" for b in bots)
        is_default = str(conv.bot_id or "") == str(bot.id)
        if is_default:
            routing = (
                "Group routing: If the latest message includes @mentions, only respond when you are mentioned. "
                "If there are no @mentions, you are the default responder and should reply. "
                "If you have nothing to add, respond with <no_reply> (this will be hidden). "
                f"Available: {slugs}"
            )
        else:
            routing = (
                "Group routing: assistants should only respond when explicitly mentioned with @slug in the latest message. "
                "If you are not mentioned, respond with <no_reply> (this will be hidden). "
                f"Available: {slugs}"
            )
        messages.append(Message(role="system", content=routing))
    if memory_summary:
        messages.append(Message(role="system", content=f"Conversation summary:\n{memory_summary}"))
    if pinned_facts:
        messages.append(Message(role="system", content=f"Pinned facts:\n{pinned_facts}"))

    for m in db_msgs[start_idx:]:
        prefix = ctx._format_group_message_prefix(
            conv=conv,
            sender_bot_id=m.sender_bot_id,
            sender_name=m.sender_name,
            fallback_role=m.role,
        )
        if m.role in ("user", "assistant"):
            messages.append(
                Message(role=m.role, content=ctx.render_template(f"{prefix}{m.content}", ctx={"meta": meta}))
            )
        elif m.role == "tool":
            try:
                obj = json.loads(m.content or "")
                if isinstance(obj, dict) and obj.get("tool") == "debug_llm_request":
                    continue
            except Exception:
                pass
            messages.append(
                Message(
                    role="system",
                    content=ctx.render_template(f"{prefix}Tool event: {m.content or ''}", ctx={"meta": meta}),
                )
            )

    try:
        while ctx.estimate_messages_tokens(messages, bot.openai_model) > HISTORY_TOKEN_BUDGET and len(messages) > 4:
            del messages[3]
    except Exception:
        pass
    return messages


def build_history_budgeted_threadsafe(
    ctx,
    *,
    bot_id: UUID,
    conversation_id: Optional[UUID],
    llm_api_key: Optional[str],
    status_cb: Optional[Callable[[str], None]] = None,
) -> list[Message]:
    with Session(ctx.engine) as session:
        bot = ctx.get_bot(session, bot_id)
        return build_history_budgeted(
            ctx,
            session=session,
            bot=bot,
            conversation_id=conversation_id,
            llm_api_key=llm_api_key,
            status_cb=status_cb,
        )


async def build_history_budgeted_async(
    ctx,
    *,
    bot_id: UUID,
    conversation_id: Optional[UUID],
    llm_api_key: Optional[str],
    status_cb: Optional[Callable[[str], None]] = None,
) -> list[Message]:
    return await asyncio.to_thread(
        build_history_budgeted_threadsafe,
        ctx,
        bot_id=bot_id,
        conversation_id=conversation_id,
        llm_api_key=llm_api_key,
        status_cb=status_cb,
    )


def get_conversation_meta(ctx, session: Session, *, conversation_id: UUID) -> dict:
    conv = ctx.get_conversation(session, conversation_id)
    meta = ctx.safe_json_loads(conv.metadata_json or "{}") or {}
    return meta if isinstance(meta, dict) else {}
