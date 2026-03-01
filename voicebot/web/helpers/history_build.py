from __future__ import annotations

import asyncio
import datetime as dt
import json
from typing import Any, Callable, Optional
from uuid import UUID

from sqlmodel import Session, delete

from voicebot.llm.openai_llm import Message
from voicebot.models import Bot, ConversationMessage
from voicebot.utils.prompt import system_prompt_with_runtime


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _wants_internal_collab(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return False
    signals = [
        "talk to each other",
        "talk with each other",
        "you guys talk",
        "you two talk",
        "discuss with each other",
        "chat with each other",
        "talk among yourselves",
        "talk among yourself",
    ]
    return any(sig in t for sig in signals)


def _build_group_system_messages(
    *,
    ctx,
    conv,
    meta: dict,
    bot: Bot,
    latest_user_text: str = "",
) -> list[Message]:
    bots = ctx._group_bots_from_conv(conv)
    if not bots:
        return []

    bot_by_id: dict[str, dict[str, str]] = {}
    for b in bots:
        bid = str(b.get("id") or "").strip()
        if bid:
            bot_by_id[bid] = b

    self_id = str(bot.id)
    default_id = str(conv.bot_id or "").strip()
    self_bot = bot_by_id.get(self_id) or {}
    default_bot = bot_by_id.get(default_id) or {}

    self_name = str(self_bot.get("name") or bot.name or "Assistant").strip() or "Assistant"
    self_slug = str(self_bot.get("slug") or "").strip().lower()
    default_slug = str(default_bot.get("slug") or "").strip().lower()

    roster_lines: list[str] = []
    roster_items: list[dict[str, Any]] = []
    for b in bots:
        bid = str(b.get("id") or "").strip()
        name = str(b.get("name") or "").strip() or "Assistant"
        slug = str(b.get("slug") or "").strip().lower()
        flags: list[str] = []
        if bid == self_id:
            flags.append("you")
        if bid == default_id:
            flags.append("default_responder")
        line = f"- @{slug}: {name}" if slug else f"- {name}"
        if flags:
            line += f" ({', '.join(flags)})"
        roster_lines.append(line)
        roster_items.append(
            {
                "id": bid,
                "name": name,
                "slug": slug,
                "is_you": bid == self_id,
                "is_default_responder": bid == default_id,
            }
        )

    is_default = self_id == default_id
    if is_default:
        routing = (
            "Group routing: if the latest message has @mentions and you are not mentioned, return <no_reply>. "
            "If there are no @mentions, you are the default responder and should reply."
        )
    else:
        routing = (
            "Group routing: respond only when you are directly mentioned with your @slug "
            "or explicitly assigned in swarm shared state. Otherwise return <no_reply>."
        )

    identity = f"You are {self_name}"
    if self_slug:
        identity += f" (@{self_slug})"
    if default_slug:
        identity += f". Default responder is @{default_slug}."
    else:
        identity += "."

    messages: list[Message] = [
        Message(role="system", content=identity),
        Message(role="system", content=f"Group members:\n{chr(10).join(roster_lines)}"),
        Message(role="system", content=routing),
        Message(
            role="system",
            content=(
                "Collaboration protocol:\n"
                "1) Use exact @slug when handing work to another assistant (not plain name).\n"
                "2) When delegating, include a concrete ask and expected output.\n"
                "3) If the user says 'other assistant' and there is only one teammate besides you, target that teammate.\n"
                "4) If the user asks assistants to talk among themselves, start an internal handoff using @slug.\n"
                "5) Never write transcript lines on behalf of other assistants; only write your own turn.\n"
                "6) If you have no meaningful contribution for this turn, output <no_reply> exactly."
            ),
        ),
        Message(
            role="system",
            content=(
                "Group roster (JSON): "
                + json.dumps(
                    {
                        "you": {"id": self_id, "name": self_name, "slug": self_slug},
                        "default_responder_id": default_id,
                        "default_responder_slug": default_slug,
                        "members": roster_items,
                    },
                    ensure_ascii=False,
                )
            ),
        ),
    ]

    swarm = meta.get("group_swarm") if isinstance(meta.get("group_swarm"), dict) else {}
    active = swarm.get("active_run") if isinstance(swarm.get("active_run"), dict) else {}
    signals = [latest_user_text]
    if active and str(active.get("sender_role") or "").strip().lower() == "user":
        signals.append(str(active.get("objective") or ""))
    internal_collab = any(_wants_internal_collab(s) for s in signals)
    teammate_slugs = [
        str(x.get("slug") or "").strip().lower()
        for x in roster_items
        if not bool(x.get("is_you")) and str(x.get("slug") or "").strip()
    ]
    if internal_collab:
        target = f"@{teammate_slugs[0]}" if teammate_slugs else "a teammate"
        messages.append(
            Message(
                role="system",
                content=(
                    "Internal collaboration mode: ON (requested by user).\n"
                    "- Do not ask the user clarifying questions in this mode.\n"
                    "- Do not output option menus.\n"
                    f"- If you need teammate input, send one direct handoff to {target} with a concrete ask.\n"
                    "- Do not simulate both sides of the conversation in one message.\n"
                    "- Keep turns short and task-focused."
                ),
            )
        )
    if not active or str(active.get("status") or "") != "running":
        return messages

    by_bot = active.get("by_bot") if isinstance(active.get("by_bot"), dict) else {}
    pending = {str(x) for x in active.get("pending_bot_ids") or [] if str(x or "").strip()}
    scheduled = {str(x) for x in active.get("scheduled_bot_ids") or [] if str(x or "").strip()}
    inflight = {str(x) for x in active.get("inflight_bot_ids") or [] if str(x or "").strip()}
    completed = {str(x) for x in active.get("completed_bot_ids") or [] if str(x or "").strip()}
    failed = {str(x) for x in active.get("failed_bot_ids") or [] if str(x or "").strip()}

    member_states: list[dict[str, Any]] = []
    for b in roster_items:
        bid = str(b.get("id") or "").strip()
        state_obj = by_bot.get(bid) if isinstance(by_bot.get(bid), dict) else {}
        state = str(state_obj.get("state") or "").strip().lower()
        if not state:
            if bid in inflight:
                state = "running"
            elif bid in scheduled:
                state = "scheduled"
            elif bid in pending:
                state = "pending"
            elif bid in completed:
                state = "completed"
            elif bid in failed:
                state = "failed"
            else:
                state = "idle"
        member_states.append({"id": bid, "slug": b.get("slug"), "name": b.get("name"), "state": state})

    bot_state = by_bot.get(self_id) if isinstance(by_bot.get(self_id), dict) else {}
    state_payload = {
        "run_id": str(active.get("run_id") or ""),
        "status": str(active.get("status") or ""),
        "objective": str(active.get("objective") or ""),
        "remaining_turns": _safe_int(active.get("remaining_turns"), default=0),
        "max_turns": _safe_int(active.get("max_turns"), default=0),
        "hop_count": _safe_int(active.get("hop_count"), default=0),
        "max_hops": _safe_int(active.get("max_hops"), default=0),
        "coordinator_bot_id": str(active.get("coordinator_bot_id") or ""),
        "assigned_to_you": bot_state,
        "member_states": member_states,
        "conflicts_recent": (active.get("conflicts") if isinstance(active.get("conflicts"), list) else [])[-8:],
    }
    messages.append(
        Message(
            role="system",
            content=f"Swarm shared state (JSON): {json.dumps(state_payload, ensure_ascii=False)}",
        )
    )
    messages.append(
        Message(
            role="system",
            content=(
                "Swarm conflict policy: coordinator has tie-break priority; avoid duplicate work; "
                "handoff with @slug when another assistant should take the next step."
            ),
        )
    )
    return messages


def build_history(ctx, session: Session, bot: Bot, conversation_id: Optional[UUID]) -> list[Message]:
    require_approval = bool(getattr(bot, "require_host_action_approval", False))
    host_actions_enabled = bool(getattr(bot, "enable_host_actions", False))
    messages: list[Message] = [
        Message(
            role="system",
            content=system_prompt_with_runtime(
                bot.system_prompt,
                require_approval=require_approval,
                host_actions_enabled=host_actions_enabled,
            ),
        )
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
                host_actions_enabled=host_actions_enabled,
            ),
        )
    ]
    if meta:
        messages.append(
            Message(role="system", content=f"Conversation metadata (JSON): {json.dumps(meta, ensure_ascii=False)}")
        )
    db_msgs = ctx.list_messages(session, conversation_id=conversation_id)
    latest_user_text = ""
    for m in reversed(db_msgs):
        if m.role == "user" and str(m.content or "").strip():
            latest_user_text = str(m.content or "")
            break
    if bool(conv.is_group):
        messages.extend(
            _build_group_system_messages(
                ctx=ctx,
                conv=conv,
                meta=meta,
                bot=bot,
                latest_user_text=latest_user_text,
            )
        )
    for m in db_msgs:
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
        host_actions_enabled = bool(getattr(bot, "enable_host_actions", False))
        return [
            Message(
                role="system",
                content=system_prompt_with_runtime(
                    bot.system_prompt,
                    require_approval=require_approval,
                    host_actions_enabled=host_actions_enabled,
                ),
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
    host_actions_enabled = bool(getattr(bot, "enable_host_actions", False))
    system_prompt = system_prompt_with_runtime(
        ctx.render_template(bot.system_prompt, ctx=ctx_obj),
        require_approval=require_approval,
        host_actions_enabled=host_actions_enabled,
    )

    db_msgs = ctx.list_messages(session, conversation_id=conversation_id)
    latest_user_text = ""
    for m in reversed(db_msgs):
        if m.role == "user" and str(m.content or "").strip():
            latest_user_text = str(m.content or "")
            break
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
        messages.extend(
            _build_group_system_messages(
                ctx=ctx,
                conv=conv,
                meta=meta,
                bot=bot,
                latest_user_text=latest_user_text,
            )
        )
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
