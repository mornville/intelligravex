from __future__ import annotations

import datetime as dt
import json
import re
from typing import Optional
from uuid import UUID

from sqlmodel import Session, delete

from voicebot.models import Bot, Conversation, ConversationMessage
from voicebot.store import (
    add_message_with_metrics,
    get_conversation,
    list_messages,
    merge_conversation_metadata,
    update_conversation_metrics,
)
from voicebot.utils.template import safe_json_loads


def slugify(value: str) -> str:
    s = (value or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "assistant"


def group_bots_from_conv(conv: Conversation) -> list[dict[str, str]]:
    raw = getattr(conv, "group_bots_json", "") or "[]"
    try:
        bots = json.loads(raw)
    except Exception:
        bots = []
    if not isinstance(bots, list):
        bots = []
    out: list[dict[str, str]] = []
    for b in bots:
        if not isinstance(b, dict):
            continue
        bid = str(b.get("id") or "").strip()
        name = str(b.get("name") or "").strip()
        slug = str(b.get("slug") or "").strip().lower()
        if not bid or not name:
            continue
        if not slug:
            slug = slugify(name)
        out.append({"id": bid, "name": name, "slug": slug})
    return out


def group_individual_map_from_conv(conv: Conversation) -> dict[str, str]:
    meta = safe_json_loads(conv.metadata_json or "{}") or {}
    if not isinstance(meta, dict):
        return {}
    mapping = meta.get("group_individual_conversations")
    if not isinstance(mapping, dict):
        return {}
    cleaned: dict[str, str] = {}
    for k, v in mapping.items():
        bid = str(k or "").strip()
        cid = str(v or "").strip()
        if bid and cid:
            cleaned[bid] = cid
    return cleaned


def ensure_group_individual_conversations(session: Session, conv: Conversation) -> dict[str, str]:
    mapping = group_individual_map_from_conv(conv)
    changed = False
    for b in group_bots_from_conv(conv):
        bid = str(b.get("id") or "").strip()
        if not bid:
            continue
        existing = mapping.get(bid)
        if existing:
            try:
                _ = get_conversation(session, UUID(existing))
                continue
            except Exception:
                pass
        now = dt.datetime.now(dt.timezone.utc)
        child = Conversation(
            bot_id=UUID(bid),
            test_flag=bool(conv.test_flag),
            is_group=False,
            metadata_json=json.dumps(
                {
                    "group_parent_id": str(conv.id),
                    "group_bot_id": bid,
                    "group_bot_name": str(b.get("name") or ""),
                },
                ensure_ascii=False,
            ),
            created_at=now,
            updated_at=now,
        )
        session.add(child)
        session.commit()
        session.refresh(child)
        mapping[bid] = str(child.id)
        changed = True
    if changed:
        merge_conversation_metadata(
            session,
            conversation_id=conv.id,
            patch={"group_individual_conversations": mapping},
        )
    return mapping


def mirror_group_message(
    session: Session,
    *,
    conv: Conversation,
    msg: ConversationMessage,
) -> None:
    if not bool(conv.is_group):
        return
    if not msg.sender_bot_id:
        return
    if msg.role not in ("assistant", "tool"):
        return
    mapping = ensure_group_individual_conversations(session, conv)
    target_id = mapping.get(str(msg.sender_bot_id))
    if not target_id:
        return
    try:
        target_uuid = UUID(str(target_id))
    except Exception:
        return
    mirror = add_message_with_metrics(
        session,
        conversation_id=target_uuid,
        role=msg.role,
        content=msg.content,
        sender_bot_id=msg.sender_bot_id,
        sender_name=msg.sender_name,
        input_tokens_est=msg.input_tokens_est,
        output_tokens_est=msg.output_tokens_est,
        cost_usd_est=msg.cost_usd_est,
        asr_ms=msg.asr_ms,
        llm_ttfb_ms=msg.llm_ttfb_ms,
        llm_total_ms=msg.llm_total_ms,
        tts_first_audio_ms=msg.tts_first_audio_ms,
        total_ms=msg.total_ms,
    )
    if mirror.role == "assistant":
        update_conversation_metrics(
            session,
            conversation_id=target_uuid,
            add_input_tokens_est=msg.input_tokens_est or 0,
            add_output_tokens_est=msg.output_tokens_est or 0,
            add_cost_usd_est=msg.cost_usd_est or 0.0,
            last_asr_ms=msg.asr_ms,
            last_llm_ttfb_ms=msg.llm_ttfb_ms,
            last_llm_total_ms=msg.llm_total_ms,
            last_tts_first_audio_ms=msg.tts_first_audio_ms,
            last_total_ms=msg.total_ms,
        )


def reset_conversation_state(session: Session, conv: Conversation, keep_meta: dict) -> None:
    session.exec(delete(ConversationMessage).where(ConversationMessage.conversation_id == conv.id))
    conv.llm_input_tokens_est = 0
    conv.llm_output_tokens_est = 0
    conv.cost_usd_est = 0.0
    conv.last_asr_ms = None
    conv.last_llm_ttfb_ms = None
    conv.last_llm_total_ms = None
    conv.last_tts_first_audio_ms = None
    conv.last_total_ms = None
    conv.last_message_at = None
    conv.last_message_preview = ""
    conv.metadata_json = json.dumps(keep_meta or {}, ensure_ascii=False)
    conv.updated_at = dt.datetime.now(dt.timezone.utc)
    session.add(conv)
    session.commit()


def group_bot_name_lookup(conv: Conversation) -> dict[str, str]:
    return {b["id"]: b["name"] for b in group_bots_from_conv(conv)}


def group_bot_slugs(conv: Conversation) -> dict[str, str]:
    return {b["slug"].lower(): b["id"] for b in group_bots_from_conv(conv)}


def group_bot_aliases(conv: Conversation) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for b in group_bots_from_conv(conv):
        bid = str(b.get("id") or "").strip()
        if not bid:
            continue
        name = str(b.get("name") or "").strip().lower()
        slug = str(b.get("slug") or "").strip().lower()
        candidate_aliases = set()
        if slug:
            candidate_aliases.add(slug)
        if name:
            candidate_aliases.add(slugify(name))
            compact_name = re.sub(r"[^a-z0-9]+", "", name)
            if compact_name:
                candidate_aliases.add(compact_name)
        for alias in list(candidate_aliases):
            compact = re.sub(r"[-_]+", "", alias)
            if compact:
                candidate_aliases.add(compact)
        for alias in candidate_aliases:
            if not alias:
                continue
            entries = aliases.setdefault(alias, [])
            if bid not in entries:
                entries.append(bid)
    return aliases


def sanitize_group_reply(text: str, conv: Conversation, bot_id: UUID) -> str:
    if not text:
        return text
    bots = group_bots_from_conv(conv)
    id_to_slug = {b["id"]: b["slug"].lower() for b in bots}
    id_to_name = {b["id"]: str(b.get("name") or "").strip().lower() for b in bots}
    self_slug = id_to_slug.get(str(bot_id), "")
    self_name = id_to_name.get(str(bot_id), "")

    m = re.match(r"^\s*\[([A-Za-z0-9 _-]{1,40})\]\s*", text)
    if m:
        tag = m.group(1).strip().lower()
        if tag and tag != self_slug and tag != self_name:
            text = text[m.end():].lstrip()

    if self_slug:
        text = re.sub(rf"@{re.escape(self_slug)}(?![A-Za-z0-9_-])", self_slug, text, flags=re.IGNORECASE)
    return text


def assert_bot_in_conversation(conv: Conversation, bot_id: UUID) -> None:
    if not bool(getattr(conv, "is_group", False)):
        if conv.bot_id != bot_id:
            raise ValueError("Conversation does not belong to bot")
        return
    bot_ids = {b["id"] for b in group_bots_from_conv(conv)}
    if bot_ids and str(bot_id) not in bot_ids:
        raise ValueError("Bot is not a member of this group")


def format_group_message_prefix(
    *,
    conv: Conversation,
    sender_bot_id: Optional[UUID],
    sender_name: Optional[str],
    fallback_role: str,
) -> str:
    if not bool(getattr(conv, "is_group", False)):
        return ""
    bots = group_bots_from_conv(conv)
    slug_by_id = {str(b.get("id") or "").strip(): str(b.get("slug") or "").strip().lower() for b in bots}
    name_by_id = {str(b.get("id") or "").strip(): str(b.get("name") or "").strip() for b in bots}

    slug = ""
    sender_id = str(sender_bot_id) if sender_bot_id else ""
    if sender_id:
        slug = slug_by_id.get(sender_id, "")
    name = (sender_name or "").strip()
    if not name and sender_id:
        name = name_by_id.get(sender_id, "")
    if not slug and name:
        lower_name = name.lower()
        for b in bots:
            b_name = str(b.get("name") or "").strip().lower()
            if b_name and b_name == lower_name:
                slug = str(b.get("slug") or "").strip().lower()
                break
    if not name:
        name = "User" if fallback_role == "user" else "Assistant"
    if fallback_role == "assistant" and slug:
        return f"[{name} (@{slug})] "
    return f"[{name}] "


def group_message_payload(m: ConversationMessage) -> dict | None:
    if m.role == "tool":
        return None
    tool_obj = safe_json_loads(m.content or "{}") if m.role == "tool" else None
    tool_name = tool_obj.get("tool") if tool_obj and isinstance(tool_obj.get("tool"), str) else None
    tool_kind = None
    if tool_obj:
        if "arguments" in tool_obj:
            tool_kind = "call"
        elif "result" in tool_obj:
            tool_kind = "result"
    try:
        citations = json.loads(getattr(m, "citations_json", "") or "[]")
        if not isinstance(citations, list):
            citations = []
    except Exception:
        citations = []
    return {
        "id": str(m.id),
        "role": m.role,
        "content": m.content,
        "created_at": m.created_at.isoformat(),
        "tool": tool_obj,
        "tool_name": tool_name,
        "tool_kind": tool_kind,
        "citations": citations,
        "sender_bot_id": str(m.sender_bot_id) if m.sender_bot_id else None,
        "sender_name": m.sender_name,
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


def group_conversation_payload(session: Session, conv: Conversation, *, include_messages: bool = True) -> dict:
    bots = group_bots_from_conv(conv)
    bot_lookup = {b["id"]: b for b in bots}
    messages: list[dict] = []
    if include_messages:
        msgs_raw = list_messages(session, conversation_id=conv.id)
        for m in msgs_raw:
            payload = group_message_payload(m)
            if payload is not None:
                messages.append(payload)

    default_bot = bot_lookup.get(str(conv.bot_id))
    individual_map = ensure_group_individual_conversations(session, conv)
    individual_items = [
        {"bot_id": bid, "conversation_id": cid} for bid, cid in individual_map.items()
    ]
    meta = safe_json_loads(conv.metadata_json or "{}") or {}
    swarm = meta.get("group_swarm") if isinstance(meta, dict) else None
    swarm_state = swarm if isinstance(swarm, dict) else None
    return {
        "conversation": {
            "id": str(conv.id),
            "title": conv.group_title or "",
            "default_bot_id": str(conv.bot_id),
            "default_bot_name": default_bot.get("name") if default_bot else None,
            "group_bots": bots,
            "swarm_state": swarm_state,
            "individual_conversations": individual_items,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
        },
        "messages": messages,
    }
