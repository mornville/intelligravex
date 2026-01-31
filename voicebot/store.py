from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import List, Optional
from uuid import UUID

from sqlmodel import Session, select

from voicebot.crypto import CryptoBox, build_hint
from voicebot.models import ApiKey, Bot, ClientKey, Conversation, ConversationMessage, IntegrationTool, GitToken


class NotFoundError(RuntimeError):
    pass


def list_keys(session: Session, *, provider: Optional[str] = None) -> List[ApiKey]:
    stmt = select(ApiKey)
    if provider:
        stmt = stmt.where(ApiKey.provider == provider)
    stmt = stmt.order_by(ApiKey.created_at.desc())
    return list(session.exec(stmt))


def get_key(session: Session, key_id: UUID) -> ApiKey:
    key = session.get(ApiKey, key_id)
    if not key:
        raise NotFoundError("API key not found")
    return key


def get_latest_key(session: Session, *, provider: str) -> ApiKey | None:
    if not provider:
        return None
    stmt = select(ApiKey).where(ApiKey.provider == provider).order_by(ApiKey.created_at.desc()).limit(1)
    return session.exec(stmt).first()


def create_key(session: Session, *, crypto: CryptoBox, provider: str, name: str, secret: str) -> ApiKey:
    k = ApiKey(
        provider=provider,
        name=name,
        hint=build_hint(secret),
        secret_ciphertext=crypto.encrypt_str(secret),
    )
    session.add(k)
    session.commit()
    session.refresh(k)
    return k


def delete_key(session: Session, key_id: UUID) -> None:
    key = session.get(ApiKey, key_id)
    if not key:
        return
    session.delete(key)
    session.commit()


def list_bots(session: Session) -> List[Bot]:
    stmt = select(Bot).order_by(Bot.updated_at.desc())
    return list(session.exec(stmt))


def bots_aggregate_metrics(session: Session) -> dict[UUID, dict]:
    from sqlmodel import func

    stmt = (
        select(
            Conversation.bot_id,
            func.count(Conversation.id),
            func.coalesce(func.sum(Conversation.llm_input_tokens_est), 0),
            func.coalesce(func.sum(Conversation.llm_output_tokens_est), 0),
            func.coalesce(func.sum(Conversation.cost_usd_est), 0.0),
            func.avg(Conversation.last_llm_ttfb_ms),
            func.avg(Conversation.last_llm_total_ms),
            func.avg(Conversation.last_total_ms),
        )
        .group_by(Conversation.bot_id)
    )
    out: dict[UUID, dict] = {}
    for bot_id, cnt, in_tok, out_tok, cost, avg_ttfb, avg_llm, avg_total in session.exec(stmt):
        out[bot_id] = {
            "conversations": int(cnt or 0),
            "input_tokens": int(in_tok or 0),
            "output_tokens": int(out_tok or 0),
            "cost_usd": float(cost or 0.0),
            "avg_llm_ttfb_ms": int(avg_ttfb) if avg_ttfb is not None else None,
            "avg_llm_total_ms": int(avg_llm) if avg_llm is not None else None,
            "avg_total_ms": int(avg_total) if avg_total is not None else None,
        }
    return out


def get_bot(session: Session, bot_id: UUID) -> Bot:
    bot = session.get(Bot, bot_id)
    if not bot:
        raise NotFoundError("Bot not found")
    return bot


def create_bot(session: Session, bot: Bot) -> Bot:
    bot.created_at = dt.datetime.now(dt.timezone.utc)
    bot.updated_at = bot.created_at
    session.add(bot)
    session.commit()
    session.refresh(bot)
    return bot


def update_bot(session: Session, bot_id: UUID, patch: dict) -> Bot:
    bot = get_bot(session, bot_id)
    for k, v in patch.items():
        setattr(bot, k, v)
    bot.updated_at = dt.datetime.now(dt.timezone.utc)
    session.add(bot)
    session.commit()
    session.refresh(bot)
    return bot


def delete_bot(session: Session, bot_id: UUID) -> None:
    bot = session.get(Bot, bot_id)
    if not bot:
        return
    # Delete dependent conversations + messages (SQLite doesn't always enforce FK cascades).
    convs = list_conversations(session, bot_id=bot_id, limit=1000000, offset=0)
    for c in convs:
        stmt = select(ConversationMessage).where(ConversationMessage.conversation_id == c.id)
        for m in session.exec(stmt):
            session.delete(m)
        session.delete(c)
    # Delete tools
    stmt = select(IntegrationTool).where(IntegrationTool.bot_id == bot_id)
    for t in session.exec(stmt):
        session.delete(t)
    session.delete(bot)
    session.commit()


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def list_client_keys(session: Session) -> List[ClientKey]:
    stmt = select(ClientKey).order_by(ClientKey.created_at.desc())
    return list(session.exec(stmt))


def get_client_key(session: Session, key_id: UUID) -> ClientKey:
    k = session.get(ClientKey, key_id)
    if not k:
        raise NotFoundError("Client key not found")
    return k


def get_git_token(session: Session, *, provider: str) -> GitToken | None:
    stmt = select(GitToken).where(GitToken.provider == provider).limit(1)
    return session.exec(stmt).first()


def upsert_git_token(session: Session, *, provider: str, token_ciphertext: bytes, hint: str) -> GitToken:
    existing = get_git_token(session, provider=provider)
    now = dt.datetime.now(dt.timezone.utc)
    if existing:
        existing.token_ciphertext = token_ciphertext
        existing.hint = hint
        existing.updated_at = now
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing
    gt = GitToken(provider=provider, token_ciphertext=token_ciphertext, hint=hint, created_at=now, updated_at=now)
    session.add(gt)
    session.commit()
    session.refresh(gt)
    return gt


def create_client_key(
    session: Session,
    *,
    name: str,
    secret: str,
    allowed_origins: str = "",
    allowed_bot_ids: Optional[list[str]] = None,
) -> ClientKey:
    k = ClientKey(
        name=name,
        hint=build_hint(secret),
        secret_hash=_sha256_hex(secret),
        allowed_origins=(allowed_origins or "").strip(),
        allowed_bot_ids_json=json.dumps(allowed_bot_ids or [], ensure_ascii=False),
    )
    session.add(k)
    session.commit()
    session.refresh(k)
    return k


def delete_client_key(session: Session, key_id: UUID) -> None:
    k = session.get(ClientKey, key_id)
    if not k:
        return
    session.delete(k)
    session.commit()


def verify_client_key(session: Session, *, secret: str) -> Optional[ClientKey]:
    if not secret:
        return None
    h = _sha256_hex(secret)
    stmt = select(ClientKey).where(ClientKey.secret_hash == h).limit(1)
    return session.exec(stmt).first()


def decrypt_provider_key(session: Session, *, crypto: CryptoBox, provider: str) -> Optional[str]:
    key = get_latest_key(session, provider=provider)
    if not key:
        return None
    return crypto.decrypt_str(key.secret_ciphertext)


def create_conversation(session: Session, *, bot_id: UUID, test_flag: bool) -> Conversation:
    now = dt.datetime.now(dt.timezone.utc)
    conv = Conversation(bot_id=bot_id, test_flag=test_flag, created_at=now, updated_at=now)
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


def get_conversation_by_external_id(
    session: Session, *, bot_id: UUID, client_key_id: Optional[UUID], external_id: str
) -> Optional[Conversation]:
    if not external_id:
        return None
    stmt = select(Conversation).where(Conversation.bot_id == bot_id).where(Conversation.external_id == external_id)
    if client_key_id:
        stmt = stmt.where(Conversation.client_key_id == client_key_id)
    return session.exec(stmt).first()


def get_or_create_conversation_by_external_id(
    session: Session, *, bot_id: UUID, test_flag: bool, client_key_id: Optional[UUID], external_id: str
) -> Conversation:
    existing = get_conversation_by_external_id(
        session, bot_id=bot_id, client_key_id=client_key_id, external_id=external_id
    )
    if existing:
        return existing
    now = dt.datetime.now(dt.timezone.utc)
    conv = Conversation(
        bot_id=bot_id,
        test_flag=test_flag,
        external_id=external_id,
        client_key_id=client_key_id,
        created_at=now,
        updated_at=now,
    )
    session.add(conv)
    session.commit()
    session.refresh(conv)
    return conv


def touch_conversation(session: Session, conv: Conversation) -> None:
    conv.updated_at = dt.datetime.now(dt.timezone.utc)
    session.add(conv)
    session.commit()

def update_conversation_metrics(
    session: Session,
    *,
    conversation_id: UUID,
    add_input_tokens_est: int,
    add_output_tokens_est: int,
    add_cost_usd_est: float,
    last_asr_ms: Optional[int],
    last_llm_ttfb_ms: Optional[int],
    last_llm_total_ms: Optional[int],
    last_tts_first_audio_ms: Optional[int],
    last_total_ms: Optional[int],
) -> None:
    conv = get_conversation(session, conversation_id)
    conv.llm_input_tokens_est = int(conv.llm_input_tokens_est or 0) + int(add_input_tokens_est or 0)
    conv.llm_output_tokens_est = int(conv.llm_output_tokens_est or 0) + int(add_output_tokens_est or 0)
    conv.cost_usd_est = float(conv.cost_usd_est or 0.0) + float(add_cost_usd_est or 0.0)
    conv.last_asr_ms = last_asr_ms
    conv.last_llm_ttfb_ms = last_llm_ttfb_ms
    conv.last_llm_total_ms = last_llm_total_ms
    conv.last_tts_first_audio_ms = last_tts_first_audio_ms
    conv.last_total_ms = last_total_ms
    touch_conversation(session, conv)

def merge_conversation_metadata(session: Session, *, conversation_id: UUID, patch: dict) -> dict:
    conv = get_conversation(session, conversation_id)
    current: dict = {}
    try:
        current = json.loads(conv.metadata_json or "{}")
        if not isinstance(current, dict):
            current = {}
    except Exception:
        current = {}
    for k, v in (patch or {}).items():
        key = str(k)
        if "." in key and key.strip(".") != key:
            # Don't treat leading/trailing dots as paths; store as-is.
            current[key] = v
            continue
        if "." in key:
            parts = [p for p in key.split(".") if p]
            if not parts:
                continue
            d = current
            for p in parts[:-1]:
                nxt = d.get(p)
                if not isinstance(nxt, dict):
                    nxt = {}
                    d[p] = nxt
                d = nxt
            d[parts[-1]] = v
        else:
            current[key] = v
    conv.metadata_json = json.dumps(current, ensure_ascii=False)
    touch_conversation(session, conv)
    return current


def list_conversations(
    session: Session,
    *,
    bot_id: Optional[UUID] = None,
    test_flag: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Conversation]:
    stmt = select(Conversation)
    if bot_id:
        stmt = stmt.where(Conversation.bot_id == bot_id)
    if test_flag is not None:
        stmt = stmt.where(Conversation.test_flag == bool(test_flag))
    stmt = stmt.order_by(Conversation.updated_at.desc())
    stmt = stmt.offset(int(offset)).limit(int(limit))
    return list(session.exec(stmt))

def count_conversations(session: Session, *, bot_id: Optional[UUID] = None, test_flag: Optional[bool] = None) -> int:
    from sqlmodel import func

    stmt = select(func.count(Conversation.id))
    if bot_id:
        stmt = stmt.where(Conversation.bot_id == bot_id)
    if test_flag is not None:
        stmt = stmt.where(Conversation.test_flag == bool(test_flag))
    return int(session.exec(stmt).one())


def get_conversation(session: Session, conversation_id: UUID) -> Conversation:
    conv = session.get(Conversation, conversation_id)
    if not conv:
        raise NotFoundError("Conversation not found")
    return conv


def add_message(session: Session, *, conversation_id: UUID, role: str, content: str) -> ConversationMessage:
    msg = ConversationMessage(conversation_id=conversation_id, role=role, content=content)
    session.add(msg)
    session.commit()
    session.refresh(msg)
    conv = session.get(Conversation, conversation_id)
    if conv:
        touch_conversation(session, conv)
    return msg


def add_message_with_metrics(
    session: Session,
    *,
    conversation_id: UUID,
    role: str,
    content: str,
    input_tokens_est: Optional[int] = None,
    output_tokens_est: Optional[int] = None,
    cost_usd_est: Optional[float] = None,
    asr_ms: Optional[int] = None,
    llm_ttfb_ms: Optional[int] = None,
    llm_total_ms: Optional[int] = None,
    tts_first_audio_ms: Optional[int] = None,
    total_ms: Optional[int] = None,
) -> ConversationMessage:
    msg = ConversationMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        input_tokens_est=input_tokens_est,
        output_tokens_est=output_tokens_est,
        cost_usd_est=cost_usd_est,
        asr_ms=asr_ms,
        llm_ttfb_ms=llm_ttfb_ms,
        llm_total_ms=llm_total_ms,
        tts_first_audio_ms=tts_first_audio_ms,
        total_ms=total_ms,
    )
    session.add(msg)
    session.commit()
    session.refresh(msg)
    conv = session.get(Conversation, conversation_id)
    if conv:
        touch_conversation(session, conv)
    return msg


def list_messages(session: Session, *, conversation_id: UUID) -> List[ConversationMessage]:
    stmt = select(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id)
    stmt = stmt.order_by(ConversationMessage.created_at.asc())
    return list(session.exec(stmt))


def list_integration_tools(session: Session, *, bot_id: UUID) -> List[IntegrationTool]:
    stmt = select(IntegrationTool).where(IntegrationTool.bot_id == bot_id)
    stmt = stmt.order_by(IntegrationTool.updated_at.desc())
    return list(session.exec(stmt))


def get_integration_tool(session: Session, tool_id: UUID) -> IntegrationTool:
    tool = session.get(IntegrationTool, tool_id)
    if not tool:
        raise NotFoundError("Tool not found")
    return tool


def get_integration_tool_by_name(session: Session, *, bot_id: UUID, name: str) -> IntegrationTool | None:
    stmt = select(IntegrationTool).where(IntegrationTool.bot_id == bot_id).where(IntegrationTool.name == name)
    return session.exec(stmt).first()


def create_integration_tool(session: Session, tool: IntegrationTool) -> IntegrationTool:
    tool.created_at = dt.datetime.now(dt.timezone.utc)
    tool.updated_at = tool.created_at
    session.add(tool)
    session.commit()
    session.refresh(tool)
    return tool


def update_integration_tool(session: Session, tool_id: UUID, patch: dict) -> IntegrationTool:
    tool = get_integration_tool(session, tool_id)
    for k, v in patch.items():
        setattr(tool, k, v)
    tool.updated_at = dt.datetime.now(dt.timezone.utc)
    session.add(tool)
    session.commit()
    session.refresh(tool)
    return tool


def delete_integration_tool(session: Session, tool_id: UUID) -> None:
    tool = session.get(IntegrationTool, tool_id)
    if not tool:
        return
    session.delete(tool)
    session.commit()
