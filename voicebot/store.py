from __future__ import annotations

import datetime as dt
import json
from typing import List, Optional
from uuid import UUID

from sqlmodel import Session, select

from voicebot.crypto import CryptoBox, build_hint
from voicebot.models import ApiKey, Bot, Conversation, ConversationMessage


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
    session.delete(bot)
    session.commit()


def decrypt_openai_key(session: Session, *, crypto: CryptoBox, bot: Bot) -> Optional[str]:
    if not bot.openai_key_id:
        return None
    key = get_key(session, bot.openai_key_id)
    if key.provider != "openai":
        return None
    return crypto.decrypt_str(key.secret_ciphertext)


def create_conversation(session: Session, *, bot_id: UUID, test_flag: bool) -> Conversation:
    now = dt.datetime.now(dt.timezone.utc)
    conv = Conversation(bot_id=bot_id, test_flag=test_flag, created_at=now, updated_at=now)
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
        current[str(k)] = v
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
