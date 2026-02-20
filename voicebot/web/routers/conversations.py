from __future__ import annotations
from uuid import UUID

from typing import Optional

from fastapi import APIRouter, Depends, Request
from sqlmodel import Session


def register(app, ctx) -> None:
    router = APIRouter()

    @router.get("/api/conversations")
    def api_list_conversations(
        request: Request,
        page: int = 1,
        page_size: int = 50,
        bot_id: Optional[UUID] = None,
        test_flag: Optional[bool] = None,
        include_groups: bool = False,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        page = max(1, int(page))
        page_size = min(200, max(10, int(page_size)))
        offset = (page - 1) * page_size
        total = ctx.count_conversations(session, bot_id=bot_id, test_flag=test_flag, include_groups=include_groups)
        convs = ctx.list_conversations(
            session,
            bot_id=bot_id,
            test_flag=test_flag,
            include_groups=include_groups,
            limit=page_size,
            offset=offset,
        )
        viewer_id = ctx._viewer_id_from_request(request)
        unread_map = ctx.count_unread_by_conversation(
            session,
            conversation_ids=[c.id for c in convs],
            viewer_id=viewer_id,
        )
        bots_by_id = {b.id: b for b in ctx.list_bots(session)}
        items = []
        for c in convs:
            b = bots_by_id.get(c.bot_id)
            items.append(
                {
                    "id": str(c.id),
                    "bot_id": str(c.bot_id),
                    "bot_name": b.name if b else None,
                    "test_flag": bool(c.test_flag),
                    "metadata_json": c.metadata_json or "{}",
                    "llm_input_tokens_est": int(c.llm_input_tokens_est or 0),
                    "llm_output_tokens_est": int(c.llm_output_tokens_est or 0),
                    "cost_usd_est": float(c.cost_usd_est or 0.0),
                    "last_asr_ms": c.last_asr_ms,
                    "last_llm_ttfb_ms": c.last_llm_ttfb_ms,
                    "last_llm_total_ms": c.last_llm_total_ms,
                    "last_tts_first_audio_ms": c.last_tts_first_audio_ms,
                    "last_total_ms": c.last_total_ms,
                    "last_message_at": c.last_message_at.isoformat() if c.last_message_at else None,
                    "last_message_preview": c.last_message_preview or "",
                    "unread_count": int(unread_map.get(c.id, 0)),
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                }
            )
        return {"items": items, "page": page, "page_size": page_size, "total": total}

    @router.get("/api/conversations/{conversation_id}")
    def api_conversation_detail(
        conversation_id: UUID,
        request: Request,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        conv = ctx.get_conversation(session, conversation_id)
        bot = ctx.get_bot(session, conv.bot_id)
        msgs_raw = ctx.list_messages(session, conversation_id=conversation_id)
        viewer_id = ctx._viewer_id_from_request(request)
        unread_map = ctx.count_unread_by_conversation(
            session,
            conversation_ids=[conversation_id],
            viewer_id=viewer_id,
        )

        def _safe_json_loads(s: str) -> dict | None:
            try:
                obj = ctx.json.loads(s)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None

        def _safe_json_list(s: str) -> list:
            try:
                obj = ctx.json.loads(s)
                return obj if isinstance(obj, list) else []
            except Exception:
                return []

        messages: list[dict] = []
        for m in msgs_raw:
            if m.role == "tool":
                continue
            tool_obj = _safe_json_loads(m.content) if m.role == "tool" else None
            tool_name = tool_obj.get("tool") if tool_obj and isinstance(tool_obj.get("tool"), str) else None
            tool_kind = None
            if tool_obj:
                if "arguments" in tool_obj:
                    tool_kind = "call"
                elif "result" in tool_obj:
                    tool_kind = "result"
            messages.append(
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                    "tool": tool_obj,
                    "tool_name": tool_name,
                    "tool_kind": tool_kind,
                    "citations": _safe_json_list(getattr(m, "citations_json", "") or "[]"),
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
            )

        return {
            "conversation": {
                "id": str(conv.id),
                "bot_id": str(conv.bot_id),
                "bot_name": bot.name,
                "test_flag": bool(conv.test_flag),
                "metadata_json": conv.metadata_json or "{}",
                "is_group": bool(conv.is_group),
                "group_title": conv.group_title or "",
                "group_bots_json": conv.group_bots_json or "[]",
                "llm_input_tokens_est": int(conv.llm_input_tokens_est or 0),
                "llm_output_tokens_est": int(conv.llm_output_tokens_est or 0),
                "cost_usd_est": float(conv.cost_usd_est or 0.0),
                "last_asr_ms": conv.last_asr_ms,
                "last_llm_ttfb_ms": conv.last_llm_ttfb_ms,
                "last_llm_total_ms": conv.last_llm_total_ms,
                "last_tts_first_audio_ms": conv.last_tts_first_audio_ms,
                "last_total_ms": conv.last_total_ms,
                "last_message_at": conv.last_message_at.isoformat() if conv.last_message_at else None,
                "last_message_preview": conv.last_message_preview or "",
                "unread_count": int(unread_map.get(conversation_id, 0)),
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            },
            "bot": ctx._bot_to_dict(bot),
            "messages": messages,
        }

    @router.get("/api/conversations/{conversation_id}/messages")
    def api_conversation_messages(
        conversation_id: UUID,
        since: Optional[str] = None,
        since_id: Optional[str] = None,
        before: Optional[str] = None,
        before_id: Optional[str] = None,
        limit: int = 200,
        order: str = "asc",
        include_tools: bool = False,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        _ = ctx.get_conversation(session, conversation_id)
        since_dt: ctx.dt.datetime | None = None
        if since:
            try:
                since_dt = ctx.dt.datetime.fromisoformat(str(since))
            except Exception:
                since_dt = None
        since_uuid: UUID | None = None
        if since_id:
            try:
                since_uuid = UUID(str(since_id))
            except Exception:
                since_uuid = None
        before_dt: ctx.dt.datetime | None = None
        if before:
            try:
                before_dt = ctx.dt.datetime.fromisoformat(str(before))
            except Exception:
                before_dt = None
        before_uuid: UUID | None = None
        if before_id:
            try:
                before_uuid = UUID(str(before_id))
            except Exception:
                before_uuid = None
        stmt = ctx.select(ctx.ConversationMessage).where(ctx.ConversationMessage.conversation_id == conversation_id)
        if not include_tools:
            stmt = stmt.where(ctx.ConversationMessage.role != "tool")
        if since_dt is not None:
            if since_uuid is not None:
                stmt = stmt.where(
                    ctx.or_(
                        ctx.ConversationMessage.created_at > since_dt,
                        ctx.and_(
                            ctx.ConversationMessage.created_at == since_dt,
                            ctx.ConversationMessage.id > since_uuid,
                        ),
                    )
                )
            else:
                stmt = stmt.where(ctx.ConversationMessage.created_at > since_dt)
        if before_dt is not None:
            if before_uuid is not None:
                stmt = stmt.where(
                    ctx.or_(
                        ctx.ConversationMessage.created_at < before_dt,
                        ctx.and_(
                            ctx.ConversationMessage.created_at == before_dt,
                            ctx.ConversationMessage.id < before_uuid,
                        ),
                    )
                )
            else:
                stmt = stmt.where(ctx.ConversationMessage.created_at < before_dt)
        if str(order).lower() == "desc":
            stmt = stmt.order_by(ctx.ConversationMessage.created_at.desc(), ctx.ConversationMessage.id.desc())
        else:
            stmt = stmt.order_by(ctx.ConversationMessage.created_at.asc(), ctx.ConversationMessage.id.asc())
        stmt = stmt.limit(min(500, max(1, int(limit))))
        msgs_raw = list(session.exec(stmt))
        messages = []
        for m in msgs_raw:
            if m.role == "tool" and not include_tools:
                continue
            tool_obj = None
            tool_name = None
            tool_kind = None
            if m.role == "tool":
                tool_obj = ctx.safe_json_loads(m.content or "{}")
                tool_name = tool_obj.get("tool") if tool_obj and isinstance(tool_obj.get("tool"), str) else None
                if tool_obj:
                    if "arguments" in tool_obj:
                        tool_kind = "call"
                    elif "result" in tool_obj:
                        tool_kind = "result"
            try:
                citations = ctx.json.loads(getattr(m, "citations_json", "") or "[]")
                if not isinstance(citations, list):
                    citations = []
            except Exception:
                citations = []
            messages.append(
                {
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
            )
        return {"conversation_id": str(conversation_id), "messages": messages}

    @router.post("/api/conversations/{conversation_id}/read")
    def api_conversation_mark_read(
        conversation_id: UUID,
        request: Request,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        _ = ctx.get_conversation(session, conversation_id)
        viewer_id = ctx._viewer_id_from_request(request)
        ctx.mark_conversation_read(session, conversation_id=conversation_id, viewer_id=viewer_id)
        return {"ok": True}

    app.include_router(router)
