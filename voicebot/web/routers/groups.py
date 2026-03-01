from __future__ import annotations
from uuid import UUID

from typing import Optional

from fastapi import APIRouter, Body, Depends, Request, WebSocket, WebSocketDisconnect
from sqlmodel import Session

from ..schemas import GroupConversationCreateRequest, GroupMessageRequest, GroupSwarmConfigRequest


def register(app, ctx) -> None:
    router = APIRouter()

    def _clamp_int(value, *, default: int, lo: int, hi: int) -> int:
        try:
            n = int(value)
        except Exception:
            n = default
        return min(hi, max(lo, n))

    def _normalize_swarm_config(raw: dict | None) -> dict:
        cfg = raw if isinstance(raw, dict) else {}
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "coordinator_mode": "coordinator_first"
            if str(cfg.get("coordinator_mode") or "").strip().lower() != "mentions_only"
            else "mentions_only",
            "max_turns_per_run": _clamp_int(cfg.get("max_turns_per_run"), default=6, lo=1, hi=40),
            "max_parallel_bots": _clamp_int(cfg.get("max_parallel_bots"), default=2, lo=1, hi=8),
            "max_hops": _clamp_int(cfg.get("max_hops"), default=3, lo=0, hi=12),
            "allow_revisit": bool(cfg.get("allow_revisit", False)),
        }

    @router.get("/api/group-conversations")
    def api_list_group_conversations(request: Request, session: Session = Depends(ctx.get_session)) -> dict:
        stmt = ctx.select(ctx.Conversation).where(ctx.Conversation.is_group == True).order_by(ctx.Conversation.updated_at.desc())  # noqa: E712
        convs = list(session.exec(stmt))
        viewer_id = ctx._viewer_id_from_request(request)
        unread_map = ctx.count_unread_by_conversation(
            session,
            conversation_ids=[c.id for c in convs],
            viewer_id=viewer_id,
        )
        items = []
        for c in convs:
            bots = ctx._group_bots_from_conv(c)
            items.append(
                {
                    "id": str(c.id),
                    "title": c.group_title or "",
                    "default_bot_id": str(c.bot_id),
                    "group_bots": bots,
                    "last_message_at": c.last_message_at.isoformat() if c.last_message_at else None,
                    "last_message_preview": c.last_message_preview or "",
                    "unread_count": int(unread_map.get(c.id, 0)),
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                }
            )
        return {"items": items}

    @router.post("/api/group-conversations")
    def api_create_group_conversation(
        payload: GroupConversationCreateRequest = Body(...),
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        title = (payload.title or "").strip()
        bot_ids = [str(b).strip() for b in (payload.bot_ids or []) if str(b).strip()]
        default_bot_id = str(payload.default_bot_id or "").strip()
        if not title:
            raise ctx.HTTPException(status_code=400, detail="Title is required")
        if not bot_ids:
            raise ctx.HTTPException(status_code=400, detail="At least one assistant is required")
        if not default_bot_id:
            raise ctx.HTTPException(status_code=400, detail="Default assistant is required")
        if default_bot_id not in bot_ids:
            raise ctx.HTTPException(status_code=400, detail="Default assistant must be in the group")

        bots: list[ctx.Bot] = []
        for bid in bot_ids:
            try:
                bots.append(ctx.get_bot(session, UUID(bid)))
            except Exception:
                raise ctx.HTTPException(status_code=404, detail=f"Assistant not found: {bid}")

        used_slugs: set[str] = set()
        group_bots: list[dict[str, str]] = []
        for b in bots:
            base = ctx._slugify(b.name)
            slug = base
            i = 2
            while slug in used_slugs:
                slug = f"{base}-{i}"
                i += 1
            used_slugs.add(slug)
            group_bots.append({"id": str(b.id), "name": b.name, "slug": slug})

        now = ctx.dt.datetime.now(ctx.dt.timezone.utc)
        metadata: dict = {}
        if isinstance(payload.swarm_config, dict):
            metadata["group_swarm"] = {"config": _normalize_swarm_config(payload.swarm_config)}
        conv = ctx.Conversation(
            bot_id=UUID(default_bot_id),
            test_flag=bool(payload.test_flag),
            is_group=True,
            group_title=title,
            group_bots_json=ctx.json.dumps(group_bots, ensure_ascii=False),
            metadata_json=ctx.json.dumps(metadata, ensure_ascii=False) if metadata else "{}",
            created_at=now,
            updated_at=now,
        )
        session.add(conv)
        session.commit()
        session.refresh(conv)
        return ctx._group_conversation_payload(session, conv)

    @router.get("/api/group-conversations/{conversation_id}")
    def api_group_conversation_detail(
        conversation_id: UUID,
        include_messages: bool = True,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        conv = ctx.get_conversation(session, conversation_id)
        if not bool(conv.is_group):
            raise ctx.HTTPException(status_code=404, detail="Group conversation not found")
        return ctx._group_conversation_payload(session, conv, include_messages=include_messages)

    @router.patch("/api/group-conversations/{conversation_id}/swarm-config")
    def api_update_group_swarm_config(
        conversation_id: UUID,
        payload: GroupSwarmConfigRequest = Body(...),
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        conv = ctx.get_conversation(session, conversation_id)
        if not bool(conv.is_group):
            raise ctx.HTTPException(status_code=404, detail="Group conversation not found")

        meta = ctx.safe_json_loads(conv.metadata_json or "{}") or {}
        if not isinstance(meta, dict):
            meta = {}
        swarm = meta.get("group_swarm")
        if not isinstance(swarm, dict):
            swarm = {}

        prev = swarm.get("config")
        if not isinstance(prev, dict):
            prev = {}
        incoming = payload.model_dump(exclude_none=True)
        merged = {**prev, **incoming}
        normalized = _normalize_swarm_config(merged)
        swarm["config"] = normalized

        active = swarm.get("active_run")
        if isinstance(active, dict) and str(active.get("status") or "") == "running":
            old_max_turns = _clamp_int(
                active.get("max_turns"),
                default=_clamp_int(prev.get("max_turns_per_run"), default=6, lo=1, hi=40),
                lo=1,
                hi=9999,
            )
            old_remaining = _clamp_int(active.get("remaining_turns"), default=0, lo=0, hi=9999)
            consumed = max(0, old_max_turns - old_remaining)
            new_max_turns = int(normalized["max_turns_per_run"])
            active["max_turns"] = new_max_turns
            active["remaining_turns"] = max(0, new_max_turns - consumed)
            active["max_hops"] = int(normalized["max_hops"])
            inflight = active.get("inflight_bot_ids")
            scheduled = active.get("scheduled_bot_ids")
            if not isinstance(inflight, list):
                inflight = []
            if not isinstance(scheduled, list):
                scheduled = []
            if int(active.get("remaining_turns") or 0) <= 0 and not inflight and not scheduled:
                active["status"] = "done"
            active["updated_at"] = ctx.dt.datetime.now(ctx.dt.timezone.utc).isoformat()
            swarm["active_run"] = active

        meta["group_swarm"] = swarm
        conv.metadata_json = ctx.json.dumps(meta, ensure_ascii=False)
        conv.updated_at = ctx.dt.datetime.now(ctx.dt.timezone.utc)
        session.add(conv)
        session.commit()
        session.refresh(conv)
        return ctx._group_conversation_payload(session, conv, include_messages=False)

    @router.get("/api/group-conversations/{conversation_id}/messages")
    def api_group_conversation_messages(
        conversation_id: UUID,
        since: Optional[str] = None,
        before: Optional[str] = None,
        before_id: Optional[str] = None,
        limit: int = 200,
        order: str = "asc",
        include_tools: bool = False,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        conv = ctx.get_conversation(session, conversation_id)
        if not bool(conv.is_group):
            raise ctx.HTTPException(status_code=404, detail="Group conversation not found")
        since_dt: ctx.dt.datetime | None = None
        if since:
            try:
                since_dt = ctx.dt.datetime.fromisoformat(str(since))
            except Exception:
                since_dt = None
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
        messages: list[dict] = []
        for m in msgs_raw:
            payload = ctx._group_message_payload(m)
            if payload is not None:
                messages.append(payload)
        return {"conversation_id": str(conversation_id), "messages": messages}

    @router.post("/api/group-conversations/{conversation_id}/messages")
    async def api_group_conversation_message(
        conversation_id: UUID,
        payload: GroupMessageRequest = Body(...),
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        conv = ctx.get_conversation(session, conversation_id)
        if not bool(conv.is_group):
            raise ctx.HTTPException(status_code=404, detail="Group conversation not found")

        text = str(payload.text or "").strip()
        if not text:
            raise ctx.HTTPException(status_code=400, detail="Empty text")

        sender_role = str(payload.sender_role or "user").strip().lower()
        if sender_role not in ("user", "assistant"):
            raise ctx.HTTPException(status_code=400, detail="Invalid sender_role")

        sender_bot_id: Optional[UUID] = None
        sender_name = (payload.sender_name or "").strip()
        if sender_role == "assistant":
            if not payload.sender_bot_id:
                raise ctx.HTTPException(status_code=400, detail="sender_bot_id is required for assistant messages")
            try:
                sender_bot_id = UUID(str(payload.sender_bot_id))
            except Exception:
                raise ctx.HTTPException(status_code=400, detail="Invalid sender_bot_id")
            if str(sender_bot_id) not in {b["id"] for b in ctx._group_bots_from_conv(conv)}:
                raise ctx.HTTPException(status_code=400, detail="Assistant is not a member of this group")
            if not sender_name:
                try:
                    sender_name = ctx.get_bot(session, sender_bot_id).name
                except Exception:
                    sender_name = "Assistant"
        else:
            if not sender_name:
                sender_name = "User"

        msg = ctx.add_message_with_metrics(
            session,
            conversation_id=conversation_id,
            role=sender_role,
            content=text,
            sender_bot_id=sender_bot_id,
            sender_name=sender_name,
        )
        if msg.role == "assistant":
            ctx._mirror_group_message(session, conv=conv, msg=msg)

        payload = ctx._group_message_payload(msg)
        if payload:
            await ctx._group_ws_broadcast(conversation_id, {"type": "message", "message": payload})

        targets = ctx._extract_group_mentions(text, conv)
        if sender_bot_id:
            targets = [bid for bid in targets if str(bid) != str(sender_bot_id)]
        if sender_role == "user" and not targets:
            if not conv.bot_id:
                raise ctx.HTTPException(status_code=400, detail="Default assistant is not configured")
            targets = [conv.bot_id]

        run_id = None
        if sender_role == "user":
            run_id, targets = ctx._start_group_swarm_run(
                conversation_id=conversation_id,
                trigger_message_id=msg.id,
                objective=text,
                requested_targets=targets,
                sender_role=sender_role,
            )

        if targets:
            ctx._schedule_group_bots(
                conversation_id,
                targets,
                run_id=run_id,
                reason="user_input",
                source_bot_id=sender_bot_id,
            )

        return ctx._group_conversation_payload(session, conv)

    @router.post("/api/group-conversations/{conversation_id}/reset")
    def api_group_conversation_reset(
        conversation_id: UUID,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        conv = ctx.get_conversation(session, conversation_id)
        if not bool(conv.is_group):
            raise ctx.HTTPException(status_code=404, detail="Group conversation not found")

        mapping = ctx._ensure_group_individual_conversations(session, conv)
        meta = ctx.safe_json_loads(conv.metadata_json or "{}") or {}
        keep = {}
        if isinstance(meta, dict):
            if "demo_seed" in meta:
                keep["demo_seed"] = meta["demo_seed"]
        keep["group_individual_conversations"] = mapping
        ctx._reset_conversation_state(session, conv, keep)

        for bid, cid in mapping.items():
            try:
                child = ctx.get_conversation(session, UUID(cid))
            except Exception:
                continue
            child_meta = ctx.safe_json_loads(child.metadata_json or "{}") or {}
            keep_child = {}
            if isinstance(child_meta, dict):
                for key in ("group_parent_id", "group_bot_id", "group_bot_name"):
                    if key in child_meta:
                        keep_child[key] = child_meta[key]
            ctx._reset_conversation_state(session, child, keep_child)

        try:
            ctx.asyncio.get_running_loop()
            ctx.asyncio.create_task(ctx._group_ws_broadcast(conversation_id, {"type": "reset"}))
        except Exception:
            pass
        return {"ok": True}

    @router.delete("/api/group-conversations/{conversation_id}")
    def api_delete_group_conversation(
        conversation_id: UUID,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        conv = ctx.get_conversation(session, conversation_id)
        if not bool(conv.is_group):
            raise ctx.HTTPException(status_code=404, detail="Group conversation not found")

        mapping = ctx._ensure_group_individual_conversations(session, conv)

        session.exec(ctx.delete(ctx.ConversationMessage).where(ctx.ConversationMessage.conversation_id == conv.id))
        session.delete(conv)

        for cid in mapping.values():
            try:
                child = ctx.get_conversation(session, UUID(cid))
            except Exception:
                continue
            session.exec(ctx.delete(ctx.ConversationMessage).where(ctx.ConversationMessage.conversation_id == child.id))
            session.delete(child)

        session.commit()
        return {"ok": True}

    @router.websocket("/ws/groups/{conversation_id}")
    async def ws_group(conversation_id: UUID, ws: WebSocket) -> None:
        await ws.accept()
        key = str(conversation_id)
        async with ctx.group_ws_lock:
            ctx.group_ws_clients.setdefault(key, set()).add(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            async with ctx.group_ws_lock:
                clients = ctx.group_ws_clients.get(key, set())
                clients.discard(ws)
                if clients:
                    ctx.group_ws_clients[key] = clients
                else:
                    ctx.group_ws_clients.pop(key, None)

    app.include_router(router)
