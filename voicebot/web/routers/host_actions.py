from __future__ import annotations
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlmodel import Session


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

        return ctx._host_action_payload(action)

    app.include_router(router)
