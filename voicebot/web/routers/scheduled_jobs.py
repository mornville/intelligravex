from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Body, Depends, Query
from sqlmodel import Session

from ..schemas import ScheduledJobCreateRequest, ScheduledJobUpdateRequest


def register(app, ctx) -> None:
    router = APIRouter()

    def _default_conversation_for_bot(session: Session, bot_id: UUID) -> UUID | None:
        rows = ctx.list_conversations(session, bot_id=bot_id, limit=1, offset=0)
        if not rows:
            return None
        return rows[0].id

    @router.get("/api/bots/{bot_id}/scheduled-jobs")
    def list_scheduled_jobs_api(
        bot_id: UUID,
        conversation_uuid: str = Query(default=""),
        enabled: str = Query(default=""),
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        _ = ctx.get_bot(session, bot_id)
        conv_id = None
        if str(conversation_uuid or "").strip():
            try:
                conv_id = UUID(str(conversation_uuid).strip())
            except Exception as exc:
                raise ctx.HTTPException(status_code=400, detail="Invalid conversation_uuid.") from exc
            conv = ctx.get_conversation(session, conv_id)
            if conv.bot_id != bot_id:
                raise ctx.HTTPException(status_code=400, detail="conversation_uuid does not belong to this bot.")
        enabled_filter = None
        text = str(enabled or "").strip().lower()
        if text in {"1", "true", "yes"}:
            enabled_filter = True
        elif text in {"0", "false", "no"}:
            enabled_filter = False
        rows = ctx.list_scheduled_jobs(
            session,
            bot_id=bot_id,
            conversation_id=conv_id,
            enabled=enabled_filter,
            limit=1000,
        )
        return {"items": [ctx._serialize_scheduled_job(r) for r in rows]}

    @router.post("/api/bots/{bot_id}/scheduled-jobs")
    def create_scheduled_job_api(
        bot_id: UUID,
        payload: ScheduledJobCreateRequest = Body(...),
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        _ = ctx.get_bot(session, bot_id)
        payload_dict = payload.model_dump(exclude_none=True)
        default_conversation_id = _default_conversation_for_bot(session, bot_id)
        if not (payload_dict.get("conversation_uuid") or default_conversation_id):
            raise ctx.HTTPException(
                status_code=400,
                detail="conversation_uuid is required when this bot has no conversations yet.",
            )
        try:
            job = ctx._create_job_from_payload(
                session,
                bot_id=bot_id,
                payload=payload_dict,
                default_conversation_id=default_conversation_id,
            )
            meta = ctx._append_job_metadata_event(
                session,
                conversation_id=job.conversation_id,
                job=job,
                status="scheduled",
                event_type="job_created",
                message="Scheduled job created from UI/API.",
            )
            return {"item": ctx._serialize_scheduled_job(job), "metadata": meta}
        except Exception as exc:
            raise ctx.HTTPException(status_code=400, detail=str(exc) or "Failed to create scheduled job.") from exc

    @router.put("/api/bots/{bot_id}/scheduled-jobs/{job_id}")
    def update_scheduled_job_api(
        bot_id: UUID,
        job_id: UUID,
        payload: ScheduledJobUpdateRequest = Body(...),
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        _ = ctx.get_bot(session, bot_id)
        payload_dict = payload.model_dump(exclude_none=True)
        try:
            job = ctx._update_job_from_payload(
                session,
                bot_id=bot_id,
                job_id=job_id,
                payload=payload_dict,
            )
            meta = ctx._append_job_metadata_event(
                session,
                conversation_id=job.conversation_id,
                job=job,
                status=job.last_status or "scheduled",
                event_type="job_updated",
                message="Scheduled job updated from UI/API.",
            )
            return {"item": ctx._serialize_scheduled_job(job), "metadata": meta}
        except Exception as exc:
            raise ctx.HTTPException(status_code=400, detail=str(exc) or "Failed to update scheduled job.") from exc

    @router.delete("/api/bots/{bot_id}/scheduled-jobs/{job_id}")
    def delete_scheduled_job_api(
        bot_id: UUID,
        job_id: UUID,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        _ = ctx.get_bot(session, bot_id)
        try:
            job = ctx.get_scheduled_job(session, job_id)
        except Exception as exc:
            raise ctx.HTTPException(status_code=404, detail="Scheduled job not found.") from exc
        if job.bot_id != bot_id:
            raise ctx.HTTPException(status_code=400, detail="Scheduled job does not belong to this bot.")
        meta = ctx._append_job_metadata_event(
            session,
            conversation_id=job.conversation_id,
            job=job,
            status="deleted",
            event_type="job_deleted",
            message="Scheduled job deleted from UI/API.",
        )
        ctx.delete_scheduled_job(session, job_id)
        return {"ok": True, "deleted": str(job_id), "metadata": meta}

    @router.post("/api/bots/{bot_id}/scheduled-jobs/{job_id}/run-now")
    def run_scheduled_job_now_api(
        bot_id: UUID,
        job_id: UUID,
        session: Session = Depends(ctx.get_session),
    ) -> dict:
        _ = ctx.get_bot(session, bot_id)
        job = ctx.get_scheduled_job(session, job_id)
        if job.bot_id != bot_id:
            raise ctx.HTTPException(status_code=400, detail="Scheduled job does not belong to this bot.")
        if bool(job.is_running):
            return {"ok": True, "item": ctx._serialize_scheduled_job(job), "message": "Job is already running."}
        updated = ctx.update_scheduled_job(
            session,
            job.id,
            {
                "enabled": True,
                "next_run_at": ctx._scheduled_jobs_utc_now(),
                "last_status": "queued",
                "last_error": "",
            },
        )
        meta = ctx._append_job_metadata_event(
            session,
            conversation_id=updated.conversation_id,
            job=updated,
            status="queued",
            event_type="job_queued",
            message="Job queued for immediate execution.",
        )
        return {"ok": True, "item": ctx._serialize_scheduled_job(updated), "metadata": meta}

    app.include_router(router)
