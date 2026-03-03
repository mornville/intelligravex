from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional
from uuid import UUID

from sqlmodel import Session

from voicebot.llm.openai_llm import Message
from voicebot.models import Bot, ScheduledJob
from voicebot.utils.template import safe_json_loads

_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")
_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_WEEKDAY_TO_INT = {name: idx for idx, name in enumerate(_WEEKDAYS)}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _as_utc(value: Optional[dt.datetime]) -> Optional[dt.datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _to_iso_z(value: Optional[dt.datetime]) -> str:
    if value is None:
        return ""
    d = _as_utc(value)
    if d is None:
        return ""
    return d.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_hhmm_utc(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    m = _HHMM_RE.fullmatch(text)
    if not m:
        raise ValueError("time_utc must be in HH:MM (24h) UTC format.")
    return text


def _parse_iso_utc(raw: Any) -> Optional[dt.datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except Exception as exc:
        raise ValueError("run_at_utc must be an ISO UTC timestamp (e.g. 2026-03-02T14:30:00Z).") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _parse_uuid(raw: Any, field_name: str) -> UUID:
    try:
        return UUID(str(raw or "").strip())
    except Exception as exc:
        raise ValueError(f"Invalid {field_name}.") from exc


def _normalize_cadence(raw: Any) -> str:
    cadence = str(raw or "").strip().lower()
    if cadence not in {"once", "daily", "weekly"}:
        raise ValueError("cadence must be one of: once, daily, weekly.")
    return cadence


def serialize_scheduled_job(job: ScheduledJob) -> dict[str, Any]:
    return {
        "id": str(job.id),
        "bot_id": str(job.bot_id),
        "assistant_id": str(job.assistant_id),
        "conversation_uuid": str(job.conversation_id),
        "what_to_do": job.what_to_do or "",
        "input_message": job.input_message or "",
        "cadence": job.cadence or "",
        "time_utc": job.time_utc or "",
        "weekday_utc": job.weekday_utc or "",
        "run_at_utc": _to_iso_z(job.run_at_utc),
        "next_run_at": _to_iso_z(job.next_run_at),
        "enabled": bool(job.enabled),
        "is_running": bool(job.is_running),
        "running_started_at": _to_iso_z(job.running_started_at),
        "last_run_at": _to_iso_z(job.last_run_at),
        "last_status": job.last_status or "",
        "last_error": job.last_error or "",
        "created_at": _to_iso_z(job.created_at),
        "updated_at": _to_iso_z(job.updated_at),
    }


def compute_next_run_at(
    *,
    cadence: str,
    time_utc: str,
    weekday_utc: str,
    run_at_utc: Optional[dt.datetime],
    from_dt: Optional[dt.datetime] = None,
) -> Optional[dt.datetime]:
    now = (from_dt or utc_now()).astimezone(dt.timezone.utc)
    cadence_norm = _normalize_cadence(cadence)
    if cadence_norm == "once":
        if run_at_utc is None:
            return None
        run_at = _as_utc(run_at_utc)
        if run_at is None:
            return None
        return run_at if run_at > now else None

    hhmm = _parse_hhmm_utc(time_utc)
    hour = int(hhmm[0:2])
    minute = int(hhmm[3:5])

    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if cadence_norm == "daily":
        if candidate <= now:
            candidate += dt.timedelta(days=1)
        return candidate

    weekday = str(weekday_utc or "").strip().lower()
    if weekday not in _WEEKDAY_TO_INT:
        raise ValueError("weekday_utc is required for weekly cadence (mon..sun).")
    target = _WEEKDAY_TO_INT[weekday]
    delta_days = (target - candidate.weekday()) % 7
    if delta_days == 0 and candidate <= now:
        delta_days = 7
    return candidate + dt.timedelta(days=delta_days)


def _event_patch_for_job(
    *,
    existing_meta: dict[str, Any],
    job: ScheduledJob,
    status: str,
    event_type: str,
    message: str = "",
    error: str = "",
) -> dict[str, Any]:
    result_event_types = {"job_executed", "job_skipped_missed"}
    meta_obj = dict(existing_meta or {})
    sched = meta_obj.get("scheduled_jobs")
    if not isinstance(sched, dict):
        sched = {}
    jobs_obj = sched.get("jobs")
    if not isinstance(jobs_obj, dict):
        jobs_obj = {}
    trail = sched.get("trail")
    if not isinstance(trail, list):
        trail = []
    trail = [t for t in trail if isinstance(t, dict) and str(t.get("type") or "") in result_event_types]

    existing_job = jobs_obj.get(str(job.id))
    if not isinstance(existing_job, dict):
        existing_job = {}
    last_result_preview = str(existing_job.get("last_result_preview") or "")
    last_result_at_utc = str(existing_job.get("last_result_at_utc") or "")
    if event_type == "job_executed":
        last_result_preview = str(message or "").strip()
        last_result_at_utc = _to_iso_z(utc_now())

    jobs_obj[str(job.id)] = {
        "job_id": str(job.id),
        "bot_id": str(job.bot_id),
        "assistant_id": str(job.assistant_id),
        "conversation_uuid": str(job.conversation_id),
        "cadence": job.cadence,
        "time_utc": job.time_utc,
        "weekday_utc": job.weekday_utc,
        "run_at_utc": _to_iso_z(job.run_at_utc),
        "next_run_at": _to_iso_z(job.next_run_at),
        "enabled": bool(job.enabled),
        "is_running": bool(job.is_running),
        "last_run_at": _to_iso_z(job.last_run_at),
        "last_status": status or (job.last_status or ""),
        "last_error": error or (job.last_error or ""),
        "last_result_preview": last_result_preview,
        "last_result_at_utc": last_result_at_utc,
        "what_to_do": job.what_to_do or "",
        "input_message": job.input_message or "",
        "updated_at": _to_iso_z(utc_now()),
    }

    if event_type in result_event_types:
        trail.append(
            {
                "at_utc": _to_iso_z(utc_now()),
                "type": event_type,
                "status": status,
                "job_id": str(job.id),
                "bot_id": str(job.bot_id),
                "assistant_id": str(job.assistant_id),
                "conversation_uuid": str(job.conversation_id),
                "cadence": job.cadence,
                "next_run_at": _to_iso_z(job.next_run_at),
                "last_run_at": _to_iso_z(job.last_run_at),
                "message": message,
                "error": error,
            }
        )
    if len(trail) > 200:
        trail = trail[-200:]
    sched["jobs"] = jobs_obj
    sched["trail"] = trail
    return {"scheduled_jobs": sched}


def append_job_metadata_event(
    ctx,
    session: Session,
    *,
    conversation_id: UUID,
    job: ScheduledJob,
    status: str,
    event_type: str,
    message: str = "",
    error: str = "",
) -> dict[str, Any]:
    conv = ctx.get_conversation(session, conversation_id)
    current = safe_json_loads(getattr(conv, "metadata_json", "") or "{}")
    if not isinstance(current, dict):
        current = {}
    patch = _event_patch_for_job(
        existing_meta=current,
        job=job,
        status=status,
        event_type=event_type,
        message=message,
        error=error,
    )
    return ctx.merge_conversation_metadata(session, conversation_id=conversation_id, patch=patch)


def _resolve_job_fields(
    *,
    payload: dict[str, Any],
    existing: Optional[ScheduledJob],
) -> tuple[str, str, str, Optional[dt.datetime], bool]:
    cadence_raw = payload.get("cadence") if "cadence" in payload else (existing.cadence if existing else "daily")
    cadence = _normalize_cadence(cadence_raw)

    time_utc = (
        _parse_hhmm_utc(payload.get("time_utc"))
        if "time_utc" in payload
        else _parse_hhmm_utc(existing.time_utc if existing else "")
    )
    weekday_utc = (
        str(payload.get("weekday_utc") or "").strip().lower()
        if "weekday_utc" in payload
        else str(existing.weekday_utc or "").strip().lower() if existing else ""
    )
    run_at_utc = _parse_iso_utc(payload.get("run_at_utc")) if "run_at_utc" in payload else (existing.run_at_utc if existing else None)
    enabled = bool(payload.get("enabled")) if "enabled" in payload else (bool(existing.enabled) if existing else True)

    if cadence == "once":
        if run_at_utc is None:
            raise ValueError("run_at_utc is required when cadence is once.")
        time_utc = ""
        weekday_utc = ""
    elif cadence == "daily":
        if not time_utc:
            raise ValueError("time_utc is required when cadence is daily.")
        weekday_utc = ""
        run_at_utc = None
    elif cadence == "weekly":
        if not time_utc:
            raise ValueError("time_utc is required when cadence is weekly.")
        if weekday_utc not in _WEEKDAY_TO_INT:
            raise ValueError("weekday_utc is required for weekly cadence (mon..sun).")
        run_at_utc = None
    return cadence, time_utc, weekday_utc, run_at_utc, enabled


def create_job_from_payload(
    ctx,
    session: Session,
    *,
    bot_id: UUID,
    payload: dict[str, Any],
    default_conversation_id: Optional[UUID],
) -> ScheduledJob:
    conversation_raw = payload.get("conversation_uuid")
    conversation_id = _parse_uuid(conversation_raw, "conversation_uuid") if conversation_raw else default_conversation_id
    if conversation_id is None:
        raise ValueError("conversation_uuid is required.")
    conv = ctx.get_conversation(session, conversation_id)
    if conv.bot_id != bot_id:
        raise ValueError("conversation_uuid does not belong to this bot.")

    assistant_raw = payload.get("assistant_id")
    assistant_id = _parse_uuid(assistant_raw, "assistant_id") if assistant_raw else bot_id

    what_to_do = str(payload.get("what_to_do") or "").strip()
    if not what_to_do:
        raise ValueError("what_to_do is required.")
    input_message = str(payload.get("input_message") or "").strip() or what_to_do

    cadence, time_utc, weekday_utc, run_at_utc, enabled = _resolve_job_fields(payload=payload, existing=None)
    next_run_at = compute_next_run_at(
        cadence=cadence,
        time_utc=time_utc,
        weekday_utc=weekday_utc,
        run_at_utc=run_at_utc,
    )
    if enabled and next_run_at is None:
        raise ValueError("The requested schedule is in the past. Please provide a future UTC time.")

    job = ScheduledJob(
        bot_id=bot_id,
        assistant_id=assistant_id,
        conversation_id=conversation_id,
        what_to_do=what_to_do,
        input_message=input_message,
        cadence=cadence,
        time_utc=time_utc,
        weekday_utc=weekday_utc,
        run_at_utc=run_at_utc,
        enabled=enabled,
        next_run_at=next_run_at if enabled else None,
        is_running=False,
        last_status="scheduled",
        last_error="",
    )
    return ctx.create_scheduled_job(session, job=job)


def update_job_from_payload(
    ctx,
    session: Session,
    *,
    bot_id: UUID,
    job_id: UUID,
    payload: dict[str, Any],
) -> ScheduledJob:
    job = ctx.get_scheduled_job(session, job_id)
    if job.bot_id != bot_id:
        raise ValueError("Scheduled job does not belong to this bot.")

    patch: dict[str, Any] = {}
    if "conversation_uuid" in payload:
        conv_id = _parse_uuid(payload.get("conversation_uuid"), "conversation_uuid")
        conv = ctx.get_conversation(session, conv_id)
        if conv.bot_id != bot_id:
            raise ValueError("conversation_uuid does not belong to this bot.")
        patch["conversation_id"] = conv_id
    if "assistant_id" in payload:
        patch["assistant_id"] = _parse_uuid(payload.get("assistant_id"), "assistant_id")
    if "what_to_do" in payload:
        what_to_do = str(payload.get("what_to_do") or "").strip()
        if not what_to_do:
            raise ValueError("what_to_do cannot be empty.")
        patch["what_to_do"] = what_to_do
    if "input_message" in payload:
        patch["input_message"] = str(payload.get("input_message") or "").strip()

    schedule_keys = {"cadence", "time_utc", "weekday_utc", "run_at_utc", "enabled"}
    if schedule_keys.intersection(payload.keys()):
        cadence, time_utc, weekday_utc, run_at_utc, enabled = _resolve_job_fields(payload=payload, existing=job)
        patch["cadence"] = cadence
        patch["time_utc"] = time_utc
        patch["weekday_utc"] = weekday_utc
        patch["run_at_utc"] = run_at_utc
        patch["enabled"] = enabled
        if enabled:
            next_run_at = compute_next_run_at(
                cadence=cadence,
                time_utc=time_utc,
                weekday_utc=weekday_utc,
                run_at_utc=run_at_utc,
            )
            if next_run_at is None:
                raise ValueError("The updated schedule is in the past. Please provide a future UTC time.")
            patch["next_run_at"] = next_run_at
        else:
            patch["next_run_at"] = None
    elif "enabled" in payload:
        enabled = bool(payload.get("enabled"))
        patch["enabled"] = enabled
        if enabled:
            next_run_at = compute_next_run_at(
                cadence=job.cadence,
                time_utc=job.time_utc,
                weekday_utc=job.weekday_utc,
                run_at_utc=job.run_at_utc,
            )
            patch["next_run_at"] = next_run_at
        else:
            patch["next_run_at"] = None

    if not patch:
        return job
    return ctx.update_scheduled_job(session, job_id, patch)


def _mark_job_terminal_state(
    ctx,
    session: Session,
    *,
    job: ScheduledJob,
    success: bool,
    error: str = "",
) -> ScheduledJob:
    now = utc_now()
    status = "success" if success else "error"
    next_run_at = None
    enabled = bool(job.enabled)
    if enabled:
        if job.cadence == "once":
            enabled = False
            next_run_at = None
        else:
            next_run_at = compute_next_run_at(
                cadence=job.cadence,
                time_utc=job.time_utc,
                weekday_utc=job.weekday_utc,
                run_at_utc=job.run_at_utc,
                from_dt=now,
            )
    patch = {
        "is_running": False,
        "running_started_at": None,
        "last_run_at": now,
        "last_status": status,
        "last_error": error.strip(),
        "enabled": enabled,
        "next_run_at": next_run_at if enabled else None,
    }
    return ctx.update_scheduled_job(session, job.id, patch)


def mark_job_missed(ctx, session: Session, *, job: ScheduledJob) -> ScheduledJob:
    now = utc_now()
    enabled = bool(job.enabled)
    next_run_at = None
    if enabled:
        if job.cadence == "once":
            enabled = False
        else:
            next_run_at = compute_next_run_at(
                cadence=job.cadence,
                time_utc=job.time_utc,
                weekday_utc=job.weekday_utc,
                run_at_utc=job.run_at_utc,
                from_dt=now,
            )
    patch = {
        "is_running": False,
        "running_started_at": None,
        "last_run_at": now,
        "last_status": "skipped_missed",
        "last_error": "",
        "enabled": enabled,
        "next_run_at": next_run_at if enabled else None,
    }
    updated = ctx.update_scheduled_job(session, job.id, patch)
    append_job_metadata_event(
        ctx,
        session,
        conversation_id=updated.conversation_id,
        job=updated,
        status="skipped_missed",
        event_type="job_skipped_missed",
        message="Skipped missed run and advanced to next slot.",
    )
    return updated


def should_skip_due_job(job: ScheduledJob, *, now: Optional[dt.datetime] = None, grace_seconds: int = 120) -> bool:
    now_utc = (now or utc_now()).astimezone(dt.timezone.utc)
    due = job.next_run_at
    if due is None:
        return False
    due_utc = _as_utc(due)
    if due_utc is None:
        return False
    return (now_utc - due_utc).total_seconds() > max(0, int(grace_seconds))


def handle_schedule_job_tool(
    ctx,
    session: Session,
    *,
    bot: Bot,
    conversation_id: UUID,
    tool_args: dict[str, Any],
) -> dict[str, Any]:
    action_raw = str(tool_args.get("action") or "create").strip().lower()
    action_alias = {
        "disable": "pause",
        "enable": "resume",
        "disable_all": "pause",
        "enable_all": "resume",
        "delete_all": "delete",
    }
    action = action_alias.get(action_raw, action_raw)
    if action not in {"create", "update", "delete", "pause", "resume", "list"}:
        return {"ok": False, "error": {"message": "Unsupported schedule_job action."}}

    try:
        conv_scope_id: Optional[UUID] = conversation_id
        if bool(tool_args.get("all_conversations")):
            conv_scope_id = None
        elif str(tool_args.get("conversation_uuid") or "").strip():
            conv_scope_id = _parse_uuid(tool_args.get("conversation_uuid"), "conversation_uuid")
            conv_scope = ctx.get_conversation(session, conv_scope_id)
            if conv_scope.bot_id != bot.id:
                raise ValueError("conversation_uuid does not belong to this bot.")

        bulk_op = bool(tool_args.get("all")) or action_raw.endswith("_all")

        if action == "list":
            rows = ctx.list_scheduled_jobs(
                session,
                bot_id=bot.id,
                conversation_id=conv_scope_id,
                limit=500,
            )
            return {
                "ok": True,
                "jobs": [serialize_scheduled_job(j) for j in rows],
                "scope": "all_conversations" if conv_scope_id is None else "current_or_selected_conversation",
                "count": len(rows),
            }

        if action == "create":
            job = create_job_from_payload(
                ctx,
                session,
                bot_id=bot.id,
                payload=tool_args,
                default_conversation_id=conv_scope_id or conversation_id,
            )
            meta = append_job_metadata_event(
                ctx,
                session,
                conversation_id=job.conversation_id,
                job=job,
                status="scheduled",
                event_type="job_created",
                message="Scheduled job created.",
            )
            return {
                "ok": True,
                "job": serialize_scheduled_job(job),
                "metadata": meta,
                "message": f"Scheduled job created for {job.cadence} at UTC {_to_iso_z(job.next_run_at)}.",
            }

        if action == "update" and bulk_op:
            return {"ok": False, "error": {"message": "Bulk update is not supported. Provide job_id for update."}}

        if action in {"delete", "pause", "resume"} and bulk_op:
            rows = ctx.list_scheduled_jobs(
                session,
                bot_id=bot.id,
                conversation_id=conv_scope_id,
                limit=1000,
            )
            if not rows:
                return {"ok": True, "count": 0, "jobs": [], "message": "No scheduled jobs found for the requested scope."}

            out_jobs: list[dict[str, Any]] = []
            deleted_ids: list[str] = []
            skipped_ids: list[str] = []
            for row in rows:
                if action == "delete":
                    append_job_metadata_event(
                        ctx,
                        session,
                        conversation_id=row.conversation_id,
                        job=row,
                        status="deleted",
                        event_type="job_deleted",
                        message="Scheduled job deleted.",
                    )
                    ctx.delete_scheduled_job(session, row.id)
                    deleted_ids.append(str(row.id))
                    continue

                if action == "pause":
                    updated = ctx.update_scheduled_job(session, row.id, {"enabled": False, "next_run_at": None})
                    append_job_metadata_event(
                        ctx,
                        session,
                        conversation_id=updated.conversation_id,
                        job=updated,
                        status="paused",
                        event_type="job_paused",
                        message="Scheduled job paused.",
                    )
                    out_jobs.append(serialize_scheduled_job(updated))
                    continue

                # resume
                next_run_at = compute_next_run_at(
                    cadence=row.cadence,
                    time_utc=row.time_utc,
                    weekday_utc=row.weekday_utc,
                    run_at_utc=row.run_at_utc,
                )
                if next_run_at is None:
                    skipped_ids.append(str(row.id))
                    continue
                updated = ctx.update_scheduled_job(
                    session,
                    row.id,
                    {"enabled": True, "next_run_at": next_run_at, "last_status": "scheduled", "last_error": ""},
                )
                append_job_metadata_event(
                    ctx,
                    session,
                    conversation_id=updated.conversation_id,
                    job=updated,
                    status="scheduled",
                    event_type="job_resumed",
                    message="Scheduled job resumed.",
                )
                out_jobs.append(serialize_scheduled_job(updated))

            if action == "delete":
                return {
                    "ok": True,
                    "count": len(deleted_ids),
                    "deleted_ids": deleted_ids,
                    "message": f"Deleted {len(deleted_ids)} scheduled job(s).",
                }
            verb = "paused" if action == "pause" else "resumed"
            msg = f"{verb.capitalize()} {len(out_jobs)} scheduled job(s)."
            if skipped_ids:
                msg += f" Skipped {len(skipped_ids)} job(s) with no future next run."
            return {
                "ok": True,
                "count": len(out_jobs),
                "jobs": out_jobs,
                "skipped_ids": skipped_ids,
                "message": msg,
            }

        target_job: ScheduledJob
        raw_job_id = str(tool_args.get("job_id") or "").strip()
        if raw_job_id:
            job_id = _parse_uuid(raw_job_id, "job_id")
            target_job = ctx.get_scheduled_job(session, job_id)
            if target_job.bot_id != bot.id:
                return {"ok": False, "error": {"message": "Scheduled job does not belong to this bot."}}
        else:
            rows = ctx.list_scheduled_jobs(
                session,
                bot_id=bot.id,
                conversation_id=conv_scope_id,
                limit=1000,
            )
            if len(rows) == 1:
                target_job = rows[0]
                job_id = target_job.id
            elif len(rows) == 0:
                return {"ok": False, "error": {"message": "No scheduled jobs found to modify."}}
            else:
                return {
                    "ok": False,
                    "error": {
                        "message": "Multiple scheduled jobs found. Provide job_id or use list first.",
                    },
                }

        if action == "delete":
            append_job_metadata_event(
                ctx,
                session,
                conversation_id=target_job.conversation_id,
                job=target_job,
                status="deleted",
                event_type="job_deleted",
                message="Scheduled job deleted.",
            )
            ctx.delete_scheduled_job(session, job_id)
            return {"ok": True, "deleted": str(job_id), "message": "Scheduled job deleted."}

        if action == "pause":
            job = ctx.update_scheduled_job(session, job_id, {"enabled": False, "next_run_at": None})
            meta = append_job_metadata_event(
                ctx,
                session,
                conversation_id=job.conversation_id,
                job=job,
                status="paused",
                event_type="job_paused",
                message="Scheduled job paused.",
            )
            return {"ok": True, "job": serialize_scheduled_job(job), "metadata": meta, "message": "Scheduled job paused."}

        if action == "resume":
            next_run_at = compute_next_run_at(
                cadence=target_job.cadence,
                time_utc=target_job.time_utc,
                weekday_utc=target_job.weekday_utc,
                run_at_utc=target_job.run_at_utc,
            )
            job = ctx.update_scheduled_job(
                session,
                job_id,
                {"enabled": True, "next_run_at": next_run_at, "last_status": "scheduled", "last_error": ""},
            )
            meta = append_job_metadata_event(
                ctx,
                session,
                conversation_id=job.conversation_id,
                job=job,
                status="scheduled",
                event_type="job_resumed",
                message="Scheduled job resumed.",
            )
            return {"ok": True, "job": serialize_scheduled_job(job), "metadata": meta, "message": "Scheduled job resumed."}

        # update
        job = update_job_from_payload(ctx, session, bot_id=bot.id, job_id=job_id, payload=tool_args)
        meta = append_job_metadata_event(
            ctx,
            session,
            conversation_id=job.conversation_id,
            job=job,
            status=job.last_status or "scheduled",
            event_type="job_updated",
            message="Scheduled job updated.",
        )
        return {"ok": True, "job": serialize_scheduled_job(job), "metadata": meta, "message": "Scheduled job updated."}
    except Exception as exc:
        return {"ok": False, "error": {"message": str(exc) or "schedule_job failed."}}


def format_schedule_job_user_reply(
    *,
    tool_args: dict[str, Any],
    tool_result: dict[str, Any],
) -> str:
    action = str((tool_args or {}).get("action") or "").strip().lower()
    alias = {
        "disable": "pause",
        "enable": "resume",
        "disable_all": "pause",
        "enable_all": "resume",
        "delete_all": "delete",
    }
    normalized_action = alias.get(action, action)

    ok = bool((tool_result or {}).get("ok"))
    if not ok:
        err_obj = (tool_result or {}).get("error")
        err_msg = ""
        if isinstance(err_obj, dict):
            err_msg = str(err_obj.get("message") or "").strip()
        if not err_msg:
            err_msg = str((tool_result or {}).get("message") or "").strip()
        if not err_msg:
            err_msg = "schedule_job failed."
        return f"Could not update scheduled jobs: {err_msg}"

    msg = str((tool_result or {}).get("message") or "").strip()
    if normalized_action in {"create", "update", "delete", "pause", "resume"}:
        if msg:
            return msg

    if normalized_action == "list":
        jobs = (tool_result or {}).get("jobs")
        if not isinstance(jobs, list) or not jobs:
            return "There are no scheduled jobs right now."
        lines: list[str] = [f"There {'is' if len(jobs) == 1 else 'are'} {len(jobs)} scheduled job{'s' if len(jobs) != 1 else ''}:"]
        for j in jobs[:10]:
            if not isinstance(j, dict):
                continue
            job_id = str(j.get("id") or j.get("job_id") or "").strip() or "unknown-id"
            cadence = str(j.get("cadence") or "").strip() or "-"
            next_run = str(j.get("next_run_at") or "").strip() or str(j.get("run_at_utc") or "").strip() or "-"
            enabled = bool(j.get("enabled"))
            status = str(j.get("last_status") or "").strip() or "-"
            lines.append(f"- {job_id}: {cadence}, next {next_run}, {'enabled' if enabled else 'disabled'}, status {status}")
        if len(jobs) > 10:
            lines.append(f"...and {len(jobs) - 10} more.")
        return "\n".join(lines)

    if "deleted" in (tool_result or {}):
        return f"Scheduled job deleted: {tool_result.get('deleted')}"
    if isinstance((tool_result or {}).get("deleted_ids"), list):
        n = len((tool_result or {}).get("deleted_ids") or [])
        return msg or f"Deleted {n} scheduled job(s)."
    if isinstance((tool_result or {}).get("jobs"), list):
        jobs = (tool_result or {}).get("jobs") or []
        if len(jobs) == 1 and isinstance(jobs[0], dict):
            job_id = str(jobs[0].get("id") or "").strip()
            if job_id:
                return msg or f"Updated scheduled job {job_id}."
        return msg or f"Updated {len(jobs)} scheduled job(s)."

    return msg or "Scheduled jobs updated."


async def execute_scheduled_job_async(ctx, *, job_id: UUID) -> None:
    outcome_error = ""
    result_preview = ""
    conv_id_for_event: Optional[UUID] = None
    try:
        with Session(ctx.engine) as session:
            job = ctx.get_scheduled_job(session, job_id)
            if not bool(job.enabled):
                _ = ctx.update_scheduled_job(session, job.id, {"is_running": False, "running_started_at": None})
                return
            bot = ctx.get_bot(session, job.bot_id)
            conv = ctx.get_conversation(session, job.conversation_id)
            if conv.bot_id != bot.id:
                raise RuntimeError("Scheduled job conversation no longer belongs to the bot.")
            conv_id_for_event = conv.id
            provider, llm_api_key, llm = ctx._require_llm_client(session, bot=bot)

            user_text = str(job.what_to_do or "").strip()
            if not user_text:
                raise RuntimeError("Scheduled job instruction is empty.")

            req_id = f"scheduled-{job.id}-{int(utc_now().timestamp())}"
            null_ws = ctx._NullWebSocket()
            history: list[Message]

            history = await ctx._build_history_budgeted_async(
                bot_id=bot.id,
                conversation_id=conv.id,
                llm_api_key=llm_api_key,
                status_cb=None,
            )
            # Scheduled execution should run as a system instruction, not as a new user turn.
            origin_text = str(job.input_message or "").strip() or user_text
            history = [
                *history,
                Message(
                    role="system",
                    content=(
                        "You are executing an already-scheduled job now.\n"
                        f"Original user request (earlier): {origin_text}\n"
                        f"Execute now: {user_text}\n"
                        "Do not create/update/delete any jobs in this run.\n"
                        "Do not call schedule_job.\n"
                        "Complete the task directly using other tools if needed."
                    ),
                ),
            ]
            tools_defs = [d for d in ctx._build_tools_for_bot(session, bot.id) if str(d.get("name") or "") != "schedule_job"]
            final_text, tool_calls, citations, timings = await ctx.talk_stream_module.run_llm_stream_with_tts(
                ws=null_ws,
                req_id=req_id,
                llm=llm,
                history=history,
                tools_defs=tools_defs,
                speak=False,
                tts_synth=None,
                bot=bot,
                openai_api_key="",
                status_cb=lambda _req_id, _stage: None,
            )
            citations_json = json.dumps(citations, ensure_ascii=False) if citations else "[]"
            if tool_calls:
                rendered_reply, _, _, _, _ = await ctx.talk_tools_module.process_talk_tool_calls(
                    ws=null_ws,
                    req_id=req_id,
                    bot_id=bot.id,
                    conv_id=conv.id,
                    tool_calls=tool_calls,
                    rendered_reply=final_text,
                    speak=False,
                    tts_synth=None,
                    status_cb=lambda _req_id, _stage: None,
                    llm=llm,
                    llm_api_key=llm_api_key,
                    provider=provider,
                    history=history,
                    citations=citations,
                    timings=timings,
                    debug_mode=False,
                )
                result_preview = (rendered_reply or "").strip()
            else:
                final_text_clean = (final_text or "").strip()
                if final_text_clean:
                    in_tok, out_tok, cost = ctx._estimate_llm_cost_for_turn(
                        session=session,
                        bot=bot,
                        provider=provider,
                        history=history,
                        assistant_text=final_text_clean,
                    )
                    ctx.add_message_with_metrics(
                        session,
                        conversation_id=conv.id,
                        role="assistant",
                        content=final_text_clean,
                        input_tokens_est=in_tok,
                        output_tokens_est=out_tok,
                        cost_usd_est=cost,
                        llm_ttfb_ms=timings.get("llm_ttfb"),
                        llm_total_ms=timings.get("llm_total"),
                        tts_first_audio_ms=timings.get("tts_first_audio"),
                        total_ms=timings.get("total"),
                        citations_json=citations_json,
                    )
                    ctx.update_conversation_metrics(
                        session,
                        conversation_id=conv.id,
                        add_input_tokens_est=in_tok,
                        add_output_tokens_est=out_tok,
                        add_cost_usd_est=cost,
                        last_asr_ms=None,
                        last_llm_ttfb_ms=timings.get("llm_ttfb"),
                        last_llm_total_ms=timings.get("llm_total"),
                        last_tts_first_audio_ms=timings.get("tts_first_audio"),
                        last_total_ms=timings.get("total"),
                    )
                result_preview = final_text_clean
    except Exception as exc:
        outcome_error = str(exc) or "Scheduled job execution failed."

    with Session(ctx.engine) as session:
        try:
            job = ctx.get_scheduled_job(session, job_id)
        except Exception:
            return
        updated = _mark_job_terminal_state(
            ctx,
            session,
            job=job,
            success=not bool(outcome_error),
            error=outcome_error,
        )
        if not result_preview and not outcome_error:
            result_preview = "Execution completed, but no assistant text output was produced."
        if conv_id_for_event is None:
            conv_id_for_event = updated.conversation_id
        append_job_metadata_event(
            ctx,
            session,
            conversation_id=conv_id_for_event,
            job=updated,
            status=updated.last_status or ("error" if outcome_error else "success"),
            event_type="job_executed",
            message=(result_preview[:2000] if result_preview else ""),
            error=outcome_error,
        )
        try:
            ctx._conversation_ws_broadcast(
                conv_id_for_event,
                {"type": "conversation_update", "conversation_id": str(conv_id_for_event)},
            )
        except Exception:
            pass


def run_scheduled_job_sync(ctx, job_id: UUID) -> None:
    asyncio.run(execute_scheduled_job_async(ctx, job_id=job_id))


def _reset_stale_running_flags(ctx) -> None:
    with Session(ctx.engine) as session:
        jobs = ctx.list_scheduled_jobs(session, limit=100000)
        changed = False
        for job in jobs:
            if bool(job.is_running):
                job.is_running = False
                job.running_started_at = None
                job.updated_at = utc_now()
                session.add(job)
                changed = True
        if changed:
            session.commit()


def scheduler_loop(ctx, *, stop_event) -> None:
    logger = ctx.logging.getLogger("voicebot.scheduler")
    poll_seconds = float(os.environ.get("VOICEBOT_SCHEDULER_POLL_SECONDS") or "2")
    max_workers = int(os.environ.get("VOICEBOT_SCHEDULER_MAX_WORKERS") or "4")
    grace_seconds = int(os.environ.get("VOICEBOT_SCHEDULER_MISSED_GRACE_SECONDS") or "120")
    if poll_seconds < 0.25:
        poll_seconds = 0.25
    if max_workers < 1:
        max_workers = 1

    _reset_stale_running_flags(ctx)
    running_by_bot: set[UUID] = set()
    in_flight: dict[Future, tuple[UUID, UUID]] = {}
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="scheduled-job")
    logger.info(
        "scheduler loop started (poll=%.2fs workers=%d grace=%ss)",
        poll_seconds,
        max_workers,
        grace_seconds,
    )
    try:
        while not stop_event.is_set():
            for fut in list(in_flight.keys()):
                if not fut.done():
                    continue
                job_id, bot_id = in_flight.pop(fut)
                running_by_bot.discard(bot_id)
                try:
                    fut.result()
                except Exception as exc:
                    logger.error("scheduled job crashed job=%s err=%s", str(job_id), str(exc))

            now = utc_now()
            with Session(ctx.engine) as session:
                due_jobs = ctx.list_due_scheduled_jobs(session, now=now, limit=200)
                for job in due_jobs:
                    if job.bot_id in running_by_bot:
                        continue
                    claimed = ctx.claim_due_scheduled_job(session, job_id=job.id, now=now)
                    if claimed is None:
                        continue
                    if should_skip_due_job(claimed, now=now, grace_seconds=grace_seconds):
                        mark_job_missed(ctx, session, job=claimed)
                        continue
                    append_job_metadata_event(
                        ctx,
                        session,
                        conversation_id=claimed.conversation_id,
                        job=claimed,
                        status="running",
                        event_type="job_started",
                        message="Scheduled job execution started.",
                    )
                    running_by_bot.add(claimed.bot_id)
                    fut = executor.submit(ctx._run_scheduled_job_sync, claimed.id)
                    in_flight[fut] = (claimed.id, claimed.bot_id)
            stop_event.wait(poll_seconds)
    finally:
        stop_event.set()
        executor.shutdown(wait=False, cancel_futures=False)
        logger.info("scheduler loop stopped")
