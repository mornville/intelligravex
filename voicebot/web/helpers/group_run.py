from __future__ import annotations

import asyncio
import datetime as dt
import re
import secrets
import time
from contextlib import nullcontext
from typing import Any, Optional
from uuid import UUID

from sqlmodel import Session

from voicebot.llm.openai_llm import CitationEvent, ToolCall
from voicebot.models import Conversation
from voicebot.web.helpers.group_tool_calls import process_group_tool_calls
from voicebot.web.helpers.history_build import build_history_budgeted_async


_SWARM_DEFAULTS = {
    "enabled": True,
    "coordinator_mode": "coordinator_first",
    "max_turns_per_run": 6,
    "max_parallel_bots": 2,
    "max_hops": 3,
    "allow_revisit": False,
}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _swarm_lock_ctx(ctx):
    lock = getattr(ctx, "group_swarm_lock", None)
    return lock if lock is not None else nullcontext()


def _to_int(value: Any, *, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except Exception:
        n = default
    return min(hi, max(lo, n))


def _normalize_swarm_config(raw: Any) -> dict[str, Any]:
    cfg = dict(_SWARM_DEFAULTS)
    if isinstance(raw, dict):
        cfg.update(raw)
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "coordinator_mode": "coordinator_first"
        if str(cfg.get("coordinator_mode") or "").strip().lower() != "mentions_only"
        else "mentions_only",
        "max_turns_per_run": _to_int(cfg.get("max_turns_per_run"), default=6, lo=1, hi=40),
        "max_parallel_bots": _to_int(cfg.get("max_parallel_bots"), default=2, lo=1, hi=8),
        "max_hops": _to_int(cfg.get("max_hops"), default=3, lo=0, hi=12),
        "allow_revisit": bool(cfg.get("allow_revisit", False)),
    }


def _safe_meta_dict(ctx, conv: Conversation) -> dict:
    meta = ctx.safe_json_loads(getattr(conv, "metadata_json", "") or "{}") or {}
    return meta if isinstance(meta, dict) else {}


def _safe_swarm_dict(meta: dict) -> dict:
    swarm = meta.get("group_swarm")
    return swarm if isinstance(swarm, dict) else {}


def _unique_member_bot_ids(conv: Conversation, bots: list[dict[str, str]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for b in bots:
        bid = str(b.get("id") or "").strip()
        if bid and bid not in seen:
            out.append(bid)
            seen.add(bid)
    default_bid = str(getattr(conv, "bot_id", "") or "").strip()
    if default_bid and default_bid not in seen:
        out.insert(0, default_bid)
    return out


def _uuid_targets_to_member_ids(targets: list[UUID], member_ids: set[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in targets:
        sid = str(t)
        if sid in member_ids and sid not in seen:
            out.append(sid)
            seen.add(sid)
    return out


def _append_conflict(active_run: dict, *, kind: str, bot_id: str = "", reason: str = "", source_bot_id: str = "") -> None:
    items = active_run.get("conflicts")
    if not isinstance(items, list):
        items = []
    entry = {
        "at": _now_iso(),
        "type": kind,
        "bot_id": bot_id or None,
        "source_bot_id": source_bot_id or None,
        "reason": reason or "",
    }
    items.append(entry)
    if len(items) > 40:
        items = items[-40:]
    active_run["conflicts"] = items


def _set_bot_state(active_run: dict, *, bot_id: str, state: str, reason: str = "", source_bot_id: str = "", error: str = "") -> None:
    by_bot = active_run.get("by_bot")
    if not isinstance(by_bot, dict):
        by_bot = {}
    cur = by_bot.get(bot_id)
    if not isinstance(cur, dict):
        cur = {}
    cur["state"] = state
    cur["updated_at"] = _now_iso()
    if reason:
        cur["reason"] = reason
    if source_bot_id:
        cur["source_bot_id"] = source_bot_id
    if error:
        cur["error"] = error[:500]
    by_bot[bot_id] = cur
    active_run["by_bot"] = by_bot


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for x in value:
        s = str(x or "").strip()
        if s:
            out.append(s)
    return out


def extract_group_mentions(ctx, text: str, conv: Conversation) -> list[UUID]:
    alias_map = ctx._group_bot_aliases(conv)
    if not alias_map:
        return []
    hits: list[UUID] = []

    def _collect_token(raw_token: str) -> None:
        token = str(raw_token or "").strip().lower()
        if not token:
            return
        candidates = {token}
        compact = re.sub(r"[-_]+", "", token)
        if compact:
            candidates.add(compact)
        for cand in candidates:
            for bot_id in alias_map.get(cand, []):
                try:
                    bid = UUID(bot_id)
                except Exception:
                    continue
                if bid not in hits:
                    hits.append(bid)

    raw_text = text or ""
    for m in re.finditer(r"@([a-zA-Z0-9][a-zA-Z0-9_-]{0,48})", raw_text):
        _collect_token(m.group(1))
    # Support handoffs like "offc-work: can you take this?" in addition to @mentions.
    for m in re.finditer(r"(?:^|[\s\(\[\{>])([a-zA-Z0-9][a-zA-Z0-9_-]{0,48})\s*:", raw_text):
        _collect_token(m.group(1))
    return hits


def start_group_swarm_run(
    ctx,
    *,
    conversation_id: UUID,
    trigger_message_id: Optional[UUID],
    objective: str,
    requested_targets: list[UUID],
    sender_role: str = "user",
) -> tuple[str, list[UUID]]:
    with _swarm_lock_ctx(ctx):
        with Session(ctx.engine) as session:
            conv = ctx.get_conversation(session, conversation_id)
            if not bool(conv.is_group):
                return "", requested_targets
            bots = ctx._group_bots_from_conv(conv)
            member_list = _unique_member_bot_ids(conv, bots)
            member_set = set(member_list)

            meta = _safe_meta_dict(ctx, conv)
            swarm = _safe_swarm_dict(meta)
            cfg = _normalize_swarm_config(swarm.get("config"))
            swarm["config"] = cfg
            if not bool(cfg.get("enabled", True)):
                requested_ids = _uuid_targets_to_member_ids(requested_targets, member_set)
                direct_targets: list[UUID] = []
                for bid in requested_ids:
                    try:
                        direct_targets.append(UUID(bid))
                    except Exception:
                        continue
                return "", direct_targets

            requested_ids = _uuid_targets_to_member_ids(requested_targets, member_set)
            coordinator_id = str(conv.bot_id or "").strip()
            if coordinator_id not in member_set and member_list:
                coordinator_id = member_list[0]

            planned_ids: list[str] = []
            if cfg.get("coordinator_mode") == "coordinator_first" and coordinator_id:
                planned_ids.append(coordinator_id)
            if requested_ids:
                for bid in requested_ids:
                    if bid not in planned_ids:
                        planned_ids.append(bid)
            elif coordinator_id and coordinator_id not in planned_ids:
                planned_ids.append(coordinator_id)
            if not planned_ids:
                planned_ids = requested_ids[:]

            run_id = secrets.token_hex(8)
            active_run = {
                "run_id": run_id,
                "status": "running",
                "started_at": _now_iso(),
                "updated_at": _now_iso(),
                "sender_role": str(sender_role or "user"),
                "objective": str(objective or "").strip()[:4000],
                "trigger_message_id": str(trigger_message_id) if trigger_message_id else None,
                "coordinator_bot_id": coordinator_id or None,
                "max_turns": int(cfg["max_turns_per_run"]),
                "remaining_turns": int(cfg["max_turns_per_run"]),
                "max_hops": int(cfg["max_hops"]),
                "hop_count": 0,
                "requested_bot_ids": requested_ids,
                "pending_bot_ids": planned_ids[:],
                "scheduled_bot_ids": [],
                "inflight_bot_ids": [],
                "completed_bot_ids": [],
                "failed_bot_ids": [],
                "conflicts": [],
                "by_bot": {},
            }
            swarm["active_run"] = active_run
            ctx.merge_conversation_metadata(session, conversation_id=conversation_id, patch={"group_swarm": swarm})

    planned_uuids: list[UUID] = []
    for bid in planned_ids:
        try:
            planned_uuids.append(UUID(bid))
        except Exception:
            continue
    return run_id, planned_uuids


def _claim_dispatch_targets(
    ctx,
    *,
    conversation_id: UUID,
    targets: list[UUID],
    run_id: Optional[str],
    reason: str,
    source_bot_id: Optional[UUID],
) -> tuple[str, list[UUID], dict[str, dict[str, str]]]:
    with _swarm_lock_ctx(ctx):
        with Session(ctx.engine) as session:
            conv = ctx.get_conversation(session, conversation_id)
            bots = ctx._group_bots_from_conv(conv)
            bot_map = {str(b.get("id") or ""): b for b in bots}
            member_list = _unique_member_bot_ids(conv, bots)
            member_set = set(member_list)

            meta = _safe_meta_dict(ctx, conv)
            swarm = _safe_swarm_dict(meta)
            cfg = _normalize_swarm_config(swarm.get("config"))
            swarm["config"] = cfg
            if not bool(cfg.get("enabled", True)):
                immediate = _uuid_targets_to_member_ids(targets, member_set)
                chosen: list[UUID] = []
                for bid in immediate:
                    try:
                        chosen.append(UUID(bid))
                    except Exception:
                        continue
                return "", chosen, bot_map

            active_run = swarm.get("active_run")
            if not isinstance(active_run, dict) or str(active_run.get("status") or "") != "running":
                if not targets:
                    return "", [], bot_map
                run_id_new = secrets.token_hex(8)
                active_run = {
                    "run_id": run_id_new,
                    "status": "running",
                    "started_at": _now_iso(),
                    "updated_at": _now_iso(),
                    "sender_role": "assistant",
                    "objective": "assistant-triggered swarm run",
                    "trigger_message_id": None,
                    "coordinator_bot_id": str(conv.bot_id or "") or None,
                    "max_turns": int(cfg["max_turns_per_run"]),
                    "remaining_turns": int(cfg["max_turns_per_run"]),
                    "max_hops": int(cfg["max_hops"]),
                    "hop_count": 0,
                    "requested_bot_ids": [],
                    "pending_bot_ids": [],
                    "scheduled_bot_ids": [],
                    "inflight_bot_ids": [],
                    "completed_bot_ids": [],
                    "failed_bot_ids": [],
                    "conflicts": [],
                    "by_bot": {},
                }

            if run_id and str(active_run.get("run_id") or "") != run_id:
                return str(active_run.get("run_id") or ""), [], bot_map

            effective_run_id = str(active_run.get("run_id") or "")
            if not effective_run_id:
                effective_run_id = secrets.token_hex(8)
                active_run["run_id"] = effective_run_id

            hop_count = _to_int(active_run.get("hop_count"), default=0, lo=0, hi=9999)
            max_hops = _to_int(active_run.get("max_hops"), default=int(cfg["max_hops"]), lo=0, hi=9999)
            if reason == "assistant_mention":
                if hop_count >= max_hops:
                    _append_conflict(
                        active_run,
                        kind="hop_limit",
                        reason=f"max_hops={max_hops}",
                        source_bot_id=str(source_bot_id) if source_bot_id else "",
                    )
                    active_run["status"] = "done"
                    active_run["updated_at"] = _now_iso()
                    swarm["active_run"] = active_run
                    ctx.merge_conversation_metadata(session, conversation_id=conversation_id, patch={"group_swarm": swarm})
                    return effective_run_id, [], bot_map
                active_run["hop_count"] = hop_count + 1

            pending = _as_str_list(active_run.get("pending_bot_ids"))
            scheduled = set(_as_str_list(active_run.get("scheduled_bot_ids")))
            inflight = set(_as_str_list(active_run.get("inflight_bot_ids")))
            completed = set(_as_str_list(active_run.get("completed_bot_ids")))
            failed = set(_as_str_list(active_run.get("failed_bot_ids")))
            remaining = _to_int(active_run.get("remaining_turns"), default=int(cfg["max_turns_per_run"]), lo=0, hi=9999)
            allow_revisit = bool(cfg.get("allow_revisit", False))

            incoming_ids = _uuid_targets_to_member_ids(targets, member_set)
            seen_pending = set(pending)
            for bid in incoming_ids:
                if bid not in seen_pending:
                    pending.append(bid)
                    seen_pending.add(bid)

            available_slots = max(0, _to_int(cfg.get("max_parallel_bots"), default=2, lo=1, hi=8) - len(inflight))
            chosen_ids: list[str] = []
            next_pending: list[str] = []

            for bid in pending:
                if bid in inflight or bid in scheduled:
                    _append_conflict(
                        active_run,
                        kind="already_in_progress",
                        bot_id=bid,
                        source_bot_id=str(source_bot_id) if source_bot_id else "",
                        reason=reason,
                    )
                    continue
                if (not allow_revisit) and (bid in completed or bid in failed):
                    _append_conflict(
                        active_run,
                        kind="already_completed",
                        bot_id=bid,
                        source_bot_id=str(source_bot_id) if source_bot_id else "",
                        reason=reason,
                    )
                    continue
                if available_slots <= 0:
                    next_pending.append(bid)
                    continue
                if remaining <= 0:
                    _append_conflict(
                        active_run,
                        kind="turn_budget_exhausted",
                        bot_id=bid,
                        source_bot_id=str(source_bot_id) if source_bot_id else "",
                        reason=reason,
                    )
                    next_pending.append(bid)
                    continue
                chosen_ids.append(bid)
                scheduled.add(bid)
                available_slots -= 1
                remaining -= 1
                _set_bot_state(
                    active_run,
                    bot_id=bid,
                    state="scheduled",
                    reason=reason,
                    source_bot_id=str(source_bot_id) if source_bot_id else "",
                )

            active_run["pending_bot_ids"] = next_pending
            active_run["scheduled_bot_ids"] = list(scheduled)
            active_run["inflight_bot_ids"] = list(inflight)
            active_run["completed_bot_ids"] = list(completed)
            active_run["failed_bot_ids"] = list(failed)
            active_run["remaining_turns"] = remaining
            if (not inflight and not scheduled and not chosen_ids) and (not next_pending or remaining <= 0):
                active_run["status"] = "done"
            active_run["updated_at"] = _now_iso()
            swarm["active_run"] = active_run
            ctx.merge_conversation_metadata(session, conversation_id=conversation_id, patch={"group_swarm": swarm})

    chosen: list[UUID] = []
    for bid in chosen_ids:
        try:
            chosen.append(UUID(bid))
        except Exception:
            continue
    return effective_run_id, chosen, bot_map


def _mark_run_bot_state(
    ctx,
    *,
    conversation_id: UUID,
    run_id: str,
    bot_id: UUID,
    state: str,
    reason: str = "",
    error: str = "",
) -> None:
    if not run_id:
        return
    with _swarm_lock_ctx(ctx):
        with Session(ctx.engine) as session:
            conv = ctx.get_conversation(session, conversation_id)
            meta = _safe_meta_dict(ctx, conv)
            swarm = _safe_swarm_dict(meta)
            active_run = swarm.get("active_run")
            if not isinstance(active_run, dict):
                return
            if str(active_run.get("run_id") or "") != str(run_id or ""):
                return

            bid = str(bot_id)
            pending = _as_str_list(active_run.get("pending_bot_ids"))
            scheduled = set(_as_str_list(active_run.get("scheduled_bot_ids")))
            inflight = set(_as_str_list(active_run.get("inflight_bot_ids")))
            completed = set(_as_str_list(active_run.get("completed_bot_ids")))
            failed = set(_as_str_list(active_run.get("failed_bot_ids")))

            if state == "running":
                scheduled.discard(bid)
                inflight.add(bid)
            elif state == "completed":
                scheduled.discard(bid)
                inflight.discard(bid)
                completed.add(bid)
            elif state == "failed":
                scheduled.discard(bid)
                inflight.discard(bid)
                failed.add(bid)
            elif state == "idle":
                inflight.discard(bid)

            active_run["pending_bot_ids"] = pending
            active_run["scheduled_bot_ids"] = list(scheduled)
            active_run["inflight_bot_ids"] = list(inflight)
            active_run["completed_bot_ids"] = list(completed)
            active_run["failed_bot_ids"] = list(failed)
            _set_bot_state(active_run, bot_id=bid, state=state, reason=reason, error=error)

            remaining = _to_int(active_run.get("remaining_turns"), default=0, lo=0, hi=9999)
            if (not pending or remaining <= 0) and not inflight and not scheduled:
                active_run["status"] = "done"
            active_run["updated_at"] = _now_iso()
            swarm["active_run"] = active_run
            ctx.merge_conversation_metadata(session, conversation_id=conversation_id, patch={"group_swarm": swarm})


async def run_group_bot_turn(
    ctx,
    *,
    bot_id: UUID,
    conversation_id: UUID,
    run_id: Optional[str] = None,
) -> str:
    with Session(ctx.engine) as session:
        bot = ctx.get_bot(session, bot_id)
        conv = ctx.get_conversation(session, conversation_id)
        provider, api_key, llm = ctx._require_llm_client(session, bot=bot)
        history = await build_history_budgeted_async(
            ctx,
            bot_id=bot.id,
            conversation_id=conversation_id,
            llm_api_key=api_key,
            status_cb=None,
        )
        tools_defs = ctx._build_tools_for_bot(session, bot.id)

    t0 = time.time()
    first_token_ts: Optional[float] = None
    tool_calls: list[ToolCall] = []
    full_text_parts: list[str] = []
    citations: list[dict] = []
    dispatch_targets: list[UUID] = []

    async for ev in ctx._aiter_from_blocking_iterator(lambda: llm.stream_text_or_tool(messages=history, tools=tools_defs)):
        if isinstance(ev, ToolCall):
            tool_calls.append(ev)
            continue
        if isinstance(ev, CitationEvent):
            citations.extend(ev.citations)
            continue
        d = str(ev)
        if d:
            if first_token_ts is None:
                first_token_ts = time.time()
            full_text_parts.append(d)

    llm_end_ts = time.time()
    rendered_reply = "".join(full_text_parts).strip()

    llm_ttfb_ms: Optional[int] = None
    if first_token_ts is not None:
        llm_ttfb_ms = int(round((first_token_ts - t0) * 1000.0))
    elif tool_calls and tool_calls[0].first_event_ts is not None:
        llm_ttfb_ms = int(round((tool_calls[0].first_event_ts - t0) * 1000.0))
    llm_total_ms = int(round((llm_end_ts - t0) * 1000.0))

    if tool_calls:
        rendered_reply, llm_ttfb_ms, llm_total_ms = await process_group_tool_calls(
            ctx,
            bot_id=bot_id,
            conversation_id=conversation_id,
            tool_calls=tool_calls,
            provider=provider,
            api_key=api_key,
            llm=llm,
            history=history,
            rendered_reply=rendered_reply,
            llm_ttfb_ms=llm_ttfb_ms,
            llm_total_ms=llm_total_ms,
        )

    if rendered_reply:
        with Session(ctx.engine) as session:
            conv = ctx.get_conversation(session, conversation_id)
        if bool(conv.is_group):
            rendered_reply = ctx._sanitize_group_reply(rendered_reply, conv, bot_id)

    payload = None
    raw_reply = rendered_reply or ""
    if re.fullmatch(r"\s*<no_reply>\s*", raw_reply, flags=re.IGNORECASE):
        rendered_reply = ""
        suppress_reply = True
    else:
        rendered_reply = re.sub(r"\s*<no_reply>\s*$", "", raw_reply, flags=re.IGNORECASE).strip()
        suppress_reply = False
    with Session(ctx.engine) as session:
        in_tok, out_tok, cost = ctx._estimate_llm_cost_for_turn(
            session=session,
            bot=bot,
            provider=provider,
            history=history,
            assistant_text=rendered_reply,
        )
        conv = ctx.get_conversation(session, conversation_id)
        assistant_msg = None
        if suppress_reply:
            # Intentionally skip writing placeholder/empty assistant messages when
            # the model returned <no_reply>. This avoids "metrics-only" noise in
            # individual bot logs while preserving run state and accounting.
            pass
        else:
            assistant_msg = ctx.add_message_with_metrics(
                session,
                conversation_id=conversation_id,
                role="assistant",
                content=rendered_reply,
                sender_bot_id=bot.id,
                sender_name=bot.name,
                input_tokens_est=in_tok,
                output_tokens_est=out_tok,
                cost_usd_est=cost,
                llm_ttfb_ms=llm_ttfb_ms,
                llm_total_ms=llm_total_ms,
                total_ms=llm_total_ms,
                citations_json=ctx.json.dumps(citations, ensure_ascii=False),
            )
            ctx._mirror_group_message(session, conv=conv, msg=assistant_msg)
        ctx.update_conversation_metrics(
            session,
            conversation_id=conversation_id,
            add_input_tokens_est=in_tok,
            add_output_tokens_est=out_tok,
            add_cost_usd_est=cost,
            last_asr_ms=None,
            last_llm_ttfb_ms=llm_ttfb_ms,
            last_llm_total_ms=llm_total_ms,
            last_tts_first_audio_ms=None,
            last_total_ms=llm_total_ms,
        )

        if assistant_msg is not None:
            payload = ctx._group_message_payload(assistant_msg)
            dispatch_targets = extract_group_mentions(ctx, rendered_reply, conv)
            dispatch_targets = [bid for bid in dispatch_targets if str(bid) != str(bot.id)]

    if payload:
        await ctx._group_ws_broadcast(conversation_id, {"type": "message", "message": payload})
    if dispatch_targets:
        schedule_group_bots(
            ctx,
            conversation_id,
            dispatch_targets,
            run_id=run_id,
            reason="assistant_mention",
            source_bot_id=bot.id,
        )

    return rendered_reply


def schedule_group_bots(
    ctx,
    conversation_id: UUID,
    targets: list[UUID],
    *,
    run_id: Optional[str] = None,
    reason: str = "dispatch",
    source_bot_id: Optional[UUID] = None,
) -> None:
    effective_run_id, claimed_targets, bot_map = _claim_dispatch_targets(
        ctx,
        conversation_id=conversation_id,
        targets=targets or [],
        run_id=run_id,
        reason=reason,
        source_bot_id=source_bot_id,
    )
    if not claimed_targets:
        return

    async def _run_target(target_id: UUID) -> None:
        binfo = bot_map.get(str(target_id), {})
        _mark_run_bot_state(
            ctx,
            conversation_id=conversation_id,
            run_id=effective_run_id,
            bot_id=target_id,
            state="running",
            reason=reason,
        )
        await ctx._group_ws_broadcast(
            conversation_id,
            {
                "type": "status",
                "bot_id": str(target_id),
                "bot_name": binfo.get("name") or "assistant",
                "state": "working",
                "run_id": effective_run_id,
            },
        )
        try:
            await run_group_bot_turn(ctx, bot_id=target_id, conversation_id=conversation_id, run_id=effective_run_id)
            _mark_run_bot_state(
                ctx,
                conversation_id=conversation_id,
                run_id=effective_run_id,
                bot_id=target_id,
                state="completed",
                reason=reason,
            )
        except Exception as exc:
            _mark_run_bot_state(
                ctx,
                conversation_id=conversation_id,
                run_id=effective_run_id,
                bot_id=target_id,
                state="failed",
                reason=reason,
                error=str(exc),
            )
        finally:
            await ctx._group_ws_broadcast(
                conversation_id,
                {
                    "type": "status",
                    "bot_id": str(target_id),
                    "bot_name": binfo.get("name") or "assistant",
                    "state": "idle",
                    "run_id": effective_run_id,
                },
            )
            schedule_group_bots(
                ctx,
                conversation_id,
                [],
                run_id=effective_run_id,
                reason="drain",
                source_bot_id=target_id,
            )

    try:
        loop = asyncio.get_running_loop()
        for bid in claimed_targets:
            loop.create_task(_run_target(bid))
    except Exception:
        pass
