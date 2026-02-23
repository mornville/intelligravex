from __future__ import annotations

import asyncio
import datetime as dt
import json
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from sqlmodel import Session

from voicebot.models import Bot


def data_agent_meta(meta: dict) -> dict:
    da = meta.get("data_agent")
    return da if isinstance(da, dict) else {}


def data_agent_container_info(ctx, *, conversation_id: UUID, container_id: str) -> tuple[str, list[dict[str, int]]]:
    name = ctx.container_name_for_conversation(conversation_id)
    ports = ctx.get_container_ports(
        container_id=container_id,
        container_name=name,
        conversation_id=conversation_id,
    )
    return name, ports


def ensure_data_agent_container(
    ctx,
    session: Session,
    *,
    bot: Bot,
    conversation_id: UUID,
    meta_current: dict,
) -> tuple[str, str, str]:
    da = data_agent_meta(meta_current)
    workspace_dir = str(da.get("workspace_dir") or "").strip() or ctx.default_workspace_dir_for_conversation(conversation_id)
    container_id = str(da.get("container_id") or "").strip()
    session_id = str(da.get("session_id") or "").strip()

    api_key = ctx._get_openai_api_key_for_bot(session, bot=bot)
    if not api_key:
        raise RuntimeError("No OpenAI API key configured for this bot (needed for Isolated Workspace).")
    auth_json = getattr(bot, "data_agent_auth_json", "") or "{}"
    git_token = ctx._get_git_token_plaintext(session, provider="github") if ctx._git_auth_mode(auth_json) == "token" else ""

    if not container_id:
        container_id = ctx.ensure_conversation_container(
            conversation_id=conversation_id,
            workspace_dir=workspace_dir,
            openai_api_key=api_key,
            git_token=git_token,
            auth_json=auth_json,
        )
    if container_id:
        container_name, ports = data_agent_container_info(ctx, conversation_id=conversation_id, container_id=container_id)
        ide_port = 0
        if ports:
            for item in ports:
                try:
                    host = int((item or {}).get("host") or 0)
                except Exception:
                    host = 0
                if host > ide_port:
                    ide_port = host
            if ide_port:
                ports = [p for p in ports if int((p or {}).get("host") or 0) != ide_port]
        meta_current = ctx.merge_conversation_metadata(
            session,
            conversation_id=conversation_id,
            patch={
                "data_agent.container_id": container_id,
                "data_agent.workspace_dir": workspace_dir,
                "data_agent.session_id": session_id,
                "data_agent.container_name": container_name,
                "data_agent.ports": ports,
                "data_agent.ide_port": ide_port,
            },
        )
    return container_id, session_id, workspace_dir


def build_data_agent_conversation_context(
    ctx,
    session: Session,
    *,
    bot: Bot,
    conversation_id: UUID,
    meta: dict,
) -> dict[str, Any]:
    summary = ""
    try:
        mem = meta.get("memory")
        if isinstance(mem, dict):
            summary = str(mem.get("summary") or "").strip()
    except Exception:
        summary = ""

    da = data_agent_meta(meta)
    container_name = str(da.get("container_name") or "").strip()
    ports = da.get("ports") if isinstance(da.get("ports"), list) else []

    msgs = ctx.list_messages(session, conversation_id=conversation_id)
    n_turns = int(getattr(bot, "history_window_turns", 16) or 16)
    max_msgs = max(8, min(96, n_turns * 2))
    tail = msgs[-max_msgs:] if len(msgs) > max_msgs else msgs
    history = [{"role": m.role, "content": m.content} for m in tail]
    ctx_obj: dict[str, Any] = {"summary": summary, "messages": history}
    if container_name or ports:
        ctx_obj["data_agent"] = {"container_name": container_name, "ports": ports}
    return ctx_obj


def initialize_data_agent_workspace(
    ctx,
    session: Session,
    *,
    bot: Bot,
    conversation_id: UUID,
    meta: dict,
) -> str:
    workspace_dir = ctx._data_agent_workspace_dir_for_conversation(session, conversation_id=conversation_id)
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)

    api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
    auth_json = (getattr(bot, "data_agent_auth_json", "") or "{}").strip() or "{}"
    sys_prompt = (getattr(bot, "data_agent_system_prompt", "") or "").strip() or ctx.DEFAULT_DATA_AGENT_SYSTEM_PROMPT
    ctx_obj = build_data_agent_conversation_context(
        ctx,
        session,
        bot=bot,
        conversation_id=conversation_id,
        meta=meta,
    )

    try:
        (ws / "api_spec.json").write_text(api_spec_text, encoding="utf-8")
        (ws / "auth.json").write_text(auth_json, encoding="utf-8")
        (ws / "AGENTS.md").write_text(sys_prompt + "\n", encoding="utf-8")
        (ws / "conversation_context.json").write_text(json.dumps(ctx_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return workspace_dir


async def kickoff_data_agent_container_if_enabled(
    ctx,
    *,
    bot_id: UUID,
    conversation_id: UUID,
) -> None:
    lock = ctx.data_agent_kickoff_locks.get(conversation_id)
    if lock is None:
        lock = asyncio.Lock()
        ctx.data_agent_kickoff_locks[conversation_id] = lock

    async with lock:
        try:
            with Session(ctx.engine) as session:
                bot = ctx.get_bot(session, bot_id)
                if not bool(getattr(bot, "enable_data_agent", False)):
                    return
                if not ctx.docker_available():
                    ctx.logger.warning(
                        "Isolated Workspace kickoff: Docker not available conv=%s bot=%s",
                        conversation_id,
                        bot_id,
                    )
                    ctx.merge_conversation_metadata(
                        session,
                        conversation_id=conversation_id,
                        patch={
                            "data_agent.init_error": "Docker is not available. Install Docker to use Isolated Workspace.",
                            "data_agent.ready": False,
                        },
                    )
                    return

                prewarm = bool(getattr(bot, "data_agent_prewarm_on_start", False))
                meta = ctx._get_conversation_meta(session, conversation_id=conversation_id)
                da = data_agent_meta(meta)
                if prewarm and (bool(da.get("ready", False)) or bool(da.get("prewarm_in_progress", False))):
                    return
                if (not prewarm) and str(da.get("container_id") or "").strip():
                    return

                api_key = ctx._get_openai_api_key_for_bot(session, bot=bot)
                if not api_key:
                    ctx.logger.warning(
                        "Isolated Workspace kickoff: missing OpenAI key conv=%s bot=%s", conversation_id, bot_id
                    )
                    ctx.merge_conversation_metadata(
                        session,
                        conversation_id=conversation_id,
                        patch={
                            "data_agent.init_error": "No OpenAI API key configured for this bot.",
                            "data_agent.ready": False,
                        },
                    )
                    return
                auth_json = getattr(bot, "data_agent_auth_json", "") or "{}"
                git_token = (
                    ctx._get_git_token_plaintext(session, provider="github")
                    if ctx._git_auth_mode(auth_json) == "token"
                    else ""
                )

                workspace_dir = (
                    str(da.get("workspace_dir") or "").strip()
                    or ctx.default_workspace_dir_for_conversation(conversation_id)
                )
                container_id = str(da.get("container_id") or "").strip()
                session_id = str(da.get("session_id") or "").strip()

            if not container_id:
                ctx.logger.info(
                    "Isolated Workspace kickoff: starting container conv=%s workspace=%s",
                    conversation_id,
                    workspace_dir,
                )
                container_id = await asyncio.to_thread(
                    ctx.ensure_conversation_container,
                    conversation_id=conversation_id,
                    workspace_dir=workspace_dir,
                    openai_api_key=api_key,
                    git_token=git_token,
                    auth_json=auth_json,
                )
                with Session(ctx.engine) as session:
                    ctx.merge_conversation_metadata(
                        session,
                        conversation_id=conversation_id,
                        patch={
                            "data_agent.container_id": container_id,
                            "data_agent.workspace_dir": workspace_dir,
                            "data_agent.session_id": session_id,
                        },
                    )

            if not prewarm:
                return

            ctx.logger.info(
                "Isolated Workspace prewarm: begin conv=%s container_id=%s session_id=%s",
                conversation_id,
                container_id,
                session_id or "",
            )
            with Session(ctx.engine) as session:
                bot = ctx.get_bot(session, bot_id)
                meta = ctx._get_conversation_meta(session, conversation_id=conversation_id)
                da = data_agent_meta(meta)
                if bool(da.get("ready", False)) or bool(da.get("prewarm_in_progress", False)):
                    return

                ctx.merge_conversation_metadata(
                    session,
                    conversation_id=conversation_id,
                    patch={
                        "data_agent.ready": False,
                        "data_agent.init_error": "",
                        "data_agent.prewarm_in_progress": True,
                        "data_agent.prewarm_started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )

                ctx_obj = build_data_agent_conversation_context(
                    ctx,
                    session,
                    bot=bot,
                    conversation_id=conversation_id,
                    meta=meta,
                )
                api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
                auth_json_raw = getattr(bot, "data_agent_auth_json", "") or "{}"
                git_token_current = (
                    ctx._get_git_token_plaintext(session, provider="github")
                    if ctx._git_auth_mode(auth_json_raw) == "token"
                    else ""
                )
                auth_json = ctx._merge_git_token_auth(auth_json_raw, git_token_current)
                sys_prompt = (
                    (getattr(bot, "data_agent_system_prompt", "") or "").strip()
                    or ctx.DEFAULT_DATA_AGENT_SYSTEM_PROMPT
                )

            init_task = (getattr(bot, "data_agent_prewarm_prompt", "") or "").strip()
            if not init_task:
                init_task = (
                    "INIT / PREWARM:\n"
                    "- Open and read: api_spec.json, auth.json, conversation_context.json.\n"
                    "- Do NOT call external APIs.\n"
                    "- Output ok=true and result_text='READY'."
                )
            try:
                res = await asyncio.to_thread(
                    ctx.run_data_agent,
                    conversation_id=conversation_id,
                    container_id=container_id,
                    session_id=session_id,
                    workspace_dir=workspace_dir,
                    api_spec_text=api_spec_text,
                    auth_json=auth_json,
                    system_prompt=sys_prompt,
                    conversation_context=ctx_obj,
                    what_to_do=init_task,
                    timeout_s=180.0,
                )
                with Session(ctx.engine) as session:
                    ctx.merge_conversation_metadata(
                        session,
                        conversation_id=conversation_id,
                        patch={
                            "data_agent.prewarm_in_progress": False,
                            "data_agent.prewarm_finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "data_agent.ready": bool(res.ok),
                            "data_agent.init_error": str(res.error or ""),
                            "data_agent.session_id": str(res.session_id or ""),
                            "data_agent.container_id": str(res.container_id or container_id),
                            "data_agent.workspace_dir": workspace_dir,
                        },
                    )
                ctx.logger.info(
                    "Isolated Workspace prewarm: done conv=%s ok=%s ready=%s session_id=%s error=%s",
                    conversation_id,
                    bool(res.ok),
                    bool(res.ok),
                    str(res.session_id or ""),
                    str(res.error or ""),
                )
            except Exception as exc:
                with Session(ctx.engine) as session:
                    ctx.merge_conversation_metadata(
                        session,
                        conversation_id=conversation_id,
                        patch={
                            "data_agent.prewarm_in_progress": False,
                            "data_agent.prewarm_finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "data_agent.ready": False,
                            "data_agent.init_error": str(exc),
                        },
                    )
                ctx.logger.info("Isolated Workspace prewarm: failed conv=%s error=%s", conversation_id, str(exc))

        except Exception:
            ctx.logger.exception("Isolated Workspace kickoff failed conv=%s bot=%s", conversation_id, bot_id)
            return


async def run_data_agent_tool_persist(
    ctx,
    *,
    conversation_id: UUID,
    bot_id: UUID,
    what_to_do: str,
    req_id: str,
    ws: Optional[Any],
    wait_reply: str,
    send_wait: bool,
    send_wait_cb: Optional[Callable[[str], Awaitable[None]]],
    stream_followup: bool,
) -> dict[str, Any]:
    outcome: dict[str, Any] = {
        "tool_result": {},
        "tool_failed": True,
        "rendered_reply": "",
        "llm_ttfb_ms": None,
        "llm_total_ms": None,
        "followup_streamed": False,
        "tool_result_persisted": False,
        "assistant_persisted": False,
    }
    loop = asyncio.get_running_loop()

    def _emit_tool_progress(text: str) -> None:
        t = (text or "").strip()
        if not t or ws is None:
            return
        asyncio.run_coroutine_threadsafe(
            ctx._ws_send_json(ws, {"type": "interim", "req_id": req_id, "kind": "tool", "text": t}),
            loop,
        )

    async def _send_wait(text: str) -> None:
        t = (text or "").strip()
        if not t or ws is None or not send_wait:
            return
        if send_wait_cb is not None:
            await send_wait_cb(t)
            return
        await ctx._ws_send_json(ws, {"type": "interim", "req_id": req_id, "kind": "wait", "text": t})

    def _strip_no_reply(text: str) -> str:
        raw = text or ""
        if re.fullmatch(r"\s*<no_reply>\s*", raw, flags=re.IGNORECASE):
            return ""
        return re.sub(r"\s*<no_reply>\s*$", "", raw, flags=re.IGNORECASE).strip()

    try:
        with Session(ctx.engine) as session:
            bot = ctx.get_bot(session, bot_id)
            if not bool(getattr(bot, "enable_data_agent", False)):
                outcome["tool_result"] = {
                    "ok": False,
                    "error": {"message": "Isolated Workspace is disabled for this bot."},
                }
                outcome["tool_failed"] = True
                raise RuntimeError("data_agent_disabled")
            if not ctx.docker_available():
                outcome["tool_result"] = {
                    "ok": False,
                    "error": {"message": "Docker is not available. Install Docker to use Isolated Workspace."},
                }
                outcome["tool_failed"] = True
                raise RuntimeError("docker_unavailable")
            if not what_to_do:
                outcome["tool_result"] = {
                    "ok": False,
                    "error": {"message": "Missing required tool arg: what_to_do"},
                }
                outcome["tool_failed"] = True
                raise RuntimeError("missing_what_to_do")

            meta_current = ctx._get_conversation_meta(session, conversation_id=conversation_id)
            da = data_agent_meta(meta_current)
            workspace_dir = (
                str(da.get("workspace_dir") or "").strip()
                or ctx.default_workspace_dir_for_conversation(conversation_id)
            )
            container_id = str(da.get("container_id") or "").strip()
            session_id = str(da.get("session_id") or "").strip()
            auth_json_raw = getattr(bot, "data_agent_auth_json", "") or "{}"
            git_token = (
                ctx._get_git_token_plaintext(session, provider="github")
                if ctx._git_auth_mode(auth_json_raw) == "token"
                else ""
            )

            if not container_id:
                api_key = ctx._get_openai_api_key_for_bot(session, bot=bot)
                if not api_key:
                    outcome["tool_result"] = {
                        "ok": False,
                        "error": {
                            "message": "No OpenAI API key configured for this bot (needed for Isolated Workspace)."
                        },
                    }
                    outcome["tool_failed"] = True
                    raise RuntimeError("missing_openai_key")
                container_id = await asyncio.to_thread(
                    ctx.ensure_conversation_container,
                    conversation_id=conversation_id,
                    workspace_dir=workspace_dir,
                    openai_api_key=api_key,
                    git_token=git_token,
                    auth_json=auth_json_raw,
                )
                meta_current = ctx.merge_conversation_metadata(
                    session,
                    conversation_id=conversation_id,
                    patch={
                        "data_agent.container_id": container_id,
                        "data_agent.workspace_dir": workspace_dir,
                    },
                )

            container_name, ports = data_agent_container_info(ctx, conversation_id=conversation_id, container_id=container_id)
            if container_name or ports:
                meta_current = ctx.merge_conversation_metadata(
                    session,
                    conversation_id=conversation_id,
                    patch={
                        "data_agent.container_name": container_name,
                        "data_agent.ports": ports,
                    },
                )

            ctx_obj = build_data_agent_conversation_context(
                ctx,
                session,
                bot=bot,
                conversation_id=conversation_id,
                meta=meta_current,
            )
            api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
            auth_json = ctx._merge_git_token_auth(auth_json_raw, git_token)
            sys_prompt = (
                (getattr(bot, "data_agent_system_prompt", "") or "").strip()
                or ctx.DEFAULT_DATA_AGENT_SYSTEM_PROMPT
            )

        task = asyncio.create_task(
            asyncio.to_thread(
                ctx.run_data_agent,
                conversation_id=conversation_id,
                container_id=container_id,
                session_id=session_id,
                workspace_dir=workspace_dir,
                api_spec_text=api_spec_text,
                auth_json=auth_json,
                system_prompt=sys_prompt,
                conversation_context=ctx_obj,
                what_to_do=what_to_do,
                on_stream=_emit_tool_progress,
            )
        )
        if send_wait and wait_reply:
            await _send_wait(wait_reply)
        last_wait = time.time()
        while not task.done():
            if send_wait and wait_reply and (time.time() - last_wait) >= 10.0:
                await _send_wait(wait_reply)
                last_wait = time.time()
            await asyncio.sleep(0.2)
        da_res = await task

        with Session(ctx.engine) as session:
            if da_res.session_id and da_res.session_id != session_id:
                ctx.merge_conversation_metadata(
                    session,
                    conversation_id=conversation_id,
                    patch={"data_agent.session_id": da_res.session_id},
                )

        tool_result = {
            "ok": bool(da_res.ok),
            "result_text": da_res.result_text,
            "data_agent_container_id": da_res.container_id,
            "data_agent_container_name": container_name,
            "data_agent_ports": ports,
            "data_agent_session_id": da_res.session_id,
            "data_agent_output_file": da_res.output_file,
            "data_agent_debug_file": da_res.debug_file,
            "error": da_res.error,
        }
        outcome["tool_result"] = tool_result
        outcome["tool_failed"] = not bool(da_res.ok)

        with Session(ctx.engine) as session:
            bot = ctx.get_bot(session, bot_id)
            conv = ctx.get_conversation(session, conversation_id)
            msg = ctx.add_message_with_metrics(
                session,
                conversation_id=conversation_id,
                role="tool",
                content=json.dumps({"tool": "give_command_to_data_agent", "result": tool_result}, ensure_ascii=False),
                sender_bot_id=bot.id,
                sender_name=bot.name,
            )
            if bool(conv.is_group):
                ctx._mirror_group_message(session, conv=conv, msg=msg)
        outcome["tool_result_persisted"] = True

        rendered_reply = ""
        llm_ttfb_ms: Optional[int] = None
        llm_total_ms: Optional[int] = None
        followup_streamed = False
        with Session(ctx.engine) as session:
            bot = ctx.get_bot(session, bot_id)
            provider, llm_api_key, _ = ctx._require_llm_client(session, bot=bot)
            base_history = await ctx._build_history_budgeted_async(
                bot_id=bot.id,
                conversation_id=conversation_id,
                llm_api_key=llm_api_key,
                status_cb=None,
            )
            history_for_cost = base_history
            if (
                bool(getattr(bot, "data_agent_return_result_directly", False))
                and bool(da_res.ok)
                and str(da_res.result_text or "").strip()
            ):
                rendered_reply = str(da_res.result_text or "").strip()
            else:
                followup_history = list(base_history)
                followup_history.append(
                    ctx.Message(
                        role="system",
                        content=(
                            ("The previous tool call failed. " if outcome["tool_failed"] else "")
                            + "Using the latest tool result(s) above, write the next assistant reply. "
                            "If the tool result contains codex_result_text, rephrase it for the user and do not mention file paths. "
                            "Do not call any tools."
                        ),
                    )
                )
                followup_model = bot.openai_model
                follow_llm = ctx._build_llm_client(
                    bot=bot,
                    api_key=llm_api_key,
                    model_override=followup_model,
                )
                target_ws = ws if (stream_followup and ws is not None) else ctx._NullWebSocket()  # type: ignore[attr-defined]
                text2, ttfb2, total2 = await ctx._stream_llm_reply(
                    ws=target_ws, req_id=req_id, llm=follow_llm, messages=followup_history
                )
                rendered_reply = text2.strip()
                llm_ttfb_ms = ttfb2
                llm_total_ms = total2
                followup_streamed = bool(stream_followup and ws is not None)
                history_for_cost = followup_history

            rendered_reply = _strip_no_reply(rendered_reply)
            if rendered_reply:
                in_tok = int(ctx.estimate_messages_tokens(history_for_cost, bot.openai_model) or 0)
                out_tok = int(ctx.estimate_text_tokens(rendered_reply, bot.openai_model) or 0)
                price = ctx._get_model_price(session, provider=provider, model=bot.openai_model)
                cost = float(
                    ctx.estimate_cost_usd(
                        model_price=price,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                    )
                    or 0.0
                )
                conv = ctx.get_conversation(session, conversation_id)
                msg = ctx.add_message_with_metrics(
                    session,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=rendered_reply,
                    input_tokens_est=in_tok or None,
                    output_tokens_est=out_tok or None,
                    cost_usd_est=cost or None,
                    llm_ttfb_ms=llm_ttfb_ms,
                    llm_total_ms=llm_total_ms,
                    total_ms=llm_total_ms,
                )
                if bool(conv.is_group):
                    ctx._mirror_group_message(session, conv=conv, msg=msg)
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
                outcome["assistant_persisted"] = True

        outcome["rendered_reply"] = rendered_reply
        outcome["llm_ttfb_ms"] = llm_ttfb_ms
        outcome["llm_total_ms"] = llm_total_ms
        outcome["followup_streamed"] = followup_streamed

    except Exception:
        if not outcome["tool_result"]:
            outcome["tool_result"] = {"ok": False, "error": {"message": "Isolated Workspace failed."}}
        try:
            with Session(ctx.engine) as session:
                bot = ctx.get_bot(session, bot_id)
                conv = ctx.get_conversation(session, conversation_id)
                msg = ctx.add_message_with_metrics(
                    session,
                    conversation_id=conversation_id,
                    role="tool",
                    content=json.dumps(
                        {"tool": "give_command_to_data_agent", "result": outcome["tool_result"]},
                        ensure_ascii=False,
                    ),
                    sender_bot_id=bot.id,
                    sender_name=bot.name,
                )
                if bool(conv.is_group):
                    ctx._mirror_group_message(session, conv=conv, msg=msg)
            outcome["tool_result_persisted"] = True
        except Exception:
            pass

    return outcome
