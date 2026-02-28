from __future__ import annotations

import asyncio
import json
from typing import Any, Optional
from uuid import UUID

from sqlmodel import Session

from voicebot.llm.openai_llm import Message, ToolCall
from voicebot.models import IntegrationTool


async def process_group_tool_calls(
    ctx,
    *,
    bot_id: UUID,
    conversation_id: UUID,
    tool_calls: list[ToolCall],
    provider: str,
    api_key: str,
    llm,
    history: list[Message],
    rendered_reply: str,
    llm_ttfb_ms: Optional[int],
    llm_total_ms: Optional[int],
) -> tuple[str, Optional[int], Optional[int]]:
    with Session(ctx.engine) as session:
        bot = ctx.get_bot(session, bot_id)
        conv = ctx.get_conversation(session, conversation_id)
        meta_current = ctx._get_conversation_meta(session, conversation_id=conversation_id)
        disabled_tools = ctx._disabled_tool_names(bot)
        final = ""
        needs_followup_llm = False
        tool_failed = False
        followup_streamed = False

        for tc in tool_calls:
            tool_name = tc.name
            if tool_name == "set_variable":
                tool_name = "set_metadata"

            suppress_tool_result = False
            try:
                tool_args = json.loads(tc.arguments_json or "{}")
                if not isinstance(tool_args, dict):
                    tool_args = {}
            except Exception:
                tool_args = {}

            tool_call_msg = ctx.add_message_with_metrics(
                session,
                conversation_id=conversation_id,
                role="tool",
                content=json.dumps({"tool": tool_name, "arguments": tool_args}, ensure_ascii=False),
                sender_bot_id=bot.id,
                sender_name=bot.name,
            )
            ctx._mirror_group_message(session, conv=conv, msg=tool_call_msg)

            next_reply = str(tool_args.get("next_reply") or "").strip()
            wait_reply = str(tool_args.get("wait_reply") or "").strip()
            follow_up = ctx._parse_follow_up_flag(tool_args.get("follow_up")) or ctx._parse_follow_up_flag(
                tool_args.get("followup")
            )
            if (
                tool_name in {"request_host_action", "capture_screenshot"}
                and "follow_up" not in tool_args
                and "followup" not in tool_args
            ):
                follow_up = True
            if tool_name in {"request_host_action", "capture_screenshot"}:
                next_reply = ""
            raw_args = tool_args.get("args")
            if isinstance(raw_args, dict):
                patch = dict(raw_args)
            else:
                patch = dict(tool_args)
                patch.pop("next_reply", None)
                patch.pop("wait_reply", None)
                patch.pop("follow_up", None)
                patch.pop("followup", None)
                patch.pop("args", None)

            tool_cfg: IntegrationTool | None = None
            response_json: Any | None = None
            if tool_name in disabled_tools:
                tool_result = {
                    "ok": False,
                    "error": {"message": f"Tool '{tool_name}' is disabled for this bot."},
                }
                tool_failed = True
                needs_followup_llm = True
                final = ""
            elif tool_name == "set_metadata":
                new_meta = ctx.merge_conversation_metadata(session, conversation_id=conversation_id, patch=patch)
                tool_result = {"ok": True, "updated": patch, "metadata": new_meta}
            elif tool_name == "web_search":
                tool_result = {
                    "ok": False,
                    "error": {"message": "web_search runs inside the model; no server tool is available."},
                }
                tool_failed = True
                needs_followup_llm = True
                final = ""
            elif tool_name == "http_request":
                tool_result = await asyncio.to_thread(
                    ctx._execute_http_request_tool, tool_args=patch, meta=meta_current
                )
                tool_failed = not bool(tool_result.get("ok", False))
                needs_followup_llm = True
                final = ""
            elif tool_name == "capture_screenshot":
                if not bool(getattr(bot, "enable_host_actions", False)):
                    tool_result = {"ok": False, "error": {"message": "Host actions are disabled for this bot."}}
                    tool_failed = True
                    needs_followup_llm = True
                    final = ""
                elif not bool(getattr(bot, "enable_host_shell", False)):
                    tool_result = {"ok": False, "error": {"message": "Shell commands are disabled for this bot."}}
                    tool_failed = True
                    needs_followup_llm = True
                    final = ""
                else:
                    try:
                        rel_path, target = ctx._prepare_screenshot_target(conv)
                    except Exception as exc:
                        tool_result = {"ok": False, "error": {"message": str(exc) or "Invalid screenshot path"}}
                        tool_failed = True
                        needs_followup_llm = True
                        final = ""
                    else:
                        ok_cmd, cmd_or_err = ctx._screencapture_command(target)
                        if not ok_cmd:
                            tool_result = {"ok": False, "error": {"message": cmd_or_err}}
                            tool_failed = True
                            needs_followup_llm = True
                            final = ""
                        else:
                            action = ctx._create_host_action(
                                session,
                                conv=conv,
                                bot=bot,
                                action_type="run_shell",
                                payload={
                                    "command": cmd_or_err,
                                    "result_path": rel_path,
                                    "result_abs_path": str(target),
                                },
                            )
                            if ctx._host_action_requires_approval(bot):
                                tool_result = ctx._build_host_action_tool_result(action, ok=True)
                                tool_result["path"] = rel_path
                                suppress_tool_result = True
                                needs_followup_llm = False
                                final = ""
                            else:
                                tool_result = await ctx._execute_host_action_and_update_async(session, action=action)
                                tool_failed = not bool(tool_result.get("ok", False))
                                if tool_failed:
                                    needs_followup_llm = True
                                    final = ""
                                else:
                                    ok, summary_text = ctx._summarize_image_file(
                                        session,
                                        bot=bot,
                                        image_path=target,
                                        prompt=str(patch.get("prompt") or "").strip(),
                                    )
                                    if not ok:
                                        tool_result["summary_error"] = summary_text
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        tool_result["summary"] = summary_text
                                        tool_result["path"] = rel_path
                                        needs_followup_llm = True
                                        final = ""
            elif tool_name == "summarize_screenshot":
                ok, summary_text, rel_path = ctx._summarize_screenshot(
                    session,
                    conv=conv,
                    bot=bot,
                    path=str(patch.get("path") or "").strip(),
                    prompt=str(patch.get("prompt") or "").strip(),
                )
                if not ok:
                    tool_result = {"ok": False, "error": {"message": summary_text}}
                    tool_failed = True
                    needs_followup_llm = True
                    final = ""
                else:
                    tool_result = {"ok": True, "summary": summary_text, "path": rel_path}
                    needs_followup_llm = True
                    final = ""
            elif tool_name == "request_host_action":
                if not bool(getattr(bot, "enable_host_actions", False)):
                    tool_result = {"ok": False, "error": {"message": "Host actions are disabled for this bot."}}
                    tool_failed = True
                    needs_followup_llm = True
                    final = ""
                else:
                    try:
                        action_type, payload = ctx._parse_host_action_args(patch)
                    except Exception as exc:
                        tool_result = {"ok": False, "error": {"message": str(exc) or "Invalid host action"}}
                        tool_failed = True
                        needs_followup_llm = True
                        final = ""
                    else:
                        if action_type in {"run_shell", "run_powershell"} and not bool(
                            getattr(bot, "enable_host_shell", False)
                        ):
                            tool_result = {
                                "ok": False,
                                "error": {"message": "Shell commands are disabled for this bot."},
                            }
                            tool_failed = True
                            needs_followup_llm = True
                            final = ""
                        else:
                            action = ctx._create_host_action(
                                session,
                                conv=conv,
                                bot=bot,
                                action_type=action_type,
                                payload=payload,
                            )
                            if ctx._host_action_requires_approval(bot):
                                tool_result = ctx._build_host_action_tool_result(action, ok=True)
                                suppress_tool_result = True
                                needs_followup_llm = False
                                final = ""
                            else:
                                tool_result = await ctx._execute_host_action_and_update_async(session, action=action)
                                tool_failed = not bool(tool_result.get("ok", False))
                                candidate = ctx._render_with_meta(next_reply, meta_current).strip()
                                if follow_up and not tool_failed:
                                    needs_followup_llm = True
                                    final = ""
                                elif candidate and not tool_failed:
                                    final = candidate
                                    needs_followup_llm = False
                                else:
                                    needs_followup_llm = True
                                    final = ""
            elif tool_name == "give_command_to_data_agent":
                if not bool(getattr(bot, "enable_data_agent", False)):
                    tool_result = {"ok": False, "error": {"message": "Isolated Workspace is disabled for this bot."}}
                    tool_failed = True
                    needs_followup_llm = True
                    final = ""
                elif not ctx.docker_available():
                    tool_result = {
                        "ok": False,
                        "error": {"message": "Docker is not available. Install Docker to use Isolated Workspace."},
                    }
                    tool_failed = True
                    needs_followup_llm = True
                    final = ""
                else:
                    what_to_do = str(patch.get("what_to_do") or "").strip()
                    if not what_to_do:
                        tool_result = {"ok": False, "error": {"message": "Missing required tool arg: what_to_do"}}
                        tool_failed = True
                        needs_followup_llm = True
                        final = ""
                    else:
                        try:
                            ctx.logger.info(
                                "Isolated Workspace tool: start conv=%s bot=%s what_to_do=%s",
                                conversation_id,
                                bot_id,
                                (what_to_do[:200] + "…") if len(what_to_do) > 200 else what_to_do,
                            )
                            da = ctx._data_agent_meta(meta_current)
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
                            container_name, ports = ctx._data_agent_container_info(
                                conversation_id=conversation_id,
                                container_id=container_id,
                            )
                            if container_name or ports:
                                meta_current = ctx.merge_conversation_metadata(
                                    session,
                                    conversation_id=conversation_id,
                                    patch={
                                        "data_agent.container_name": container_name,
                                        "data_agent.ports": ports,
                                    },
                                )

                            ctx_obj = ctx._build_data_agent_conversation_context(
                                session, bot=bot, conversation_id=conversation_id, meta=meta_current
                            )
                            api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
                            auth_json = ctx._merge_git_token_auth(auth_json_raw, git_token)
                            sys_prompt = (
                                (getattr(bot, "data_agent_system_prompt", "") or "").strip()
                                or ctx.DEFAULT_DATA_AGENT_SYSTEM_PROMPT
                            )
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
                                what_to_do=what_to_do,
                            )
                            tool_result = {
                                "ok": bool(res.ok),
                                "error": {"message": str(res.error or "")} if res.error else None,
                                "session_id": str(res.session_id or ""),
                                "container_id": str(res.container_id or container_id),
                                "duration_ms": int(getattr(res, "duration_ms", 0) or 0),
                                "result_text": str(res.result_text or ""),
                            }
                            if res.output_dir:
                                tool_result["output_dir"] = str(res.output_dir)
                            if res.output_file:
                                tool_result["output_file"] = str(res.output_file)
                            if res.result_file:
                                tool_result["result_file"] = str(res.result_file)
                            if res.stderr:
                                tool_result["stderr"] = str(res.stderr)
                            if res.trace:
                                tool_result["trace"] = str(res.trace)
                            meta_current = ctx.merge_conversation_metadata(
                                session,
                                conversation_id=conversation_id,
                                patch={
                                    "data_agent.session_id": str(res.session_id or session_id),
                                    "data_agent.container_id": str(res.container_id or container_id),
                                    "data_agent.workspace_dir": workspace_dir,
                                    "data_agent.ready": bool(res.ok),
                                    "data_agent.last_error": str(res.error or ""),
                                },
                            )
                            tool_failed = not bool(res.ok)
                            needs_followup_llm = True
                            final = ""
                        except Exception as exc:
                            tool_result = {"ok": False, "error": {"message": str(exc) or "Data agent failed"}}
                            tool_failed = True
                            needs_followup_llm = True
                            final = ""
            else:
                tool_cfg = ctx.get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                if not tool_cfg:
                    raise RuntimeError(f"Unknown tool: {tool_name}")
                if not bool(getattr(tool_cfg, "enabled", True)):
                    response_json = {
                        "__tool_args_error__": {
                            "missing": [],
                            "message": f"Tool '{tool_name}' is disabled for this bot.",
                        }
                    }
                else:
                    task = asyncio.create_task(
                        asyncio.to_thread(
                            ctx._execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                        )
                    )
                    while True:
                        try:
                            response_json = await asyncio.wait_for(asyncio.shield(task), timeout=60.0)
                            break
                        except asyncio.TimeoutError:
                            continue
                if isinstance(response_json, dict) and "__tool_args_error__" in response_json:
                    err = response_json["__tool_args_error__"] or {}
                    tool_result = {"ok": False, "error": err}
                    tool_failed = True
                    needs_followup_llm = True
                    final = ""
                elif isinstance(response_json, dict) and "__http_error__" in response_json:
                    err = response_json["__http_error__"] or {}
                    tool_result = {"ok": False, "error": err}
                    tool_failed = True
                    needs_followup_llm = True
                    final = ""
                else:
                    pagination_info = None
                    if isinstance(response_json, dict) and "__igx_pagination__" in response_json:
                        pagination_info = response_json.pop("__igx_pagination__", None)
                    if bool(getattr(tool_cfg, "use_codex_response", False)):
                        tool_result = {"ok": True}
                        new_meta = meta_current
                    else:
                        mapped = ctx._apply_response_mapper(
                            mapper_json=tool_cfg.response_mapper_json,
                            response_json=response_json,
                            meta=meta_current,
                            tool_args=patch,
                        )
                        new_meta = ctx.merge_conversation_metadata(
                            session, conversation_id=conversation_id, patch=mapped
                        )
                        meta_current = new_meta
                        tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}
                    if pagination_info:
                        tool_result["pagination"] = pagination_info

                    static_preview = ""
                    if (tool_cfg.static_reply_template or "").strip():
                        try:
                            static_preview = ctx._render_static_reply(
                                template_text=tool_cfg.static_reply_template,
                                meta=new_meta or meta_current,
                                response_json=response_json,
                                tool_args=patch,
                            ).strip()
                        except Exception:
                            static_preview = ""
                    if (not static_preview) and bool(getattr(tool_cfg, "use_codex_response", False)):
                        fields_required = str(patch.get("fields_required") or "").strip()
                        why_api_was_called = str(patch.get("why_api_was_called") or "").strip()
                        what_to_search_for = str(patch.get("what_to_search_for") or "").strip()
                        if not fields_required:
                            fields_required = what_to_search_for
                        if not fields_required or not why_api_was_called:
                            tool_result["codex_error"] = "Missing fields_required / why_api_was_called."
                        else:
                            fields_required_for_codex = fields_required
                            if what_to_search_for and what_to_search_for not in fields_required_for_codex:
                                fields_required_for_codex = (
                                    f"{fields_required_for_codex}\n\n(what_to_search_for) {what_to_search_for}"
                                )
                            did_postprocess = False
                            if (tool_cfg.postprocess_python or "").strip():
                                try:
                                    py_res = await asyncio.to_thread(
                                        ctx.run_python_postprocess,
                                        python_code=tool_cfg.postprocess_python,
                                        response_json=response_json,
                                        meta=new_meta or meta_current,
                                        args=patch,
                                        fields_required=fields_required_for_codex,
                                        why_api_was_called=why_api_was_called,
                                        timeout_s=60,
                                    )
                                    tool_result["python_ok"] = bool(getattr(py_res, "ok", False))
                                    tool_result["python_duration_ms"] = int(getattr(py_res, "duration_ms", 0) or 0)
                                    if getattr(py_res, "error", None):
                                        tool_result["python_error"] = str(getattr(py_res, "error"))
                                    if getattr(py_res, "stderr", None):
                                        tool_result["python_stderr"] = str(getattr(py_res, "stderr"))
                                    if py_res.ok:
                                        did_postprocess = True
                                        tool_result["postprocess_mode"] = "python"
                                        tool_result["codex_ok"] = True
                                        tool_result["codex_result_text"] = str(
                                            getattr(py_res, "result_text", "") or ""
                                        )
                                        mp = getattr(py_res, "metadata_patch", None)
                                        if isinstance(mp, dict) and mp:
                                            try:
                                                meta_current = ctx.merge_conversation_metadata(
                                                    session,
                                                    conversation_id=conversation_id,
                                                    patch=mp,
                                                )
                                                tool_result["python_metadata_patch"] = mp
                                            except Exception:
                                                pass
                                except Exception as exc:
                                    tool_result["python_ok"] = False
                                    tool_result["python_error"] = str(exc)

                            if not did_postprocess:
                                codex_model = (
                                    (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                    or "gpt-5.1-codex-mini"
                                )
                                try:
                                    agent_res = await asyncio.to_thread(
                                        ctx.run_codex_http_agent_one_shot,
                                        api_key=api_key or "",
                                        model=codex_model,
                                        response_json=response_json,
                                        fields_required=fields_required_for_codex,
                                        why_api_was_called=why_api_was_called,
                                        response_schema_json=getattr(tool_cfg, "response_schema_json", "") or "",
                                        conversation_id=str(conversation_id) if conversation_id is not None else None,
                                        req_id=req_id,
                                        tool_codex_prompt=getattr(tool_cfg, "codex_prompt", "") or "",
                                        progress_fn=lambda _p: None,
                                    )
                                    tool_result["postprocess_mode"] = "codex"
                                    tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                    tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                    tool_result["codex_output_file"] = getattr(agent_res, "result_text_path", "")
                                    tool_result["codex_debug_file"] = getattr(agent_res, "debug_json_path", "")
                                    tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                    tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                    tool_result["codex_continue_reason"] = getattr(agent_res, "continue_reason", "")
                                    tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                    err = getattr(agent_res, "error", None)
                                    if err:
                                        tool_result["codex_error"] = str(err)
                                except Exception as exc:
                                    tool_result["codex_ok"] = False
                                    tool_result["codex_error"] = str(exc)

            if tool_name == "capture_screenshot" and tool_failed:
                msg = ctx._tool_error_message(tool_result, fallback="Screenshot failed.")
                final = f"Screenshot failed: {msg}"
                needs_followup_llm = False
            if tool_name == "request_host_action" and tool_failed:
                msg = ctx._tool_error_message(tool_result, fallback="Host action failed.")
                final = f"Host action failed: {msg}"
                needs_followup_llm = False

            if not suppress_tool_result:
                tool_result_msg = ctx.add_message_with_metrics(
                    session,
                    conversation_id=conversation_id,
                    role="tool",
                    content=json.dumps({"tool": tool_name, "result": tool_result}, ensure_ascii=False),
                    sender_bot_id=bot.id,
                    sender_name=bot.name,
                )
                ctx._mirror_group_message(session, conv=conv, msg=tool_result_msg)
            if isinstance(tool_result, dict):
                meta_current = tool_result.get("metadata") or meta_current

            if tool_failed:
                break

            candidate = ""
            if tool_name != "set_metadata" and tool_cfg:
                static_text = ""
                if (tool_cfg.static_reply_template or "").strip():
                    static_text = ctx._render_static_reply(
                        template_text=tool_cfg.static_reply_template,
                        meta=meta_current,
                        response_json=response_json,
                        tool_args=patch,
                    ).strip()
                if static_text:
                    needs_followup_llm = False
                    final = static_text
                else:
                    if bool(getattr(tool_cfg, "use_codex_response", False)):
                        if bool(getattr(tool_cfg, "return_result_directly", False)) and isinstance(tool_result, dict):
                            direct = str(tool_result.get("codex_result_text") or "").strip()
                            if direct:
                                needs_followup_llm = False
                                final = direct
                            else:
                                needs_followup_llm = True
                                final = ""
                        else:
                            needs_followup_llm = True
                            final = ""
                    else:
                        needs_followup_llm = ctx._should_followup_llm_for_tool(
                            tool=tool_cfg, static_rendered=static_text
                        )
                        candidate = ctx._render_with_meta(next_reply, meta_current).strip()
                        if candidate:
                            final = candidate
                            needs_followup_llm = False
                        else:
                            final = ""
            else:
                candidate = ctx._render_with_meta(next_reply, meta_current).strip()
                final = candidate or final

        if needs_followup_llm:
            followup_history = await ctx._build_history_budgeted_async(
                bot_id=bot.id,
                conversation_id=conversation_id,
                llm_api_key=api_key,
                status_cb=None,
            )
            followup_history.append(
                Message(
                    role="system",
                    content=(
                        ("The previous tool call failed. " if tool_failed else "")
                        + "Using the latest tool result(s) above, write the next assistant reply. "
                        "If the tool result contains codex_result_text, rephrase it for the user and do not mention file paths. "
                        "Do not call any tools."
                    ),
                )
            )
            text2 = ""
            async for d in ctx._aiter_from_blocking_iterator(lambda: llm.stream_text(messages=followup_history)):
                if d:
                    text2 += str(d)
            rendered_reply = text2.strip()
            followup_streamed = True
            llm_ttfb_ms = None
            llm_total_ms = None
        else:
            rendered_reply = final

    _ = followup_streamed
    return rendered_reply, llm_ttfb_ms, llm_total_ms
