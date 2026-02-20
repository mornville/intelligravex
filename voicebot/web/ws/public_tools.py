from __future__ import annotations

def bind_ctx(ctx):
    globals().update(ctx.__dict__)

async def process_public_tool_calls(
    *,
    session,
    ws,
    req_id: str,
    bot,
    conv_id,
    tool_calls,
    rendered_reply: str,
    llm_ttfb_ms,
    llm_total_ms,
    citations,
    citations_json: str,
    llm,
    llm_api_key: str,
    provider: str,
    history,
):
    conv = get_conversation(session, conv_id) if conv_id else None
    meta_current = _get_conversation_meta(session, conversation_id=conv_id)
    disabled_tools = _disabled_tool_names(bot)
    final = ""
    needs_followup_llm = False
    tool_failed = False
    followup_streamed = False

    for tc in tool_calls:
        tool_name = tc.name
        if tool_name == "set_variable":
            tool_name = "set_metadata"
        skip_tool_result_persist = False

        tool_args = json.loads(tc.arguments_json or "{}")
        if not isinstance(tool_args, dict):
            tool_args = {}

        add_message_with_metrics(
            session,
            conversation_id=conv_id,
            role="tool",
            content=json.dumps({"tool": tool_name, "arguments": tool_args}, ensure_ascii=False),
        )

        next_reply = str(tool_args.get("next_reply") or "").strip()
        wait_reply = str(tool_args.get("wait_reply") or "").strip()
        follow_up = _parse_follow_up_flag(tool_args.get("follow_up")) or _parse_follow_up_flag(
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
        elif tool_name in {"request_host_action", "capture_screenshot"} and not conv:
            tool_result = {
                "ok": False,
                "error": {"message": "Conversation not found."},
            }
            tool_failed = True
            needs_followup_llm = True
            final = ""
        elif tool_name == "set_metadata":
            new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=patch)
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
                _execute_http_request_tool, tool_args=patch, meta=meta_current
            )
            tool_failed = not bool(tool_result.get("ok", False))
            needs_followup_llm = True
            final = ""
        elif tool_name == "capture_screenshot":
            if not bool(getattr(bot, "enable_host_actions", False)):
                tool_result = {
                    "ok": False,
                    "error": {"message": "Host actions are disabled for this bot."},
                }
                tool_failed = True
                needs_followup_llm = True
                final = ""
            elif not bool(getattr(bot, "enable_host_shell", False)):
                tool_result = {
                    "ok": False,
                    "error": {"message": "Shell commands are disabled for this bot."},
                }
                tool_failed = True
                needs_followup_llm = True
                final = ""
            else:
                try:
                    rel_path, target = _prepare_screenshot_target(conv)
                except Exception as exc:
                    tool_result = {
                        "ok": False,
                        "error": {"message": str(exc) or "Invalid screenshot path"},
                    }
                    tool_failed = True
                    needs_followup_llm = True
                    final = ""
                else:
                    ok_cmd, cmd_or_err = _screencapture_command(target)
                    if not ok_cmd:
                        tool_result = {"ok": False, "error": {"message": cmd_or_err}}
                        tool_failed = True
                        needs_followup_llm = True
                        final = ""
                    else:
                        action = _create_host_action(
                            session,
                            conv=conv,
                            bot=bot,
                            action_type="run_shell",
                            payload={"command": cmd_or_err},
                        )
                        if _host_action_requires_approval(bot):
                            tool_result = _build_host_action_tool_result(action, ok=True)
                            tool_result["path"] = rel_path
                            if follow_up:
                                final = ""
                                needs_followup_llm = True
                            else:
                                candidate = _render_with_meta(next_reply, meta_current).strip()
                                if candidate:
                                    final = candidate
                                    needs_followup_llm = False
                                else:
                                    final = (
                                        "Approve the screenshot capture in the Action Queue, then ask me to analyze it."
                                    )
                                    needs_followup_llm = False
                        else:
                            tool_result = await _execute_host_action_and_update_async(
                                session, action=action
                            )
                            tool_failed = not bool(tool_result.get("ok", False))
                            if tool_failed:
                                needs_followup_llm = True
                                final = ""
                            else:
                                ok, summary_text = _summarize_image_file(
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
                                    if follow_up:
                                        final = ""
                                        needs_followup_llm = True
                                    else:
                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                        if candidate:
                                            final = candidate
                                            needs_followup_llm = False
                                        else:
                                            final = summary_text
                                            needs_followup_llm = False
        elif tool_name == "summarize_screenshot":
            ok, summary_text, rel_path = _summarize_screenshot(
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
                candidate = _render_with_meta(next_reply, meta_current).strip()
                if candidate:
                    final = candidate
                    needs_followup_llm = False
                else:
                    final = summary_text
                    needs_followup_llm = False
        elif tool_name == "request_host_action":
            if not bool(getattr(bot, "enable_host_actions", False)):
                tool_result = {
                    "ok": False,
                    "error": {"message": "Host actions are disabled for this bot."},
                }
                tool_failed = True
                needs_followup_llm = True
                final = ""
            else:
                try:
                    action_type, payload = _parse_host_action_args(patch)
                except Exception as exc:
                    tool_result = {"ok": False, "error": {"message": str(exc) or "Invalid host action"}}
                    tool_failed = True
                    needs_followup_llm = True
                    final = ""
                else:
                    if action_type == "run_shell" and not bool(getattr(bot, "enable_host_shell", False)):
                        tool_result = {
                            "ok": False,
                            "error": {"message": "Shell commands are disabled for this bot."},
                        }
                        tool_failed = True
                        needs_followup_llm = True
                        final = ""
                    else:
                        action = _create_host_action(
                            session,
                            conv=conv,
                            bot=bot,
                            action_type=action_type,
                            payload=payload,
                        )
                        if _host_action_requires_approval(bot):
                            tool_result = _build_host_action_tool_result(action, ok=True)
                            if follow_up:
                                final = ""
                                needs_followup_llm = True
                            else:
                                candidate = _render_with_meta(next_reply, meta_current).strip()
                                if candidate:
                                    final = candidate
                                    needs_followup_llm = False
                                else:
                                    needs_followup_llm = True
                                    final = ""
                        else:
                            tool_result = await _execute_host_action_and_update_async(
                                session, action=action
                            )
                            tool_failed = not bool(tool_result.get("ok", False))
                            candidate = _render_with_meta(next_reply, meta_current).strip()
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
            what_to_do = str(patch.get("what_to_do") or "").strip()
            if not what_to_do:
                tool_result = {"ok": False, "error": {"message": "Missing required tool arg: what_to_do"}}
                tool_failed = True
                needs_followup_llm = True
                final = ""
            else:
                data_task = asyncio.create_task(
                    _run_data_agent_tool_persist(
                        conversation_id=conv_id,
                        bot_id=bot.id,
                        what_to_do=what_to_do,
                        req_id=req_id,
                        ws=ws,
                        wait_reply=wait_reply,
                        send_wait=True,
                        send_wait_cb=lambda t: _public_send_interim(
                            ws, req_id=req_id, kind="wait", text=t
                        ),
                        stream_followup=True,
                    )
                )
                outcome = await asyncio.shield(data_task)
                tool_result = outcome.get("tool_result") or {}
                tool_failed = bool(outcome.get("tool_failed"))
                final = str(outcome.get("rendered_reply") or "")
                needs_followup_llm = False
                if outcome.get("followup_streamed"):
                    followup_streamed = True
                if outcome.get("assistant_persisted"):
                    followup_persisted = True
                skip_tool_result_persist = bool(outcome.get("tool_result_persisted"))
        else:
            tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
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
                        _execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                    )
                )
                if wait_reply:
                    await _public_send_interim(ws, req_id=req_id, kind="wait", text=wait_reply)
                while True:
                    try:
                        response_json = await asyncio.wait_for(asyncio.shield(task), timeout=60.0)
                        break
                    except asyncio.TimeoutError:
                        if wait_reply:
                            await _public_send_interim(
                                ws, req_id=req_id, kind="wait", text=wait_reply
                            )
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
                    # Avoid bloating LLM-visible conversation metadata in Codex mode.
                    tool_result = {"ok": True}
                    new_meta = meta_current
                else:
                    mapped = _apply_response_mapper(
                        mapper_json=tool_cfg.response_mapper_json,
                        response_json=response_json,
                        meta=meta_current,
                        tool_args=patch,
                    )
                    new_meta = merge_conversation_metadata(
                        session, conversation_id=conv_id, patch=mapped
                    )
                    meta_current = new_meta
                    tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}
                if pagination_info:
                    tool_result["pagination"] = pagination_info

                # Optional Codex HTTP agent (post-process the raw response JSON).
                # Static reply (if configured) takes priority.
                static_preview = ""
                if (tool_cfg.static_reply_template or "").strip():
                    try:
                        static_preview = _render_static_reply(
                            template_text=tool_cfg.static_reply_template,
                            meta=new_meta or meta_current,
                            response_json=response_json,
                            tool_args=patch,
                        ).strip()
                    except Exception:
                        static_preview = ""
                if (not static_preview) and bool(getattr(tool_cfg, "use_codex_response", False)):
                    fields_required = str(patch.get("fields_required") or "").strip()
                    what_to_search_for = str(patch.get("what_to_search_for") or "").strip()
                    if not fields_required:
                        fields_required = what_to_search_for
                    why_api_was_called = str(patch.get("why_api_was_called") or "").strip()
                    if not why_api_was_called:
                        why_api_was_called = str(patch.get("why_to_search_for") or "").strip()
                    if not fields_required or not why_api_was_called:
                        tool_result["codex_ok"] = False
                        tool_result["codex_error"] = "Missing fields_required / why_api_was_called."
                    else:
                        fields_required_for_codex = fields_required
                        if what_to_search_for:
                            fields_required_for_codex = (
                                f"{fields_required_for_codex}\n\n(what_to_search_for) {what_to_search_for}"
                            )
                        did_postprocess = False
                        postprocess_python = str(getattr(tool_cfg, "postprocess_python", "") or "").strip()
                        if postprocess_python:
                            py_task = asyncio.create_task(
                                asyncio.to_thread(
                                    run_python_postprocessor,
                                    python_code=postprocess_python,
                                    payload={
                                        "response_json": response_json,
                                        "meta": new_meta or meta_current,
                                        "args": patch,
                                        "fields_required": fields_required,
                                        "why_api_was_called": why_api_was_called,
                                    },
                                    timeout_s=60,
                                )
                            )
                            if wait_reply:
                                await _public_send_interim(
                                    ws, req_id=req_id, kind="wait", text=wait_reply
                                )
                            last_wait = time.time()
                            wait_interval_s = 15.0
                            while not py_task.done():
                                now = time.time()
                                if wait_reply and (now - last_wait) >= wait_interval_s:
                                    await _public_send_interim(
                                        ws, req_id=req_id, kind="wait", text=wait_reply
                                    )
                                    last_wait = now
                                await asyncio.sleep(0.2)
                            try:
                                py_res = await py_task
                                tool_result["python_ok"] = bool(getattr(py_res, "ok", False))
                                tool_result["python_duration_ms"] = int(
                                    getattr(py_res, "duration_ms", 0) or 0
                                )
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
                                            meta_current = merge_conversation_metadata(
                                                session,
                                                conversation_id=conv_id,
                                                patch=mp,
                                            )
                                            tool_result["python_metadata_patch"] = mp
                                        except Exception:
                                            pass
                                    try:
                                        append_saved_run_index(
                                            conversation_id=str(conv_id),
                                            event={
                                                "kind": "integration_python_postprocess",
                                                "tool_name": tool_name,
                                                "req_id": req_id,
                                                "python_ok": tool_result.get("python_ok"),
                                                "python_duration_ms": tool_result.get(
                                                    "python_duration_ms"
                                                ),
                                            },
                                        )
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
                            progress_q: "queue.Queue[str]" = queue.Queue()

                            def _progress(s: str) -> None:
                                try:
                                    progress_q.put_nowait(str(s))
                                except Exception:
                                    return

                            agent_task = asyncio.create_task(
                                asyncio.to_thread(
                                    run_codex_http_agent_one_shot,
                                    api_key=api_key or "",
                                    model=codex_model,
                                    response_json=response_json,
                                    fields_required=fields_required_for_codex,
                                    why_api_was_called=why_api_was_called,
                                    response_schema_json=getattr(tool_cfg, "response_schema_json", "")
                                    or "",
                                    conversation_id=str(conv_id) if conv_id is not None else None,
                                    req_id=req_id,
                                    tool_codex_prompt=getattr(tool_cfg, "codex_prompt", "") or "",
                                    progress_fn=_progress,
                                )
                            )
                            if wait_reply:
                                await _public_send_interim(
                                    ws, req_id=req_id, kind="wait", text=wait_reply
                                )
                            last_wait = time.time()
                            last_progress = last_wait
                            wait_interval_s = 15.0
                            while not agent_task.done():
                                try:
                                    while True:
                                        p = progress_q.get_nowait()
                                        if p:
                                            await _public_send_interim(
                                                ws, req_id=req_id, kind="progress", text=p
                                            )
                                            last_progress = time.time()
                                except queue.Empty:
                                    pass
                                now = time.time()
                                if (
                                    wait_reply
                                    and (now - last_wait) >= wait_interval_s
                                    and (now - last_progress) >= wait_interval_s
                                ):
                                    await _public_send_interim(
                                        ws, req_id=req_id, kind="wait", text=wait_reply
                                    )
                                    last_wait = now
                                await asyncio.sleep(0.2)
                            try:
                                agent_res = await agent_task
                                tool_result["postprocess_mode"] = "codex"
                                tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                tool_result["codex_output_file"] = getattr(
                                    agent_res, "result_text_path", ""
                                )
                                tool_result["codex_debug_file"] = getattr(agent_res, "debug_json_path", "")
                                tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                tool_result["codex_continue_reason"] = getattr(
                                    agent_res, "continue_reason", ""
                                )
                                tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                saved_input_json_path = getattr(agent_res, "input_json_path", "")
                                saved_schema_json_path = getattr(agent_res, "schema_json_path", "")
                                err = getattr(agent_res, "error", None)
                                if err:
                                    tool_result["codex_error"] = str(err)
                            except Exception as exc:
                                tool_result["codex_ok"] = False
                                tool_result["codex_error"] = str(exc)
                                saved_input_json_path = ""
                                saved_schema_json_path = ""

                            try:
                                append_saved_run_index(
                                    conversation_id=str(conv_id),
                                    event={
                                        "kind": "integration_http",
                                        "tool_name": tool_name,
                                        "req_id": req_id,
                                        "input_json_path": saved_input_json_path,
                                        "schema_json_path": saved_schema_json_path,
                                        "fields_required": fields_required,
                                        "why_api_was_called": why_api_was_called,
                                        "codex_output_dir": tool_result.get("codex_output_dir"),
                                        "codex_ok": tool_result.get("codex_ok"),
                                    },
                                )
                            except Exception:
                                pass

        if not skip_tool_result_persist:
            add_message_with_metrics(
                session,
                conversation_id=conv_id,
                role="tool",
                content=json.dumps({"tool": tool_name, "result": tool_result}, ensure_ascii=False),
            )
        if isinstance(tool_result, dict):
            meta_current = tool_result.get("metadata") or meta_current

        if tool_failed:
            break

        candidate = ""
        if tool_name != "set_metadata" and tool_cfg:
            static_text = ""
            if (tool_cfg.static_reply_template or "").strip():
                static_text = _render_static_reply(
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
                    if bool(getattr(tool_cfg, "return_result_directly", False)) and isinstance(
                        tool_result, dict
                    ):
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
                    needs_followup_llm = _should_followup_llm_for_tool(
                        tool=tool_cfg, static_rendered=static_text
                    )
                    candidate = _render_with_meta(next_reply, meta_current).strip()
                    if candidate:
                        final = candidate
                        needs_followup_llm = False
                    else:
                        final = ""
        else:
            candidate = _render_with_meta(next_reply, meta_current).strip()
            final = candidate or final

    if needs_followup_llm:
        followup_history = await _build_history_budgeted_async(
            bot_id=bot.id,
            conversation_id=conv_id,
            llm_api_key=llm_api_key,
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
        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
        text2, ttfb2, total2 = await _stream_llm_reply(
            ws=ws, req_id=req_id, llm=llm, messages=followup_history
        )
        rendered_reply = text2.strip()
        followup_streamed = True
        llm_ttfb_ms = ttfb2
        llm_total_ms = total2
    else:
        rendered_reply = final

    if rendered_reply and not followup_streamed:
        await _ws_send_json(
            ws, {"type": "text_delta", "req_id": req_id, "delta": rendered_reply}
        )

    return rendered_reply, llm_ttfb_ms, llm_total_ms
