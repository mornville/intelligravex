from __future__ import annotations

from voicebot.web.ws import talk_tool_integrations as integration_tools_module

def bind_ctx(ctx):
    globals().update(ctx.__dict__)

async def process_talk_tool_calls(
    *,
    ws,
    req_id: str,
    bot_id,
    conv_id,
    tool_calls,
    rendered_reply: str,
    speak: bool,
    tts_synth,
    status_cb,
    llm,
    llm_api_key: str,
    provider: str,
    history,
    citations,
    timings,
    debug_mode: bool = False,
):
    rendered_reply = ""
    llm_ttfb_ms = timings.get("llm_ttfb") if isinstance(timings, dict) else None
    llm_total_ms = timings.get("llm_total") if isinstance(timings, dict) else None
    tool_error: Optional[str] = None
    needs_followup_llm = False
    tool_failed = False
    followup_streamed = False
    followup_persisted = False
    tts_busy_until: float = 0.0
    last_wait_text: str | None = None
    last_wait_ts: float = 0.0
    wait_repeat_s: float = 45.0

    async def _send_interim(text: str, *, kind: str) -> None:
        nonlocal tts_busy_until, last_wait_text, last_wait_ts
        t = (text or "").strip()
        if not t:
            return
        if kind == "wait":
            now = time.time()
            if last_wait_text == t and (now - last_wait_ts) < wait_repeat_s:
                return
            last_wait_text = t
            last_wait_ts = now
        await _ws_send_json(
            ws,
            {"type": "interim", "req_id": req_id, "kind": kind, "text": t},
        )
        if not speak:
            return
        now = time.time()
        if now < tts_busy_until:
            await asyncio.sleep(tts_busy_until - now)
        if status_cb:
            status_cb(req_id, "tts")
        try:
            wav, sr = await asyncio.to_thread(tts_synth, t)
            tts_busy_until = time.time() + _estimate_wav_seconds(wav, sr) + 0.15
            await _ws_send_json(
                ws,
                {
                    "type": "audio_wav",
                    "req_id": req_id,
                    "wav_base64": base64.b64encode(wav).decode(),
                    "sr": sr,
                },
            )
        except Exception:
            return
    with Session(engine) as session:
        bot = get_bot(session, bot_id)
        conv = get_conversation(session, conv_id) if conv_id else None
        meta_current = _get_conversation_meta(session, conversation_id=conv_id)
        disabled_tools = _disabled_tool_names(bot)

        for tc in tool_calls:
            tool_name = tc.name
            if tool_name == "set_variable":
                tool_name = "set_metadata"
            skip_tool_result_persist = False
            suppress_tool_result = False

            await _ws_send_json(
                ws,
                {
                    "type": "tool_call",
                    "req_id": req_id,
                    "name": tool_name,
                    "arguments_json": tc.arguments_json,
                },
            )
            tool_parse_error = False
            try:
                tool_args = json.loads(tc.arguments_json or "{}")
                if not isinstance(tool_args, dict):
                    raise ValueError("tool args must be an object")
            except Exception as exc:
                tool_parse_error = True
                tool_error = str(exc) or "Invalid tool args"
                tool_args = {}

            add_message_with_metrics(
                session,
                conversation_id=conv_id,
                role="tool",
                content=json.dumps({"tool": tool_name, "arguments": tool_args}, ensure_ascii=False),
            )

            next_reply = str(tool_args.get("next_reply") or "").strip()
            wait_reply = str(tool_args.get("wait_reply") or "").strip() or "Working on it…"
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

            if tool_parse_error:
                tool_result = {
                    "ok": False,
                    "error": {"message": tool_error, "status_code": 500},
                }
                tool_failed = True
                needs_followup_llm = True
                rendered_reply = ""
            elif tool_name in {"request_host_action", "capture_screenshot", "summarize_screenshot"} and not conv:
                tool_result = {
                    "ok": False,
                    "error": {"message": "Conversation not found."},
                }
                tool_failed = True
                needs_followup_llm = True
                rendered_reply = ""
            elif tool_name in disabled_tools:
                tool_result = {
                    "ok": False,
                    "error": {"message": f"Tool '{tool_name}' is disabled for this bot."},
                }
                tool_failed = True
                needs_followup_llm = True
                rendered_reply = ""
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
                rendered_reply = ""
            elif tool_name == "http_request":
                task = asyncio.create_task(
                    asyncio.to_thread(
                        _execute_http_request_tool, tool_args=patch, meta=meta_current
                    )
                )
                if wait_reply:
                    await _send_interim(wait_reply, kind="wait")
                while True:
                    try:
                        tool_result = await asyncio.wait_for(
                            asyncio.shield(task), timeout=60.0
                        )
                        break
                    except asyncio.TimeoutError:
                        if wait_reply:
                            await _send_interim(wait_reply, kind="wait")
                        continue
                tool_failed = not bool(tool_result.get("ok", False))
                needs_followup_llm = True
                rendered_reply = ""
            elif tool_name == "capture_screenshot":
                if not bool(getattr(bot, "enable_host_actions", False)):
                    tool_result = {
                        "ok": False,
                        "error": {"message": "Host actions are disabled for this bot."},
                    }
                    tool_failed = True
                    needs_followup_llm = True
                    rendered_reply = ""
                elif not bool(getattr(bot, "enable_host_shell", False)):
                    tool_result = {
                        "ok": False,
                        "error": {"message": "Shell commands are disabled for this bot."},
                    }
                    tool_failed = True
                    needs_followup_llm = True
                    rendered_reply = ""
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
                        rendered_reply = ""
                    else:
                        ok_cmd, cmd_or_err = _screencapture_command(target)
                        if not ok_cmd:
                            tool_result = {"ok": False, "error": {"message": cmd_or_err}}
                            tool_failed = True
                            needs_followup_llm = True
                            rendered_reply = ""
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
                                suppress_tool_result = True
                                skip_tool_result_persist = True
                                rendered_reply = ""
                                needs_followup_llm = False
                            else:
                                if wait_reply:
                                    await _send_interim(wait_reply, kind="wait")
                                tool_result = await _execute_host_action_and_update_async(
                                    session, action=action
                                )
                                tool_failed = not bool(tool_result.get("ok", False))
                                if tool_failed:
                                    needs_followup_llm = True
                                    rendered_reply = ""
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
                                        rendered_reply = ""
                                    else:
                                        tool_result["summary"] = summary_text
                                        tool_result["path"] = rel_path
                                        if follow_up:
                                            rendered_reply = ""
                                            needs_followup_llm = True
                                        else:
                                            candidate = _render_with_meta(next_reply, meta_current).strip()
                                            if candidate:
                                                rendered_reply = candidate
                                                needs_followup_llm = False
                                            else:
                                                rendered_reply = summary_text
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
                    rendered_reply = ""
                else:
                    tool_result = {"ok": True, "summary": summary_text, "path": rel_path}
                    candidate = _render_with_meta(next_reply, meta_current).strip()
                    if candidate:
                        rendered_reply = candidate
                        needs_followup_llm = False
                    else:
                        rendered_reply = summary_text
                        needs_followup_llm = False
            elif tool_name == "request_host_action":
                if not bool(getattr(bot, "enable_host_actions", False)):
                    tool_result = {
                        "ok": False,
                        "error": {"message": "Host actions are disabled for this bot."},
                    }
                    tool_failed = True
                    needs_followup_llm = True
                    rendered_reply = ""
                else:
                    try:
                        action_type, payload = _parse_host_action_args(patch)
                    except Exception as exc:
                        tool_result = {"ok": False, "error": {"message": str(exc) or "Invalid host action"}}
                        tool_failed = True
                        needs_followup_llm = True
                        rendered_reply = ""
                    else:
                        if action_type == "run_shell" and not bool(getattr(bot, "enable_host_shell", False)):
                            tool_result = {
                                "ok": False,
                                "error": {"message": "Shell commands are disabled for this bot."},
                            }
                            tool_failed = True
                            needs_followup_llm = True
                            rendered_reply = ""
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
                                suppress_tool_result = True
                                skip_tool_result_persist = True
                                rendered_reply = ""
                                needs_followup_llm = False
                            else:
                                if wait_reply:
                                    await _send_interim(wait_reply, kind="wait")
                                tool_result = await _execute_host_action_and_update_async(
                                    session, action=action
                                )
                                tool_failed = not bool(tool_result.get("ok", False))
                                candidate = _render_with_meta(next_reply, meta_current).strip()
                                if follow_up and not tool_failed:
                                    needs_followup_llm = True
                                    rendered_reply = ""
                                elif candidate and not tool_failed:
                                    rendered_reply = candidate
                                    needs_followup_llm = False
                                else:
                                    needs_followup_llm = True
                                    rendered_reply = ""
            elif tool_name == "give_command_to_data_agent":
                what_to_do = str(patch.get("what_to_do") or "").strip()
                if not what_to_do:
                    tool_result = {
                        "ok": False,
                        "error": {"message": "Missing required tool arg: what_to_do"},
                    }
                    tool_failed = True
                    needs_followup_llm = True
                    rendered_reply = ""
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
                            send_wait_cb=lambda t: _send_interim(t, kind="wait"),
                            stream_followup=False,
                        )
                    )
                    outcome = await asyncio.shield(data_task)
                    tool_result = outcome.get("tool_result") or {}
                    tool_failed = bool(outcome.get("tool_failed"))
                    rendered_reply = str(outcome.get("rendered_reply") or "")
                    needs_followup_llm = False
                    if outcome.get("followup_streamed"):
                        followup_streamed = True
                    if outcome.get("assistant_persisted"):
                        followup_persisted = True
                    skip_tool_result_persist = bool(outcome.get("tool_result_persisted"))
            else:
                (
                    tool_cfg,
                    response_json,
                    tool_result,
                    tool_failed,
                    needs_followup_llm,
                    rendered_reply,
                    meta_current,
                ) = await integration_tools_module.handle_integration_tool(
                    session=session,
                    bot=bot,
                    conv_id=conv_id,
                    tool_name=tool_name,
                    patch=patch,
                    meta_current=meta_current,
                    wait_reply=wait_reply,
                    speak=speak,
                    tts_busy_until=tts_busy_until,
                    send_interim=_send_interim,
                    req_id=req_id,
                    api_key=llm_api_key,
                )
            if speak:
                now = time.time()
                if now < tts_busy_until:
                    await asyncio.sleep(tts_busy_until - now)

            if tool_name == "capture_screenshot" and tool_failed:
                msg = _tool_error_message(tool_result, fallback="Screenshot failed.")
                rendered_reply = f"Screenshot failed: {msg}"
                needs_followup_llm = False
            if tool_name == "request_host_action" and tool_failed:
                msg = _tool_error_message(tool_result, fallback="Host action failed.")
                rendered_reply = f"Host action failed: {msg}"
                needs_followup_llm = False

            if not skip_tool_result_persist:
                add_message_with_metrics(
                    session,
                    conversation_id=conv_id,
                    role="tool",
                    content=json.dumps({"tool": tool_name, "result": tool_result}, ensure_ascii=False),
                )
            if isinstance(tool_result, dict):
                meta_current = tool_result.get("metadata") or meta_current

            if not suppress_tool_result:
                await _ws_send_json(
                    ws,
                    {"type": "tool_result", "req_id": req_id, "name": tool_name, "result": tool_result},
                )

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
                    rendered_reply = static_text
                else:
                    if bool(getattr(tool_cfg, "use_codex_response", False)):
                        if bool(getattr(tool_cfg, "return_result_directly", False)) and isinstance(
                            tool_result, dict
                        ):
                            direct = str(tool_result.get("codex_result_text") or "").strip()
                            if direct:
                                needs_followup_llm = False
                                rendered_reply = direct
                            else:
                                needs_followup_llm = True
                                rendered_reply = ""
                        else:
                            needs_followup_llm = True
                            rendered_reply = ""
                    else:
                        needs_followup_llm = _should_followup_llm_for_tool(
                            tool=tool_cfg, static_rendered=static_text
                        )
                        candidate = _render_with_meta(next_reply, meta_current).strip()
                        if candidate:
                            rendered_reply = candidate
                            needs_followup_llm = False
                        else:
                            rendered_reply = ""
            else:
                candidate = _render_with_meta(next_reply, meta_current).strip()
                rendered_reply = candidate or rendered_reply

    # If static reply is missing/empty for an integration tool, ask the LLM again
    # with tool call + tool result already persisted in history.
    if needs_followup_llm and conv_id is not None:
        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
        with Session(engine) as session:
            followup_bot = get_bot(session, bot_id)
            followup_history = await _build_history_budgeted_async(
                bot_id=followup_bot.id,
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
        followup_model = followup_bot.openai_model
        follow_llm = _build_llm_client(
            bot=followup_bot,
            api_key=llm_api_key,
            model_override=followup_model,
        )
        if debug_mode:
            await _emit_llm_debug_payload(
                ws=ws,
                req_id=req_id,
                conversation_id=conv_id,
                phase="tool_followup_llm",
                payload=follow_llm.build_request_payload(messages=followup_history, stream=True),
            )
        text2, ttfb2, total2 = await _stream_llm_reply(
            ws=ws, req_id=req_id, llm=follow_llm, messages=followup_history
        )
        rendered_reply = text2.strip()
        llm_ttfb_ms = ttfb2
        llm_total_ms = total2
        if rendered_reply:
            followup_streamed = True
            in_tok = int(estimate_messages_tokens(followup_history, followup_model) or 0)
            out_tok = int(estimate_text_tokens(rendered_reply, followup_model) or 0)
            with Session(engine) as session:
                price = _get_model_price(session, provider=provider, model=followup_model)
            cost = float(
                estimate_cost_usd(
                    model_price=price,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                )
                or 0.0
            )
            with Session(engine) as session:
                add_message_with_metrics(
                    session,
                    conversation_id=conv_id,
                    role="assistant",
                    content=rendered_reply,
                    input_tokens_est=in_tok or None,
                    output_tokens_est=out_tok or None,
                    cost_usd_est=cost or None,
                    llm_ttfb_ms=ttfb2,
                    llm_total_ms=total2,
                    total_ms=total2,
                )
                update_conversation_metrics(
                    session,
                    conversation_id=conv_id,
                    add_input_tokens_est=in_tok,
                    add_output_tokens_est=out_tok,
                    add_cost_usd_est=cost,
                    last_asr_ms=None,
                    last_llm_ttfb_ms=ttfb2,
                    last_llm_total_ms=total2,
                    last_tts_first_audio_ms=None,
                    last_total_ms=total2,
                )
            followup_persisted = True
            await _ws_send_json(
                ws,
                {
                    "type": "metrics",
                    "req_id": req_id,
                    "timings_ms": {"llm_ttfb": ttfb2, "llm_total": total2, "total": total2},
                },
            )

    if tool_error:
        await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": tool_error})

    if rendered_reply:
        if not followup_streamed:
            await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": rendered_reply})
        try:
            if not followup_persisted:
                with Session(engine) as session:
                    in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                        session=session,
                        bot=bot,
                        provider=provider,
                        history=history,
                        assistant_text=rendered_reply,
                    )
                    add_message_with_metrics(
                        session,
                        conversation_id=conv_id,
                        role="assistant",
                        content=rendered_reply,
                        input_tokens_est=in_tok,
                        output_tokens_est=out_tok,
                        cost_usd_est=cost,
                        llm_ttfb_ms=timings.get("llm_ttfb"),
                        llm_total_ms=timings.get("llm_total"),
                        total_ms=timings.get("total"),
                    )
                    update_conversation_metrics(
                        session,
                        conversation_id=conv_id,
                        add_input_tokens_est=in_tok,
                        add_output_tokens_est=out_tok,
                        add_cost_usd_est=cost,
                        last_asr_ms=None,
                        last_llm_ttfb_ms=timings.get("llm_ttfb"),
                        last_llm_total_ms=timings.get("llm_total"),
                        last_tts_first_audio_ms=None,
                        last_total_ms=timings.get("total"),
                    )
        except Exception:
            pass

        if speak:
            if status_cb:
                status_cb(req_id, "tts")
            if tts_synth is None:
                tts_synth = _get_tts_synth_fn(bot, openai_api_key)
            wav, sr = await asyncio.to_thread(tts_synth, rendered_reply)
            await _ws_send_json(
                ws,
                {
                    "type": "audio_wav",
                    "req_id": req_id,
                    "wav_base64": base64.b64encode(wav).decode(),
                    "sr": sr,
                },
            )

    await _ws_send_json(
        ws,
        {"type": "done", "req_id": req_id, "text": rendered_reply, "citations": citations},
    )
    return rendered_reply, llm_ttfb_ms, llm_total_ms, followup_streamed, followup_persisted
