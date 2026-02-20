from __future__ import annotations

from voicebot.web.ws import talk_tools as talk_tools_module
from voicebot.web.ws import talk_stream as talk_stream_module

def bind_ctx(ctx):
    globals().update(ctx.__dict__)

async def talk_ws(bot_id: UUID, ws: WebSocket) -> None:  # pyright: ignore[reportGeneralTypeIssues]
    if not _basic_auth_ok(_ws_auth_header(ws)):
        await ws.accept()
        await _ws_send_json(ws, {"type": "error", "error": "Unauthorized"})
        await ws.close(code=4401)
        return
    await ws.accept()
    loop = asyncio.get_running_loop()

    def status(req_id: str, stage: str) -> None:
        asyncio.run_coroutine_threadsafe(
            _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": stage}), loop
        )

    active_req_id: Optional[str] = None
    audio_buf = bytearray()
    conv_id: Optional[UUID] = None
    speak = True
    test_flag = True
    debug_mode = False
    stop_ts: Optional[float] = None
    accepting_audio = False
    tts_synth: Optional[Callable[[str], tuple[bytes, int]]] = None

    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    await _ws_send_json(ws, {"type": "error", "error": "Invalid JSON"})
                    continue

                msg_type = payload.get("type")
                req_id = str(payload.get("req_id") or "")
                if msg_type == "init":
                    if not req_id:
                        await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                        continue
                    if active_req_id is not None:
                        await _ws_send_json(
                            ws,
                            {
                                "type": "error",
                                "req_id": req_id,
                                "error": "Another request is already in progress",
                            },
                        )
                        continue
                    active_req_id = req_id
                    speak = bool(payload.get("speak", True))
                    test_flag = bool(payload.get("test_flag", True))
                    debug_mode = bool(payload.get("debug", False))
                    accepting_audio = False
                    conversation_id_str = str(payload.get("conversation_id") or "").strip()
                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "init"})
                    if conversation_id_str:
                        try:
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                conv_id = UUID(conversation_id_str)
                                conv = get_conversation(session, conv_id)
                                if conv.bot_id != bot.id:
                                    raise HTTPException(status_code=400, detail="Conversation does not belong to bot")
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue
                        await _ws_send_json(
                            ws,
                            {
                                "type": "conversation",
                                "req_id": req_id,
                                "conversation_id": str(conv_id),
                                "id": str(conv_id),
                            },
                        )
                        await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        accepting_audio = False
                        continue
                    try:
                        conv_id = await _init_conversation_and_greet(
                            bot_id=bot_id,
                            speak=speak,
                            test_flag=test_flag,
                            ws=ws,
                            req_id=req_id,
                            debug=debug_mode,
                        )
                    except Exception as exc:
                        await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        conv_id = None
                        continue
                    await _ws_send_json(
                        ws,
                        {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)},
                    )
                    await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                    active_req_id = None
                    accepting_audio = False
                    continue

                if msg_type == "start":
                    if not req_id:
                        await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                        continue
                    if active_req_id is not None:
                        await _ws_send_json(
                            ws,
                            {
                                "type": "error",
                                "req_id": req_id,
                                "error": "Another request is already in progress",
                            },
                        )
                        continue

                    active_req_id = req_id
                    audio_buf = bytearray()
                    debug_mode = bool(payload.get("debug", False))
                    speak = bool(payload.get("speak", True))
                    test_flag = bool(payload.get("test_flag", True))
                    accepting_audio = True

                    conversation_id_str = str(payload.get("conversation_id") or "").strip()

                    try:
                        with Session(engine) as session:
                            bot = get_bot(session, bot_id)
                            if conversation_id_str:
                                conv_id = UUID(conversation_id_str)
                                conv = get_conversation(session, conv_id)
                                if conv.bot_id != bot.id:
                                    raise HTTPException(
                                        status_code=400, detail="Conversation does not belong to bot"
                                    )
                            else:
                                conv = create_conversation(session, bot_id=bot.id, test_flag=test_flag)
                                conv_id = conv.id
                    except Exception as exc:
                        await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                        active_req_id = None
                        conv_id = None
                        continue

                    await _ws_send_json(
                        ws,
                        {
                            "type": "conversation",
                            "req_id": req_id,
                            "conversation_id": str(conv_id),
                            "id": str(conv_id),
                        },
                    )
                    asyncio.create_task(
                        _kickoff_data_agent_container_if_enabled(bot_id=bot_id, conversation_id=conv_id)
                    )
                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "recording"})

                elif msg_type == "chat":
                    # Text-only chat turn (for when Speak is disabled).
                    if not req_id:
                        await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                        continue
                    if active_req_id is not None:
                        await _ws_send_json(
                            ws,
                            {
                                "type": "error",
                                "req_id": req_id,
                                "error": "Another request is already in progress",
                            },
                        )
                        continue

                    active_req_id = req_id
                    speak = bool(payload.get("speak", True))
                    test_flag = bool(payload.get("test_flag", True))
                    debug_mode = bool(payload.get("debug", False))
                    user_text = str(payload.get("text") or "").strip()
                    conversation_id_str = str(payload.get("conversation_id") or "").strip()
                    accepting_audio = False
                    if not user_text:
                        await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": "Empty text"})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        continue

                    try:
                        with Session(engine) as session:
                            bot = get_bot(session, bot_id)
                            if conversation_id_str:
                                conv_id = UUID(conversation_id_str)
                                conv = get_conversation(session, conv_id)
                                if conv.bot_id != bot.id:
                                    raise HTTPException(status_code=400, detail="Conversation does not belong to bot")
                            else:
                                conv = create_conversation(session, bot_id=bot.id, test_flag=test_flag)
                                conv_id = conv.id
                    except Exception as exc:
                        await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        conv_id = None
                        continue

                    await _ws_send_json(
                        ws,
                        {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)},
                    )
                    asyncio.create_task(_kickoff_data_agent_container_if_enabled(bot_id=bot_id, conversation_id=conv_id))

                    if user_text.lstrip().startswith("!"):
                        cmd = user_text.lstrip()[1:].strip()
                        if not cmd:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": "Empty command"})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            continue
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                        try:
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                add_message_with_metrics(session, conversation_id=conv_id, role="user", content=user_text)
                                meta = _get_conversation_meta(session, conversation_id=conv_id)
                                da = _data_agent_meta(meta)
                                workspace_dir = (
                                    str(da.get("workspace_dir") or "").strip()
                                    or default_workspace_dir_for_conversation(conv_id)
                                )
                                container_id = str(da.get("container_id") or "").strip()
                                if not container_id:
                                    api_key = _get_openai_api_key_for_bot(session, bot=bot)
                                    if not api_key:
                                        raise RuntimeError(
                                            "No OpenAI API key configured for this bot (needed to start Isolated Workspace container)."
                                        )
                                    auth_json_raw = getattr(bot, "data_agent_auth_json", "") or "{}"
                                    git_token = (
                                        _get_git_token_plaintext(session, provider="github")
                                        if _git_auth_mode(auth_json_raw) == "token"
                                        else ""
                                    )
                                    container_id = ensure_conversation_container(
                                        conversation_id=conv_id,
                                        workspace_dir=workspace_dir,
                                        openai_api_key=api_key,
                                        git_token=git_token,
                                        auth_json=auth_json_raw,
                                    )
                                    merge_conversation_metadata(
                                        session,
                                        conversation_id=conv_id,
                                        patch={
                                            "data_agent.container_id": container_id,
                                            "data_agent.workspace_dir": workspace_dir,
                                        },
                                    )
                            res = await asyncio.to_thread(run_container_command, container_id=container_id, command=cmd)
                            out = res.stdout
                            if res.stderr:
                                out = (out + "\n" if out else "") + f"[stderr]\\n{res.stderr}"
                            if not out:
                                out = f"(exit {res.exit_code})"
                            if len(out) > 8000:
                                out = out[:8000] + "…"
                            with Session(engine) as session:
                                add_message_with_metrics(session, conversation_id=conv_id, role="assistant", content=out)
                            await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": out})
                            await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": out})
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        continue

                    try:
                        with Session(engine) as session:
                            bot = get_bot(session, bot_id)
                            provider, llm_api_key, llm = _require_llm_client(session, bot=bot)
                            openai_api_key: Optional[str] = None
                            if speak:
                                openai_api_key = _get_openai_api_key_for_bot(session, bot=bot)
                                if not openai_api_key:
                                    raise HTTPException(
                                        status_code=400,
                                        detail="No OpenAI key configured for this bot (needed for TTS).",
                                    )

                            add_message_with_metrics(
                                session,
                                conversation_id=conv_id,
                                role="user",
                                content=user_text,
                            )
                            loop = asyncio.get_running_loop()

                            def _status_cb(stage: str) -> None:
                                asyncio.run_coroutine_threadsafe(
                                    _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": stage}), loop
                                )

                            history = await _build_history_budgeted_async(
                                bot_id=bot.id,
                                conversation_id=conv_id,
                                llm_api_key=llm_api_key,
                                status_cb=_status_cb,
                            )
                            tools_defs = _build_tools_for_bot(session, bot.id)
                            if speak:
                                tts_synth = _get_tts_synth_fn(bot, openai_api_key)
                            if debug_mode:
                                await _emit_llm_debug_payload(
                                    ws=ws,
                                    req_id=req_id,
                                    conversation_id=conv_id,
                                    phase="chat_llm",
                                    payload=llm.build_request_payload(
                                        messages=history, tools=tools_defs, stream=True
                                    ),
                                )
                    except Exception as exc:
                        await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        conv_id = None
                        continue

                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                    final_text, tool_calls, citations, timings = await talk_stream_module.run_llm_stream_with_tts(
                        ws=ws,
                        req_id=req_id,
                        llm=llm,
                        history=history,
                        tools_defs=tools_defs,
                        speak=speak,
                        tts_synth=tts_synth,
                        bot=bot,
                        openai_api_key=openai_api_key,
                        status_cb=status,
                    )
                    citations_json = json.dumps(citations, ensure_ascii=False) if citations else "[]"
                    if tool_calls and conv_id is not None:
                        rendered_reply, llm_ttfb_ms, llm_total_ms, followup_streamed, followup_persisted = await talk_tools_module.process_talk_tool_calls(
                            ws=ws,
                            req_id=req_id,
                            bot_id=bot_id,
                            conv_id=conv_id,
                            tool_calls=tool_calls,
                            rendered_reply=final_text,
                            speak=speak,
                            tts_synth=tts_synth,
                            status_cb=status,
                            llm=llm,
                            llm_api_key=llm_api_key,
                            provider=provider,
                            history=history,
                            citations=citations,
                            timings=timings,
                            debug_mode=debug_mode,
                        )
                    else:
                        # Store assistant response.
                        try:
                            with Session(engine) as session:
                                in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                                    session=session,
                                    bot=bot,
                                    provider=provider,
                                    history=history,
                                    assistant_text=final_text,
                                )
                                add_message_with_metrics(
                                    session,
                                    conversation_id=conv_id,
                                    role="assistant",
                                    content=final_text,
                                    input_tokens_est=in_tok,
                                    output_tokens_est=out_tok,
                                    cost_usd_est=cost,
                                    llm_ttfb_ms=timings.get("llm_ttfb"),
                                    llm_total_ms=timings.get("llm_total"),
                                    tts_first_audio_ms=timings.get("tts_first_audio"),
                                    total_ms=timings.get("total"),
                                    citations_json=citations_json,
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
                                    last_tts_first_audio_ms=timings.get("tts_first_audio"),
                                    last_total_ms=timings.get("total"),
                                )
                        except Exception:
                            pass

                        await _ws_send_json(
                            ws,
                            {"type": "done", "req_id": req_id, "text": final_text, "citations": citations},
                        )

                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                    active_req_id = None
                    conv_id = None
                    accepting_audio = False
                    continue

                elif msg_type == "stop":
                    if not req_id or active_req_id != req_id:
                        await _ws_send_json(
                            ws, {"type": "error", "req_id": req_id or None, "error": "Unknown req_id"}
                        )
                        continue
                    accepting_audio = False
                    if not conv_id:
                        await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": "No conversation"})
                        active_req_id = None
                        continue

                    stop_ts = time.time()
                    if not audio_buf:
                        await _ws_send_json(ws, {"type": "asr", "req_id": req_id, "text": ""})
                        await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        conv_id = None
                        continue

                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "asr"})

                    asr_start_ts = time.time()
                    try:
                        with Session(engine) as session:
                            bot = get_bot(session, bot_id)
                            openai_api_key = _get_openai_api_key_for_bot(session, bot=bot)
                            if not openai_api_key:
                                await _ws_send_json(
                                    ws,
                                    {
                                        "type": "error",
                                        "req_id": req_id,
                                        "error": "No OpenAI key configured for this bot.",
                                    },
                                )
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                                active_req_id = None
                                conv_id = None
                                continue

                            pcm16 = bytes(audio_buf)

                            asr = await asyncio.to_thread(
                                _get_asr(openai_api_key, bot.openai_asr_model, bot.language).transcribe_pcm16,
                                pcm16=pcm16,
                                sample_rate=16000,
                            )
                            asr_end_ts = time.time()

                        user_text = (asr.text or "").strip()
                        await _ws_send_json(ws, {"type": "asr", "req_id": req_id, "text": user_text})
                        if not user_text:
                            await _ws_send_json(
                                ws,
                                {
                                    "type": "metrics",
                                    "req_id": req_id,
                                    "timings_ms": {
                                        "asr": int(round((asr_end_ts - asr_start_ts) * 1000.0)),
                                        "total": int(
                                            round((time.time() - (stop_ts or asr_start_ts)) * 1000.0)
                                        ),
                                    },
                                },
                            )
                            await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue

                        add_message_with_metrics(
                            session,
                            conversation_id=conv_id,
                            role="user",
                            content=user_text,
                            asr_ms=int(round((asr_end_ts - asr_start_ts) * 1000.0)),
                        )
                        loop = asyncio.get_running_loop()

                        def _status_cb(stage: str) -> None:
                            asyncio.run_coroutine_threadsafe(
                                _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": stage}), loop
                            )

                        provider, llm_api_key, llm = _require_llm_client(session, bot=bot)
                        history = await _build_history_budgeted_async(
                            bot_id=bot.id,
                            conversation_id=conv_id,
                            llm_api_key=llm_api_key,
                            status_cb=_status_cb,
                        )
                        tools_defs = _build_tools_for_bot(session, bot.id)
                        if debug_mode:
                            await _emit_llm_debug_payload(
                                ws=ws,
                                req_id=req_id,
                                conversation_id=conv_id,
                                phase="asr_turn_llm",
                                payload=llm.build_request_payload(messages=history, tools=tools_defs, stream=True),
                            )
                    except Exception as exc:
                        await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        conv_id = None
                        continue

                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                    final_text, tool_calls, citations, timings = await talk_stream_module.run_llm_stream_with_tts(
                        ws=ws,
                        req_id=req_id,
                        llm=llm,
                        history=history,
                        tools_defs=tools_defs,
                        speak=speak,
                        tts_synth=tts_synth,
                        bot=bot,
                        openai_api_key=openai_api_key,
                        status_cb=status,
                    )
                    citations_json = json.dumps(citations, ensure_ascii=False) if citations else "[]"
                    if tool_calls and conv_id is not None:
                        rendered_reply, llm_ttfb_ms, llm_total_ms, followup_streamed, followup_persisted = await talk_tools_module.process_talk_tool_calls(
                            ws=ws,
                            req_id=req_id,
                            bot_id=bot_id,
                            conv_id=conv_id,
                            tool_calls=tool_calls,
                            rendered_reply=final_text,
                            speak=speak,
                            tts_synth=tts_synth,
                            status_cb=status,
                            llm=llm,
                            llm_api_key=llm_api_key,
                            provider=provider,
                            history=history,
                            citations=citations,
                            timings=timings,
                            debug_mode=debug_mode,
                        )
                    else:
                        await _ws_send_json(
                            ws,
                            {"type": "done", "req_id": req_id, "text": final_text, "citations": citations},
                        )

                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})

                    active_req_id = None
                    conv_id = None
                    audio_buf = bytearray()
                    accepting_audio = False

                else:
                    await _ws_send_json(
                        ws,
                        {"type": "error", "req_id": req_id or None, "error": f"Unknown message type: {msg_type}"},
                    )

            elif "bytes" in msg and msg["bytes"] is not None:
                # Be tolerant to stray/late audio frames (browser worklet flush, etc.)
                if active_req_id is None or not accepting_audio:
                    continue
                audio_buf.extend(msg["bytes"])
            else:
                # ignore
                pass

    except WebSocketDisconnect:
        return
    except RuntimeError:
        # Starlette can raise RuntimeError if receive() is called after disconnect was already processed.
        return
    except Exception as exc:
        try:
            await _ws_send_json(ws, {"type": "error", "error": f"Server error: {exc}"})
        except Exception:
            pass
        return
