from __future__ import annotations

from voicebot.web.ws import public_tools as public_tools_module

def bind_ctx(ctx):
    globals().update(ctx.__dict__)

async def public_chat_ws(bot_id: UUID, ws: WebSocket) -> None:
    if not _basic_auth_ok(_ws_auth_header(ws)):
        await ws.accept()
        await _ws_send_json(ws, {"type": "error", "error": "Unauthorized"})
        await ws.close(code=4401)
        return
    await ws.accept()

    key_secret = (ws.query_params.get("key") or "").strip()
    external_id = (ws.query_params.get("user_conversation_id") or "").strip()
    if not key_secret or not external_id:
        await _ws_send_json(ws, {"type": "error", "error": "Missing key or user_conversation_id"})
        await ws.close(code=4400)
        return

    origin = ws.headers.get("origin")
    conv_id: Optional[UUID] = None
    with Session(engine) as session:
        ck = verify_client_key(session, secret=key_secret)
        if not ck:
            await _ws_send_json(ws, {"type": "error", "error": "Invalid client key"})
            await ws.close(code=4401)
            return
        if not _origin_allowed(ck, origin):
            await _ws_send_json(ws, {"type": "error", "error": "Origin not allowed"})
            await ws.close(code=4403)
            return
        if not _bot_allowed(ck, bot_id):
            await _ws_send_json(ws, {"type": "error", "error": "Bot not allowed for this key"})
            await ws.close(code=4403)
            return
        # Create (or load) the conversation immediately on connect so we can prewarm the Isolated Workspace
        # as soon as the conversation exists (before the first user message).
        try:
            bot = get_bot(session, bot_id)
            conv = get_or_create_conversation_by_external_id(
                session,
                bot_id=bot.id,
                test_flag=False,
                client_key_id=ck.id,
                external_id=external_id,
            )
            conv_id = conv.id
        except Exception:
            conv_id = None

    if conv_id is not None:
        asyncio.create_task(_kickoff_data_agent_container_if_enabled(bot_id=bot_id, conversation_id=conv_id))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                payload = json.loads(raw)
            except Exception:
                await _ws_send_json(ws, {"type": "error", "error": "Invalid JSON"})
                continue

            msg_type = str(payload.get("type") or "")
            req_id = str(payload.get("req_id") or "")
            if not req_id:
                await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                continue

            if msg_type == "start":
                with Session(engine) as session:
                    bot = get_bot(session, bot_id)
                    ck = verify_client_key(session, secret=key_secret)
                    if not ck:
                        raise HTTPException(status_code=401, detail="Invalid client key")
                    conv = get_or_create_conversation_by_external_id(
                        session, bot_id=bot.id, test_flag=False, client_key_id=ck.id, external_id=external_id
                    )
                    conv_id = conv.id

                    await _ws_send_json(
                        ws,
                        {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)},
                    )
                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})

                    if len(list_messages(session, conversation_id=conv_id)) == 0:
                        need_llm = not (
                            bot.start_message_mode == "static" and (bot.start_message_text or "").strip()
                        )
                        provider = _llm_provider_for_bot(bot)
                        llm_api_key = ""
                        if need_llm:
                            provider, llm_api_key, _ = _require_llm_client(session, bot=bot)
                        text, metrics = await _public_send_greeting(
                            ws=ws,
                            req_id=req_id,
                            bot=bot,
                            conv_id=conv_id,
                            provider=provider,
                            llm_api_key=llm_api_key,
                        )
                        await _public_send_done(ws, req_id=req_id, text=text, metrics=metrics)
                    else:
                        await _public_send_done(ws, req_id=req_id, text="", metrics={})

                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                continue

            if msg_type == "chat":
                user_text = str(payload.get("text") or "").strip()
                if not user_text:
                    await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": "Empty text"})
                    continue

                with Session(engine) as session:
                    bot = get_bot(session, bot_id)
                    ck = verify_client_key(session, secret=key_secret)
                    if not ck:
                        raise HTTPException(status_code=401, detail="Invalid client key")
                    conv = get_or_create_conversation_by_external_id(
                        session, bot_id=bot.id, test_flag=False, client_key_id=ck.id, external_id=external_id
                    )
                    conv_id = conv.id
                    await _ws_send_json(
                        ws, {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)}
                    )

                    provider, llm_api_key, llm = _require_llm_client(session, bot=bot)

                    add_message_with_metrics(session, conversation_id=conv_id, role="user", content=user_text)
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

                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                    t0 = time.time()
                    first_token_ts: Optional[float] = None
                    full_text_parts: list[str] = []
                    tool_calls: list[ToolCall] = []
                    citations: list[dict[str, Any]] = []

                    async for ev in _aiter_from_blocking_iterator(
                        lambda: llm.stream_text_or_tool(messages=history, tools=tools_defs)
                    ):
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
                            await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})

                    llm_end_ts = time.time()
                    rendered_reply = "".join(full_text_parts).strip()
                    citations_json = json.dumps(citations, ensure_ascii=False) if citations else "[]"

                    llm_ttfb_ms: Optional[int] = None
                    if first_token_ts is not None:
                        llm_ttfb_ms = int(round((first_token_ts - t0) * 1000.0))
                    elif tool_calls and tool_calls[0].first_event_ts is not None:
                        llm_ttfb_ms = int(round((tool_calls[0].first_event_ts - t0) * 1000.0))
                    llm_total_ms = int(round((llm_end_ts - t0) * 1000.0))

                    if tool_calls:
                        rendered_reply, llm_ttfb_ms, llm_total_ms = await public_tools_module.process_public_tool_calls(
                            session=session,
                            ws=ws,
                            req_id=req_id,
                            bot=bot,
                            conv_id=conv_id,
                            tool_calls=tool_calls,
                            rendered_reply=rendered_reply,
                            llm_ttfb_ms=llm_ttfb_ms,
                            llm_total_ms=llm_total_ms,
                            citations=citations,
                            citations_json=citations_json,
                            llm=llm,
                            llm_api_key=llm_api_key,
                            provider=provider,
                            history=history,
                        )
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
                        llm_ttfb_ms=llm_ttfb_ms,
                        llm_total_ms=llm_total_ms,
                        total_ms=llm_total_ms,
                        citations_json=citations_json,
                    )
                    update_conversation_metrics(
                        session,
                        conversation_id=conv_id,
                        add_input_tokens_est=in_tok,
                        add_output_tokens_est=out_tok,
                        add_cost_usd_est=cost,
                        last_asr_ms=None,
                        last_llm_ttfb_ms=llm_ttfb_ms,
                        last_llm_total_ms=llm_total_ms,
                        last_tts_first_audio_ms=None,
                        last_total_ms=llm_total_ms,
                    )

                    metrics = {
                        "model": bot.openai_model,
                        "input_tokens_est": in_tok,
                        "output_tokens_est": out_tok,
                        "cost_usd_est": cost,
                        "llm_ttfb_ms": llm_ttfb_ms,
                        "llm_total_ms": llm_total_ms,
                    }
                    await _public_send_done(
                        ws, req_id=req_id, text=rendered_reply, metrics=metrics, citations=citations
                    )
                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                continue

            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        return
    except RuntimeError:
        return
