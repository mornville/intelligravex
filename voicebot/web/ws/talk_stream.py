from __future__ import annotations


def bind_ctx(ctx):
    globals().update(ctx.__dict__)


async def run_llm_stream_with_tts(
    *,
    ws,
    req_id: str,
    llm,
    history,
    tools_defs,
    speak: bool,
    tts_synth,
    bot,
    openai_api_key: str,
    status_cb,
):
    llm_start_ts = time.time()

    delta_q_client: "queue.Queue[Optional[str]]" = queue.Queue()
    delta_q_tts: "queue.Queue[Optional[str]]" = queue.Queue()
    audio_q: "queue.Queue[Optional[tuple[bytes, int]]]" = queue.Queue()
    tool_calls: list[ToolCall] = []
    error_q: "queue.Queue[Optional[str]]" = queue.Queue()
    full_text_parts: list[str] = []
    citations_collected: list[dict[str, Any]] = []
    metrics_lock = threading.Lock()
    citations_lock = threading.Lock()
    first_token_ts: Optional[float] = None
    tts_start_ts: Optional[float] = None
    first_audio_ts: Optional[float] = None

    def llm_thread() -> None:
        try:
            for ev in llm.stream_text_or_tool(messages=history, tools=tools_defs):
                if isinstance(ev, ToolCall):
                    tool_calls.append(ev)
                    continue
                if isinstance(ev, CitationEvent):
                    with citations_lock:
                        citations_collected.extend(ev.citations)
                    continue
                d = ev
                full_text_parts.append(d)
                delta_q_client.put(d)
                if speak:
                    delta_q_tts.put(d)
        except Exception as exc:
            error_q.put(str(exc))
        finally:
            delta_q_client.put(None)
            if speak:
                delta_q_tts.put(None)

    def tts_thread() -> None:
        nonlocal tts_start_ts
        if not speak:
            audio_q.put(None)
            return
        try:
            synth = tts_synth or _get_tts_synth_fn(bot, openai_api_key)
            for text_to_speak in _iter_tts_chunks(delta_q_tts):
                if not text_to_speak:
                    continue
                with metrics_lock:
                    if tts_start_ts is None:
                        tts_start_ts = time.time()
                status_cb(req_id, "tts")
                wav, sr = synth(text_to_speak)
                audio_q.put((wav, sr))
        except Exception as exc:
            error_q.put(f"TTS failed: {exc}")
        finally:
            audio_q.put(None)

    t1 = threading.Thread(target=llm_thread, daemon=True)
    t2 = threading.Thread(target=tts_thread, daemon=True)
    t1.start()
    t2.start()

    open_deltas = True
    open_audio = True
    while open_deltas or open_audio:
        try:
            err = error_q.get_nowait()
            if err:
                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": err})
                open_deltas = False
                open_audio = False
                break
        except queue.Empty:
            pass

        try:
            d = delta_q_client.get_nowait()
            if d is None:
                open_deltas = False
            else:
                if first_token_ts is None:
                    first_token_ts = time.time()
                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
        except queue.Empty:
            pass

        if speak:
            try:
                item = audio_q.get_nowait()
                if item is None:
                    open_audio = False
                else:
                    wav, sr = item
                    if first_audio_ts is None:
                        first_audio_ts = time.time()
                    await _ws_send_json(
                        ws,
                        {
                            "type": "audio_wav",
                            "req_id": req_id,
                            "wav_base64": base64.b64encode(wav).decode(),
                            "sr": sr,
                        },
                    )
            except queue.Empty:
                pass
        else:
            open_audio = False

        if (open_deltas or open_audio) and first_token_ts is None:
            await asyncio.sleep(0.01)
        else:
            await asyncio.sleep(0.005)

    t1.join()
    t2.join()

    llm_end_ts = time.time()
    final_text = "".join(full_text_parts).strip()
    with citations_lock:
        citations = list(citations_collected)

    timings: dict[str, int] = {"total": int(round((llm_end_ts - llm_start_ts) * 1000.0))}
    if first_token_ts is not None:
        timings["llm_ttfb"] = int(round((first_token_ts - llm_start_ts) * 1000.0))
    elif tool_calls and tool_calls[0].first_event_ts is not None:
        timings["llm_ttfb"] = int(round((tool_calls[0].first_event_ts - llm_start_ts) * 1000.0))
    timings["llm_total"] = int(round((llm_end_ts - llm_start_ts) * 1000.0))
    if first_audio_ts is not None and tts_start_ts is not None:
        timings["tts_first_audio"] = int(round((first_audio_ts - tts_start_ts) * 1000.0))

    await _ws_send_json(ws, {"type": "metrics", "req_id": req_id, "timings_ms": timings})

    return final_text, tool_calls, citations, timings
