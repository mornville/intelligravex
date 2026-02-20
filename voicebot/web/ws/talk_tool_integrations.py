from __future__ import annotations

import asyncio
import queue
import time
from typing import Any


def bind_ctx(ctx):
    globals().update(ctx.__dict__)


async def handle_integration_tool(
    *,
    session,
    bot,
    conv_id,
    tool_name: str,
    patch: dict,
    meta_current: dict,
    wait_reply: str,
    speak: bool,
    tts_busy_until: float,
    send_interim,
    req_id: str,
    api_key: str,
):
    tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
    response_json: Any | None = None
    tool_result: dict[str, Any] = {}
    tool_failed = False
    needs_followup_llm = False
    rendered_reply = ""

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
            await send_interim(wait_reply, kind="wait")
        while True:
            try:
                response_json = await asyncio.wait_for(asyncio.shield(task), timeout=60.0)
                break
            except asyncio.TimeoutError:
                if wait_reply:
                    await send_interim(wait_reply, kind="wait")
                continue

    if speak:
        now = time.time()
        if now < tts_busy_until:
            await asyncio.sleep(tts_busy_until - now)

    if isinstance(response_json, dict) and "__tool_args_error__" in response_json:
        err = response_json["__tool_args_error__"] or {}
        tool_result = {"ok": False, "error": err}
        tool_failed = True
        needs_followup_llm = True
        rendered_reply = ""
    elif isinstance(response_json, dict) and "__http_error__" in response_json:
        err = response_json["__http_error__"] or {}
        tool_result = {"ok": False, "error": err}
        tool_failed = True
        needs_followup_llm = True
        rendered_reply = ""
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
            new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=mapped)
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
                        await send_interim(wait_reply, kind="wait")
                    last_wait = time.time()
                    wait_interval_s = 15.0
                    while not py_task.done():
                        now = time.time()
                        if wait_reply and (now - last_wait) >= wait_interval_s:
                            await send_interim(wait_reply, kind="wait")
                            last_wait = now
                        await asyncio.sleep(0.2)
                    try:
                        py_res = await py_task
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
                            tool_result["codex_result_text"] = str(getattr(py_res, "result_text", "") or "")
                            mp = getattr(py_res, "metadata_patch", None)
                            if isinstance(mp, dict) and mp:
                                try:
                                    meta_current = merge_conversation_metadata(
                                        session, conversation_id=conv_id, patch=mp
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
                                        "python_duration_ms": tool_result.get("python_duration_ms"),
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
                            response_schema_json=getattr(tool_cfg, "response_schema_json", "") or "",
                            conversation_id=str(conv_id) if conv_id is not None else None,
                            req_id=req_id,
                            tool_codex_prompt=getattr(tool_cfg, "codex_prompt", "") or "",
                            progress_fn=_progress,
                        )
                    )
                    if wait_reply:
                        await send_interim(wait_reply, kind="wait")
                    last_wait = time.time()
                    last_progress = last_wait
                    wait_interval_s = 15.0
                    while not agent_task.done():
                        try:
                            while True:
                                p = progress_q.get_nowait()
                                if p:
                                    await send_interim(p, kind="progress")
                                    last_progress = time.time()
                        except queue.Empty:
                            pass
                        now = time.time()
                        if (
                            wait_reply
                            and (now - last_wait) >= wait_interval_s
                            and (now - last_progress) >= wait_interval_s
                        ):
                            await send_interim(wait_reply, kind="wait")
                            last_wait = now
                        await asyncio.sleep(0.2)
                    try:
                        agent_res = await agent_task
                        tool_result["postprocess_mode"] = "codex"
                        tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                        tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                        tool_result["codex_output_file"] = getattr(agent_res, "result_text_path", "")
                        tool_result["codex_debug_file"] = getattr(agent_res, "debug_json_path", "")
                        tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                        tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                        tool_result["codex_continue_reason"] = getattr(agent_res, "continue_reason", "")
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

    return (
        tool_cfg,
        response_json,
        tool_result,
        tool_failed,
        needs_followup_llm,
        rendered_reply,
        meta_current,
    )
