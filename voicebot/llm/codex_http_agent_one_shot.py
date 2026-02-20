from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Optional

from .codex_http_agent_helpers import (
    CodexAgentResult,
    NextStep,
    ProgressFn,
    _call_codex_json,
    _coerce_json_schema,
    _json_tree_schema,
    _parse_status_line,
    _run_python_script,
    _schema_matches_payload_top,
    _write_text,
)


def run_codex_http_agent_one_shot(
    *,
    api_key: str,
    model: str,
    response_json: Any,
    fields_required: str,
    why_api_was_called: str,
    response_schema_json: Any = None,
    conversation_id: str | None = None,
    req_id: str | None = None,
    tool_codex_prompt: str | None = None,
    progress_fn: Optional[ProgressFn] = None,
) -> CodexAgentResult:
    """
    Single-step Codex workflow:
    - Save HTTP response JSON to a local file.
    - Send ONLY schema + file paths + intent to Codex.
    - Codex returns a Python script that writes RESULT_TEXT_PATH (human readable) and prints __IGX_STATUS__.
    """
    m = (model or "").strip()
    if not m:
        raise ValueError("model is required")
    fields = (fields_required or "").strip()
    why_called = (why_api_was_called or "").strip()
    if not fields or not why_called:
        raise ValueError("fields_required and why_api_was_called are required")

    conv = (conversation_id or "").strip() or "no_conversation"
    rid = (req_id or "").strip() or str(int(time.time() * 1000))
    out_dir = os.path.join(tempfile.gettempdir(), "igx_codex_one_shot", conv, rid)
    os.makedirs(out_dir, exist_ok=True)

    debug_path = os.path.join(out_dir, "debug.json")
    input_path = os.path.join(out_dir, "input_response.json")
    schema_path = os.path.join(out_dir, "input_schema.json")
    result_text_path = os.path.join(out_dir, "result.txt")

    # Keep these for compatibility with existing tool result shape.
    filtered_path = os.path.join(out_dir, "filtered.json")
    validated_path = os.path.join(out_dir, "validated.json")

    debug_events: list[dict[str, Any]] = []

    def _debug(event: dict[str, Any]) -> None:
        try:
            debug_events.append(event)
            _write_text(debug_path, json.dumps(debug_events, ensure_ascii=False, indent=2))
        except Exception:
            return

    if progress_fn:
        progress_fn("Got results-organizing them…")
    _write_text(input_path, json.dumps(response_json, ensure_ascii=False, indent=2))

    provided_schema = _coerce_json_schema(response_schema_json)
    schema_source = "provided_json_schema" if provided_schema is not None else "derived_tree"
    if provided_schema is not None and _schema_matches_payload_top(provided_schema, response_json):
        schema_obj: Any = provided_schema
    else:
        if progress_fn:
            progress_fn("Making sure I capture the right details…")
        schema_obj = _json_tree_schema(response_json)
        schema_source = "derived_tree"
    _write_text(schema_path, json.dumps(schema_obj, ensure_ascii=False, indent=2))
    schema_inline = json.dumps(schema_obj, ensure_ascii=False)[:24000]
    _debug({"phase": "schema", "schema_source": schema_source})

    tool_prompt = (tool_codex_prompt or "").strip()
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a Python data extraction + summarization agent.\n"
                "Return ONLY JSON matching the schema.\n"
                "Write a Python script that reads a JSON file and writes RESULT_TEXT_PATH (a UTF-8 text file).\n"
                "The RESULT_TEXT_PATH must contain a single human-readable string that answers FIELDS_REQUIRED and WHY_API_WAS_CALLED.\n"
                "Goal: Get only what is needed to answer FIELDS_REQUIRED from the response.\n"
                "The response is already filtered by the upstream API; do NOT write filtering/query logic. Only extract + aggregate the needed fields.\n"
                "Use INPUT_SCHEMA_JSON to understand the structure and pick the minimal set of fields needed.\n"
                "Write small, efficient, readable code.\n"
                "The script MUST accept file paths via either:\n"
                "- argv[1] = INPUT_JSON_PATH, argv[2] = RESULT_TEXT_PATH, OR\n"
                "- environment variables INPUT_JSON_PATH and RESULT_TEXT_PATH.\n"
                "At the end of the script, print exactly one line: __IGX_STATUS__=<JSON>.\n"
                "The <JSON> must be an object with keys: phase, next_step, STOP_REASON, CONTINUE_REASON.\n"
                "For this one-shot workflow, always set next_step to STOP.\n"
                "Constraints:\n"
                "- Use ONLY Python standard library\n"
                "- No network calls, no subprocess, deterministic\n"
                "- Do not print raw file contents\n"
                + (f"\nTOOL_CODEX_PROMPT:\n{tool_prompt}\n" if tool_prompt else "")
            ),
        },
        {
            "role": "user",
            "content": (
                f"FIELDS_REQUIRED: {fields}\n"
                f"WHY_API_WAS_CALLED: {why_called}\n"
                f"INPUT_JSON_PATH: {input_path}\n"
                f"INPUT_SCHEMA_JSON_PATH: {schema_path}\n"
                f"INPUT_SCHEMA_JSON: {schema_inline}\n"
                f"RESULT_TEXT_PATH: {result_text_path}\n"
            ),
        },
    ]

    if progress_fn:
        progress_fn("Thinking…")
    step = _call_codex_json(
        api_key=api_key,
        model=m,
        messages=messages,
        schema_name="codex_http_one_shot",
        progress_fn=None,
    )
    _debug({"phase": "one_shot_llm", "model": m, "codex_response": step})
    stop_reason = str(step.get("STOP_REASON") or "")
    continue_reason = str(step.get("CONTINUE_REASON") or "")
    next_step: NextStep = "STOP" if str(step.get("next_step") or "").upper() == "STOP" else "CONTINUE"
    script = str(step.get("script") or "")
    if not script.strip():
        return CodexAgentResult(
            ok=False,
            output_dir=out_dir,
            debug_json_path=debug_path,
            input_json_path=input_path,
            schema_json_path=schema_path,
            filtered_json_path=filtered_path,
            validated_json_path=validated_path,
            result_text_path=result_text_path,
            result_text="",
            stop_reason=stop_reason,
            continue_reason=continue_reason,
            next_step=next_step,
            error="Codex returned an empty script.",
        )

    py_path = os.path.join(out_dir, "one_shot.py")
    _write_text(py_path, script)
    if progress_fn:
        progress_fn("Formatting the answer…")
    rc, out, err = _run_python_script(
        script_path=py_path,
        cwd=out_dir,
        timeout_s=35.0,
        argv=[input_path, result_text_path],
        env_extra={"INPUT_JSON_PATH": input_path, "RESULT_TEXT_PATH": result_text_path},
    )
    _debug({"phase": "one_shot_run", "rc": rc, "stdout": out[-4000:], "stderr": err[-4000:]})

    status = _parse_status_line(out)
    if status is not None:
        _debug({"phase": "one_shot_status", "status": status})
        stop_reason = str(status.get("STOP_REASON") or stop_reason)
        continue_reason = str(status.get("CONTINUE_REASON") or continue_reason)
        next_step = "STOP" if str(status.get("next_step") or "").upper() == "STOP" else next_step

    if rc != 0:
        return CodexAgentResult(
            ok=False,
            output_dir=out_dir,
            debug_json_path=debug_path,
            input_json_path=input_path,
            schema_json_path=schema_path,
            filtered_json_path=filtered_path,
            validated_json_path=validated_path,
            result_text_path=result_text_path,
            result_text="",
            stop_reason=stop_reason,
            continue_reason=continue_reason,
            next_step=next_step,
            error=f"One-shot script failed (exit {rc}). stderr: {err[:2000]} stdout: {out[:2000]}",
        )

    try:
        with open(result_text_path, "r", encoding="utf-8") as f:
            result_text = (f.read() or "").strip()
    except Exception as exc:
        return CodexAgentResult(
            ok=False,
            output_dir=out_dir,
            debug_json_path=debug_path,
            input_json_path=input_path,
            schema_json_path=schema_path,
            filtered_json_path=filtered_path,
            validated_json_path=validated_path,
            result_text_path=result_text_path,
            result_text="",
            stop_reason=stop_reason,
            continue_reason=continue_reason,
            next_step=next_step,
            error=f"Missing/invalid result.txt at {result_text_path}: {exc}",
        )

    return CodexAgentResult(
        ok=True,
        output_dir=out_dir,
        debug_json_path=debug_path,
        input_json_path=input_path,
        schema_json_path=schema_path,
        filtered_json_path=filtered_path,
        validated_json_path=validated_path,
        result_text_path=result_text_path,
        result_text=result_text,
        stop_reason=stop_reason or "ONE_SHOT_COMPLETED",
        continue_reason=continue_reason,
        next_step="STOP",
        error=None,
    )


def run_codex_http_agent_one_shot_from_paths(
    *,
    api_key: str,
    model: str,
    input_json_path: str,
    input_schema_json_path: str | None = None,
    fields_required: str,
    why_api_was_called: str,
    conversation_id: str | None = None,
    req_id: str | None = None,
    tool_codex_prompt: str | None = None,
    progress_fn: Optional[ProgressFn] = None,
) -> CodexAgentResult:
    """
    One-shot Codex workflow over an existing saved JSON file (no HTTP call).
    Uses a saved schema if available; otherwise derives a tree schema from the payload locally.
    """
    m = (model or "").strip()
    if not m:
        raise ValueError("model is required")
    in_path = (input_json_path or "").strip()
    if not in_path:
        raise ValueError("input_json_path is required")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Saved response JSON not found: {in_path}")
    fields = (fields_required or "").strip()
    why_called = (why_api_was_called or "").strip()
    if not fields or not why_called:
        raise ValueError("fields_required and why_api_was_called are required")

    conv = (conversation_id or "").strip() or "no_conversation"
    rid = (req_id or "").strip() or str(int(time.time() * 1000))
    out_dir = os.path.join(tempfile.gettempdir(), "igx_codex_one_shot", conv, rid)
    os.makedirs(out_dir, exist_ok=True)

    debug_path = os.path.join(out_dir, "debug.json")
    schema_path = os.path.join(out_dir, "input_schema.json")
    result_text_path = os.path.join(out_dir, "result.txt")
    filtered_path = os.path.join(out_dir, "filtered.json")
    validated_path = os.path.join(out_dir, "validated.json")

    debug_events: list[dict[str, Any]] = []

    def _debug(event: dict[str, Any]) -> None:
        try:
            debug_events.append(event)
            _write_text(debug_path, json.dumps(debug_events, ensure_ascii=False, indent=2))
        except Exception:
            return

    if progress_fn:
        progress_fn("Reviewing previous results…")

    schema_obj: Any = None
    schema_source = "derived_tree"
    sch_path = (input_schema_json_path or "").strip()
    if sch_path and os.path.exists(sch_path):
        try:
            with open(sch_path, "r", encoding="utf-8") as f:
                schema_obj = json.loads(f.read() or "null")
            schema_source = "saved_schema"
        except Exception:
            schema_obj = None
            schema_source = "derived_tree"

    if schema_obj is None:
        if progress_fn:
            progress_fn("Making sure I capture the right details…")
        try:
            with open(in_path, "r", encoding="utf-8") as f:
                payload = json.loads(f.read() or "null")
        except Exception as exc:
            raise RuntimeError(f"Failed to load saved JSON payload at {in_path}: {exc}") from exc
        schema_obj = _json_tree_schema(payload)
        schema_source = "derived_tree"

    _write_text(schema_path, json.dumps(schema_obj, ensure_ascii=False, indent=2))
    schema_inline = json.dumps(schema_obj, ensure_ascii=False)[:24000]
    _debug({"phase": "schema", "schema_source": schema_source})

    tool_prompt = (tool_codex_prompt or "").strip()
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a Python data extraction + summarization agent.\n"
                "Return ONLY JSON matching the schema.\n"
                "Write a Python script that reads a JSON file and writes RESULT_TEXT_PATH (a UTF-8 text file).\n"
                "The RESULT_TEXT_PATH must contain a single human-readable string that answers FIELDS_REQUIRED and WHY_API_WAS_CALLED.\n"
                "Goal: Get only what is needed to answer FIELDS_REQUIRED from the response.\n"
                "The response is already filtered by the upstream API; do NOT write filtering/query logic. Only extract + aggregate the needed fields.\n"
                "Use INPUT_SCHEMA_JSON to understand the structure and pick the minimal set of fields needed.\n"
                "Write small, efficient, readable code.\n"
                "The script MUST accept file paths via either:\n"
                "- argv[1] = INPUT_JSON_PATH, argv[2] = RESULT_TEXT_PATH, OR\n"
                "- environment variables INPUT_JSON_PATH and RESULT_TEXT_PATH.\n"
                "At the end of the script, print exactly one line: __IGX_STATUS__=<JSON>.\n"
                "The <JSON> must be an object with keys: phase, next_step, STOP_REASON, CONTINUE_REASON.\n"
                "For this one-shot workflow, always set next_step to STOP.\n"
                "Constraints:\n"
                "- Use ONLY Python standard library\n"
                "- No network calls, no subprocess, deterministic\n"
                "- Do not print raw file contents\n"
                + (f"\nTOOL_CODEX_PROMPT:\n{tool_prompt}\n" if tool_prompt else "")
            ),
        },
        {
            "role": "user",
            "content": (
                f"FIELDS_REQUIRED: {fields}\n"
                f"WHY_API_WAS_CALLED: {why_called}\n"
                f"INPUT_JSON_PATH: {in_path}\n"
                f"INPUT_SCHEMA_JSON_PATH: {schema_path}\n"
                f"INPUT_SCHEMA_JSON: {schema_inline}\n"
                f"RESULT_TEXT_PATH: {result_text_path}\n"
            ),
        },
    ]

    if progress_fn:
        progress_fn("Thinking…")
    step = _call_codex_json(
        api_key=api_key,
        model=m,
        messages=messages,
        schema_name="codex_http_one_shot",
        progress_fn=None,
    )
    _debug({"phase": "one_shot_llm", "model": m, "codex_response": step})
    stop_reason = str(step.get("STOP_REASON") or "")
    continue_reason = str(step.get("CONTINUE_REASON") or "")
    next_step: NextStep = "STOP" if str(step.get("next_step") or "").upper() == "STOP" else "CONTINUE"
    script = str(step.get("script") or "")
    if not script.strip():
        return CodexAgentResult(
            ok=False,
            output_dir=out_dir,
            debug_json_path=debug_path,
            input_json_path=in_path,
            schema_json_path=schema_path,
            filtered_json_path=filtered_path,
            validated_json_path=validated_path,
            result_text_path=result_text_path,
            result_text="",
            stop_reason=stop_reason,
            continue_reason=continue_reason,
            next_step=next_step,
            error="Codex returned an empty script.",
        )

    py_path = os.path.join(out_dir, "one_shot.py")
    _write_text(py_path, script)
    if progress_fn:
        progress_fn("Formatting the answer…")
    rc, out, err = _run_python_script(
        script_path=py_path,
        cwd=out_dir,
        timeout_s=35.0,
        argv=[in_path, result_text_path],
        env_extra={"INPUT_JSON_PATH": in_path, "RESULT_TEXT_PATH": result_text_path},
    )
    _debug({"phase": "one_shot_run", "rc": rc, "stdout": out[-4000:], "stderr": err[-4000:]})

    status = _parse_status_line(out)
    if status is not None:
        _debug({"phase": "one_shot_status", "status": status})
        stop_reason = str(status.get("STOP_REASON") or stop_reason)
        continue_reason = str(status.get("CONTINUE_REASON") or continue_reason)
        next_step = "STOP" if str(status.get("next_step") or "").upper() == "STOP" else next_step

    if rc != 0:
        return CodexAgentResult(
            ok=False,
            output_dir=out_dir,
            debug_json_path=debug_path,
            input_json_path=in_path,
            schema_json_path=schema_path,
            filtered_json_path=filtered_path,
            validated_json_path=validated_path,
            result_text_path=result_text_path,
            result_text="",
            stop_reason=stop_reason,
            continue_reason=continue_reason,
            next_step=next_step,
            error=f"One-shot script failed (exit {rc}). stderr: {err[:2000]} stdout: {out[:2000]}",
        )

    try:
        with open(result_text_path, "r", encoding="utf-8") as f:
            result_text = (f.read() or "").strip()
    except Exception as exc:
        return CodexAgentResult(
            ok=False,
            output_dir=out_dir,
            debug_json_path=debug_path,
            input_json_path=in_path,
            schema_json_path=schema_path,
            filtered_json_path=filtered_path,
            validated_json_path=validated_path,
            result_text_path=result_text_path,
            result_text="",
            stop_reason=stop_reason,
            continue_reason=continue_reason,
            next_step=next_step,
            error=f"Missing/invalid result.txt at {result_text_path}: {exc}",
        )

    return CodexAgentResult(
        ok=True,
        output_dir=out_dir,
        debug_json_path=debug_path,
        input_json_path=in_path,
        schema_json_path=schema_path,
        filtered_json_path=filtered_path,
        validated_json_path=validated_path,
        result_text_path=result_text_path,
        result_text=result_text,
        stop_reason=stop_reason or "RECALL_ONE_SHOT_COMPLETED",
        continue_reason=continue_reason,
        next_step="STOP",
        error=None,
    )
