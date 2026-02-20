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


def run_codex_http_agent(
    *,
    api_key: str,
    model: str,
    response_json: Any,
    what_to_search_for: str,
    why_to_search_for: str,
    response_schema_json: Any = None,
    input_json_path: str | None = None,
    conversation_artifacts_index_path: str | None = None,
    tool_codex_prompt: str | None = None,
    progress_fn: Optional[ProgressFn] = None,
    max_cycles: int = 2,
    max_attempts_per_phase: int = 3,
) -> CodexAgentResult:
    """
    Runs a Codex-driven extraction/validation loop over a raw HTTP response JSON.

    The agent writes scripts to a temp dir, executes them, and returns:
    - filtered_json_path: extracted data relevant to the user query
    - validated_json_path: validated/normalized result JSON
    - result_text: a concise string summary (for the main chat model to rephrase)
    """
    m = (model or "").strip()
    if not m:
        raise ValueError("model is required")
    what = (what_to_search_for or "").strip()
    why = (why_to_search_for or "").strip()
    if not what or not why:
        raise ValueError("what_to_search_for and why_to_search_for are required")

    out_dir = tempfile.mkdtemp(prefix="igx_codex_http_")
    debug_path = os.path.join(out_dir, "debug.json")
    input_path = os.path.join(out_dir, "input_response.json")
    schema_path = os.path.join(out_dir, "input_schema.json")
    filtered_path = os.path.join(out_dir, "filtered.json")
    validated_path = os.path.join(out_dir, "validated.json")
    result_text_path = os.path.join(out_dir, "result.txt")
    index_path = os.path.join(out_dir, "artifacts_index.json")

    # If caller provided a path to the raw JSON, reuse it; otherwise write it.
    if input_json_path and os.path.exists(input_json_path):
        input_path = input_json_path
    else:
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

    debug_events: list[dict[str, Any]] = []

    def _debug(event: dict[str, Any]) -> None:
        try:
            debug_events.append(event)
            _write_text(debug_path, json.dumps(debug_events, ensure_ascii=False, indent=2))
        except Exception:
            return

    _debug({"phase": "schema", "schema_source": schema_source})

    tool_prompt = (tool_codex_prompt or "").strip()
    system_prompt = (
        "You are a Python data extraction + summarization agent.\n"
        "Return ONLY JSON matching the schema.\n"
        "Write a Python script that reads a JSON file and writes RESULT_TEXT_PATH (a UTF-8 text file).\n"
        "The RESULT_TEXT_PATH must contain a single human-readable string that answers WHAT_TO_SEARCH_FOR and WHY_TO_SEARCH_FOR.\n"
        "Goal: Get only what is needed to answer WHAT_TO_SEARCH_FOR from the response.\n"
        "The response is already filtered by the upstream API; do NOT write filtering/query logic. Only extract + aggregate the needed fields.\n"
        "Use INPUT_SCHEMA_JSON to understand the structure and pick the minimal set of fields needed.\n"
        "Write small, efficient, readable code.\n"
        "The script MUST accept file paths via either:\n"
        "- argv[1] = INPUT_JSON_PATH, argv[2] = FILTERED_JSON_PATH, argv[3] = VALIDATED_JSON_PATH, argv[4] = RESULT_TEXT_PATH, OR\n"
        "- environment variables INPUT_JSON_PATH, FILTERED_JSON_PATH, VALIDATED_JSON_PATH, RESULT_TEXT_PATH.\n"
        "At the end of the script, print exactly one line: __IGX_STATUS__=<JSON>.\n"
        "The <JSON> must be an object with keys: phase, next_step, STOP_REASON, CONTINUE_REASON.\n"
        "Constraints:\n"
        "- Use ONLY Python standard library\n"
        "- No network calls, no subprocess, deterministic\n"
        + (f"\nTOOL_CODEX_PROMPT:\n{tool_prompt}\n" if tool_prompt else "")
    )

    user_prompt = (
        f"WHAT_TO_SEARCH_FOR: {what}\n"
        f"WHY_TO_SEARCH_FOR: {why}\n"
        f"INPUT_JSON_PATH: {input_path}\n"
        f"INPUT_SCHEMA_JSON_PATH: {schema_path}\n"
        f"INPUT_SCHEMA_JSON: {schema_inline}\n"
        f"FILTERED_JSON_PATH: {filtered_path}\n"
        f"VALIDATED_JSON_PATH: {validated_path}\n"
        f"RESULT_TEXT_PATH: {result_text_path}\n"
    )

    last_stop_reason = ""
    last_continue_reason = ""
    last_next_step: NextStep = "STOP"
    last_result_text = ""

    for cycle in range(max_cycles):
        if progress_fn:
            progress_fn("Thinking…")
        step = _call_codex_json(
            api_key=api_key,
            model=m,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            schema_name="codex_http_agent",
            progress_fn=None,
        )
        _debug({"phase": "llm", "cycle": cycle, "model": m, "codex_response": step})

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

        py_path = os.path.join(out_dir, f"cycle_{cycle}.py")
        _write_text(py_path, script)
        if progress_fn:
            progress_fn("Formatting the answer…")
        rc, out, err = _run_python_script(
            script_path=py_path,
            cwd=out_dir,
            timeout_s=35.0,
            argv=[input_path, filtered_path, validated_path, result_text_path],
            env_extra={
                "INPUT_JSON_PATH": input_path,
                "FILTERED_JSON_PATH": filtered_path,
                "VALIDATED_JSON_PATH": validated_path,
                "RESULT_TEXT_PATH": result_text_path,
            },
        )
        _debug({"phase": "run", "cycle": cycle, "rc": rc, "stdout": out[-4000:], "stderr": err[-4000:]})

        status = _parse_status_line(out)
        if status is not None:
            _debug({"phase": "status", "cycle": cycle, "status": status})
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
                error=f"Script failed (exit {rc}). stderr: {err[:2000]} stdout: {out[:2000]}",
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

        if next_step == "STOP":
            _debug({"phase": "done", "cycle": cycle, "stop_reason": stop_reason})
            last_stop_reason = stop_reason
            last_continue_reason = continue_reason
            last_next_step = next_step
            last_result_text = result_text
            # Write a small index if requested (for later tooling).
            if conversation_artifacts_index_path:
                try:
                    _write_text(
                        conversation_artifacts_index_path,
                        json.dumps(
                            {
                                "input_json_path": input_path,
                                "schema_json_path": schema_path,
                                "filtered_json_path": filtered_path,
                                "validated_json_path": validated_path,
                                "result_text_path": result_text_path,
                                "result_text": result_text,
                                "debug_json_path": debug_path,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    )
                except Exception:
                    pass
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
                stop_reason=stop_reason or "DONE",
                continue_reason=continue_reason,
                next_step=next_step,
                error=None,
            )

        last_stop_reason = stop_reason
        last_continue_reason = continue_reason
        last_next_step = next_step
        last_result_text = result_text

        if progress_fn:
            progress_fn("Refining…")

    return CodexAgentResult(
        ok=False,
        output_dir=out_dir,
        debug_json_path=debug_path,
        input_json_path=input_path,
        schema_json_path=schema_path,
        filtered_json_path=filtered_path,
        validated_json_path=validated_path,
        result_text_path=result_text_path,
        result_text=last_result_text,
        stop_reason=last_stop_reason,
        continue_reason=last_continue_reason,
        next_step=last_next_step,
        error=f"Codex requested CONTINUE after max_cycles={max_cycles}.",
    )
