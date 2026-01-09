from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional


ProgressFn = Callable[[str], None]
NextStep = Literal["STOP", "CONTINUE"]
_STATUS_PREFIX = "__IGX_STATUS__="


@dataclass(frozen=True)
class CodexAgentResult:
    ok: bool
    output_dir: str
    debug_json_path: str
    input_json_path: str
    schema_json_path: str
    filtered_json_path: str
    validated_json_path: str
    result_text_path: str
    result_text: str
    stop_reason: str
    continue_reason: str
    next_step: NextStep
    error: Optional[str] = None


def _to_responses_input(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role") or "user"
        text = (m.get("content") or "").strip()
        if not text:
            continue
        items.append({"role": role, "content": [{"type": "input_text", "text": text}]})
    return items


def _coerce_json_schema(obj: Any) -> Optional[dict[str, Any]]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str) and obj.strip():
        try:
            parsed = json.loads(obj)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _schema_matches_payload_top(schema: dict[str, Any], payload: Any) -> bool:
    """
    Best-effort, shallow validation:
    - If schema declares top-level type object/array, ensure payload matches.
    - Otherwise return True.
    """
    t = schema.get("type")
    types: list[str] = []
    if isinstance(t, str):
        types = [t]
    elif isinstance(t, list):
        types = [x for x in t if isinstance(x, str)]
    if not types:
        return True
    if "object" in types:
        return isinstance(payload, dict)
    if "array" in types:
        return isinstance(payload, list)
    return True


def _parse_status_line(stdout: str) -> Optional[dict[str, Any]]:
    if not stdout:
        return None
    lines = stdout.splitlines()
    for line in reversed(lines):
        s = (line or "").strip()
        if not s.startswith(_STATUS_PREFIX):
            continue
        raw = s[len(_STATUS_PREFIX) :].strip()
        try:
            obj = json.loads(raw)
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None
    return None


def _json_tree_schema(value: Any, *, max_depth: int = 8, max_list_items: int = 25) -> Any:
    """
    Produces a compact, tree-structured "schema-like" representation:
    - objects: {key: subtree, ...} plus "__type__":"object"
    - arrays: {"__type__":"array","items":subtree}
    - primitives: {"__type__":"string|number|boolean|null"}
    """
    if max_depth <= 0:
        return {"__type__": "unknown"}
    if value is None:
        return {"__type__": "null"}
    if isinstance(value, bool):
        return {"__type__": "boolean"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {"__type__": "number"}
    if isinstance(value, str):
        return {"__type__": "string"}
    if isinstance(value, dict):
        out: dict[str, Any] = {"__type__": "object"}
        for k in sorted([str(x) for x in value.keys()])[:800]:
            try:
                v = value.get(k)
            except Exception:
                continue
            out[k] = _json_tree_schema(v, max_depth=max_depth - 1, max_list_items=max_list_items)
        return out
    if isinstance(value, list):
        merged: Any = None
        for it in value[: max(0, int(max_list_items))]:
            sch = _json_tree_schema(it, max_depth=max_depth - 1, max_list_items=max_list_items)
            if merged is None:
                merged = sch
            else:
                merged = _merge_tree_schema(merged, sch)
        return {"__type__": "array", "items": merged if merged is not None else {"__type__": "unknown"}}
    return {"__type__": type(value).__name__}


def _merge_tree_schema(a: Any, b: Any) -> Any:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return a
    ta = a.get("__type__")
    tb = b.get("__type__")
    if ta != tb:
        return {"__type__": f"{ta}|{tb}"}
    if ta == "object":
        out = {"__type__": "object"}
        keys = {k for k in a.keys() if k != "__type__"} | {k for k in b.keys() if k != "__type__"}
        for k in sorted(keys)[:800]:
            if k in a and k in b:
                out[k] = _merge_tree_schema(a.get(k), b.get(k))
            elif k in a:
                out[k] = a.get(k)
            else:
                out[k] = b.get(k)
        return out
    if ta == "array":
        return {
            "__type__": "array",
            "items": _merge_tree_schema(a.get("items"), b.get("items")),
        }
    return a


def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _run_python_script(
    *,
    script_path: str,
    cwd: str,
    timeout_s: float = 20.0,
    argv: Optional[list[str]] = None,
    env_extra: Optional[dict[str, str]] = None,
) -> tuple[int, str, str]:
    env = dict(os.environ)
    for k in (
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "OPENAI_ORG",
        "OPENAI_ORGANIZATION",
        "OPENAI_PROJECT",
        "OPENAI_BASE_URL",
    ):
        env.pop(k, None)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if env_extra:
        for k, v in env_extra.items():
            if isinstance(k, str) and k.strip():
                env[str(k)] = str(v)
    proc = subprocess.run(
        [sys.executable, script_path] + ([str(x) for x in (argv or [])]),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return int(proc.returncode), proc.stdout or "", proc.stderr or ""


def _call_codex_json(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    schema_name: str,
    progress_fn: Optional[ProgressFn] = None,
) -> dict[str, Any]:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"openai sdk not installed: {exc}") from exc

    client = OpenAI(api_key=api_key)
    if progress_fn:
        progress_fn(f"Codex: calling model {model}…")
    resp = client.responses.create(
        model=model,
        input=_to_responses_input(messages),
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "STOP_REASON": {"type": "string"},
                        "CONTINUE_REASON": {"type": "string"},
                        "next_step": {"type": "string", "enum": ["STOP", "CONTINUE"]},
                        "script": {"type": "string"},
                    },
                    "required": ["STOP_REASON", "CONTINUE_REASON", "next_step", "script"],
                },
            }
        },
    )
    raw = (getattr(resp, "output_text", "") or "").strip()
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise RuntimeError(f"Codex did not return valid JSON: {exc}; raw={raw[:500]}") from exc
    if not isinstance(obj, dict):
        raise RuntimeError("Codex JSON must be an object.")
    return obj


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
    input_path = (input_json_path or "").strip() or os.path.join(out_dir, "input_response.json")
    schema_path = os.path.join(out_dir, "input_schema.json")
    filtered_path = os.path.join(out_dir, "filtered.json")
    validated_path = os.path.join(out_dir, "validated.json")
    result_text_path = os.path.join(out_dir, "result.txt")

    debug_events: list[dict[str, Any]] = []

    def _debug(event: dict[str, Any]) -> None:
        try:
            debug_events.append(event)
            _write_text(debug_path, json.dumps(debug_events, ensure_ascii=False, indent=2))
        except Exception:
            return

    if progress_fn:
        progress_fn("Codex agent: saving HTTP response…")
    if not (input_json_path or "").strip():
        _write_text(input_path, json.dumps(response_json, ensure_ascii=False, indent=2))
    schema_obj: Any = None
    schema_source = "derived_tree"
    provided_schema = _coerce_json_schema(response_schema_json)
    if provided_schema is not None and _schema_matches_payload_top(provided_schema, response_json):
        schema_obj = provided_schema
        schema_source = "provided_json_schema"
    else:
        if progress_fn:
            progress_fn("Making sure I capture the right details…")
        schema_obj = _json_tree_schema(response_json)
    _write_text(schema_path, json.dumps(schema_obj, ensure_ascii=False, indent=2))
    schema_inline = json.dumps(schema_obj, ensure_ascii=False)[:24000]
    _debug({"phase": "schema", "schema_source": schema_source})

    last_stop_reason = ""
    last_continue_reason = ""
    last_next_step: NextStep = "CONTINUE"
    last_result_text = ""
    last_status: dict[str, Any] | None = None
    tool_prompt = (tool_codex_prompt or "").strip()

    for cycle in range(1, max(1, int(max_cycles)) + 1):
        if progress_fn:
            progress_fn(f"Codex agent: cycle {cycle}…")

        # 1) Extraction script
        extract_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a Python data extraction agent.\n"
                    "Return ONLY JSON matching the schema.\n"
                    "Write a Python script that reads INPUT_JSON_PATH (a JSON file) and writes FILTERED_JSON_PATH (a JSON file).\n"
                    "At the end of the script, print exactly one line: __IGX_STATUS__=<JSON>.\n"
                    "The <JSON> must be an object with keys: phase, next_step, STOP_REASON, CONTINUE_REASON.\n"
                    "You MAY read additional prior response files if needed, using CONVERSATION_ARTIFACTS_INDEX_PATH.\n"
                    "If CONVERSATION_ARTIFACTS_INDEX_PATH is set, it points to index.json with shape:\n"
                    "{ \"version\": 1, \"items\": [ {\"manifest_path\": \"manifests/<id>.json\", \"tool_name\": \"...\", \"called_at\": \"...\", \"what_to_search_for\": \"...\"} ] }\n"
                    "Each manifest JSON contains at least: { \"response_path\": \"responses/<id>.json\", \"what_to_search_for\": \"...\", \"why_to_search_for\": \"...\" }.\n"
                    "Resolve relative paths against os.path.dirname(CONVERSATION_ARTIFACTS_INDEX_PATH).\n"
                    "Do NOT print raw file contents.\n"
                    "Constraints:\n"
                    "- Use ONLY Python standard library\n"
                    "- No network calls, no subprocess, no filesystem writes except FILTERED_JSON_PATH\n"
                    "- Deterministic output; exit with code 0 on success\n"
                    "- FILTERED_JSON_PATH must contain only data needed to answer WHAT_TO_SEARCH_FOR.\n"
                    "- Use WHY_TO_SEARCH_FOR to decide which fields to retain.\n"
                    + (f"\nTOOL_CODEX_PROMPT:\n{tool_prompt}\n" if tool_prompt else "")
                ),
            },
            {
                "role": "user",
                "content": (
                    f"WHAT_TO_SEARCH_FOR: {what}\n"
                    f"WHY_TO_SEARCH_FOR: {why}\n"
                    f"INPUT_JSON_PATH: {input_path}\n"
                    f"FILTERED_JSON_PATH: {filtered_path}\n"
                    f"INPUT_SCHEMA_JSON_PATH: {schema_path}\n"
                    f"INPUT_SCHEMA_JSON: {schema_inline}\n"
                    f"CONVERSATION_ARTIFACTS_INDEX_PATH: {(conversation_artifacts_index_path or '').strip()}\n"
                    f"PRIOR_CONTINUE_REASON: {last_continue_reason}\n"
                    f"PRIOR_RESULT_TEXT_PREVIEW: {last_result_text[:1200]}\n"
                    f"PRIOR_VALIDATED_JSON_PATH: {validated_path}\n"
                    f"PRIOR_STATUS_JSON: {json.dumps(last_status, ensure_ascii=False) if last_status else ''}\n"
                ),
            },
        ]
        filtered_obj: Any = None
        last_extract_error: str = ""
        for attempt in range(1, max(1, int(max_attempts_per_phase)) + 1):
            if progress_fn:
                if attempt == 1:
                    progress_fn("Codex agent: requesting extraction script…")
                else:
                    progress_fn(f"Codex agent: retrying extraction (attempt {attempt})…")
            if last_extract_error:
                extract_messages = list(extract_messages) + [
                    {"role": "user", "content": f"PREVIOUS_EXTRACTION_ERROR:\n{last_extract_error[:3000]}\n"},
                    {
                        "role": "user",
                        "content": (
                            "Please return a corrected script. Reminder: standard library only; "
                            "read INPUT_JSON_PATH and write FILTERED_JSON_PATH as valid JSON.\n"
                        ),
                    },
                ]
            step1 = _call_codex_json(
                api_key=api_key,
                model=m,
                messages=extract_messages,
                schema_name="codex_http_extract",
                progress_fn=progress_fn,
            )
            _debug(
                {
                    "phase": "extract_llm",
                    "cycle": cycle,
                    "attempt": attempt,
                    "model": m,
                    "codex_response": step1,
                }
            )
            last_stop_reason = str(step1.get("STOP_REASON") or "")
            last_continue_reason = str(step1.get("CONTINUE_REASON") or "")
            last_next_step = "STOP" if str(step1.get("next_step") or "").upper() == "STOP" else "CONTINUE"
            extract_script = str(step1.get("script") or "")
            if not extract_script.strip():
                last_extract_error = "Codex returned an empty extraction script."
                continue
            extract_py = os.path.join(out_dir, f"extract_cycle_{cycle}_attempt_{attempt}.py")
            _write_text(extract_py, extract_script)
            if progress_fn:
                progress_fn("Codex agent: running extraction script…")
            rc, out, err = _run_python_script(
                script_path=extract_py,
                cwd=out_dir,
                timeout_s=25.0,
                argv=[input_path, filtered_path, schema_path],
                env_extra={
                    "INPUT_JSON_PATH": input_path,
                    "FILTERED_JSON_PATH": filtered_path,
                    "INPUT_SCHEMA_JSON_PATH": schema_path,
                },
            )
            _debug(
                {
                    "phase": "extract_run",
                    "cycle": cycle,
                    "attempt": attempt,
                    "script_path": extract_py,
                    "rc": rc,
                    "stdout": out[-4000:],
                    "stderr": err[-4000:],
                }
            )
            status = _parse_status_line(out)
            if status is not None:
                last_status = status
                _debug({"phase": "extract_status", "cycle": cycle, "attempt": attempt, "status": status})
            if rc != 0:
                last_extract_error = f"Extraction script failed (exit {rc}). stderr: {err[:2000]} stdout: {out[:2000]}"
                continue
            try:
                with open(filtered_path, "r", encoding="utf-8") as f:
                    filtered_obj = json.load(f)
                last_extract_error = ""
                break
            except Exception as exc:
                last_extract_error = f"Extraction did not produce valid JSON at {filtered_path}: {exc}"
                continue
        if filtered_obj is None or last_extract_error:
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
                stop_reason=last_stop_reason,
                continue_reason=last_continue_reason,
                next_step=last_next_step,
                error=last_extract_error or "Extraction failed.",
            )

        # 2) Validation + summary script
        validate_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a Python validation + summarization agent.\n"
                    "Return ONLY JSON matching the schema.\n"
                    "Write a Python script that reads FILTERED_JSON_PATH (a JSON file) and writes:\n"
                    "- VALIDATED_JSON_PATH (a JSON file)\n"
                    "- RESULT_TEXT_PATH (a UTF-8 text file containing a concise summary string)\n"
                    "At the end of the script, print exactly one line: __IGX_STATUS__=<JSON>.\n"
                    "The <JSON> must be an object with keys: phase, next_step, STOP_REASON, CONTINUE_REASON.\n"
                    "You MAY read additional prior response files if needed, using CONVERSATION_ARTIFACTS_INDEX_PATH.\n"
                    "If CONVERSATION_ARTIFACTS_INDEX_PATH is set, it points to index.json with shape:\n"
                    "{ \"version\": 1, \"items\": [ {\"manifest_path\": \"manifests/<id>.json\", \"tool_name\": \"...\", \"called_at\": \"...\", \"what_to_search_for\": \"...\"} ] }\n"
                    "Each manifest JSON contains at least: { \"response_path\": \"responses/<id>.json\", \"what_to_search_for\": \"...\", \"why_to_search_for\": \"...\" }.\n"
                    "Resolve relative paths against os.path.dirname(CONVERSATION_ARTIFACTS_INDEX_PATH).\n"
                    "Do NOT print raw file contents.\n"
                    "Constraints:\n"
                    "- Use ONLY Python standard library\n"
                    "- No network calls, no subprocess, deterministic\n"
                    "- Validate that the filtered data answers WHAT_TO_SEARCH_FOR; if not, normalize the data and still write files.\n"
                    "- RESULT_TEXT_PATH must be user-facing and concise.\n"
                    + (f"\nTOOL_CODEX_PROMPT:\n{tool_prompt}\n" if tool_prompt else "")
                ),
            },
            {
                "role": "user",
                "content": (
                    f"WHAT_TO_SEARCH_FOR: {what}\n"
                    f"WHY_TO_SEARCH_FOR: {why}\n"
                    f"FILTERED_JSON_PATH: {filtered_path}\n"
                    f"VALIDATED_JSON_PATH: {validated_path}\n"
                    f"RESULT_TEXT_PATH: {result_text_path}\n"
                    f"FILTERED_SCHEMA_JSON: {json.dumps(_json_tree_schema(filtered_obj), ensure_ascii=False)[:12000]}\n"
                    f"CONVERSATION_ARTIFACTS_INDEX_PATH: {(conversation_artifacts_index_path or '').strip()}\n"
                    f"PRIOR_STATUS_JSON: {json.dumps(last_status, ensure_ascii=False) if last_status else ''}\n"
                ),
            },
        ]
        last_validate_error: str = ""
        for attempt in range(1, max(1, int(max_attempts_per_phase)) + 1):
            if progress_fn:
                if attempt == 1:
                    progress_fn("Codex agent: requesting validation script…")
                else:
                    progress_fn(f"Codex agent: retrying validation (attempt {attempt})…")
            if last_validate_error:
                validate_messages = list(validate_messages) + [
                    {"role": "user", "content": f"PREVIOUS_VALIDATION_ERROR:\n{last_validate_error[:3000]}\n"},
                    {
                        "role": "user",
                        "content": (
                            "Please return a corrected script. Reminder: standard library only; "
                            "read FILTERED_JSON_PATH and write VALIDATED_JSON_PATH (valid JSON) and RESULT_TEXT_PATH (text).\n"
                        ),
                    },
                ]
            step2 = _call_codex_json(
                api_key=api_key,
                model=m,
                messages=validate_messages,
                schema_name="codex_http_validate",
                progress_fn=progress_fn,
            )
            _debug(
                {
                    "phase": "validate_llm",
                    "cycle": cycle,
                    "attempt": attempt,
                    "model": m,
                    "codex_response": step2,
                }
            )
            last_stop_reason = str(step2.get("STOP_REASON") or "")
            last_continue_reason = str(step2.get("CONTINUE_REASON") or "")
            last_next_step = "STOP" if str(step2.get("next_step") or "").upper() == "STOP" else "CONTINUE"
            validate_script = str(step2.get("script") or "")
            if not validate_script.strip():
                last_validate_error = "Codex returned an empty validation script."
                continue
            validate_py = os.path.join(out_dir, f"validate_cycle_{cycle}_attempt_{attempt}.py")
            _write_text(validate_py, validate_script)
            if progress_fn:
                progress_fn("Codex agent: running validation script…")
            rc2, out2, err2 = _run_python_script(
                script_path=validate_py,
                cwd=out_dir,
                timeout_s=25.0,
                argv=[filtered_path, validated_path, result_text_path],
                env_extra={
                    "FILTERED_JSON_PATH": filtered_path,
                    "VALIDATED_JSON_PATH": validated_path,
                    "RESULT_TEXT_PATH": result_text_path,
                },
            )
            _debug(
                {
                    "phase": "validate_run",
                    "cycle": cycle,
                    "attempt": attempt,
                    "script_path": validate_py,
                    "rc": rc2,
                    "stdout": out2[-4000:],
                    "stderr": err2[-4000:],
                }
            )
            status2 = _parse_status_line(out2)
            if status2 is not None:
                last_status = status2
                _debug({"phase": "validate_status", "cycle": cycle, "attempt": attempt, "status": status2})
            if rc2 != 0:
                last_validate_error = f"Validation script failed (exit {rc2}). stderr: {err2[:2000]} stdout: {out2[:2000]}"
                continue
            try:
                with open(validated_path, "r", encoding="utf-8") as f:
                    _ = json.load(f)
            except Exception as exc:
                last_validate_error = f"Validation did not produce valid JSON at {validated_path}: {exc}"
                continue
            try:
                with open(result_text_path, "r", encoding="utf-8") as f:
                    last_result_text = (f.read() or "").strip()
            except Exception:
                last_result_text = ""
            last_validate_error = ""
            break
        if last_validate_error:
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
                stop_reason=last_stop_reason,
                continue_reason=last_continue_reason,
                next_step=last_next_step,
                error=last_validate_error,
            )
        if progress_fn and last_result_text:
            progress_fn("Codex agent: summary ready.")

        # If the script explicitly told us to stop, honor it (this is still a Codex decision).
        if isinstance(last_status, dict) and str(last_status.get("next_step") or "").upper() == "STOP":
            last_stop_reason = str(last_status.get("STOP_REASON") or last_stop_reason or "")
            last_continue_reason = str(last_status.get("CONTINUE_REASON") or last_continue_reason or "")
            last_next_step = "STOP"
            return CodexAgentResult(
                ok=True,
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
                error=None,
            )

        # 3) Decide STOP / CONTINUE based on outputs.
        decide_messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are deciding whether the extraction is complete.\n"
                    "Return ONLY JSON matching the schema.\n"
                    "If next_step is STOP, script can be empty.\n"
                    "If next_step is CONTINUE, explain what is missing in CONTINUE_REASON.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"WHAT_TO_SEARCH_FOR: {what}\n"
                    f"WHY_TO_SEARCH_FOR: {why}\n"
                    f"INPUT_JSON_PATH: {input_path}\n"
                    f"FILTERED_JSON_PATH: {filtered_path}\n"
                    f"VALIDATED_JSON_PATH: {validated_path}\n"
                    f"RESULT_TEXT_PATH: {result_text_path}\n"
                    f"CONVERSATION_ARTIFACTS_INDEX_PATH: {(conversation_artifacts_index_path or '').strip()}\n"
                    f"RESULT_TEXT_PREVIEW: {last_result_text[:2000]}\n"
                    f"CYCLE: {cycle}/{max_cycles}\n"
                ),
            },
        ]
        step3 = _call_codex_json(
            api_key=api_key,
            model=m,
            messages=decide_messages,
            schema_name="codex_http_decide",
            progress_fn=progress_fn,
        )
        _debug(
            {
                "phase": "decide_llm",
                "cycle": cycle,
                "model": m,
                "codex_response": step3,
                "result_text_preview": last_result_text[:2000],
            }
        )
        last_stop_reason = str(step3.get("STOP_REASON") or "")
        last_continue_reason = str(step3.get("CONTINUE_REASON") or "")
        last_next_step = "STOP" if str(step3.get("next_step") or "").upper() == "STOP" else "CONTINUE"
        if last_next_step == "STOP":
            return CodexAgentResult(
                ok=True,
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
                error=None,
            )

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
        progress_fn("Codex agent: saving HTTP response…")
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
        progress_fn(f"Codex: calling model {m}…")
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
        progress_fn("Codex agent: running script…")
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
        progress_fn("Recall agent: loading saved schema…")

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
    _debug(
        {
            "phase": "recall_schema",
            "schema_source": schema_source,
            "source_input_json_path": in_path,
            "source_input_schema_json_path": sch_path,
        }
    )

    tool_prompt = (tool_codex_prompt or "").strip()
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a Python data extraction + summarization agent.\n"
                "Return ONLY JSON matching the schema.\n"
                "Write a Python script that reads a JSON file and writes RESULT_TEXT_PATH (a UTF-8 text file).\n"
                "The RESULT_TEXT_PATH must contain a single human-readable string that answers FIELDS_REQUIRED and WHY_API_WAS_CALLED.\n"
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
        progress_fn(f"Codex: calling model {m}…")
    step = _call_codex_json(
        api_key=api_key,
        model=m,
        messages=messages,
        schema_name="codex_http_recall_one_shot",
        progress_fn=None,
    )
    _debug({"phase": "recall_one_shot_llm", "model": m, "codex_response": step})
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
        progress_fn("Recall agent: running script…")
    rc, out, err = _run_python_script(
        script_path=py_path,
        cwd=out_dir,
        timeout_s=35.0,
        argv=[in_path, result_text_path],
        env_extra={"INPUT_JSON_PATH": in_path, "RESULT_TEXT_PATH": result_text_path},
    )
    _debug({"phase": "recall_one_shot_run", "rc": rc, "stdout": out[-4000:], "stderr": err[-4000:]})

    status = _parse_status_line(out)
    if status is not None:
        _debug({"phase": "recall_one_shot_status", "status": status})
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


@dataclass(frozen=True)
class CodexExportResult:
    ok: bool
    output_dir: str
    debug_json_path: str
    input_json_path: str
    schema_json_path: str
    export_file_path: str
    export_format: str
    stop_reason: str
    error: Optional[str] = None


def run_codex_export_from_paths(
    *,
    api_key: str,
    model: str,
    input_json_path: str,
    input_schema_json_path: str | None = None,
    export_request: str,
    output_format: str = "csv",
    conversation_id: str | None = None,
    req_id: str | None = None,
    progress_fn: Optional[ProgressFn] = None,
) -> CodexExportResult:
    """
    Codex-powered export over an existing saved JSON file.
    The script writes a single export file to OUTPUT_FILE_PATH.
    """
    m = (model or "").strip()
    if not m:
        raise ValueError("model is required")
    in_path = (input_json_path or "").strip()
    if not in_path:
        raise ValueError("input_json_path is required")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Saved response JSON not found: {in_path}")

    req = (export_request or "").strip()
    if not req:
        raise ValueError("export_request is required")

    fmt = (output_format or "csv").strip().lower()
    if fmt not in ("csv", "json"):
        raise ValueError("output_format must be csv or json")

    conv = (conversation_id or "").strip() or "no_conversation"
    rid = (req_id or "").strip() or str(int(time.time() * 1000))
    out_dir = os.path.join(tempfile.gettempdir(), "igx_exports", conv, rid)
    os.makedirs(out_dir, exist_ok=True)

    debug_path = os.path.join(out_dir, "debug.json")
    schema_path = os.path.join(out_dir, "input_schema.json")
    export_path = os.path.join(out_dir, f"export.{fmt}")

    debug_events: list[dict[str, Any]] = []

    def _debug(event: dict[str, Any]) -> None:
        try:
            debug_events.append(event)
            _write_text(debug_path, json.dumps(debug_events, ensure_ascii=False, indent=2))
        except Exception:
            return

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
    _debug(
        {
            "phase": "export_schema",
            "schema_source": schema_source,
            "source_input_json_path": in_path,
            "source_input_schema_json_path": sch_path,
            "output_format": fmt,
        }
    )

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a Python export agent.\n"
                "Return ONLY JSON matching the schema.\n"
                "Write a small, efficient Python script that reads INPUT_JSON_PATH and writes a single export file to OUTPUT_FILE_PATH.\n"
                "Goal: Get only what is needed to satisfy EXPORT_REQUEST from the response.\n"
                "The upstream API response is already filtered; do NOT write any filtering/querying logic. Only extract and aggregate.\n"
                "Use INPUT_SCHEMA_JSON to understand the structure and pick the minimal set of fields needed.\n"
                "If OUTPUT_FORMAT is csv:\n"
                "- Write a header row\n"
                "- Use UTF-8\n"
                "- One row per record when possible; otherwise write a single row summary\n"
                "If OUTPUT_FORMAT is json:\n"
                "- Write valid JSON (object or array)\n"
                "The script MUST accept paths via either argv[1]=INPUT_JSON_PATH argv[2]=OUTPUT_FILE_PATH, OR env vars INPUT_JSON_PATH/OUTPUT_FILE_PATH.\n"
                "At the end of the script, print exactly one line: __IGX_STATUS__=<JSON>.\n"
                "The <JSON> must be an object with keys: phase, next_step, STOP_REASON, CONTINUE_REASON.\n"
                "For this export workflow, always set next_step to STOP.\n"
                "Constraints:\n"
                "- Use ONLY Python standard library\n"
                "- No network calls, no subprocess, deterministic\n"
                "- Do not print raw file contents\n"
            ),
        },
        {
            "role": "user",
            "content": (
                f"EXPORT_REQUEST: {req}\n"
                f"OUTPUT_FORMAT: {fmt}\n"
                f"INPUT_JSON_PATH: {in_path}\n"
                f"INPUT_SCHEMA_JSON_PATH: {schema_path}\n"
                f"INPUT_SCHEMA_JSON: {schema_inline}\n"
                f"OUTPUT_FILE_PATH: {export_path}\n"
            ),
        },
    ]

    if progress_fn:
        progress_fn(f"Codex: calling model {m}…")
    step = _call_codex_json(
        api_key=api_key,
        model=m,
        messages=messages,
        schema_name="codex_http_export_one_shot",
        progress_fn=None,
    )
    _debug({"phase": "export_one_shot_llm", "model": m, "codex_response": step})
    stop_reason = str(step.get("STOP_REASON") or "")
    script = str(step.get("script") or "")
    if not script.strip():
        return CodexExportResult(
            ok=False,
            output_dir=out_dir,
            debug_json_path=debug_path,
            input_json_path=in_path,
            schema_json_path=schema_path,
            export_file_path=export_path,
            export_format=fmt,
            stop_reason=stop_reason,
            error="Codex returned an empty script.",
        )

    py_path = os.path.join(out_dir, "export.py")
    _write_text(py_path, script)
    if progress_fn:
        progress_fn("Export agent: running script…")
    rc, out, err = _run_python_script(
        script_path=py_path,
        cwd=out_dir,
        timeout_s=45.0,
        argv=[in_path, export_path],
        env_extra={"INPUT_JSON_PATH": in_path, "OUTPUT_FILE_PATH": export_path},
    )
    _debug({"phase": "export_one_shot_run", "rc": rc, "stdout": out[-4000:], "stderr": err[-4000:]})

    if rc != 0:
        return CodexExportResult(
            ok=False,
            output_dir=out_dir,
            debug_json_path=debug_path,
            input_json_path=in_path,
            schema_json_path=schema_path,
            export_file_path=export_path,
            export_format=fmt,
            stop_reason=stop_reason,
            error=f"Export script failed (exit {rc}). stderr: {err[:2000]} stdout: {out[:2000]}",
        )
    if not os.path.exists(export_path):
        return CodexExportResult(
            ok=False,
            output_dir=out_dir,
            debug_json_path=debug_path,
            input_json_path=in_path,
            schema_json_path=schema_path,
            export_file_path=export_path,
            export_format=fmt,
            stop_reason=stop_reason,
            error="Export script did not create OUTPUT_FILE_PATH.",
        )

    return CodexExportResult(
        ok=True,
        output_dir=out_dir,
        debug_json_path=debug_path,
        input_json_path=in_path,
        schema_json_path=schema_path,
        export_file_path=export_path,
        export_format=fmt,
        stop_reason=stop_reason or "EXPORT_WRITTEN",
        error=None,
    )
