from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Optional

from .codex_http_agent_helpers import (
    ProgressFn,
    _call_codex_json,
    _json_tree_schema,
    _parse_status_line,
    _run_python_script,
    _write_text,
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
    _debug({"phase": "schema", "schema_source": schema_source})

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a Python data export agent.\n"
                "Return ONLY JSON matching the schema.\n"
                "Write a Python script that reads a JSON file and writes OUTPUT_FILE_PATH in the requested format.\n"
                "The script MUST accept file paths via either:\n"
                "- argv[1] = INPUT_JSON_PATH, argv[2] = OUTPUT_FILE_PATH, OR\n"
                "- environment variables INPUT_JSON_PATH and OUTPUT_FILE_PATH.\n"
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
        progress_fn("Thinking…")
    step = _call_codex_json(
        api_key=api_key,
        model=m,
        messages=messages,
        schema_name="codex_http_export",
        progress_fn=None,
    )
    _debug({"phase": "export_llm", "model": m, "codex_response": step})
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
        progress_fn("Formatting the export…")
    rc, out, err = _run_python_script(
        script_path=py_path,
        cwd=out_dir,
        timeout_s=60.0,
        argv=[in_path, export_path],
        env_extra={"INPUT_JSON_PATH": in_path, "OUTPUT_FILE_PATH": export_path},
    )
    _debug({"phase": "export_run", "rc": rc, "stdout": out[-4000:], "stderr": err[-4000:]})

    status = _parse_status_line(out)
    if status is not None:
        _debug({"phase": "export_status", "status": status})
        stop_reason = str(status.get("STOP_REASON") or stop_reason)

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
            error=f"Missing export file at {export_path}",
        )

    return CodexExportResult(
        ok=True,
        output_dir=out_dir,
        debug_json_path=debug_path,
        input_json_path=in_path,
        schema_json_path=schema_path,
        export_file_path=export_path,
        export_format=fmt,
        stop_reason=stop_reason or "EXPORT_COMPLETED",
        error=None,
    )
