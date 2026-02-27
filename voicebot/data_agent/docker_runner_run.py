from __future__ import annotations

import json
import shlex
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from .docker_runner_auth import (
    _build_ssh_env_prefix,
    _ensure_preferred_repo,
    _extract_preferred_repo,
    _repo_name_from_url,
    _setup_ssh_from_auth,
    _write_json,
    _write_text,
)
from .docker_runner_constants import DEFAULT_DATA_AGENT_SYSTEM_PROMPT, logger
from .docker_runner_stream import (
    _extract_codex_stream_text,
    _run_stream_jsonl,
    _safe_str,
)


@dataclass(frozen=True)
class DataAgentRunResult:
    ok: bool
    result_text: str
    container_id: str
    session_id: str
    output_file: str
    debug_file: str
    error: str = ""


def _append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _utc_run_tag() -> str:
    # File-name safe UTC timestamp.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _extract_thread_id_from_jsonl(stdout: str) -> str:
    for line in (stdout or "").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            ev = json.loads(s)
        except Exception:
            continue
        if not isinstance(ev, dict):
            continue
        t = str(ev.get("type") or "")
        if t == "thread.started":
            tid = str(ev.get("thread_id") or "")
            if tid:
                return tid
    return ""


def run_data_agent(
    *,
    conversation_id: UUID,
    container_id: str,
    session_id: str,
    workspace_dir: str,
    api_spec_text: str,
    auth_json: str,
    system_prompt: str,
    conversation_context: dict[str, Any],
    what_to_do: str,
    openai_base_url: str | None = None,
    openai_api_key: str | None = None,
    data_agent_model: str | None = None,
    data_agent_reasoning_effort: str | None = None,
    timeout_s: float = 600.0,
    on_stream: Callable[[str], None] | None = None,
) -> DataAgentRunResult:
    """
    Runs (or resumes) a Codex CLI session inside the per-conversation container.

    Returns a strict JSON response validated by Codex via --output-schema.
    """
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    run_tag = _utc_run_tag()

    # Files visible to Codex.
    api_spec_path = ws / "api_spec.json"
    legacy_api_spec_path = ws / "API_SPEC.md"
    auth_path = ws / "auth.json"
    agents_path = ws / "AGENTS.md"
    ctx_path = ws / "conversation_context.json"

    _write_text(api_spec_path, api_spec_text or "")
    # Remove legacy file if present to avoid confusion when inspecting the workspace.
    try:
        if legacy_api_spec_path.exists():
            legacy_api_spec_path.unlink()
    except Exception:
        pass
    _write_text(auth_path, (auth_json or "{}").strip() or "{}")
    _setup_ssh_from_auth(ws, auth_json or "{}")
    _ensure_preferred_repo(container_id, ws=ws, auth_json=auth_json or "{}")
    sys_prompt = (system_prompt or "").strip() or DEFAULT_DATA_AGENT_SYSTEM_PROMPT
    _write_text(agents_path, sys_prompt + "\n")
    _write_json(ctx_path, conversation_context or {})

    # Strict response schema and output.
    schema_path = ws / "output_schema.json"
    output_path = ws / "output.json"
    debug_path = ws / "debug.json"
    activity_path = ws / "activity.jsonl"
    runs_dir = ws / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    output_snapshot_path = runs_dir / f"output_{run_tag}.json"
    debug_snapshot_path = runs_dir / f"debug_{run_tag}.json"
    schema_snapshot_path = runs_dir / f"output_schema_{run_tag}.json"

    schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "result_text": {"type": "string"},
        },
        "required": ["ok", "result_text"],
        # OpenAI Structured Outputs requires additionalProperties=false (and to be supplied) for object schemas.
        "additionalProperties": False,
    }
    _write_json(schema_path, schema)

    repo_url, _repo_cache, _repo_source = _extract_preferred_repo(auth_json or "{}")
    repo_name = _repo_name_from_url(repo_url) if repo_url else ""
    repo_note = ""
    if repo_url and repo_name:
        repo_note = (
            "\nPreferred repo cache:\n"
            f"- Repo URL: {repo_url}\n"
            f"- Workspace path: /work/{repo_name}\n"
            "- If the repo already exists, avoid recloning; use fetch/checkout instead.\n"
        )

    prompt = (
        "You are the Isolated Workspace for this conversation.\n"
        f"- Task (what_to_do): {what_to_do}\n\n"
        "Context files:\n"
        f"- API spec: {api_spec_path.name}\n"
        f"- Auth JSON: {auth_path.name}\n"
        f"- Conversation context: {ctx_path.name}\n\n"
        f"{repo_note}"
        "Rules:\n"
        "- Use the API spec and auth JSON if you need to call external APIs.\n"
        "- Keep the response concise and directly answer the task.\n"
        "- Output MUST match the provided JSON schema.\n"
    )

    # Build codex command.
    # We run from /work (mounted workspace), and write the final structured output to output.json.
    #
    # Note: Options like --output-schema / --output-last-message are `codex exec` options, so they must
    # appear before the optional `resume` subcommand.
    model = (data_agent_model or "gpt-5.2").strip() or "gpt-5.2"
    reasoning = (data_agent_reasoning_effort or "high").strip()
    cmd: list[str] = [
        "codex",
        "exec",
        "--model",
        model,
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
        "--output-schema",
        "/work/output_schema.json",
        "--output-last-message",
        "/work/output.json",
    ]
    if reasoning:
        cmd.extend(["--config", f'model_reasoning_effort="{reasoning}"'])
    if session_id:
        cmd += ["resume", session_id]
    cmd.append(prompt)
    # Use a shell to avoid PATH issues in minimal images, but keep args safely quoted.
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    env_parts: list[str] = []
    env_prefix = _build_ssh_env_prefix(ws)
    if env_prefix:
        env_parts.append(env_prefix)
    if openai_base_url:
        env_parts.append(f"OPENAI_BASE_URL={shlex.quote(openai_base_url)}")
    if openai_api_key:
        env_parts.append(f"OPENAI_API_KEY={shlex.quote(openai_api_key)}")
        env_parts.append(f"CODEX_API_KEY={shlex.quote(openai_api_key)}")
    if env_parts:
        cmd_str = f"{' '.join(env_parts)} {cmd_str}"

    logger.info(
        "Isolated Workspace run: conv=%s container=%s session_id=%s timeout_s=%s",
        conversation_id,
        container_id,
        session_id or "",
        timeout_s,
    )
    started_at = time.time()
    thread_id_box: dict[str, str] = {"thread_id": ""}
    token_box: dict[str, Any] = {"total": None, "input": None, "output": None}

    def _on_event(ev: dict[str, Any]) -> None:
        t = str(ev.get("type") or "")
        if on_stream is not None:
            try:
                text = _extract_codex_stream_text(ev)
                if text:
                    on_stream(text)
            except Exception:
                pass
        if t == "thread.started" and not thread_id_box["thread_id"]:
            tid = str(ev.get("thread_id") or "")
            if tid:
                thread_id_box["thread_id"] = tid
        if t == "event_msg":
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            if str(payload.get("type") or "") == "token_count":
                info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
                usage = info.get("total_token_usage") if isinstance(info.get("total_token_usage"), dict) else {}
                token_box["total"] = usage.get("total_tokens")
                token_box["input"] = usage.get("input_tokens")
                token_box["output"] = usage.get("output_tokens")

    log_prefix = f"DataAgent stream conv={conversation_id} container={container_id}"
    if session_id:
        log_prefix += f" session={session_id}"

    # Run codex inside the container.
    cmd_exec = ["docker", "exec", "-i", container_id, "sh", "-lc", cmd_str]
    res = _run_stream_jsonl(
        cmd_exec,
        timeout_s=timeout_s,
        log_prefix=log_prefix,
        on_event=_on_event,
    )
    elapsed_s = time.time() - started_at
    thread_id = thread_id_box.get("thread_id") or _extract_thread_id_from_jsonl(res.stdout or "")

    # Always snapshot raw output/debug.
    try:
        output_snapshot_path.write_text((output_path.read_text(encoding="utf-8") or ""), encoding="utf-8")
    except Exception:
        pass
    try:
        debug_snapshot_path.write_text((debug_path.read_text(encoding="utf-8") or ""), encoding="utf-8")
    except Exception:
        pass
    try:
        schema_snapshot_path.write_text((schema_path.read_text(encoding="utf-8") or ""), encoding="utf-8")
    except Exception:
        pass

    if res.returncode != 0:
        return DataAgentRunResult(
            ok=False,
            result_text="",
            container_id=container_id,
            session_id=session_id or thread_id or "",
            output_file=str(output_path),
            debug_file=str(debug_path),
            error=_safe_str((res.stderr.strip() or res.stdout.strip() or "Isolated Workspace failed."), limit=2000),
        )

    out_obj: dict[str, Any] = {}
    try:
        out_obj = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return DataAgentRunResult(
            ok=False,
            result_text="",
            container_id=container_id,
            session_id=session_id or thread_id or "",
            output_file=str(output_path),
            debug_file=str(debug_path),
            error=f"Failed to parse Isolated Workspace output.json: {exc}",
        )

    ok = bool(out_obj.get("ok", True))
    result_text = str(out_obj.get("result_text") or "").strip()
    if not result_text:
        ok = False
    # Append a durable, per-conversation activity record.
    _append_jsonl(
        activity_path,
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_tag": run_tag,
            "conversation_id": str(conversation_id),
            "container_id": container_id,
            "session_id": session_id or thread_id or "",
            "thread_id": thread_id or "",
            "ok": bool(ok),
            "what_to_do": _safe_str(what_to_do, limit=400),
            "elapsed_s": round(elapsed_s, 3),
            "token_usage": {"total": token_box["total"], "input": token_box["input"], "output": token_box["output"]},
            "output_snapshot": str(output_snapshot_path),
            "debug_snapshot": str(debug_snapshot_path),
            "schema_snapshot": str(schema_snapshot_path),
            "result_preview": _safe_str(result_text, limit=800),
        },
    )
    return DataAgentRunResult(
        ok=ok,
        result_text=result_text,
        container_id=container_id,
        session_id=session_id or thread_id or "",
        output_file=str(output_path),
        debug_file=str(debug_path),
        error="" if ok else (str(out_obj.get("error") or "") or "Empty result_text"),
    )


def default_workspace_dir_for_conversation(conversation_id: UUID) -> str:
    # NOTE: stored in OS temp for now; TODO: add retention/cleanup + configurable persistent volume for prod/Kubernetes.
    root = Path(tempfile.gettempdir()) / "igx_data_agent"
    return str(root / str(conversation_id))
