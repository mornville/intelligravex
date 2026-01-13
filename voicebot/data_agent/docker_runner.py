from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import UUID


DEFAULT_DATA_AGENT_SYSTEM_PROMPT = (
    "You are given a task (what_to_do), API spec, authorization tokens, and conversation context. "
    "Call any API if needed, satisfy what_to_do, and respond back with a simple response."
)

DATA_AGENT_IMAGE = "igx-data-agent:latest"

logger = logging.getLogger("voicebot.data_agent")

_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsk-[A-Za-z0-9]{8,}\b"), "sk-[REDACTED]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\\-]{10,}\b"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)(Incorrect API key provided: )([^\\s\"']+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key\\s*[:=]\\s*)([^\\s,;\"']+)"), r"\1[REDACTED]"),
]


def _redact(text: str) -> str:
    s = text or ""
    for pat, repl in _REDACT_PATTERNS:
        s = pat.sub(repl, s)
    return s


def _safe_str(x: Any, *, limit: int = 240) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    s = _redact(s)
    if len(s) > limit:
        s = s[:limit] + "…"
    return s


def _summarize_codex_event(ev: dict[str, Any]) -> str | None:
    """
    Convert a Codex --json JSONL event to a short, safe log line.

    We intentionally avoid logging full prompts, message contents, or tool outputs.
    """
    t = str(ev.get("type") or "").strip()
    payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}

    if t in ("thread.started", "thread.resumed"):
        tid = _safe_str(ev.get("thread_id") or payload.get("thread_id"), limit=80)
        return f"{t} thread_id={tid}" if tid else t

    if t.startswith("turn."):
        if t == "turn.failed":
            err = payload.get("error") if isinstance(payload.get("error"), dict) else {}
            msg = _safe_str(err.get("message") or payload.get("message") or "", limit=200)
            code = _safe_str(err.get("code") or payload.get("code") or "", limit=64)
            parts = [t]
            if code:
                parts.append(f"code={code}")
            if msg:
                parts.append(f"message={msg}")
            return " ".join(parts)
        return t

    # "event_msg" frequently carries token usage, tool status, etc.
    if t == "event_msg":
        pt = str(payload.get("type") or "").strip()
        if pt == "token_count":
            info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
            usage = info.get("total_token_usage") if isinstance(info.get("total_token_usage"), dict) else {}
            total = usage.get("total_tokens")
            inp = usage.get("input_tokens")
            out = usage.get("output_tokens")
            if isinstance(total, int) or isinstance(inp, int) or isinstance(out, int):
                return f"token_count total={total} in={inp} out={out}"
            return "token_count"
        # Avoid logging pt == "user_message" (contains full prompt).
        if pt in ("tool_call", "tool_result", "tool_error"):
            tool = _safe_str(payload.get("tool") or payload.get("name") or "", limit=80)
            status = _safe_str(payload.get("status") or "", limit=40)
            parts = [pt]
            if tool:
                parts.append(f"tool={tool}")
            if status:
                parts.append(f"status={status}")
            return " ".join(parts)
        return None

    if t == "response_item":
        # Avoid logging user content; only log assistant messages as counts.
        if str(payload.get("type") or "").strip() == "message" and str(payload.get("role") or "").strip() == "assistant":
            content = payload.get("content")
            chars = 0
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "output_text":
                        chars += len(str(item.get("text") or ""))
            return f"assistant_message chars={chars}"
        return None

    return None


@dataclass(frozen=True)
class _StreamRunResult:
    returncode: int
    stdout: str
    stderr: str


def _run_stream_jsonl(
    cmd: list[str],
    *,
    timeout_s: float,
    log_prefix: str,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> _StreamRunResult:
    """
    Run a process and stream stdout/stderr line-by-line.

    - stdout is expected to be JSONL when Codex is invoked with --json.
    - Emits condensed INFO logs for selected events.
    - Returns captured stdout/stderr (bounded) for error reporting / debug tails.
    """
    logger.debug("Running command (stream): %s", " ".join(shlex.quote(x) for x in cmd))
    p = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )

    stdout_buf: list[str] = []
    stderr_buf: list[str] = []
    stdout_chars = 0
    stderr_chars = 0
    max_chars = 200_000  # keep bounded tails

    def _append(buf: list[str], s: str, *, is_stdout: bool) -> None:
        nonlocal stdout_chars, stderr_chars
        if is_stdout:
            stdout_chars += len(s)
            while stdout_chars > max_chars and buf:
                dropped = buf.pop(0)
                stdout_chars -= len(dropped)
        else:
            stderr_chars += len(s)
            while stderr_chars > max_chars and buf:
                dropped = buf.pop(0)
                stderr_chars -= len(dropped)
        buf.append(s)

    def _read_stdout() -> None:
        if not p.stdout:
            return
        for line in iter(p.stdout.readline, ""):
            _append(stdout_buf, line, is_stdout=True)
            s = line.strip()
            if not s:
                continue
            try:
                ev = json.loads(s)
            except Exception:
                continue
            if isinstance(ev, dict):
                msg = _summarize_codex_event(ev)
                if msg:
                    logger.info("%s %s", log_prefix, msg)
                if on_event is not None:
                    try:
                        on_event(ev)
                    except Exception:
                        # Never let logging callbacks crash the run.
                        logger.debug("%s on_event failed", log_prefix, exc_info=True)

    def _read_stderr() -> None:
        if not p.stderr:
            return
        key_counts: dict[str, int] = {}
        key_samples: dict[str, str] = {}
        logged_keys: set[str] = set()

        for line in iter(p.stderr.readline, ""):
            _append(stderr_buf, line, is_stdout=False)
            # Stderr often contains error summaries; keep it condensed and redacted.
            s = line.strip()
            if not s:
                continue
            lower = s.lower()
            important = ("error" in lower) or ("failed" in lower) or ("unauthorized" in lower) or ("bad request" in lower)
            if not important:
                logger.debug("%s stderr=%s", log_prefix, _safe_str(s, limit=240))
                continue

            # Normalize stderr so retries don't spam logs.
            #
            # Codex may retry and include changing request ids/timestamps, so we collapse common HTTP errors.
            # Collapse by HTTP status code when present.
            m = re.search(r"\\bhttp\\s+(\\d{3})\\b", lower)
            if m:
                key = f"http {m.group(1)}"
            else:
                key = re.sub(r"^\\d{4}-\\d{2}-\\d{2}[tT][0-9:.]+[zZ]\\s+", "", s)

            key_counts[key] = key_counts.get(key, 0) + 1
            key_samples.setdefault(key, s)

            # Log only the first occurrence of each key (keep logs condensed).
            if key not in logged_keys and len(logged_keys) < 1:
                logger.warning("%s stderr=%s", log_prefix, _safe_str(s, limit=240))
                logged_keys.add(key)

        # Summarize repeated stderr keys at the end (best-effort).
        for key, n in key_counts.items():
            if n > 1:
                logger.warning("%s stderr_summary key=%s count=%s", log_prefix, _safe_str(key, limit=120), n)

    t1 = threading.Thread(target=_read_stdout, daemon=True)
    t2 = threading.Thread(target=_read_stderr, daemon=True)
    t1.start()
    t2.start()

    try:
        rc = p.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            p.kill()
        except Exception:
            pass
        rc = -9

    # Ensure pipes are drained.
    try:
        if p.stdout:
            p.stdout.close()
        if p.stderr:
            p.stderr.close()
    except Exception:
        pass
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    return _StreamRunResult(returncode=rc, stdout="".join(stdout_buf), stderr="".join(stderr_buf))


@dataclass(frozen=True)
class DataAgentRunResult:
    ok: bool
    result_text: str
    container_id: str
    session_id: str
    output_file: str
    debug_file: str
    error: str = ""


def _run(cmd: list[str], *, timeout_s: float = 300.0) -> subprocess.CompletedProcess[str]:
    logger.debug("Running command: %s", " ".join(shlex.quote(x) for x in cmd))
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )


def _docker_available() -> bool:
    try:
        p = _run(["docker", "version"], timeout_s=10.0)
        if p.returncode != 0:
            logger.warning("Docker not available: docker version rc=%s stderr=%s", p.returncode, (p.stderr or "").strip())
        return p.returncode == 0
    except Exception:
        logger.exception("Docker not available: exception when running docker version")
        return False


def ensure_image_built() -> None:
    if not _docker_available():
        raise RuntimeError("Docker is not available (cannot start Data Agent runtime).")
    p = _run(["docker", "image", "inspect", DATA_AGENT_IMAGE], timeout_s=10.0)
    if p.returncode == 0:
        logger.info("Data Agent image present: %s", DATA_AGENT_IMAGE)
        return
    logger.info("Data Agent image missing; building: %s", DATA_AGENT_IMAGE)
    dockerfile = Path(__file__).with_name("Dockerfile")
    context_dir = dockerfile.parent
    build = _run(
        ["docker", "build", "-t", DATA_AGENT_IMAGE, "-f", str(dockerfile), str(context_dir)],
        timeout_s=900.0,
    )
    if build.returncode != 0:
        logger.error(
            "Failed to build Data Agent image rc=%s stdout_tail=%s stderr_tail=%s",
            build.returncode,
            (build.stdout or "")[-2000:],
            (build.stderr or "")[-2000:],
        )
        raise RuntimeError(f"Failed to build Data Agent image: {build.stderr.strip() or build.stdout.strip()}")
    logger.info("Built Data Agent image: %s", DATA_AGENT_IMAGE)


def _container_name_for_conversation(conversation_id: UUID) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(conversation_id))
    return f"igx-data-agent-{s}"


def _get_existing_container_id(name: str) -> str:
    p = _run(["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.ID}}"], timeout_s=10.0)
    if p.returncode != 0:
        return ""
    return (p.stdout or "").strip()


def _ensure_container_running(container_id: str) -> bool:
    if not container_id:
        return False
    p = _run(["docker", "inspect", "-f", "{{.State.Running}}", container_id], timeout_s=10.0)
    if p.returncode != 0:
        return False
    if (p.stdout or "").strip().lower() == "true":
        return True
    started = _run(["docker", "start", container_id], timeout_s=20.0)
    return started.returncode == 0


def ensure_conversation_container(
    *,
    conversation_id: UUID,
    workspace_dir: str,
    openai_api_key: str,
) -> str:
    ensure_image_built()
    name = _container_name_for_conversation(conversation_id)
    existing_id = _get_existing_container_id(name)
    if existing_id:
        if _ensure_container_running(existing_id):
            logger.info("Reusing Data Agent container %s for conversation %s", existing_id, conversation_id)
            return existing_id
        logger.warning("Data Agent container exists but is not running; recreating conv=%s container_id=%s", conversation_id, existing_id)

    Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Starting Data Agent container for conversation %s (workspace=%s)", conversation_id, workspace_dir)

    p = _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-e",
            f"CODEX_API_KEY={openai_api_key}",
            "-e",
            "HOME=/work",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-v",
            f"{workspace_dir}:/work",
            "-w",
            "/work",
            DATA_AGENT_IMAGE,
            "sh",
            "-lc",
            "mkdir -p /work/.codex && tail -f /dev/null",
        ],
        timeout_s=30.0,
    )
    if p.returncode != 0:
        logger.error(
            "Failed to start Data Agent container rc=%s stdout_tail=%s stderr_tail=%s",
            p.returncode,
            (p.stdout or "")[-2000:],
            (p.stderr or "")[-2000:],
        )
        raise RuntimeError(f"Failed to start Data Agent container: {p.stderr.strip() or p.stdout.strip()}")
    cid = (p.stdout or "").strip()
    if not cid:
        logger.error("Failed to start Data Agent container: no container id returned (stdout=%s)", (p.stdout or "").strip())
        raise RuntimeError("Failed to start Data Agent container: no container id returned.")
    logger.info("Started Data Agent container %s for conversation %s", cid, conversation_id)
    return cid


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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
    timeout_s: float = 600.0,
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
    auth_path = ws / "auth.json"
    agents_path = ws / "AGENTS.md"
    ctx_path = ws / "conversation_context.json"

    _write_text(api_spec_path, api_spec_text or "")
    _write_text(auth_path, (auth_json or "{}").strip() or "{}")
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

    prompt = (
        "You are the Data Agent for this conversation.\n"
        f"- Task (what_to_do): {what_to_do}\n\n"
        "Context files:\n"
        f"- API spec: {api_spec_path.name}\n"
        f"- Auth JSON: {auth_path.name}\n"
        f"- Conversation context: {ctx_path.name}\n\n"
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
    cmd: list[str] = [
        "codex",
        "exec",
        "--model",
        "gpt-5.2",
        "--config",
        'model_reasoning_effort="low"',
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
        "--output-schema",
        "/work/output_schema.json",
        "--output-last-message",
        "/work/output.json",
    ]
    if session_id:
        cmd += ["resume", session_id]
    cmd.append(prompt)
    # Use a shell to avoid PATH issues in minimal images, but keep args safely quoted.
    cmd_str = " ".join(shlex.quote(x) for x in cmd)

    logger.info(
        "Data Agent run: conv=%s container=%s session_id=%s timeout_s=%s",
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
    p = _run_stream_jsonl(
        ["docker", "exec", "-i", container_id, "sh", "-lc", cmd_str],
        timeout_s=timeout_s,
        log_prefix=log_prefix,
        on_event=_on_event,
    )
    elapsed_s = time.time() - started_at
    logger.info("Data Agent finished: conv=%s rc=%s elapsed_s=%.2f", conversation_id, p.returncode, elapsed_s)
    logger.debug(
        "Data Agent finished (tails): conv=%s stdout_tail=%s stderr_tail=%s",
        conversation_id,
        _safe_str((p.stdout or "")[-500:], limit=500),
        _safe_str((p.stderr or "")[-500:], limit=500),
    )

    thread_id = thread_id_box["thread_id"] or _extract_thread_id_from_jsonl(p.stdout or "")

    # Preserve per-invocation artifacts so output/debug don't get overwritten.
    try:
        shutil.copyfile(schema_path, schema_snapshot_path)
    except Exception:
        logger.debug("Data Agent: failed to snapshot output schema conv=%s", conversation_id, exc_info=True)
    try:
        if output_path.exists():
            shutil.copyfile(output_path, output_snapshot_path)
    except Exception:
        logger.debug("Data Agent: failed to snapshot output.json conv=%s", conversation_id, exc_info=True)
    try:
        if debug_path.exists():
            shutil.copyfile(debug_path, debug_snapshot_path)
    except Exception:
        logger.debug("Data Agent: failed to snapshot debug.json conv=%s", conversation_id, exc_info=True)

    # Debug info (temporary; cleanup TODO).
    cmd_for_debug = list(cmd)
    if cmd_for_debug:
        # Replace the prompt (last arg) with a short placeholder; prompts can be large and may contain sensitive data.
        cmd_for_debug[-1] = f"<PROMPT len={len(prompt)}>"
    debug_obj = {
        "ts": time.time(),
        "elapsed_s": elapsed_s,
        "cmd": cmd_for_debug,
        "returncode": int(p.returncode),
        "stdout_tail": _redact((p.stdout or "")[-4000:]),
        "stderr_tail": _redact((p.stderr or "")[-4000:]),
        "thread_id": thread_id or "",
        "session_id": session_id or "",
        "token_usage": {"total": token_box["total"], "input": token_box["input"], "output": token_box["output"]},
    }
    _write_json(debug_path, debug_obj)

    if p.returncode != 0:
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
                "ok": False,
                "what_to_do": _safe_str(what_to_do, limit=400),
                "elapsed_s": round(elapsed_s, 3),
                "token_usage": {"total": token_box["total"], "input": token_box["input"], "output": token_box["output"]},
                "output_snapshot": str(output_snapshot_path),
                "debug_snapshot": str(debug_snapshot_path),
                "schema_snapshot": str(schema_snapshot_path),
                "error": _safe_str((p.stderr.strip() or p.stdout.strip() or "Data Agent failed."), limit=2000),
            },
        )
        return DataAgentRunResult(
            ok=False,
            result_text="",
            container_id=container_id,
            session_id=session_id or thread_id or "",
            output_file=str(output_path),
            debug_file=str(debug_path),
            error=_safe_str((p.stderr.strip() or p.stdout.strip() or "Data Agent failed."), limit=2000),
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
            error=f"Failed to parse Data Agent output.json: {exc}",
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
