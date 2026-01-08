from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import UUID


DEFAULT_DATA_AGENT_SYSTEM_PROMPT = (
    "You are given a task (what_to_do), API spec, authorization tokens, and conversation context. "
    "Call any API if needed, satisfy what_to_do, and respond back with a simple response."
)

DATA_AGENT_IMAGE = "igx-data-agent:latest"


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
        return p.returncode == 0
    except Exception:
        return False


def ensure_image_built() -> None:
    if not _docker_available():
        raise RuntimeError("Docker is not available (cannot start Data Agent runtime).")
    p = _run(["docker", "image", "inspect", DATA_AGENT_IMAGE], timeout_s=10.0)
    if p.returncode == 0:
        return
    dockerfile = Path(__file__).with_name("Dockerfile")
    context_dir = dockerfile.parent
    build = _run(
        ["docker", "build", "-t", DATA_AGENT_IMAGE, "-f", str(dockerfile), str(context_dir)],
        timeout_s=900.0,
    )
    if build.returncode != 0:
        raise RuntimeError(f"Failed to build Data Agent image: {build.stderr.strip() or build.stdout.strip()}")


def _container_name_for_conversation(conversation_id: UUID) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(conversation_id))
    return f"igx-data-agent-{s}"


def _get_existing_container_id(name: str) -> str:
    p = _run(["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.ID}}"], timeout_s=10.0)
    if p.returncode != 0:
        return ""
    return (p.stdout or "").strip()


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
        return existing_id

    Path(workspace_dir).mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError(f"Failed to start Data Agent container: {p.stderr.strip() or p.stdout.strip()}")
    cid = (p.stdout or "").strip()
    if not cid:
        raise RuntimeError("Failed to start Data Agent container: no container id returned.")
    return cid


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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

    # Files visible to Codex.
    api_spec_path = ws / "API_SPEC.md"
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

    schema = {
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "result_text": {"type": "string"},
        },
        "required": ["ok", "result_text"],
        "additionalProperties": True,
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
    base = [
        "codex",
        "exec",
    ]
    if session_id:
        base = ["codex", "exec", "resume", session_id]

    # Use strict, non-interactive behavior.
    # --yolo bypasses approvals/sandbox; --search allows web search; --json emits JSONL events; -o writes the last message.
    cmd = (
        base
        + [
            "--yolo",
            "--search",
            "--skip-git-repo-check",
            "--json",
            "--output-schema",
            "/work/output_schema.json",
            "-o",
            "/work/output.json",
            prompt,
        ]
    )
    # Use a shell to avoid PATH issues in minimal images, but keep args safely quoted.
    cmd_str = " ".join(shlex.quote(x) for x in cmd)

    started_at = time.time()
    p = _run(["docker", "exec", "-i", container_id, "sh", "-lc", cmd_str], timeout_s=timeout_s)
    elapsed_s = time.time() - started_at

    thread_id = _extract_thread_id_from_jsonl(p.stdout or "")

    # Debug info (temporary; cleanup TODO).
    debug_obj = {
        "ts": time.time(),
        "elapsed_s": elapsed_s,
        "cmd": cmd,
        "returncode": p.returncode,
        "stdout_tail": (p.stdout or "")[-4000:],
        "stderr_tail": (p.stderr or "")[-4000:],
    }
    _write_json(debug_path, debug_obj)

    if p.returncode != 0:
        return DataAgentRunResult(
            ok=False,
            result_text="",
            container_id=container_id,
            session_id=session_id or thread_id or "",
            output_file=str(output_path),
            debug_file=str(debug_path),
            error=(p.stderr.strip() or p.stdout.strip() or "Data Agent failed."),
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

