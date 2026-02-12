from __future__ import annotations

import base64
import json
import uuid
import logging
import os
import re
import shlex
import shutil
import socket
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

DEFAULT_DATA_AGENT_IMAGE = "ghcr.io/mornville/data-agent:latest"

logger = logging.getLogger("voicebot.data_agent")

_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsk-[A-Za-z0-9]{8,}\b"), "sk-[REDACTED]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\\-]{10,}\b"), "Bearer [REDACTED]"),
    (re.compile(r"(?i)(Incorrect API key provided: )([^\\s\"']+)"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(api[_-]?key\\s*[:=]\\s*)([^\\s,;\"']+)"), r"\1[REDACTED]"),
]

_PORT_ALLOC_LOCK = threading.Lock()
_DEFAULT_PORT_RANGE = (8000, 8100)
_DEFAULT_PORTS_PER_CONTAINER = 5


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


def _extract_codex_stream_text(ev: dict[str, Any]) -> str | None:
    t = str(ev.get("type") or "").strip()
    payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
    if t in ("item.started", "item.completed"):
        item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
        if isinstance(item, dict) and str(item.get("type") or "").strip() == "command_execution":
            cmd = _safe_str(item.get("command") or "", limit=160)
            if not cmd:
                return None
            if t == "item.completed":
                exit_code = item.get("exit_code")
                if isinstance(exit_code, int):
                    return f"cmd done (exit {exit_code}): {cmd}"
                return f"cmd done: {cmd}"
            return f"cmd start: {cmd}"
    if t == "response_item":
        if str(payload.get("type") or "").strip() == "message" and str(payload.get("role") or "").strip() == "assistant":
            content = payload.get("content")
            parts: list[str] = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "output_text":
                        text = str(item.get("text") or "")
                        if text:
                            parts.append(text)
            if parts:
                s = _redact("".join(parts).strip())
                if len(s) > 1000:
                    s = s[:1000] + "…"
                return s if s else None
    if t == "event_msg":
        pt = str(payload.get("type") or "").strip()
        if pt in ("tool_call", "tool_result", "tool_error", "status"):
            tool = _safe_str(payload.get("tool") or payload.get("name") or "", limit=80)
            status = _safe_str(payload.get("status") or "", limit=60)
            msg = " ".join(x for x in [pt, tool, status] if x)
            return msg if msg else None
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


@dataclass(frozen=True)
class ContainerCommandResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int


def _run(cmd: list[str], *, timeout_s: float = 300.0) -> subprocess.CompletedProcess[str]:
    logger.debug("Running command: %s", " ".join(shlex.quote(x) for x in cmd))
    try:
        return subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        logger.warning("Command not found: %s", cmd[0] if cmd else "<empty>")
        return subprocess.CompletedProcess(cmd, returncode=127, stdout="", stderr=str(exc))
    except Exception as exc:
        logger.exception("Failed to execute command: %s", " ".join(shlex.quote(x) for x in cmd))
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr=str(exc))


def _docker_available() -> bool:
    try:
        p = _run(["docker", "version"], timeout_s=10.0)
        if p.returncode != 0:
            logger.warning("Docker not available: docker version rc=%s stderr=%s", p.returncode, (p.stderr or "").strip())
        return p.returncode == 0
    except Exception:
        logger.exception("Docker not available: exception when running docker version")
        return False


def docker_available() -> bool:
    return _docker_available()


def _data_agent_state_dir() -> Path:
    raw = (
        os.environ.get("VOICEBOT_DATA_DIR")
        or os.environ.get("DATA_DIR")
        or str(Path.home() / ".gravexstudio")
    )
    path = Path(raw).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path


def _port_allocations_path() -> Path:
    return _data_agent_state_dir() / "data_agent_ports.json"


def _load_port_allocations() -> dict[str, Any]:
    path = _port_allocations_path()
    if not path.exists():
        return {"version": 1, "ports": {}}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "ports": {}}
    if not isinstance(obj, dict):
        return {"version": 1, "ports": {}}
    ports = obj.get("ports")
    if not isinstance(ports, dict):
        obj["ports"] = {}
    if "version" not in obj:
        obj["version"] = 1
    return obj


def _save_port_allocations(obj: dict[str, Any]) -> None:
    path = _port_allocations_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _parse_port_range(raw: str) -> tuple[int, int]:
    s = (raw or "").strip()
    if not s:
        return _DEFAULT_PORT_RANGE
    m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", s)
    if not m:
        return _DEFAULT_PORT_RANGE
    try:
        start = int(m.group(1))
        end = int(m.group(2))
    except Exception:
        return _DEFAULT_PORT_RANGE
    if start < 1 or end < 1 or start > 65535 or end > 65535:
        return _DEFAULT_PORT_RANGE
    if end < start:
        start, end = end, start
    return start, end


def _port_range() -> tuple[int, int]:
    raw = (
        os.environ.get("IGX_DATA_AGENT_PORT_RANGE")
        or os.environ.get("VOICEBOT_DATA_AGENT_PORT_RANGE")
        or ""
    )
    return _parse_port_range(raw)


def _ports_per_container() -> int:
    raw = (
        os.environ.get("IGX_DATA_AGENT_PORTS_PER_CONTAINER")
        or os.environ.get("VOICEBOT_DATA_AGENT_PORTS_PER_CONTAINER")
        or ""
    )
    try:
        val = int(raw)
    except Exception:
        val = _DEFAULT_PORTS_PER_CONTAINER
    if val < 0:
        val = 0
    if val > 50:
        val = 50
    return val


def _port_available(port: int) -> bool:
    if port <= 0 or port > 65535:
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
        return True
    except Exception:
        return False


def _ports_for_container(data: dict[str, Any], *, container_id: str, container_name: str) -> list[dict[str, int]]:
    ports_obj = data.get("ports")
    if not isinstance(ports_obj, dict):
        return []
    items: list[dict[str, int]] = []
    for host_str, entry in ports_obj.items():
        if not isinstance(entry, dict):
            continue
        match_id = container_id and entry.get("container_id") == container_id
        match_name = container_name and entry.get("container_name") == container_name
        if not (match_id or match_name):
            continue
        try:
            host_port = int(host_str)
        except Exception:
            continue
        try:
            container_port = int(entry.get("container_port") or host_port)
        except Exception:
            container_port = host_port
        items.append({"host": host_port, "container": container_port})
    items.sort(key=lambda x: x["host"])
    return items


def reserve_container_ports(
    *,
    conversation_id: UUID,
    container_name: str,
    container_id: str = "",
) -> list[dict[str, int]]:
    name = (container_name or "").strip()
    cid = (container_id or "").strip()
    if not name:
        return []
    with _PORT_ALLOC_LOCK:
        data = _load_port_allocations()
        existing = _ports_for_container(data, container_id=cid, container_name=name)
        if existing:
            return existing
        count = _ports_per_container()
        if count <= 0:
            return []
        start, end = _port_range()
        ports_obj = data.get("ports")
        if not isinstance(ports_obj, dict):
            ports_obj = {}
            data["ports"] = ports_obj
        used = set()
        for key in ports_obj.keys():
            try:
                used.add(int(key))
            except Exception:
                continue
        picked: list[int] = []
        for p in range(start, end + 1):
            if p in used:
                continue
            if not _port_available(p):
                continue
            picked.append(p)
            if len(picked) >= count:
                break
        if not picked:
            return []
        now = datetime.now(timezone.utc).isoformat()
        for p in picked:
            ports_obj[str(p)] = {
                "container_port": p,
                "container_id": cid,
                "container_name": name,
                "conversation_id": str(conversation_id),
                "assigned_at": now,
            }
        _save_port_allocations(data)
        return [{"host": p, "container": p} for p in picked]


def assign_container_id_to_ports(*, container_id: str, container_name: str) -> None:
    cid = (container_id or "").strip()
    name = (container_name or "").strip()
    if not cid or not name:
        return
    with _PORT_ALLOC_LOCK:
        data = _load_port_allocations()
        ports_obj = data.get("ports")
        if not isinstance(ports_obj, dict):
            return
        changed = False
        for entry in ports_obj.values():
            if not isinstance(entry, dict):
                continue
            if entry.get("container_name") != name:
                continue
            if entry.get("container_id") != cid:
                entry["container_id"] = cid
                changed = True
        if changed:
            _save_port_allocations(data)


def release_container_ports(*, container_id: str = "", container_name: str = "") -> list[dict[str, int]]:
    cid = (container_id or "").strip()
    name = (container_name or "").strip()
    if not cid and not name:
        return []
    removed: list[dict[str, int]] = []
    with _PORT_ALLOC_LOCK:
        data = _load_port_allocations()
        ports_obj = data.get("ports")
        if not isinstance(ports_obj, dict):
            return []
        next_ports: dict[str, Any] = {}
        for host_str, entry in ports_obj.items():
            if not isinstance(entry, dict):
                continue
            match_id = cid and entry.get("container_id") == cid
            match_name = name and entry.get("container_name") == name
            if match_id or match_name:
                try:
                    host_port = int(host_str)
                except Exception:
                    host_port = 0
                try:
                    container_port = int(entry.get("container_port") or host_port)
                except Exception:
                    container_port = host_port
                if host_port:
                    removed.append({"host": host_port, "container": container_port})
                continue
            next_ports[host_str] = entry
        data["ports"] = next_ports
        _save_port_allocations(data)
    removed.sort(key=lambda x: x["host"])
    return removed


def _inspect_container_ports(container_id: str) -> list[dict[str, int]]:
    cid = (container_id or "").strip()
    if not cid:
        return []
    p = _run(["docker", "inspect", cid], timeout_s=10.0)
    if p.returncode != 0:
        return []
    try:
        obj = json.loads(p.stdout or "")
    except Exception:
        return []
    if not isinstance(obj, list) or not obj:
        return []
    data = obj[0] if isinstance(obj[0], dict) else {}
    ports = data.get("NetworkSettings", {}).get("Ports", {})
    if not isinstance(ports, dict):
        return []
    out: list[dict[str, int]] = []
    for key, host_list in ports.items():
        if not isinstance(key, str) or not isinstance(host_list, list):
            continue
        try:
            container_port = int(key.split("/", 1)[0])
        except Exception:
            continue
        for host in host_list:
            if not isinstance(host, dict):
                continue
            try:
                host_port = int(host.get("HostPort") or 0)
            except Exception:
                host_port = 0
            if host_port:
                out.append({"host": host_port, "container": container_port})
    dedup: dict[int, dict[str, int]] = {}
    for item in out:
        dedup[item["host"]] = item
    return sorted(dedup.values(), key=lambda x: x["host"])


def sync_container_ports_from_docker(
    *,
    container_id: str,
    container_name: str,
    conversation_id: UUID | None = None,
) -> list[dict[str, int]]:
    cid = (container_id or "").strip()
    name = (container_name or "").strip()
    if not cid or not name:
        return []
    mappings = _inspect_container_ports(cid)
    if not mappings:
        return []
    with _PORT_ALLOC_LOCK:
        data = _load_port_allocations()
        ports_obj = data.get("ports")
        if not isinstance(ports_obj, dict):
            ports_obj = {}
            data["ports"] = ports_obj
        now = datetime.now(timezone.utc).isoformat()
        for item in mappings:
            host_port = int(item.get("host") or 0)
            if not host_port:
                continue
            ports_obj[str(host_port)] = {
                "container_port": int(item.get("container") or host_port),
                "container_id": cid,
                "container_name": name,
                "conversation_id": str(conversation_id) if conversation_id else "",
                "assigned_at": now,
            }
        _save_port_allocations(data)
        return _ports_for_container(data, container_id=cid, container_name=name)


def get_container_ports(
    *,
    container_id: str = "",
    container_name: str = "",
    conversation_id: UUID | None = None,
) -> list[dict[str, int]]:
    cid = (container_id or "").strip()
    name = (container_name or "").strip()
    with _PORT_ALLOC_LOCK:
        data = _load_port_allocations()
        existing = _ports_for_container(data, container_id=cid, container_name=name)
    if existing:
        return existing
    if cid and name:
        return sync_container_ports_from_docker(
            container_id=cid,
            container_name=name,
            conversation_id=conversation_id,
        )
    return []


def list_data_agent_containers() -> dict:
    if not _docker_available():
        return {"docker_available": False, "items": []}
    p = _run(
        ["docker", "ps", "--filter", "name=^igx-data-agent-", "--format", "{{json .}}"],
        timeout_s=10.0,
    )
    if p.returncode != 0:
        logger.warning("docker ps failed rc=%s stderr=%s", p.returncode, (p.stderr or "").strip())
        return {"docker_available": True, "items": [], "error": (p.stderr or "").strip()}
    items: list[dict] = []
    ids: list[str] = []
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        name = str(data.get("Names") or "")
        cid = str(data.get("ID") or "")
        if cid:
            ids.append(cid)
        items.append(
            {
                "id": cid,
                "name": name,
                "image": str(data.get("Image") or ""),
                "status": str(data.get("Status") or ""),
                "created_at": str(data.get("CreatedAt") or ""),
                "running_for": str(data.get("RunningFor") or ""),
                "conversation_id": _conversation_id_from_container_name(name),
            }
        )
    stats_map: dict[str, dict] = {}
    if ids:
        stats = _run(
            ["docker", "stats", "--no-stream", "--format", "{{json .}}"] + ids,
            timeout_s=10.0,
        )
        if stats.returncode == 0:
            for line in (stats.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except Exception:
                    continue
                sid = str(s.get("ID") or "")
                if not sid:
                    continue
                stats_map[sid] = {
                    "cpu": str(s.get("CPUPerc") or ""),
                    "mem": str(s.get("MemUsage") or ""),
                    "mem_perc": str(s.get("MemPerc") or ""),
                }
    for item in items:
        stat = stats_map.get(item.get("id") or "")
        if stat:
            item.update(stat)
    return {"docker_available": True, "items": items}


def stop_data_agent_container(container_id_or_name: str) -> dict:
    if not _docker_available():
        return {"docker_available": False, "stopped": False, "error": "Docker is not available"}
    target = (container_id_or_name or "").strip()
    if not target:
        return {"docker_available": True, "stopped": False, "error": "Missing container id"}
    container_name = ""
    try:
        inspect = _run(["docker", "inspect", "-f", "{{.Name}}", target], timeout_s=5.0)
        if inspect.returncode == 0:
            container_name = (inspect.stdout or "").strip().lstrip("/")
    except Exception:
        container_name = ""
    p = _run(["docker", "rm", "-f", target], timeout_s=20.0)
    if p.returncode != 0:
        return {
            "docker_available": True,
            "stopped": False,
            "error": (p.stderr or p.stdout or "").strip(),
        }
    try:
        release_container_ports(container_id=target, container_name=container_name)
    except Exception:
        logger.debug("Failed to release Isolated Workspace ports for %s", target, exc_info=True)
    return {"docker_available": True, "stopped": True}


def _get_data_agent_image() -> str:
    return (
        os.environ.get("IGX_DATA_AGENT_IMAGE")
        or os.environ.get("VOICEBOT_DATA_AGENT_IMAGE")
        or DEFAULT_DATA_AGENT_IMAGE
    )


def ensure_image_pulled() -> str:
    if not _docker_available():
        raise RuntimeError("Docker is not available (cannot start Isolated Workspace runtime).")
    image = _get_data_agent_image().strip() or DEFAULT_DATA_AGENT_IMAGE
    p = _run(["docker", "image", "inspect", image], timeout_s=10.0)
    if p.returncode == 0:
        logger.info("Isolated Workspace image present: %s", image)
        return image
    logger.info("Isolated Workspace image missing; pulling: %s", image)
    pull = _run(["docker", "pull", image], timeout_s=900.0)
    if pull.returncode != 0:
        logger.error(
            "Failed to pull Isolated Workspace image rc=%s stdout_tail=%s stderr_tail=%s",
            pull.returncode,
            (pull.stdout or "")[-2000:],
            (pull.stderr or "")[-2000:],
        )
        raise RuntimeError(
            "Failed to pull Isolated Workspace image. "
            "Check your network/login or set IGX_DATA_AGENT_IMAGE to a reachable image."
        )
    logger.info("Pulled Isolated Workspace image: %s", image)
    return image


def _container_name_for_conversation(conversation_id: UUID) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(conversation_id))
    return f"igx-data-agent-{s}"


def container_name_for_conversation(conversation_id: UUID) -> str:
    return _container_name_for_conversation(conversation_id)


def _conversation_id_from_container_name(name: str) -> str:
    prefix = "igx-data-agent-"
    if not name.startswith(prefix):
        return ""
    raw = name[len(prefix) :].strip()
    if not raw:
        return ""
    candidate = raw.replace("_", "-")
    try:
        return str(uuid.UUID(candidate))
    except Exception:
        return ""


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


def get_container_status(
    *,
    conversation_id: Optional[UUID] = None,
    container_id: str = "",
    container_name: str = "",
) -> dict:
    name = (container_name or "").strip()
    if not name and conversation_id is not None:
        name = _container_name_for_conversation(conversation_id)

    cid = (container_id or "").strip()
    if not cid and name:
        cid = _get_existing_container_id(name)

    if not _docker_available():
        return {
            "docker_available": False,
            "exists": False,
            "running": False,
            "status": "",
            "container_id": cid,
            "container_name": name,
            "error": "Docker not available",
        }

    if not cid:
        return {
            "docker_available": True,
            "exists": False,
            "running": False,
            "status": "",
            "container_id": "",
            "container_name": name,
        }

    p = _run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.State.Status}}|{{.State.Running}}|{{.State.StartedAt}}|{{.State.FinishedAt}}|{{.Name}}|{{.Id}}",
            cid,
        ],
        timeout_s=10.0,
    )
    if p.returncode != 0:
        return {
            "docker_available": True,
            "exists": False,
            "running": False,
            "status": "",
            "container_id": cid,
            "container_name": name,
            "error": (p.stderr or p.stdout or "").strip(),
        }

    parts = (p.stdout or "").strip().split("|")
    status = parts[0].strip() if len(parts) > 0 else ""
    running = parts[1].strip().lower() == "true" if len(parts) > 1 else False
    started_at = parts[2].strip() if len(parts) > 2 else ""
    finished_at = parts[3].strip() if len(parts) > 3 else ""
    name_out = parts[4].strip().lstrip("/") if len(parts) > 4 else ""
    cid_out = parts[5].strip() if len(parts) > 5 else cid

    return {
        "docker_available": True,
        "exists": True,
        "running": running,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "container_id": cid_out or cid,
        "container_name": name_out or name,
    }


def run_container_command(*, container_id: str, command: str, timeout_s: float = 60.0) -> ContainerCommandResult:
    cmd = ["docker", "exec", "-i", container_id, "sh", "-lc", command]
    p = _run(cmd, timeout_s=timeout_s)
    return ContainerCommandResult(
        ok=p.returncode == 0,
        stdout=(p.stdout or "").strip(),
        stderr=(p.stderr or "").strip(),
        exit_code=int(p.returncode),
    )


def ensure_conversation_container(
    *,
    conversation_id: UUID,
    workspace_dir: str,
    openai_api_key: str,
    git_token: str = "",
    auth_json: str = "",
) -> str:
    image = ensure_image_pulled()
    name = _container_name_for_conversation(conversation_id)
    existing_id = _get_existing_container_id(name)
    if existing_id:
        if _ensure_container_running(existing_id):
            logger.info("Reusing Isolated Workspace container %s for conversation %s", existing_id, conversation_id)
            try:
                sync_container_ports_from_docker(
                    container_id=existing_id,
                    container_name=name,
                    conversation_id=conversation_id,
                )
            except Exception:
                logger.debug("Failed to sync Isolated Workspace ports for %s", existing_id, exc_info=True)
            return existing_id
        logger.warning("Isolated Workspace container exists but is not running; recreating conv=%s container_id=%s", conversation_id, existing_id)

    Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    # Back-compat cleanup: older versions wrote API spec to API_SPEC.md.
    # Keep the workspace consistent with the new api_spec.json naming.
    try:
        legacy = Path(workspace_dir) / "API_SPEC.md"
        if legacy.exists():
            legacy.unlink()
    except Exception:
        pass
    logger.info("Starting Isolated Workspace container for conversation %s (workspace=%s)", conversation_id, workspace_dir)

    repo_url, repo_cache_path, repo_source_path = _extract_preferred_repo(auth_json or "")
    host_cache_path = _normalize_host_path(repo_cache_path)
    host_source_path = _normalize_host_path(repo_source_path)
    if host_cache_path:
        try:
            Path(host_cache_path).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    ports = reserve_container_ports(conversation_id=conversation_id, container_name=name)
    cmd = [
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
    ]
    for mapping in ports:
        host_port = int(mapping.get("host") or 0)
        container_port = int(mapping.get("container") or 0)
        if host_port > 0 and container_port > 0:
            cmd.extend(["-p", f"127.0.0.1:{host_port}:{container_port}"])
    if git_token:
        cmd.extend(["-e", f"GIT_TOKEN={git_token}", "-e", f"GITHUB_TOKEN={git_token}"])
    if host_cache_path:
        cmd.extend(["-v", f"{host_cache_path}:/work/.repo_cache/preferred.git"])
    if host_source_path:
        cmd.extend(["-v", f"{host_source_path}:/work/.repo_source:ro"])
    cmd.extend(
        [
            "-v",
            f"{workspace_dir}:/work",
            "-w",
            "/work",
            image,
            "sh",
            "-lc",
            "mkdir -p /work/.codex && tail -f /dev/null",
        ]
    )
    p = _run(cmd, timeout_s=30.0)
    if p.returncode != 0:
        # If another concurrent kickoff created the container after we checked, docker run may fail
        # with a name conflict. In that case, reuse the existing container instead of failing.
        stderr = (p.stderr or "")
        if "Conflict. The container name" in stderr and name in stderr:
            existing_id = _get_existing_container_id(name)
            if existing_id and _ensure_container_running(existing_id):
                release_container_ports(container_name=name)
                try:
                    sync_container_ports_from_docker(
                        container_id=existing_id,
                        container_name=name,
                        conversation_id=conversation_id,
                    )
                except Exception:
                    logger.debug("Failed to sync Isolated Workspace ports for %s", existing_id, exc_info=True)
                logger.warning(
                    "Isolated Workspace container name conflict; reusing existing container %s for conversation %s",
                    existing_id,
                    conversation_id,
                )
                return existing_id
        release_container_ports(container_name=name)
        logger.error(
            "Failed to start Isolated Workspace container rc=%s stdout_tail=%s stderr_tail=%s",
            p.returncode,
            (p.stdout or "")[-2000:],
            (p.stderr or "")[-2000:],
        )
        raise RuntimeError(f"Failed to start Isolated Workspace container: {p.stderr.strip() or p.stdout.strip()}")
    cid = (p.stdout or "").strip()
    if not cid:
        release_container_ports(container_name=name)
        logger.error("Failed to start Isolated Workspace container: no container id returned (stdout=%s)", (p.stdout or "").strip())
        raise RuntimeError("Failed to start Isolated Workspace container: no container id returned.")
    assign_container_id_to_ports(container_id=cid, container_name=name)
    logger.info("Started Isolated Workspace container %s for conversation %s", cid, conversation_id)
    return cid


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _setup_ssh_from_auth(ws: Path, auth_json: str) -> None:
    try:
        auth_obj = json.loads((auth_json or "").strip() or "{}")
    except Exception:
        return
    if not isinstance(auth_obj, dict):
        return

    key = str(auth_obj.get("ssh_private_key") or auth_obj.get("ssh_key") or auth_obj.get("ssh_private_key_pem") or "")
    if not key:
        key_path = str(auth_obj.get("ssh_private_key_path") or auth_obj.get("ssh_key_path") or "").strip()
        if key_path:
            try:
                key = Path(key_path).read_text(encoding="utf-8")
            except Exception:
                key = ""
    if not key:
        key_b64 = str(auth_obj.get("ssh_private_key_b64") or auth_obj.get("ssh_private_key_base64") or "")
        if key_b64:
            try:
                key = base64.b64decode(key_b64).decode("utf-8", errors="ignore")
            except Exception:
                key = ""
    if not key.strip():
        return

    ssh_dir = ws / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    key_name = str(auth_obj.get("ssh_key_filename") or "id_ed25519").strip() or "id_ed25519"
    key_path = ssh_dir / key_name
    _write_text(key_path, key.strip() + "\n")
    try:
        os.chmod(ssh_dir, 0o700)
        os.chmod(key_path, 0o600)
    except Exception:
        pass

    pub_key = str(auth_obj.get("ssh_public_key") or "")
    if not pub_key:
        pub_key_path = str(auth_obj.get("ssh_public_key_path") or "").strip()
        if pub_key_path:
            try:
                pub_key = Path(pub_key_path).read_text(encoding="utf-8")
            except Exception:
                pub_key = ""
    if pub_key.strip():
        _write_text(ssh_dir / f"{key_name}.pub", pub_key.strip() + "\n")

    known_hosts = str(auth_obj.get("ssh_known_hosts") or auth_obj.get("known_hosts") or "").strip()
    if not known_hosts:
        known_hosts_path = str(auth_obj.get("ssh_known_hosts_path") or "").strip()
        if known_hosts_path:
            try:
                known_hosts = Path(known_hosts_path).read_text(encoding="utf-8").strip()
            except Exception:
                known_hosts = ""
    if known_hosts:
        _write_text(ssh_dir / "known_hosts", known_hosts + "\n")

    passphrase = str(
        auth_obj.get("ssh_key_passphrase")
        or auth_obj.get("ssh_private_key_passphrase")
        or auth_obj.get("ssh_passphrase")
        or ""
    )
    passphrase = passphrase.strip()
    if passphrase:
        pass_path = ssh_dir / "passphrase"
        _write_text(pass_path, passphrase)
        askpass_path = ssh_dir / "askpass.sh"
        _write_text(askpass_path, "#!/bin/sh\ncat /work/.ssh/passphrase\n")
        try:
            os.chmod(pass_path, 0o600)
            os.chmod(askpass_path, 0o700)
        except Exception:
            pass

    ssh_user = str(auth_obj.get("ssh_user") or auth_obj.get("git_ssh_user") or "git").strip() or "git"
    ssh_host = str(auth_obj.get("ssh_host") or auth_obj.get("git_host") or "github.com").strip() or "github.com"
    strict = str(auth_obj.get("ssh_strict_host_key_checking") or "").strip().lower()
    if not strict:
        strict = "yes" if known_hosts else "accept-new"

    config = (
        f"Host {ssh_host}\n"
        f"  HostName {ssh_host}\n"
        f"  User {ssh_user}\n"
        f"  IdentityFile /work/.ssh/{key_name}\n"
        "  IdentitiesOnly yes\n"
        f"  StrictHostKeyChecking {strict}\n"
        "  UserKnownHostsFile /work/.ssh/known_hosts\n"
    )
    _write_text(ssh_dir / "config", config)


def _parse_auth_json(auth_json: str) -> dict[str, Any]:
    try:
        obj = json.loads((auth_json or "").strip() or "{}")
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    return obj


def _extract_preferred_repo(auth_json: str) -> tuple[str, str, str]:
    obj = _parse_auth_json(auth_json)
    repo_url = str(
        obj.get("preferred_repo_url")
        or obj.get("git_preferred_repo_url")
        or obj.get("git_repo_url")
        or obj.get("preferred_repo")
        or ""
    ).strip()
    cache_path = str(
        obj.get("preferred_repo_cache_path")
        or obj.get("git_repo_cache_path")
        or obj.get("preferred_repo_path")
        or ""
    ).strip()
    source_path = str(
        obj.get("preferred_repo_source_path")
        or obj.get("git_repo_source_path")
        or obj.get("preferred_repo_working_path")
        or ""
    ).strip()
    return repo_url, cache_path, source_path


def _repo_name_from_url(url: str) -> str:
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    if u.endswith(".git"):
        u = u[: -len(".git")]
    if ":" in u and "://" not in u:
        u = u.split(":", 1)[1]
    if "/" in u:
        u = u.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", u) or "repo"


def _build_ssh_env_prefix(ws: Path) -> str:
    env_prefix: list[str] = []
    ssh_config = ws / ".ssh" / "config"
    if ssh_config.exists():
        env_prefix.append('GIT_SSH_COMMAND="ssh -F /work/.ssh/config"')
    askpass = ws / ".ssh" / "askpass.sh"
    if askpass.exists():
        env_prefix.append("SSH_ASKPASS=/work/.ssh/askpass.sh")
        env_prefix.append("SSH_ASKPASS_REQUIRE=force")
        env_prefix.append("DISPLAY=1")
    return " ".join(env_prefix)


def _normalize_host_path(raw: str) -> str:
    if not raw:
        return ""
    p = Path(raw).expanduser()
    if p.is_absolute():
        return str(p)
    try:
        return str((Path.cwd() / p).resolve())
    except Exception:
        return str(p)


def _ensure_preferred_repo(container_id: str, *, ws: Path, auth_json: str) -> None:
    repo_url, cache_path, source_path = _extract_preferred_repo(auth_json)
    if not repo_url or (not cache_path and not source_path):
        return
    repo_name = _repo_name_from_url(repo_url)
    if not repo_name:
        return
    logger.info(
        "Preferred repo setup: repo=%s cache=%s source=%s",
        _safe_str(repo_url, limit=120),
        _safe_str(cache_path, limit=120),
        _safe_str(source_path, limit=120),
    )
    mirror_path = "/work/.repo_cache/preferred.git"
    source_mount = "/work/.repo_source"
    if source_path:
        check = run_container_command(
            container_id=container_id,
            command=f"[ -d {source_mount}/.git ] || [ -f {source_mount}/HEAD ]",
            timeout_s=5.0,
        )
        if not check.ok:
            logger.warning(
                "Preferred repo source path not mounted; reuse requires a new container. source=%s",
                _safe_str(source_path, limit=120),
            )
            return
    env_prefix = _build_ssh_env_prefix(ws)
    # Ensure mirror exists/updated, then ensure working tree exists.
    ref_expr = (
        f"REF=''; "
        f"if [ -d {source_mount}/.git ] || [ -f {source_mount}/HEAD ]; then REF='{source_mount}'; "
        f"elif [ -d {mirror_path}/objects ]; then REF='{mirror_path}'; "
        f"fi; "
        f"REF_ARG=''; if [ -n \"$REF\" ]; then REF_ARG=\"--reference $REF\"; fi; "
    )
    mirror_setup = (
        f"mkdir -p /work/.repo_cache && "
        f"if [ -d {mirror_path}/objects ]; then "
        f"  git -C {mirror_path} remote update --prune; "
        f"elif [ -z \"$REF\" ]; then "
        f"  git clone --mirror {shlex.quote(repo_url)} {mirror_path}; "
        f"fi; "
    )
    clone_setup = (
        f"if [ -d /work/{repo_name}/.git ]; then "
        f"  git -C /work/{repo_name} fetch --all --prune; "
        f"else "
        f"  git clone $REF_ARG {shlex.quote(repo_url)} /work/{repo_name}; "
        f"fi"
    )
    script = ref_expr + mirror_setup + clone_setup
    if env_prefix:
        script = f"{env_prefix} {script}"
    res = run_container_command(container_id=container_id, command=script, timeout_s=600.0)
    if not res.ok:
        logger.warning(
            "Preferred repo prefetch failed: repo=%s cache=%s exit=%s stderr=%s",
            _safe_str(repo_url, limit=120),
            _safe_str(cache_path, limit=120),
            res.exit_code,
            _safe_str(res.stderr, limit=240),
        )


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
    cmd: list[str] = [
        "codex",
        "exec",
        "--model",
        "gpt-5.2",
        "--config",
        'model_reasoning_effort="high"',
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
    env_prefix = _build_ssh_env_prefix(ws)
    if env_prefix:
        cmd_str = f"{env_prefix} {cmd_str}"

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
    p = _run_stream_jsonl(
        ["docker", "exec", "-i", container_id, "sh", "-lc", cmd_str],
        timeout_s=timeout_s,
        log_prefix=log_prefix,
        on_event=_on_event,
    )
    elapsed_s = time.time() - started_at
    logger.info("Isolated Workspace finished: conv=%s rc=%s elapsed_s=%.2f", conversation_id, p.returncode, elapsed_s)
    logger.debug(
        "Isolated Workspace finished (tails): conv=%s stdout_tail=%s stderr_tail=%s",
        conversation_id,
        _safe_str((p.stdout or "")[-500:], limit=500),
        _safe_str((p.stderr or "")[-500:], limit=500),
    )

    thread_id = thread_id_box["thread_id"] or _extract_thread_id_from_jsonl(p.stdout or "")

    # Preserve per-invocation artifacts so output/debug don't get overwritten.
    try:
        shutil.copyfile(schema_path, schema_snapshot_path)
    except Exception:
        logger.debug("Isolated Workspace: failed to snapshot output schema conv=%s", conversation_id, exc_info=True)
    try:
        if output_path.exists():
            shutil.copyfile(output_path, output_snapshot_path)
    except Exception:
        logger.debug("Isolated Workspace: failed to snapshot output.json conv=%s", conversation_id, exc_info=True)
    try:
        if debug_path.exists():
            shutil.copyfile(debug_path, debug_snapshot_path)
    except Exception:
        logger.debug("Isolated Workspace: failed to snapshot debug.json conv=%s", conversation_id, exc_info=True)

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
                "error": _safe_str((p.stderr.strip() or p.stdout.strip() or "Isolated Workspace failed."), limit=2000),
            },
        )
        return DataAgentRunResult(
            ok=False,
            result_text="",
            container_id=container_id,
            session_id=session_id or thread_id or "",
            output_file=str(output_path),
            debug_file=str(debug_path),
            error=_safe_str((p.stderr.strip() or p.stdout.strip() or "Isolated Workspace failed."), limit=2000),
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
