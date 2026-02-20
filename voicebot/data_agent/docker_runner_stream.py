from __future__ import annotations

import json
import re
import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .docker_runner_constants import logger


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
