from __future__ import annotations

import asyncio
import base64
import datetime as dt
import logging
import mimetypes
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import quote as _url_quote

from sqlmodel import Session

from voicebot.llm.openai_llm import OpenAILLM
from voicebot.models import Bot, Conversation, HostAction
from voicebot.utils.template import safe_json_loads
from voicebot.web.helpers.files import (
    is_path_within_root,
    resolve_workspace_target_for_conversation,
    sanitize_upload_path,
    workspace_dir_for_conversation,
)
from voicebot.web.helpers.llm_keys import get_openai_api_key_global


def parse_host_action_args(patch: dict) -> tuple[str, dict]:
    action = str(patch.get("action") or patch.get("action_type") or "").strip().lower()
    if action not in {"run_shell", "run_applescript"}:
        raise ValueError("Unsupported host action")
    if action == "run_shell":
        command = str(patch.get("command") or patch.get("cmd") or "").strip()
        if not command:
            raise ValueError("Missing command")
        return action, {"command": command}
    script = str(patch.get("script") or patch.get("applescript") or "").strip()
    if not script:
        raise ValueError("Missing script")
    return action, {"script": script}


def host_action_payload(action: HostAction) -> dict:
    return {
        "id": str(action.id),
        "conversation_id": str(action.conversation_id),
        "requested_by_bot_id": str(action.requested_by_bot_id) if action.requested_by_bot_id else None,
        "requested_by_name": action.requested_by_name,
        "action_type": action.action_type,
        "payload": safe_json_loads(action.payload_json or "{}") or {},
        "status": action.status,
        "stdout": action.stdout,
        "stderr": action.stderr,
        "exit_code": action.exit_code,
        "error": action.error,
        "created_at": action.created_at.isoformat() if action.created_at else None,
        "updated_at": action.updated_at.isoformat() if action.updated_at else None,
        "executed_at": action.executed_at.isoformat() if action.executed_at else None,
    }


def create_host_action(
    session: Session,
    *,
    conv: Conversation,
    bot: Bot,
    action_type: str,
    payload: dict,
) -> HostAction:
    now = dt.datetime.now(dt.timezone.utc)
    action = HostAction(
        conversation_id=conv.id,
        requested_by_bot_id=bot.id,
        requested_by_name=bot.name,
        action_type=action_type,
        payload_json=json.dumps(payload, ensure_ascii=False),
    )
    action.status = "pending"
    action.created_at = now
    action.updated_at = now
    session.add(action)
    session.commit()
    session.refresh(action)
    return action


def summarize_screenshot_tool_def() -> dict:
    return {
        "type": "function",
        "name": "summarize_screenshot",
        "description": "Summarize an image stored in the Isolated Workspace workspace (e.g., a captured screenshot).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the image file in the workspace (e.g. screenshots/screenshot-123.png).",
                },
                "prompt": {
                    "type": "string",
                    "description": "Optional prompt to guide the summary.",
                },
            },
            "required": ["path"],
        },
        "strict": False,
    }


def capture_screenshot_tool_def() -> dict:
    return {
        "type": "function",
        "name": "capture_screenshot",
        "description": "Capture a screenshot on the host machine and save it to the Isolated Workspace workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path (within workspace) to write the screenshot file.",
                },
                "follow_up": {
                    "type": "boolean",
                    "description": "If true, force a follow-up assistant reply after the screenshot is ready.",
                },
            },
            "required": ["path"],
        },
        "strict": False,
    }


def screenshot_base_dir() -> Path:
    return Path(os.environ.get("IGX_SCREENSHOT_DIR") or "").expanduser().resolve() if os.environ.get("IGX_SCREENSHOT_DIR") else Path.home() / "Pictures" / "IGX"


def user_screenshot_dir() -> Path:
    base = screenshot_base_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base


def prepare_screenshot_target(conv: Conversation) -> tuple[str, Path]:
    rel = ""
    try:
        rel = sanitize_upload_path("screenshots/screenshot.png")
    except Exception:
        rel = ""
    if not rel:
        rel = "screenshots/screenshot.png"
    try:
        _root, rel, target = resolve_workspace_target_for_conversation(conv, path=rel, include_hidden=True)
    except Exception:
        workspace_dir = workspace_dir_for_conversation(conv)
        target = Path(workspace_dir).resolve() / "screenshots" / "screenshot.png"
        rel = "screenshots/screenshot.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    return rel, target


def copy_screenshot_to_user_dir(source: Path) -> Optional[Path]:
    try:
        base = user_screenshot_dir()
    except Exception:
        return None
    if not source.exists() or not source.is_file():
        return None
    try:
        dest = base / source.name
        idx = 2
        while dest.exists():
            stem = source.stem
            suffix = source.suffix
            dest = base / f"{stem}-{idx}{suffix}"
            idx += 1
        dest.write_bytes(source.read_bytes())
        return dest
    except Exception:
        return None


def maybe_copy_screenshot_from_command(command: str) -> Optional[Path]:
    logger = logging.getLogger("voicebot.web")
    if not command:
        return None
    parts = [p.strip() for p in command.split(" ") if p.strip()]
    if not parts:
        return None
    target: Optional[str] = None
    for part in reversed(parts):
        if part.startswith("-"):
            continue
        target = part
        break
    if not target:
        logger.warning("capture_screenshot: no output target parsed from command=%s", command)
        return None
    src = Path(target)
    if not src.exists():
        logger.warning("capture_screenshot: target missing %s", src)
        return None
    try:
        size = src.stat().st_size
        if size <= 0:
            logger.warning("capture_screenshot: target empty %s", src)
    except Exception:
        logger.exception("capture_screenshot: failed to stat %s", src)
    dest = copy_screenshot_to_user_dir(src)
    if dest:
        logger.info("capture_screenshot: copied to %s", dest)
    else:
        logger.warning("capture_screenshot: copy failed for %s", target)
    return dest


def tool_error_message(tool_result: dict, *, fallback: str) -> str:
    msg = ""
    if isinstance(tool_result, dict):
        err = tool_result.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message") or "").strip()
        elif isinstance(err, str):
            msg = err.strip()
        if not msg:
            summary_error = tool_result.get("summary_error")
            if isinstance(summary_error, str):
                msg = summary_error.strip()
    return msg or fallback


def screencapture_command(target: Path) -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "Screenshot capture is only supported on macOS."
    cmd = f"/usr/sbin/screencapture -x -t png {shlex.quote(str(target))}"
    return True, cmd


def summarize_image_file(
    session: Session,
    *,
    bot: Bot,
    image_path: Path,
    prompt: str,
) -> tuple[bool, str]:
    api_key = get_openai_api_key_global(session)
    if not api_key:
        return False, "OpenAI API key not configured."
    if not image_path.exists() or not image_path.is_file():
        return False, "Image file not found."
    mt, _ = mimetypes.guess_type(str(image_path))
    if not mt or not mt.startswith("image/"):
        ext = str(image_path.suffix or "").lower()
        if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return False, "Unsupported image type."
        mt = "image/png" if ext == ".png" else "image/jpeg"
    try:
        data = image_path.read_bytes()
    except Exception:
        return False, "Failed to read image."
    if not data:
        return False, "Image is empty."
    b64 = base64.b64encode(data).decode("ascii")
    image_url = f"data:{mt};base64,{b64}"
    model = (getattr(bot, "openai_model", "") or "o4-mini").strip() or "o4-mini"
    summary_prompt = (prompt or "").strip() or "Summarize the screenshot. Be concise and structured."
    try:
        llm = OpenAILLM(model=model, api_key=api_key)
        text = llm.complete_vision(prompt=summary_prompt, image_url=image_url)
        return True, text.strip()
    except Exception as exc:
        return False, f"Vision summary failed: {exc}"


def summarize_screenshot(
    session: Session,
    *,
    conv: Conversation,
    bot: Bot,
    path: str,
    prompt: str,
) -> tuple[bool, str, Optional[str]]:
    if not bool(getattr(bot, "enable_data_agent", False)):
        return False, "Isolated Workspace is disabled for this bot.", None
    api_key = get_openai_api_key_global(session)
    if not api_key:
        return False, "OpenAI API key not configured.", None
    if not path:
        return False, "Missing path.", None
    try:
        _root, req_rel, target = resolve_workspace_target_for_conversation(
            conv,
            path=path,
            include_hidden=False,
        )
    except Exception as exc:
        return False, str(exc) or "Invalid path.", None
    if not target.exists() or not target.is_file():
        return False, "Image file not found.", None
    mt, _ = mimetypes.guess_type(str(target))
    if not mt or not mt.startswith("image/"):
        ext = str(target.suffix or "").lower()
        if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return False, "Unsupported image type.", None
        mt = "image/png" if ext == ".png" else "image/jpeg"
    try:
        data = target.read_bytes()
    except Exception:
        return False, "Failed to read image.", None
    if not data:
        return False, "Image is empty.", None
    b64 = base64.b64encode(data).decode("ascii")
    image_url = f"data:{mt};base64,{b64}"
    model = (getattr(bot, "openai_model", "") or "o4-mini").strip() or "o4-mini"
    summary_prompt = (prompt or "").strip() or "Summarize the screenshot. Be concise and structured."
    try:
        llm = OpenAILLM(model=model, api_key=api_key)
        text = llm.complete_vision(prompt=summary_prompt, image_url=image_url)
        return True, text.strip(), req_rel
    except Exception as exc:
        return False, f"Vision summary failed: {exc}", req_rel


def execute_host_action(
    action: HostAction,
    *,
    workspace_dir: Optional[str] = None,
) -> tuple[bool, str, str, Optional[int], str, dict]:
    logger = logging.getLogger("voicebot.web")
    payload = safe_json_loads(action.payload_json or "{}") or {}
    action_type = str(action.action_type or "").strip()
    stdout = ""
    stderr = ""
    exit_code: Optional[int] = None
    error = ""
    result_payload: dict = {}
    is_screencapture = False
    try:
        if action_type == "run_shell":
            command = str(payload.get("command") or "").strip()
            is_screencapture = "screencapture" in command
            if is_screencapture and command:
                logger.info("capture_screenshot: command=%s", command)
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            stdout, stderr, exit_code = res.stdout or "", res.stderr or "", res.returncode
            if exit_code == 0 and is_screencapture:
                copied = maybe_copy_screenshot_from_command(command)
                if copied:
                    result_payload["saved_user_path"] = str(copied)
        elif action_type == "run_applescript":
            if sys.platform != "darwin":
                raise RuntimeError("AppleScript is only supported on macOS")
            script = str(payload.get("script") or "").strip()
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=30)
            stdout, stderr, exit_code = res.stdout or "", res.stderr or "", res.returncode
        else:
            raise RuntimeError("Unknown host action")
        if exit_code is None:
            exit_code = 0
        ok = exit_code == 0
        if not ok:
            error = stderr or stdout or "Host action failed"
            if action_type == "run_shell" and is_screencapture:
                logger.warning("capture_screenshot: screencapture failed exit_code=%s error=%s", exit_code, error)
        return ok, stdout, stderr, exit_code, error, result_payload
    except Exception as exc:
        return False, stdout, stderr, exit_code, str(exc), result_payload


def host_action_requires_approval(bot: Bot) -> bool:
    return bool(getattr(bot, "require_host_action_approval", False))


def build_host_action_tool_result(action: HostAction, *, ok: bool) -> dict:
    return {
        "ok": ok,
        "action_id": str(action.id),
        "status": action.status,
        "action_type": action.action_type,
        "payload": safe_json_loads(action.payload_json or "{}") or {},
        "stdout": action.stdout or "",
        "stderr": action.stderr or "",
        "exit_code": action.exit_code,
        "error": action.error or "",
    }


def finalize_host_action_run(
    session: Session,
    *,
    action: HostAction,
    ok: bool,
    stdout: str,
    stderr: str,
    exit_code: Optional[int],
    error: str,
    result_payload: dict,
) -> dict:
    action.status = "done" if ok else "error"
    action.stdout = stdout or ""
    action.stderr = stderr or ""
    action.exit_code = exit_code
    action.error = error or ""
    if result_payload:
        payload = safe_json_loads(action.payload_json or "{}") or {}
        payload.update(result_payload)
        if result_payload.get("result_path"):
            payload["result_download_url"] = (
                f"/api/conversations/{action.conversation_id}/files/download?path="
                f"{_url_quote(str(result_payload.get('result_path') or ''))}"
            )
        try:
            action.payload_json = json.dumps(payload, ensure_ascii=False)
        except Exception:
            pass
    now = dt.datetime.now(dt.timezone.utc)
    action.executed_at = now
    action.updated_at = now
    session.add(action)
    session.commit()
    session.refresh(action)
    return build_host_action_tool_result(action, ok=ok)


def execute_host_action_and_update(session: Session, *, action: HostAction) -> dict:
    action.status = "running"
    action.updated_at = dt.datetime.now(dt.timezone.utc)
    session.add(action)
    session.commit()
    ok, stdout, stderr, exit_code, error, result_payload = execute_host_action(action)
    return finalize_host_action_run(
        session,
        action=action,
        ok=ok,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        error=error,
        result_payload=result_payload,
    )


async def execute_host_action_and_update_async(session: Session, *, action: HostAction) -> dict:
    action.status = "running"
    action.updated_at = dt.datetime.now(dt.timezone.utc)
    session.add(action)
    session.commit()
    ok, stdout, stderr, exit_code, error, result_payload = await asyncio.to_thread(execute_host_action, action)
    return finalize_host_action_run(
        session,
        action=action,
        ok=ok,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        error=error,
        result_payload=result_payload,
    )
