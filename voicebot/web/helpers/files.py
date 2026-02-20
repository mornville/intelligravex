from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException
from sqlmodel import Session

from voicebot.data_agent.docker_runner import default_workspace_dir_for_conversation
from voicebot.models import Bot, Conversation
from voicebot.store import get_conversation
from voicebot.utils.template import safe_json_loads


def sanitize_upload_path(raw_name: str) -> str:
    name = (raw_name or "").replace("\\", "/").strip()
    if re.match(r"^[A-Za-z]:/", name):
        name = name[2:]
    name = name.lstrip("/").strip()
    if not name:
        return ""
    parts = []
    for part in name.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            return ""
        parts.append(part)
    return "/".join(parts)


def is_path_within_root(root: Path, child: Path) -> bool:
    root_abs = root.resolve()
    child_abs = child.resolve()
    return child_abs == root_abs or root_abs in child_abs.parents


def should_hide_workspace_path(rel: str, *, include_hidden: bool) -> bool:
    r = (rel or "").lstrip("/").strip()
    if not r:
        return False
    parts = [p for p in r.split("/") if p]
    if not parts:
        return False
    if parts[0] == ".codex":
        return True
    if (not include_hidden) and any(p.startswith(".") for p in parts):
        return True
    deny = {
        "auth.json",
        "AGENTS.md",
        "api_spec.json",
        "output_schema.json",
    }
    if parts[-1] in deny:
        return True
    return False


def workspace_dir_for_conversation(conv: Conversation) -> str:
    meta = safe_json_loads(conv.metadata_json or "{}") or {}
    da = meta.get("data_agent") if isinstance(meta, dict) else {}
    if isinstance(da, dict):
        workspace_dir = str(da.get("workspace_dir") or "").strip()
    else:
        workspace_dir = ""
    return workspace_dir or default_workspace_dir_for_conversation(conv.id)


def resolve_workspace_target_for_conversation(
    conv: Conversation,
    *,
    path: str,
    include_hidden: bool,
) -> tuple[Path, str, Path]:
    rel = sanitize_upload_path(path)
    if not rel:
        raise ValueError("Invalid path")
    if should_hide_workspace_path(rel, include_hidden=bool(include_hidden)):
        raise ValueError("Path not allowed")
    root = Path(workspace_dir_for_conversation(conv)).resolve()
    target = (root / rel).resolve()
    if not is_path_within_root(root, target):
        raise ValueError("Invalid path")
    return root, rel, target


def data_agent_workspace_dir_for_conversation(meta: dict, *, conversation_id: UUID) -> str:
    da = meta.get("data_agent") if isinstance(meta, dict) else {}
    if isinstance(da, dict):
        workspace_dir = str(da.get("workspace_dir") or "").strip()
    else:
        workspace_dir = ""
    return workspace_dir or default_workspace_dir_for_conversation(conversation_id)


def resolve_data_agent_target(
    session: Session,
    *,
    conversation_id: UUID,
    path: str,
    include_hidden: bool,
) -> tuple[Path, str, Path]:
    conv = get_conversation(session, conversation_id)
    meta = safe_json_loads(conv.metadata_json or "{}") or {}
    if not isinstance(meta, dict):
        meta = {}
    workspace_dir = data_agent_workspace_dir_for_conversation(meta, conversation_id=conversation_id)
    root = Path(workspace_dir).resolve()
    req_rel = (path or "").lstrip("/").strip()
    target = (root / req_rel).resolve()
    if not is_path_within_root(root, target):
        raise HTTPException(status_code=400, detail="Invalid path")
    if req_rel and should_hide_workspace_path(req_rel, include_hidden=bool(include_hidden)):
        raise HTTPException(status_code=403, detail="Path not allowed")
    return root, req_rel, target


def conversation_files_payload(
    *,
    session: Session,
    conversation_id: UUID,
    conv: Conversation,
    bot: Bot,
    path: str,
    recursive: bool,
    include_hidden: bool,
    download_url_for,
) -> dict:
    root, req_rel, target = resolve_data_agent_target(
        session,
        conversation_id=conversation_id,
        path=path,
        include_hidden=include_hidden,
    )
    max_items = 2000
    if not target.exists():
        if not req_rel:
            return {
                "conversation_id": str(conversation_id),
                "bot_id": str(conv.bot_id),
                "bot_name": bot.name,
                "external_id": conv.external_id,
                "workspace_dir": str(root),
                "path": req_rel,
                "recursive": bool(recursive),
                "items": [],
                "max_items": max_items,
            }
        raise HTTPException(status_code=404, detail="Path not found")

    items: list[dict] = []

    def _add_item(p: Path) -> None:
        nonlocal items
        try:
            rel = str(p.relative_to(root)).replace(os.sep, "/")
        except Exception:
            return
        if should_hide_workspace_path(rel, include_hidden=bool(include_hidden)):
            return
        try:
            st = p.stat()
        except Exception:
            return
        is_dir = p.is_dir()
        download_url = None
        if not is_dir:
            download_url = download_url_for(rel)
        items.append(
            {
                "path": rel,
                "name": p.name,
                "is_dir": bool(is_dir),
                "size_bytes": int(st.st_size) if not is_dir else None,
                "mtime": dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).isoformat(),
                "download_url": download_url,
            }
        )

    if target.is_file():
        _add_item(target)
    else:
        _add_item(target)
        if recursive:
            for p in sorted(target.rglob("*")):
                if len(items) >= max_items:
                    break
                _add_item(p)
        else:
            for p in sorted(target.iterdir()):
                if len(items) >= max_items:
                    break
                _add_item(p)

    return {
        "conversation_id": str(conversation_id),
        "bot_id": str(conv.bot_id),
        "bot_name": bot.name,
        "external_id": conv.external_id,
        "workspace_dir": str(root),
        "path": req_rel,
        "recursive": bool(recursive),
        "items": items,
        "max_items": max_items,
    }
