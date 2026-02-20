from __future__ import annotations

import datetime as dt
import html
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote as _url_quote
from uuid import UUID

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from sqlmodel import Session

from voicebot.data_agent.docker_runner import default_workspace_dir_for_conversation
from voicebot.models import Bot, Conversation, ConversationMessage
from voicebot.store import get_bot, get_conversation, list_messages, verify_client_key
from voicebot.utils.template import safe_json_loads
from voicebot.web.helpers.files import is_path_within_root


def parse_allowed_bot_ids(k) -> set[str]:
    try:
        ids = json.loads(getattr(k, "allowed_bot_ids_json", "") or "[]")
        if not isinstance(ids, list):
            return set()
        return {str(x) for x in ids if isinstance(x, str) and str(x).strip()}
    except Exception:
        return set()


def origin_allowed(k, origin: Optional[str]) -> bool:
    allowed = (getattr(k, "allowed_origins", "") or "").strip()
    if not allowed:
        return True
    origin_val = (origin or "").strip()
    allowset = {o.strip() for o in allowed.split(",") if o.strip()}
    return origin_val in allowset


def bot_allowed(k, bot_id: UUID) -> bool:
    allowset = parse_allowed_bot_ids(k)
    if not allowset:
        return True
    return str(bot_id) in allowset


def require_public_conversation_access(
    *,
    session: Session,
    request: Request,
    conversation_id: UUID,
    key: str,
) -> tuple[Any, Any, Any]:
    key_secret = (key or "").strip()
    if not key_secret:
        raise HTTPException(status_code=401, detail="Missing key")
    ck = verify_client_key(session, secret=key_secret)
    if not ck:
        raise HTTPException(status_code=401, detail="Invalid key")
    origin = request.headers.get("origin")
    if origin and (not origin_allowed(ck, origin)):
        raise HTTPException(status_code=403, detail="Origin not allowed")
    try:
        conv = get_conversation(session, conversation_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if conv.client_key_id != ck.id:
        raise HTTPException(status_code=403, detail="Conversation not accessible with this key")
    if not bot_allowed(ck, conv.bot_id):
        raise HTTPException(status_code=403, detail="Bot not allowed for this key")
    bot = get_bot(session, conv.bot_id)
    return ck, conv, bot


def conversation_messages_payload(
    *,
    session: Session,
    conversation_id: UUID,
    include_tools: bool,
    include_system: bool,
) -> dict:
    conv = get_conversation(session, conversation_id)
    bot = get_bot(session, conv.bot_id)
    msgs_raw = list_messages(session, conversation_id=conversation_id)

    def _safe_json_loads(s: str) -> dict | None:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _safe_json_list(s: str) -> list:
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, list) else []
        except Exception:
            return []

    messages: list[dict] = []
    for m in msgs_raw:
        if m.role == "tool" and not include_tools:
            continue
        if m.role == "system" and not include_system:
            continue
        if m.role not in ("user", "assistant", "tool", "system"):
            continue

        tool_obj = _safe_json_loads(m.content) if m.role == "tool" else None
        tool_name = tool_obj.get("tool") if tool_obj and isinstance(tool_obj.get("tool"), str) else None
        tool_kind = None
        if tool_obj:
            if "arguments" in tool_obj:
                tool_kind = "call"
            elif "result" in tool_obj:
                tool_kind = "result"
        messages.append(
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
                "tool": tool_obj,
                "tool_name": tool_name,
                "tool_kind": tool_kind,
                "citations": _safe_json_list(getattr(m, "citations_json", "") or "[]"),
                "sender_bot_id": str(m.sender_bot_id) if m.sender_bot_id else None,
                "sender_name": m.sender_name,
            }
        )

    return {
        "conversation": {
            "id": str(conv.id),
            "bot_id": str(conv.bot_id),
            "bot_name": bot.name,
            "external_id": conv.external_id,
            "is_group": bool(conv.is_group),
            "group_title": conv.group_title or "",
            "group_bots_json": conv.group_bots_json or "[]",
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
        },
        "messages": messages,
    }


def render_conversation_html(
    *,
    title: str,
    conversation_id: UUID,
    key: str,
    include_tools: bool,
    include_system: bool,
    payload: dict,
) -> str:
    key_q = key
    include_tools_q = "1" if include_tools else "0"
    include_system_q = "1" if include_system else "0"

    conv = payload.get("conversation") if isinstance(payload.get("conversation"), dict) else {}
    bot_name = str(conv.get("bot_name") or "")
    external_id = str(conv.get("external_id") or "")

    def _q(**params: str) -> str:
        parts = []
        for k, v in params.items():
            parts.append(f"{k}={v}")
        return "&".join(parts)

    base_params = {"key": key_q}
    chat_href = f"/conversations/{conversation_id}?{_q(**base_params, include_tools='0', include_system='0')}"
    tools_href = f"/conversations/{conversation_id}?{_q(**base_params, include_tools='1', include_system=include_system_q)}"
    system_on_href = f"/conversations/{conversation_id}?{_q(**base_params, include_tools=include_tools_q, include_system='1')}"
    system_off_href = f"/conversations/{conversation_id}?{_q(**base_params, include_tools=include_tools_q, include_system='0')}"
    json_href = f"/conversations/{conversation_id}?{_q(**base_params, include_tools=include_tools_q, include_system=include_system_q, format='json')}"

    rows_html: list[str] = []
    msgs = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        created_at = str(m.get("created_at") or "")
        content = str(m.get("content") or "")
        tool_name = str(m.get("tool_name") or "")
        tool_kind = str(m.get("tool_kind") or "")

        display_role = role
        if role == "tool" and tool_name:
            display_role = f"tool:{tool_name}"
            if tool_kind:
                display_role = f"{display_role} ({tool_kind})"

        pretty = content
        if role == "tool":
            try:
                obj = json.loads(content)
                pretty = json.dumps(obj, ensure_ascii=False, indent=2)
            except Exception:
                pretty = content

        rows_html.append(
            "<tr>"
            f"<td class='c-role'>{html.escape(display_role)}</td>"
            f"<td class='c-time'>{html.escape(created_at)}</td>"
            f"<td class='c-msg'><pre>{html.escape(pretty)}</pre></td>"
            "</tr>"
        )

    subtitle_parts = []
    if bot_name:
        subtitle_parts.append(f"Bot: {bot_name}")
    if external_id:
        subtitle_parts.append(f"External ID: {external_id}")
    subtitle = " • ".join(subtitle_parts)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0b1020;
      --panel: #111936;
      --border: rgba(255,255,255,0.12);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.70);
      --link: #7dd3fc;
      --chip: rgba(255,255,255,0.08);
    }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: var(--bg); color: var(--text); }}
    .wrap {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
    .header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }}
    h1 {{ font-size: 18px; margin: 0 0 6px; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }}
    a.btn {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 10px; border: 1px solid var(--border); border-radius: 10px; background: var(--chip); color: var(--text); text-decoration: none; font-size: 13px; }}
    a.btn:hover {{ border-color: rgba(255,255,255,0.25); }}
    a.btn.primary {{ border-color: rgba(125, 211, 252, 0.6); color: var(--link); }}
    .panel {{ margin-top: 14px; background: var(--panel); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }}
    table {{ width: 100%; border-collapse: collapse; }}
    thead th {{ position: sticky; top: 0; background: rgba(17,25,54,0.92); backdrop-filter: blur(10px); text-align: left; font-size: 12px; color: var(--muted); padding: 10px; border-bottom: 1px solid var(--border); }}
    tbody td {{ vertical-align: top; padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
    .c-role {{ width: 190px; white-space: nowrap; color: var(--muted); font-size: 12px; }}
    .c-time {{ width: 210px; white-space: nowrap; color: var(--muted); font-size: 12px; }}
    .c-msg pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 13px; line-height: 1.35; }}
    .footer {{ margin-top: 12px; font-size: 12px; color: var(--muted); }}
    .footer a {{ color: var(--link); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div>
        <h1>Conversation {html.escape(str(conversation_id))}</h1>
        <div class="sub">{html.escape(subtitle)}</div>
      </div>
      <div class="actions">
        <a class="btn primary" href="{html.escape(chat_href)}">Chat view</a>
        <a class="btn" href="{html.escape(tools_href)}">Include tools</a>
        <a class="btn" href="{html.escape(system_on_href)}">Include system</a>
        <a class="btn" href="{html.escape(system_off_href)}">Hide system</a>
        <a class="btn" href="{html.escape(json_href)}">JSON</a>
      </div>
    </div>
    <div class="panel">
      <table>
        <thead>
          <tr>
            <th>Role</th>
            <th>Time</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
    </div>
    <div class="footer">
      View options: include_tools={include_tools_q}, include_system={include_system_q}.
    </div>
  </div>
</body>
</html>
"""


def data_agent_workspace_dir_for_conversation(session: Session, *, conversation_id: UUID) -> str:
    meta = safe_json_loads(get_conversation(session, conversation_id).metadata_json or "{}") or {}
    da = meta.get("data_agent") if isinstance(meta, dict) else {}
    if isinstance(da, dict):
        workspace_dir = str(da.get("workspace_dir") or "").strip()
    else:
        workspace_dir = ""
    return workspace_dir or default_workspace_dir_for_conversation(conversation_id)


def should_hide_data_agent_path(rel: str, *, include_hidden: bool) -> bool:
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


def resolve_data_agent_target(
    session: Session,
    *,
    conversation_id: UUID,
    path: str,
    include_hidden: bool,
) -> tuple[Path, str, Path]:
    workspace_dir = data_agent_workspace_dir_for_conversation(session, conversation_id=conversation_id)
    root = Path(workspace_dir).resolve()
    req_rel = (path or "").lstrip("/").strip()
    target = (root / req_rel).resolve()
    if not is_path_within_root(root, target):
        raise HTTPException(status_code=400, detail="Invalid path")
    if req_rel and should_hide_data_agent_path(req_rel, include_hidden=bool(include_hidden)):
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
    download_url_for: Callable[[str], Optional[str]],
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

    items: list[dict[str, Any]] = []

    def _add_item(p: Path) -> None:
        nonlocal items
        try:
            rel = str(p.relative_to(root)).replace(os.sep, "/")
        except Exception:
            return
        if should_hide_data_agent_path(rel, include_hidden=bool(include_hidden)):
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
