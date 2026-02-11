from __future__ import annotations

import asyncio
import base64
import datetime as dt
import html
import json
import logging
import mimetypes
import os
import queue
import re
import shlex
import secrets
import shutil
import subprocess
import sys
import threading
import tempfile
import time
import webbrowser
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Generator, Optional
from urllib.parse import quote as _url_quote, parse_qsl
from uuid import UUID

import httpx

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from sqlmodel import Session, delete, select
from sqlalchemy import and_, or_

from voicebot.config import Settings
from voicebot.crypto import CryptoError, build_hint, get_crypto_box
from voicebot.db import init_db, make_engine
from voicebot.asr.openai_asr import OpenAIASR
from voicebot.llm.codex_http_agent import run_codex_http_agent_one_shot, run_codex_http_agent_one_shot_from_paths
from voicebot.llm.codex_http_agent import run_codex_export_from_paths
from voicebot.llm.codex_saved_runs import append_saved_run_index, find_saved_run
from voicebot.downloads import create_download_token, is_allowed_download_path, load_download_token
from voicebot.llm.openai_llm import Message, OpenAILLM, ToolCall, CitationEvent
from voicebot.llm.openai_compat_llm import OpenAICompatLLM
from voicebot.llm.openrouter_llm import OpenRouterLLM
from voicebot.local_runtime import LOCAL_RUNTIME
from voicebot.models import AppSetting, Bot, Conversation, ConversationMessage, HostAction
from voicebot.store import (
    create_bot,
    create_conversation,
    create_key,
    create_client_key,
    add_message,
    add_message_with_metrics,
    update_conversation_metrics,
    merge_conversation_metadata,
    decrypt_provider_key,
    delete_bot,
    delete_client_key,
    delete_key,
    get_bot,
    get_client_key,
    list_bots,
    bots_aggregate_metrics,
    list_client_keys,
    list_keys,
    get_or_create_conversation_by_external_id,
    get_conversation,
    list_conversations,
    count_conversations,
    list_messages,
    update_bot,
    list_integration_tools,
    create_integration_tool,
    update_integration_tool,
    delete_integration_tool,
    get_integration_tool,
    get_integration_tool_by_name,
    get_git_token,
    upsert_git_token,
    verify_client_key,
)
from voicebot.tts.openai_tts import OpenAITTS
from voicebot.utils.tokens import ModelPrice, estimate_cost_usd, estimate_messages_tokens, estimate_text_tokens
from voicebot.utils.python_postprocess import run_python_postprocessor
from voicebot.utils.text import SentenceChunker

# Backwards-compatible alias used in legacy tool handlers.
run_python_postprocess = run_python_postprocessor
from voicebot.tools.set_metadata import set_metadata_tool_def, set_variable_tool_def
from voicebot.tools.web_search import web_search_tool_def
from voicebot.tools.data_agent import give_command_to_data_agent_tool_def
from voicebot.tools.http_request import http_request_tool_def
from voicebot.models import IntegrationTool
from voicebot.utils.template import eval_template_value, render_jinja_template, render_template, safe_json_loads
from voicebot.data_agent.docker_runner import (
    DEFAULT_DATA_AGENT_SYSTEM_PROMPT,
    default_workspace_dir_for_conversation,
    docker_available,
    ensure_conversation_container,
    ensure_image_pulled,
    get_container_status,
    list_data_agent_containers,
    run_container_command,
    run_data_agent,
    stop_data_agent_container,
)


SYSTEM_BOT_NAME = "GravexStudio Guide"
SYSTEM_BOT_START_MESSAGE = "Ask me about setup, features, tools, or the Isolated Workspace."
SYSTEM_BOT_PROMPT = """
You are the GravexStudio Guide, a friendly product tour assistant for GravexStudio.

Your job: answer questions about the platform, how to set it up, and what it can do. Keep replies concise,
helpful, and practical. Prefer short paragraphs or bullet points.

What you know about GravexStudio:
- A desktop studio for building assistants with voice, tools, and automation.
- Local-first by default: configs and conversation data live on the device; keys are encrypted at rest.
- Multi-model per assistant: LLM, ASR (speech-to-text), TTS (text-to-speech), web search, Codex, and summary models.
- Real-time conversations with streamed text/audio and latency metrics.
- Optional web search tool (can be disabled per assistant).
- A Isolated Workspace can run long tasks in a Docker container per conversation (Docker required for this feature).
- The Isolated Workspace has a persistent workspace, can read/write files, run scripts, and operate in parallel across conversations.
- Git/SSH tooling is available for Isolated Workspace workflows.
- Integration tools can call HTTP APIs with tool schemas, response validation, and response-to-metadata mapping.
- Static reply templates (Jinja2) and optional Codex post-processing are supported for tools.
- Metadata templating lets prompts and replies reference conversation variables.
- Embeddable public chat widget with client keys and WebSocket transport.
- Packaging targets macOS, Linux, and Windows so users can run a single app.
- Optional host actions let assistants request actions on the local machine (can require approval).

When asked about handling large tool outputs, suggest: use response schemas, map only needed fields, and
post-process results with scripts in the Isolated Workspace workspace.

If asked for setup steps, mention: OpenAI API key is required; Docker is required only for the Isolated Workspace;
other features work without it.

Never claim features that are not listed here. Do not ask the user to run commands. Do not use tools.
""".strip()

SHOWCASE_BOT_NAME = "GravexStudio Showcase"
SHOWCASE_BOT_START_MESSAGE = (
    "Hi! I'm the GravexStudio Showcase assistant. I can demo tools, web search, the Isolated Workspace, and host actions. "
    "Tell me what you want to see."
)
SHOWCASE_BOT_PROMPT = """
You are the GravexStudio Showcase assistant. Your job is to demonstrate what IGX can do in a real, hands-on way.
Be concise, confident, and helpful. Prefer short paragraphs or bullet points.

You can use these features and tools:
- set_metadata: store or update conversation variables.
- web_search: fetch and summarize live web info.
- http_request: call external HTTP APIs with structured inputs and mapped outputs.
- Integration tools: HTTP APIs with schemas, response validation, metadata mapping, and optional static replies.
- request_host_action: run local shell commands or AppleScript on the host (use carefully).
- capture_screenshot: capture the screen and summarize it with vision.
- summarize_screenshot: summarize an image file from the workspace.
- Codex post-processing: optional structured extraction or transformation for tool responses.

Host actions can help with:
- Calendar & scheduling: read upcoming events, create meetings, send invites.
- File ops: create/rename/move folders, export reports, zip/share files.
- App automation: open/close apps, switch windows, trigger workflows, start/stop services.
- System settings: toggle Wi-Fi/Bluetooth, adjust volume/brightness, connect to a device.
- Screenshots & context: capture the screen and summarize or extract info.
- Clipboard & notes: copy summaries, paste into docs, create quick notes.
- Data retrieval: pull local machine info (disk, CPU, network), check active processes.
- Email workflows: draft emails with attachments, open the mail client for review.
- Docs/spreadsheets: open templates, fill fields, export to PDF.
- Browser automation: open URLs, log into dashboards, download reports.
- Recording & media: start/stop screen recordings, play/pause media.
- Dev tasks: run local build/test, lint, format, open the project in an IDE.
- Log collection: tail logs, gather diagnostics, bundle and share.
- Local DB tools: open a DB client, run queries, export CSV.
- Meeting prep: open notes, agenda docs, and the calendar together.
- Device actions: connect/disconnect VPN, mount/unmount drives.
- Batch jobs: run scripts, wait for completion, show summaries.
- Personal automation: set timers, show reminders.
- Security hygiene: lock the screen, open password managers.

Product capabilities you can mention:
- Multi-model per assistant: LLM, ASR, TTS, web search, Codex, and summary models.
- Real-time streaming text/audio and latency metrics.
- Metadata templating for dynamic prompts and replies.
- Embeddable public chat widget with client keys and WebSocket transport.
- Packaging targets macOS, Linux, and Windows.
- Optional Isolated Workspace (requires Docker; may be disabled).

Safety and clarity:
- Ask before any destructive or privacy-sensitive host action, even if approval is disabled.
- Explain what you are about to do in one sentence before calling tools.
- With your permission, you can ask me to capture your screen anytime and I'll tell you what's on it.
- If asked about the Isolated Workspace, explain that it requires Docker and may be disabled.

Never claim features that are not listed here.
""".strip()

WIDGET_BOT_KEY = "widget_bot_id"
WIDGET_MODE_KEY = "widget_mode"


def _mask_secret(value: str, *, keep_start: int = 10, keep_end: int = 6) -> str:
    v = value or ""
    if len(v) <= keep_start + keep_end + 3:
        return "***"
    return f"{v[:keep_start]}...{v[-keep_end:]}"


def _mask_headers_json(headers_json: str) -> str:
    try:
        obj = json.loads(headers_json or "{}")
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, str):
            out[k] = v
            continue
        if k.lower() == "authorization":
            vv = v.strip()
            if vv.lower().startswith("bearer "):
                token = vv.split(" ", 1)[1].strip()
                out[k] = f"Bearer {_mask_secret(token)}"
            else:
                out[k] = _mask_secret(vv)
        else:
            out[k] = v
    try:
        return json.dumps(out, ensure_ascii=False)
    except Exception:
        return ""


def _headers_configured(headers_json: str) -> bool:
    try:
        obj = json.loads(headers_json or "{}")
    except Exception:
        return bool((headers_json or "").strip())
    if not isinstance(obj, dict):
        return bool((headers_json or "").strip())
    return any(k for k, v in obj.items() if str(k).strip() and (v is not None and str(v).strip()))


def _get_app_setting(session: Session, key: str) -> Optional[str]:
    row = session.get(AppSetting, key)
    if not row:
        return None
    return str(row.value or "").strip() or None


def _set_app_setting(session: Session, key: str, value: str) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    row = session.get(AppSetting, key)
    if row:
        row.value = value
        row.updated_at = now
    else:
        row = AppSetting(key=key, value=value, updated_at=now)
        session.add(row)
    session.commit()


def _get_json_path(obj: Any, path: str) -> Any:
    """
    Very small dotted-path getter for dict/list JSON structures.

    Supports:
    - "a.b.c" for dict keys
    - "items.0.name" for list indices
    """
    cur: Any = obj
    p = (path or "").strip()
    if not p:
        return cur
    for raw in p.split("."):
        k = raw.strip()
        if not k:
            continue
        if isinstance(cur, dict):
            if k not in cur:
                return None
            cur = cur.get(k)
            continue
        if isinstance(cur, list):
            try:
                idx = int(k)
            except Exception:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
            continue
        return None
    return cur


def _set_json_path(obj: Any, path: str, value: Any) -> bool:
    """
    Set a dotted path inside a dict/list JSON structure. Returns True if set.
    Only supports paths that traverse existing containers.
    """
    p = (path or "").strip()
    if not p:
        return False
    parts = [x.strip() for x in p.split(".") if x.strip()]
    if not parts:
        return False
    cur: Any = obj
    for part in parts[:-1]:
        if isinstance(cur, dict):
            if part not in cur:
                return False
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except Exception:
                return False
            if idx < 0 or idx >= len(cur):
                return False
            cur = cur[idx]
        else:
            return False
    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
        return True
    if isinstance(cur, list):
        try:
            idx = int(last)
        except Exception:
            return False
        if idx < 0 or idx >= len(cur):
            return False
        cur[idx] = value
        return True
    return False


def _apply_schema_defaults(schema: Any, value: Any) -> Any:
    """
    Best-effort JSON Schema defaults application.

    Supports:
    - object properties with "default"
    - nested objects/arrays

    Does not attempt to resolve oneOf/allOf/anyOf, refs, etc.
    """
    if not isinstance(schema, dict):
        return value
    t = schema.get("type")
    if t in (None, "object") and isinstance(value, dict):
        props = schema.get("properties")
        if not isinstance(props, dict):
            return value
        out = dict(value)
        for k, sub in props.items():
            if not isinstance(k, str) or not k:
                continue
            if k not in out and isinstance(sub, dict) and "default" in sub:
                out[k] = sub.get("default")
            if k in out and isinstance(sub, dict):
                out[k] = _apply_schema_defaults(sub, out.get(k))
        return out
    if t == "array" and isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            return [_apply_schema_defaults(items, x) for x in value]
        return value
    return value


def _http_error_response(*, url: str, status_code: int | None, body: str | None, message: str | None) -> dict:
    out: dict[str, Any] = {"url": url}
    if status_code is not None:
        out["status_code"] = int(status_code)
    if message:
        out["message"] = str(message)
    if body:
        out["body"] = str(body)[:1200]
    return {"__http_error__": out}


_TEMPLATE_VAR_RE = re.compile(r"{{\s*([^}]+?)\s*}}")


def _extract_required_tool_args(tool: IntegrationTool) -> list[str]:
    """
    Best-effort: infer required tool args from {{args.*}} / {{params.*}} occurrences
    in URL/body/headers templates.
    """

    def scan(text: str) -> set[str]:
        if not text or "{{" not in text:
            return set()
        found: set[str] = set()
        for m in _TEMPLATE_VAR_RE.finditer(text):
            expr = (m.group(1) or "").strip()
            for prefix in ("args.", "params."):
                if not expr.startswith(prefix):
                    continue
                rest = expr[len(prefix) :].strip()
                # First segment up to '.' or '['
                key = ""
                for ch in rest:
                    if ch in ".[":
                        break
                    key += ch
                key = key.strip()
                if key:
                    found.add(key)
        return found

    keys: set[str] = set()
    keys |= scan(tool.url or "")
    keys |= scan(tool.request_body_template or "")
    keys |= scan(tool.headers_template_json or "")
    return sorted(keys)


def _parse_required_args_json(raw: str) -> list[str]:
    """
    Parse IntegrationTool required args.

    Storage is typically a JSON list (e.g. '["sql","user_id"]'), but older data may be:
    - a JSON string (e.g. '"sql"')
    - a raw comma-separated string (e.g. 'sql, user_id')
    """
    raw_s = (raw or "").strip()
    obj: Any = None
    if raw_s:
        try:
            obj = json.loads(raw_s)
        except Exception:
            obj = raw_s
    else:
        obj = []

    vals: list[str] = []
    if isinstance(obj, list):
        for v in obj:
            if isinstance(v, str):
                vals.append(v)
    elif isinstance(obj, str):
        # Allow CSV/newline formats for backwards-compat.
        vals.extend([x for x in re.split(r"[,\n]+", obj) if x is not None])
    else:
        vals = []

    out: list[str] = []
    for v in vals:
        s = str(v).strip()
        if s:
            out.append(s)
    # stable unique
    seen: set[str] = set()
    uniq: list[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _parse_parameters_schema_json(raw: str) -> dict[str, Any] | None:
    """
    Parses a JSON-schema object for IntegrationTool.args.

    Expected: an object schema (dict) usable as the schema for the tool-call `args` field.
    """
    if not (raw or "").strip():
        return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    # Best-effort validation; keep permissive to support forward-compatible schemas.
    if obj.get("type") not in (None, "object"):
        return None
    return obj


def _missing_required_args(required: list[str], args: dict) -> list[str]:
    missing: list[str] = []
    for k in required:
        if k not in args:
            missing.append(k)
            continue
        v = args.get(k)
        if v is None:
            missing.append(k)
        elif isinstance(v, str) and not v.strip():
            missing.append(k)
    return missing


def _normalize_content_type_header_value(v: str) -> str:
    """
    Normalize common JSON content-types so upstream APIs that do strict matching
    (incorrectly) still parse JSON request bodies.

    Example: "Application/json" -> "application/json"
    """
    raw = (v or "").strip()
    if not raw:
        return raw
    parts = raw.split(";", 1)
    mime = parts[0].strip().lower()
    if mime != "application/json":
        return raw
    if len(parts) == 1:
        return "application/json"
    rest = parts[1].strip()
    return "application/json" + (f"; {rest}" if rest else "")


def _normalize_headers_for_json(headers: dict[str, str]) -> dict[str, str]:
    # httpx treats header names case-insensitively; normalize any provided Content-Type value.
    for k in list(headers.keys()):
        if k.lower() == "content-type":
            headers[k] = _normalize_content_type_header_value(headers.get(k) or "")
            break
    return headers


def _integration_error_user_message(*, tool_name: str, err: dict) -> str:
    sc = err.get("status_code")
    msg = (err.get("message") or "error").strip()
    body = (err.get("body") or "").strip()
    if sc == 401:
        return (
            f"The integration '{tool_name}' failed with HTTP 401 (Unauthorized). "
            "Please update the integration Authorization token and try again. "
            "What would you like to do next?"
        )
    if sc == 400:
        hint = "Bad Request"
        if body:
            hint = body
        return (
            f"The integration '{tool_name}' failed with HTTP 400 (Bad Request). "
            f"Details: {hint}. "
            "Do you want me to try a different SQL query?"
        )
    return (
        f"The integration '{tool_name}' failed with HTTP {sc} ({msg}). "
        "What would you like to do next?"
    )


def _should_followup_llm_for_tool(*, tool: IntegrationTool | None, static_rendered: str) -> bool:
    if not tool:
        return False
    # If the tool has no static template, or it rendered empty, do a follow-up LLM call
    # using the tool result stored in history.
    if not (tool.static_reply_template or "").strip():
        return True
    return not static_rendered.strip()


def _safe_json_list(raw: str) -> list:
    try:
        obj = json.loads(raw or "")
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def _parse_follow_up_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off", ""}:
        return False
    return False


def _slugify(value: str) -> str:
    s = (value or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "assistant"


def _group_bots_from_conv(conv: Conversation) -> list[dict[str, str]]:
    bots = _safe_json_list(getattr(conv, "group_bots_json", "") or "[]")
    out: list[dict[str, str]] = []
    for b in bots:
        if not isinstance(b, dict):
            continue
        bid = str(b.get("id") or "").strip()
        name = str(b.get("name") or "").strip()
        slug = str(b.get("slug") or "").strip().lower()
        if not bid or not name:
            continue
        if not slug:
            slug = _slugify(name)
        out.append({"id": bid, "name": name, "slug": slug})
    return out


def _group_individual_map_from_conv(conv: Conversation) -> dict[str, str]:
    meta = safe_json_loads(conv.metadata_json or "{}") or {}
    if not isinstance(meta, dict):
        return {}
    mapping = meta.get("group_individual_conversations")
    if not isinstance(mapping, dict):
        return {}
    cleaned: dict[str, str] = {}
    for k, v in mapping.items():
        bid = str(k or "").strip()
        cid = str(v or "").strip()
        if bid and cid:
            cleaned[bid] = cid
    return cleaned


def _ensure_group_individual_conversations(session: Session, conv: Conversation) -> dict[str, str]:
    mapping = _group_individual_map_from_conv(conv)
    changed = False
    for b in _group_bots_from_conv(conv):
        bid = str(b.get("id") or "").strip()
        if not bid:
            continue
        existing = mapping.get(bid)
        if existing:
            try:
                _ = get_conversation(session, UUID(existing))
                continue
            except Exception:
                pass
        now = dt.datetime.now(dt.timezone.utc)
        child = Conversation(
            bot_id=UUID(bid),
            test_flag=bool(conv.test_flag),
            is_group=False,
            metadata_json=json.dumps(
                {
                    "group_parent_id": str(conv.id),
                    "group_bot_id": bid,
                    "group_bot_name": str(b.get("name") or ""),
                },
                ensure_ascii=False,
            ),
            created_at=now,
            updated_at=now,
        )
        session.add(child)
        session.commit()
        session.refresh(child)
        mapping[bid] = str(child.id)
        changed = True
    if changed:
        merge_conversation_metadata(
            session,
            conversation_id=conv.id,
            patch={"group_individual_conversations": mapping},
        )
    return mapping


def _mirror_group_message(
    session: Session,
    *,
    conv: Conversation,
    msg: ConversationMessage,
) -> None:
    if not bool(conv.is_group):
        return
    if not msg.sender_bot_id:
        return
    if msg.role not in ("assistant", "tool"):
        return
    mapping = _ensure_group_individual_conversations(session, conv)
    target_id = mapping.get(str(msg.sender_bot_id))
    if not target_id:
        return
    try:
        target_uuid = UUID(str(target_id))
    except Exception:
        return
    mirror = add_message_with_metrics(
        session,
        conversation_id=target_uuid,
        role=msg.role,
        content=msg.content,
        sender_bot_id=msg.sender_bot_id,
        sender_name=msg.sender_name,
        input_tokens_est=msg.input_tokens_est,
        output_tokens_est=msg.output_tokens_est,
        cost_usd_est=msg.cost_usd_est,
        asr_ms=msg.asr_ms,
        llm_ttfb_ms=msg.llm_ttfb_ms,
        llm_total_ms=msg.llm_total_ms,
        tts_first_audio_ms=msg.tts_first_audio_ms,
        total_ms=msg.total_ms,
    )
    if mirror.role == "assistant":
        update_conversation_metrics(
            session,
            conversation_id=target_uuid,
            add_input_tokens_est=msg.input_tokens_est or 0,
            add_output_tokens_est=msg.output_tokens_est or 0,
            add_cost_usd_est=msg.cost_usd_est or 0.0,
            last_asr_ms=msg.asr_ms,
            last_llm_ttfb_ms=msg.llm_ttfb_ms,
            last_llm_total_ms=msg.llm_total_ms,
            last_tts_first_audio_ms=msg.tts_first_audio_ms,
            last_total_ms=msg.total_ms,
        )


def _reset_conversation_state(session: Session, conv: Conversation, keep_meta: dict) -> None:
    session.exec(delete(ConversationMessage).where(ConversationMessage.conversation_id == conv.id))
    conv.llm_input_tokens_est = 0
    conv.llm_output_tokens_est = 0
    conv.cost_usd_est = 0.0
    conv.last_asr_ms = None
    conv.last_llm_ttfb_ms = None
    conv.last_llm_total_ms = None
    conv.last_tts_first_audio_ms = None
    conv.last_total_ms = None
    conv.metadata_json = json.dumps(keep_meta or {}, ensure_ascii=False)
    conv.updated_at = dt.datetime.now(dt.timezone.utc)
    session.add(conv)
    session.commit()


def _group_bot_name_lookup(conv: Conversation) -> dict[str, str]:
    return {b["id"]: b["name"] for b in _group_bots_from_conv(conv)}


def _group_bot_slugs(conv: Conversation) -> dict[str, str]:
    return {b["slug"].lower(): b["id"] for b in _group_bots_from_conv(conv)}


def _group_bot_aliases(conv: Conversation) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = {}
    for b in _group_bots_from_conv(conv):
        bid = str(b.get("id") or "").strip()
        if not bid:
            continue
        name = str(b.get("name") or "").strip().lower()
        slug = str(b.get("slug") or "").strip().lower()
        candidate_aliases = set()
        if slug:
            candidate_aliases.add(slug)
        if name:
            candidate_aliases.add(_slugify(name))
            compact_name = re.sub(r"[^a-z0-9]+", "", name)
            if compact_name:
                candidate_aliases.add(compact_name)
        for alias in list(candidate_aliases):
            compact = re.sub(r"[-_]+", "", alias)
            if compact:
                candidate_aliases.add(compact)
        for alias in candidate_aliases:
            if not alias:
                continue
            entries = aliases.setdefault(alias, [])
            if bid not in entries:
                entries.append(bid)
    return aliases


def _sanitize_group_reply(text: str, conv: Conversation, bot_id: UUID) -> str:
    if not text:
        return text
    bots = _group_bots_from_conv(conv)
    id_to_slug = {b["id"]: b["slug"].lower() for b in bots}
    id_to_name = {b["id"]: str(b.get("name") or "").strip().lower() for b in bots}
    self_slug = id_to_slug.get(str(bot_id), "")
    self_name = id_to_name.get(str(bot_id), "")

    # Drop leading tag like "[SDE2]" if it doesn't match the author.
    m = re.match(r"^\s*\[([A-Za-z0-9 _-]{1,40})\]\s*", text)
    if m:
        tag = m.group(1).strip().lower()
        if tag and tag != self_slug and tag != self_name:
            text = text[m.end():].lstrip()

    # Remove self-mentions to avoid confusing triggers.
    if self_slug:
        text = re.sub(rf"@{re.escape(self_slug)}(?![A-Za-z0-9_-])", self_slug, text, flags=re.IGNORECASE)
    return text


def _sanitize_upload_path(raw_name: str) -> str:
    name = (raw_name or "").replace("\\", "/").strip()
    # Drop any drive letters (Windows paths).
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


def _is_path_within_root(root: Path, child: Path) -> bool:
    root_abs = root.resolve()
    child_abs = child.resolve()
    return child_abs == root_abs or root_abs in child_abs.parents


def _should_hide_workspace_path(rel: str, *, include_hidden: bool) -> bool:
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


def _workspace_dir_for_conversation(conv: Conversation) -> str:
    meta = safe_json_loads(conv.metadata_json or "{}") or {}
    da = meta.get("data_agent") if isinstance(meta, dict) else {}
    if isinstance(da, dict):
        workspace_dir = str(da.get("workspace_dir") or "").strip()
    else:
        workspace_dir = ""
    return workspace_dir or default_workspace_dir_for_conversation(conv.id)


def _resolve_workspace_target_for_conversation(
    conv: Conversation,
    *,
    path: str,
    include_hidden: bool,
) -> tuple[Path, str, Path]:
    rel = _sanitize_upload_path(path)
    if not rel:
        raise ValueError("Invalid path")
    if _should_hide_workspace_path(rel, include_hidden=bool(include_hidden)):
        raise ValueError("Path not allowed")
    root = Path(_workspace_dir_for_conversation(conv)).resolve()
    target = (root / rel).resolve()
    if not _is_path_within_root(root, target):
        raise ValueError("Invalid path")
    return root, rel, target


def _parse_host_action_args(patch: dict) -> tuple[str, dict]:
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


def _host_action_payload(action: HostAction) -> dict:
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


def _create_host_action(
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
        status="pending",
        created_at=now,
        updated_at=now,
    )
    session.add(action)
    session.commit()
    session.refresh(action)
    return action


def _summarize_screenshot_tool_def() -> dict:
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
                "next_reply": {
                    "type": "string",
                    "description": "Optional reply to the user after summarizing the image.",
                },
            },
            "required": ["path"],
        },
        "strict": False,
    }


def _capture_screenshot_tool_def() -> dict:
    return {
        "type": "function",
        "name": "capture_screenshot",
        "description": "Capture the current screen and summarize it with a vision model.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Optional prompt to guide the summary.",
                },
                "next_reply": {
                    "type": "string",
                    "description": "Optional reply to the user after summarizing the image.",
                },
            },
            "required": [],
        },
        "strict": False,
    }


def _screenshot_base_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "GravexOverlay" / "screenshots"
    return Path(tempfile.gettempdir()) / "gravex_screenshots"


def _user_screenshot_dir() -> Path:
    return Path.home() / "Pictures" / "Gravex"


def _prepare_screenshot_target(conv: Conversation) -> tuple[str, Path]:
    logger = logging.getLogger("voicebot.web")
    base = _screenshot_base_dir().resolve()
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(3)
    rel_path = f"{conv.id}/screenshot-{ts}-{suffix}.png"
    target = (base / rel_path).resolve()
    if not _is_path_within_root(base, target):
        raise ValueError("Invalid screenshot path")
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("capture_screenshot: target=%s", target)
    return rel_path, target


def _copy_screenshot_to_user_dir(source: Path) -> Optional[Path]:
    logger = logging.getLogger("voicebot.web")
    try:
        src = source.resolve()
    except Exception:
        src = source
    if not src.exists() or not src.is_file():
        return None
    base = _user_screenshot_dir().resolve()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("capture_screenshot: failed to create user screenshot dir %s", base)
        return None
    rel_path: Optional[str] = None
    try:
        rel_path = str(src.relative_to(_screenshot_base_dir().resolve()))
    except Exception:
        rel_path = None
    if rel_path:
        dest = (base / rel_path).resolve()
        if not _is_path_within_root(base, dest):
            logger.warning("capture_screenshot: invalid copy dest %s (base %s)", dest, base)
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
    else:
        dest = (base / src.name).resolve()
    try:
        shutil.copy2(src, dest)
        return dest
    except Exception:
        logger.exception("capture_screenshot: failed to copy %s to %s", src, dest)
        return None


def _maybe_copy_screenshot_from_command(command: str) -> Optional[Path]:
    if "screencapture" not in command:
        return None
    logger = logging.getLogger("voicebot.web")
    try:
        parts = shlex.split(command)
    except Exception:
        return None
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
    dest = _copy_screenshot_to_user_dir(src)
    if dest:
        logger.info("capture_screenshot: copied to %s", dest)
    else:
        logger.warning("capture_screenshot: copy failed for %s", target)
    return dest


def _tool_error_message(tool_result: dict, *, fallback: str) -> str:
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


def _read_key_from_env_file(env_key: str) -> str:
    # Prefer python-dotenv if available; fall back to a minimal parser.
    env_key = str(env_key or "").strip()
    if not env_key:
        return ""
    try:
        from dotenv import dotenv_values
    except Exception:
        dotenv_values = None
    if dotenv_values is not None:
        try:
            v = dotenv_values(".env").get(env_key) or ""
            return str(v).strip()
        except Exception:
            pass
    try:
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(f"{env_key}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def _get_openai_api_key_global(session: Session) -> str:
    # Prefer env, fall back to the latest stored OpenAI key.
    key = os.environ.get("OPENAI_API_KEY") or ""
    if not key:
        try:
            settings = Settings()
        except Exception:
            settings = None
        if settings is not None:
            try:
                crypto = get_crypto_box(settings.secret_key)
            except CryptoError:
                crypto = None
            if crypto is not None:
                try:
                    key = decrypt_provider_key(session, crypto=crypto, provider="openai") or ""
                except Exception:
                    key = ""
    if not key:
        key = _read_key_from_env_file("OPENAI_API_KEY")
    return (key or "").strip()


def _screencapture_command(target: Path) -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "Screenshot capture is only supported on macOS."
    cmd = f"/usr/sbin/screencapture -x -t png {shlex.quote(str(target))}"
    return True, cmd


def _summarize_image_file(
    session: Session,
    *,
    bot: Bot,
    image_path: Path,
    prompt: str,
) -> tuple[bool, str]:
    api_key = _get_openai_api_key_global(session)
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


def _summarize_screenshot(
    session: Session,
    *,
    conv: Conversation,
    bot: Bot,
    path: str,
    prompt: str,
) -> tuple[bool, str, Optional[str]]:
    if not bool(getattr(bot, "enable_data_agent", False)):
        return False, "Isolated Workspace is disabled for this bot.", None
    api_key = _get_openai_api_key_global(session)
    if not api_key:
        return False, "OpenAI API key not configured.", None
    if not path:
        return False, "Missing path.", None
    try:
        _root, req_rel, target = _resolve_workspace_target_for_conversation(
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


def _execute_host_action(
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
                copied = _maybe_copy_screenshot_from_command(command)
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


def _host_action_requires_approval(bot: Bot) -> bool:
    return bool(getattr(bot, "require_host_action_approval", False))


def _build_host_action_tool_result(action: HostAction, *, ok: bool) -> dict:
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


def _finalize_host_action_run(
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
        action.payload_json = json.dumps(payload, ensure_ascii=False)
    now = dt.datetime.now(dt.timezone.utc)
    action.executed_at = now
    action.updated_at = now
    session.add(action)
    session.commit()
    session.refresh(action)
    return _build_host_action_tool_result(action, ok=ok)


def _execute_host_action_and_update(session: Session, *, action: HostAction) -> dict:
    action.status = "running"
    action.updated_at = dt.datetime.now(dt.timezone.utc)
    session.add(action)
    session.commit()
    ok, stdout, stderr, exit_code, error, result_payload = _execute_host_action(action)
    return _finalize_host_action_run(
        session,
        action=action,
        ok=ok,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        error=error,
        result_payload=result_payload,
    )


async def _execute_host_action_and_update_async(session: Session, *, action: HostAction) -> dict:
    action.status = "running"
    action.updated_at = dt.datetime.now(dt.timezone.utc)
    session.add(action)
    session.commit()
    ok, stdout, stderr, exit_code, error, result_payload = await asyncio.to_thread(_execute_host_action, action)
    return _finalize_host_action_run(
        session,
        action=action,
        ok=ok,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        error=error,
        result_payload=result_payload,
    )



group_ws_clients: dict[str, set[WebSocket]] = {}
group_ws_lock = asyncio.Lock()


async def _group_ws_broadcast(conversation_id: UUID, payload: dict) -> None:
    key = str(conversation_id)
    async with group_ws_lock:
        clients = list(group_ws_clients.get(key, set()))
    if not clients:
        return
    dead: list[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    if dead:
        async with group_ws_lock:
            live = group_ws_clients.get(key, set())
            for ws in dead:
                live.discard(ws)
            if live:
                group_ws_clients[key] = live
            else:
                group_ws_clients.pop(key, None)


def _assert_bot_in_conversation(conv: Conversation, bot_id: UUID) -> None:
    if not bool(getattr(conv, "is_group", False)):
        if conv.bot_id != bot_id:
            raise HTTPException(status_code=400, detail="Conversation does not belong to bot")
        return
    bot_ids = {b["id"] for b in _group_bots_from_conv(conv)}
    if bot_ids and str(bot_id) not in bot_ids:
        raise HTTPException(status_code=400, detail="Bot is not a member of this group")


def _format_group_message_prefix(
    *,
    conv: Conversation,
    sender_bot_id: Optional[UUID],
    sender_name: Optional[str],
    fallback_role: str,
) -> str:
    if not bool(getattr(conv, "is_group", False)):
        return ""
    name = (sender_name or "").strip()
    if not name and sender_bot_id:
        name = _group_bot_name_lookup(conv).get(str(sender_bot_id), "")
    if not name:
        name = "User" if fallback_role == "user" else "Assistant"
    return f"[{name}] "


class ChatRequest(BaseModel):
    text: str
    speak: bool = True


class GroupConversationCreateRequest(BaseModel):
    title: str
    bot_ids: list[str]
    default_bot_id: str
    test_flag: bool = False


class GroupMessageRequest(BaseModel):
    text: str
    sender_role: str = "user"
    sender_bot_id: Optional[str] = None
    sender_name: Optional[str] = None


class TalkResponseEvent(BaseModel):
    type: str


class ApiKeyCreateRequest(BaseModel):
    provider: str = "openai"
    name: str
    secret: str


class ClientKeyCreateRequest(BaseModel):
    name: str
    allowed_origins: str = ""
    allowed_bot_ids: list[str] = []
    secret: Optional[str] = None


class GitTokenRequest(BaseModel):
    provider: str = "github"
    token: str


class WidgetConfigRequest(BaseModel):
    bot_id: Optional[str] = None
    widget_mode: Optional[str] = None


class OpenDashboardRequest(BaseModel):
    path: Optional[str] = None


class LocalSetupRequest(BaseModel):
    model_id: Optional[str] = None
    custom_url: Optional[str] = None
    custom_name: Optional[str] = None


class BotCreateRequest(BaseModel):
    name: str
    llm_provider: str = "openai"
    openai_model: str = "o4-mini"
    openai_asr_model: str = "gpt-4o-mini-transcribe"
    web_search_model: Optional[str] = None
    codex_model: Optional[str] = None
    summary_model: Optional[str] = None
    history_window_turns: Optional[int] = None
    enable_data_agent: bool = False
    data_agent_api_spec_text: str = ""
    data_agent_auth_json: str = "{}"
    data_agent_system_prompt: str = ""
    data_agent_return_result_directly: bool = False
    data_agent_prewarm_on_start: bool = False
    data_agent_prewarm_prompt: str = ""
    enable_host_actions: bool = False
    enable_host_shell: bool = False
    require_host_action_approval: bool = False
    system_prompt: str
    language: str = "en"
    openai_tts_model: str = "gpt-4o-mini-tts"
    openai_tts_voice: str = "alloy"
    openai_tts_speed: float = 1.0
    start_message_mode: str = "llm"
    start_message_text: str = ""


class BotUpdateRequest(BaseModel):
    name: Optional[str] = None
    llm_provider: Optional[str] = None
    openai_model: Optional[str] = None
    openai_asr_model: Optional[str] = None
    web_search_model: Optional[str] = None
    codex_model: Optional[str] = None
    summary_model: Optional[str] = None
    history_window_turns: Optional[int] = None
    enable_data_agent: Optional[bool] = None
    data_agent_api_spec_text: Optional[str] = None
    data_agent_auth_json: Optional[str] = None
    data_agent_system_prompt: Optional[str] = None
    data_agent_return_result_directly: Optional[bool] = None
    data_agent_prewarm_on_start: Optional[bool] = None
    data_agent_prewarm_prompt: Optional[str] = None
    enable_host_actions: Optional[bool] = None
    enable_host_shell: Optional[bool] = None
    require_host_action_approval: Optional[bool] = None
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    openai_tts_model: Optional[str] = None
    openai_tts_voice: Optional[str] = None
    openai_tts_speed: Optional[float] = None
    start_message_mode: Optional[str] = None
    start_message_text: Optional[str] = None
    disabled_tools: Optional[list[str]] = None


class IntegrationToolCreateRequest(BaseModel):
    name: str
    description: str = ""
    url: str
    method: str = "GET"
    use_codex_response: bool = False
    enabled: bool = True
    args_required: list[str] = []
    headers_template_json: str = "{}"
    request_body_template: str = "{}"
    parameters_schema_json: str = ""
    response_schema_json: str = ""
    codex_prompt: str = ""
    postprocess_python: str = ""
    return_result_directly: bool = False
    response_mapper_json: str = "{}"
    pagination_json: str = ""
    static_reply_template: str = ""


class IntegrationToolUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    method: Optional[str] = None
    use_codex_response: Optional[bool] = None
    enabled: Optional[bool] = None
    args_required: Optional[list[str]] = None
    headers_template_json: Optional[str] = None
    request_body_template: Optional[str] = None
    parameters_schema_json: Optional[str] = None
    response_schema_json: Optional[str] = None
    codex_prompt: Optional[str] = None
    postprocess_python: Optional[str] = None
    return_result_directly: Optional[bool] = None
    response_mapper_json: Optional[str] = None
    pagination_json: Optional[str] = None
    static_reply_template: Optional[str] = None


def create_app() -> FastAPI:
    logger = logging.getLogger("voicebot.web")
    start = time.monotonic()
    logger.info("create_app: start")
    t0 = time.monotonic()
    settings = Settings()
    logger.info("create_app: settings loaded (%.2fs)", time.monotonic() - t0)
    t0 = time.monotonic()
    engine = make_engine(settings.db_url)
    logger.info("create_app: db engine ready (%.2fs)", time.monotonic() - t0)
    t0 = time.monotonic()
    init_db(engine)
    logger.info("create_app: init_db done (%.2fs)", time.monotonic() - t0)

    def _get_or_create_system_bot(session: Session) -> Bot:
        stmt = select(Bot).where(Bot.name == SYSTEM_BOT_NAME).limit(1)
        bot = session.exec(stmt).first()
        if bot:
            updated = False
            if bot.system_prompt != SYSTEM_BOT_PROMPT:
                bot.system_prompt = SYSTEM_BOT_PROMPT
                updated = True
            if bot.start_message_mode != "static":
                bot.start_message_mode = "static"
                updated = True
            if bot.start_message_text != SYSTEM_BOT_START_MESSAGE:
                bot.start_message_text = SYSTEM_BOT_START_MESSAGE
                updated = True
            if bot.enable_data_agent:
                bot.enable_data_agent = False
                updated = True
            if not bot.enable_host_actions:
                bot.enable_host_actions = True
                updated = True
            if not bot.enable_host_shell:
                bot.enable_host_shell = True
                updated = True
            if bot.require_host_action_approval:
                bot.require_host_action_approval = False
                updated = True
            if bot.disabled_tools_json != "[]":
                bot.disabled_tools_json = "[]"
                updated = True
            if updated:
                bot.updated_at = dt.datetime.now(dt.timezone.utc)
                session.add(bot)
                session.commit()
                session.refresh(bot)
            return bot
        bot = Bot(
            name=SYSTEM_BOT_NAME,
            system_prompt=SYSTEM_BOT_PROMPT,
            start_message_mode="static",
            start_message_text=SYSTEM_BOT_START_MESSAGE,
            enable_data_agent=False,
            enable_host_actions=True,
            enable_host_shell=True,
            require_host_action_approval=False,
            disabled_tools_json="[]",
        )
        return create_bot(session, bot)

    def _get_or_create_showcase_bot(session: Session) -> Bot:
        stmt = select(Bot).where(Bot.name == SHOWCASE_BOT_NAME).limit(1)
        bot = session.exec(stmt).first()
        if bot:
            updated = False
            if bot.system_prompt != SHOWCASE_BOT_PROMPT:
                bot.system_prompt = SHOWCASE_BOT_PROMPT
                updated = True
            if bot.start_message_mode != "static":
                bot.start_message_mode = "static"
                updated = True
            if bot.start_message_text != SHOWCASE_BOT_START_MESSAGE:
                bot.start_message_text = SHOWCASE_BOT_START_MESSAGE
                updated = True
            if not bot.enable_data_agent:
                bot.enable_data_agent = True
                updated = True
            if not bot.enable_host_actions:
                bot.enable_host_actions = True
                updated = True
            if not bot.enable_host_shell:
                bot.enable_host_shell = True
                updated = True
            if bot.require_host_action_approval:
                bot.require_host_action_approval = False
                updated = True
            if bot.disabled_tools_json != "[]":
                bot.disabled_tools_json = "[]"
                updated = True
            if updated:
                bot.updated_at = dt.datetime.now(dt.timezone.utc)
                session.add(bot)
                session.commit()
                session.refresh(bot)
            return bot
        bot = Bot(
            name=SHOWCASE_BOT_NAME,
            system_prompt=SHOWCASE_BOT_PROMPT,
            start_message_mode="static",
            start_message_text=SHOWCASE_BOT_START_MESSAGE,
            enable_data_agent=True,
            enable_host_actions=True,
            enable_host_shell=True,
            require_host_action_approval=False,
            disabled_tools_json="[]",
        )
        return create_bot(session, bot)

    try:
        with Session(engine) as session:
            _get_or_create_system_bot(session)
            _get_or_create_showcase_bot(session)
        logger.info("create_app: system bots ensured (%.2fs)", time.monotonic() - t0)
    except Exception:
        logger.exception("create_app: failed to ensure system bots")

    t0 = time.monotonic()
    app = FastAPI(title="GravexStudio")
    logger.info("create_app: FastAPI init done (%.2fs)", time.monotonic() - t0)
    data_agent_kickoff_locks: dict[UUID, asyncio.Lock] = {}

    download_base_url = (getattr(settings, "download_base_url", "") or "127.0.0.1:8000").strip()

    def _download_url_for_token(token: str) -> str:
        """
        Builds an absolute download URL for /api/downloads/{token}.

        Configure via VOICEBOT_DOWNLOAD_BASE_URL (supports full URL or host[:port]).
        """
        base = (download_base_url or "").strip()
        if not base:
            return f"/api/downloads/{token}"
        if not (base.startswith("http://") or base.startswith("https://")):
            base = "http://" + base
        base = base.rstrip("/")
        return f"{base}/api/downloads/{token}"

    basic_user = (settings.basic_auth_user or "").strip()
    basic_pass = (settings.basic_auth_pass or "").strip()
    basic_auth_enabled = bool(basic_user and basic_pass)

    def _basic_auth_ok(auth_header: str) -> bool:
        if not basic_auth_enabled:
            return True
        if not auth_header:
            return False
        if not auth_header.lower().startswith("basic "):
            return False
        token = auth_header.split(" ", 1)[1].strip()
        try:
            decoded = base64.b64decode(token).decode("utf-8")
        except Exception:
            return False
        user, sep, pwd = decoded.partition(":")
        if not sep:
            return False
        return secrets.compare_digest(user, basic_user) and secrets.compare_digest(pwd, basic_pass)

    def _ws_auth_header(ws: WebSocket) -> str:
        header = (ws.headers.get("authorization") or "").strip()
        if header:
            return header
        token = (ws.query_params.get("auth") or "").strip()
        if not token:
            return ""
        if token.lower().startswith("basic "):
            return token
        return f"Basic {token}"

    @app.middleware("http")
    async def _basic_auth_middleware(request: Request, call_next):  # type: ignore[override]
        if not basic_auth_enabled or request.method == "OPTIONS":
            return await call_next(request)
        if _basic_auth_ok(request.headers.get("authorization", "")):
            return await call_next(request)
        return Response(status_code=401, headers={"WWW-Authenticate": "Basic"})

    cors_raw = (os.environ.get("VOICEBOT_CORS_ORIGINS") or "").strip()
    cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()] if cors_raw else []
    cors_origin_regex: str | None = None
    if not cors_origins:
        # Dev-friendly defaults; override via VOICEBOT_CORS_ORIGINS for production.
        cors_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001",
        ]
        # Also allow other dev ports on common local hostnames.
        cors_origin_regex = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=cors_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ui_options = {
        "openai_models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            # GPT-5 family (see OpenAI docs for naming; used by Responses API)
            "gpt-5.2",
            "gpt-5.2-chat-latest",
            "gpt-5.2-pro",
            "gpt-5.1",
            "gpt-5.1-chat-latest",
            "gpt-5.1-mini",
            "gpt-5.1-nano",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex-mini",
            "gpt-5.1-codex",
            "gpt-5-codex",
            "o4-mini",
            "gpt-5",
            "gpt-5-chat-latest",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.1",
            "gpt-5.1-chat-latest",
        ],
        "openai_asr_models": ["gpt-4o-mini-transcribe", "whisper-1"],
        "languages": ["auto", "en", "hi", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl"],
        "openai_tts_models": ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
        "openai_tts_voices": [
            "alloy",
            "ash",
            "ballad",
            "coral",
            "echo",
            "fable",
            "nova",
            "onyx",
            "sage",
            "shimmer",
            "verse",
            # Extra voices observed in some SDK/model versions.
            "marin",
            "cedar",
        ],
    }

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    ui_dir_env = (os.environ.get("VOICEBOT_UI_DIR") or "").strip()
    ui_dir = Path(ui_dir_env) if ui_dir_env else (Path(__file__).parent / "ui")
    ui_index = ui_dir / "index.html"

    def _accepts_html(accept_header: str) -> bool:
        accept = (accept_header or "").lower()
        if not accept:
            return True
        if "text/html" in accept:
            return True
        if "*/*" in accept:
            return True
        return False

    # Best-effort: keep the model dropdown up-to-date by periodically fetching available models
    # from the OpenAI API, if a key is configured in the environment.
    openai_models_cache: dict[str, Any] = {"ts": 0.0, "models": []}
    openrouter_models_cache: dict[str, Any] = {"ts": 0.0, "models": [], "pricing": {}}

    def get_session() -> Generator[Session, None, None]:
        with Session(engine) as s:
            yield s

    def require_crypto():
        try:
            return get_crypto_box(settings.secret_key)
        except CryptoError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/system-bot")
    def api_system_bot(session: Session = Depends(get_session)) -> dict:
        bot = _get_or_create_system_bot(session)
        return {"id": str(bot.id), "name": bot.name}

    @app.get("/api/widget-config")
    def api_widget_config(session: Session = Depends(get_session)) -> dict:
        bot_id = _get_app_setting(session, WIDGET_BOT_KEY)
        widget_mode = (_get_app_setting(session, WIDGET_MODE_KEY) or "").strip().lower()
        if widget_mode not in ("mic", "text"):
            widget_mode = "mic"
        bot_name = None
        if bot_id:
            try:
                bot = get_bot(session, UUID(bot_id))
                bot_name = bot.name
            except Exception:
                bot_id = None
        return {"bot_id": bot_id, "bot_name": bot_name, "widget_mode": widget_mode}

    @app.post("/api/widget-config")
    def api_widget_config_update(
        payload: WidgetConfigRequest = Body(...),
        session: Session = Depends(get_session),
    ) -> dict:
        bot_id: Optional[str] = None
        bot_name: Optional[str] = None
        if payload.bot_id is not None:
            raw_bot_id = str(payload.bot_id or "").strip()
            if not raw_bot_id:
                _set_app_setting(session, WIDGET_BOT_KEY, "")
            else:
                bot = get_bot(session, UUID(raw_bot_id))
                _set_app_setting(session, WIDGET_BOT_KEY, str(bot.id))
                bot_id = str(bot.id)
                bot_name = bot.name

        if payload.widget_mode is not None:
            raw_mode = str(payload.widget_mode or "").strip().lower()
            if raw_mode not in ("mic", "text"):
                raise HTTPException(status_code=400, detail="widget_mode must be 'mic' or 'text'")
            _set_app_setting(session, WIDGET_MODE_KEY, raw_mode)

        if bot_id is None:
            stored_bot_id = _get_app_setting(session, WIDGET_BOT_KEY)
            if stored_bot_id:
                try:
                    bot = get_bot(session, UUID(stored_bot_id))
                    bot_id = str(bot.id)
                    bot_name = bot.name
                except Exception:
                    bot_id = None
                    bot_name = None

        widget_mode = (_get_app_setting(session, WIDGET_MODE_KEY) or "").strip().lower()
        if widget_mode not in ("mic", "text"):
            widget_mode = "mic"
        return {"bot_id": bot_id, "bot_name": bot_name, "widget_mode": widget_mode}

    @app.post("/api/open-dashboard")
    def api_open_dashboard(payload: Optional[OpenDashboardRequest] = Body(None)) -> dict:
        host = (os.environ.get("VOICEBOT_LAUNCH_HOST") or "127.0.0.1").strip() or "127.0.0.1"
        port = (os.environ.get("VOICEBOT_LAUNCH_PORT") or "8000").strip() or "8000"
        requested = ""
        if payload and payload.path:
            requested = str(payload.path or "").strip()
        path = requested or (os.environ.get("VOICEBOT_OPEN_PATH") or "/dashboard").strip() or "/dashboard"
        if "://" in path:
            path = "/dashboard"
        if not path.startswith("/"):
            path = f"/{path}"
        url = f"http://{host}:{port}{path}"
        try:
            webbrowser.open(url)
        except Exception:
            return {"ok": False, "url": url}
        return {"ok": True, "url": url}

    @app.get("/", include_in_schema=False)
    def root(request: Request):
        if ui_index.exists() and _accepts_html(request.headers.get("accept") or ""):
            return FileResponse(str(ui_index))
        return {"ok": True, "api_base": "/api", "public_widget_js": "/public/widget.js", "docs": "/docs"}

    @app.exception_handler(StarletteHTTPException)
    async def spa_fallback(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            path = request.url.path or ""
            if not path.startswith(("/api", "/public", "/static", "/ws", "/docs", "/openapi.json")):
                if ui_index.exists() and _accepts_html(request.headers.get("accept") or ""):
                    return FileResponse(str(ui_index))
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    class SpaStaticFiles(StaticFiles):
        def __init__(self, *args, index_file: Path, **kwargs):
            super().__init__(*args, **kwargs)
            self._index_file = index_file

        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            if response.status_code != 404:
                return response
            try:
                headers = Headers(scope=scope)
                accept = headers.get("accept") or ""
            except Exception:
                accept = ""
            if self._index_file.exists() and _accepts_html(accept) and "." not in path:
                return FileResponse(str(self._index_file))
            return response

    def _ndjson(obj: dict) -> bytes:
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

    def _wav_bytes(audio, sample_rate: int) -> bytes:
        import io

        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio, samplerate=sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _decode_wav_bytes_to_pcm16_16k(wav_bytes: bytes) -> bytes:
        import io

        import numpy as np
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if isinstance(data, np.ndarray) and data.ndim > 1:
            data = data[:, 0]
        audio_f32 = np.asarray(data, dtype=np.float32)
        if sr != 16000:
            # Simple linear resample to 16k (good enough for short test turns).
            ratio = 16000.0 / float(sr)
            n_out = int(round(len(audio_f32) * ratio))
            if n_out <= 0:
                return b""
            x_old = np.linspace(0.0, 1.0, num=len(audio_f32), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
            audio_f32 = np.interp(x_new, x_old, audio_f32).astype(np.float32)
        audio_i16 = np.clip(audio_f32, -1.0, 1.0)
        audio_i16 = (audio_i16 * 32767.0).astype(np.int16)
        return audio_i16.tobytes()

    @lru_cache(maxsize=16)
    def _get_asr(api_key: str, model_name: str, language: str) -> OpenAIASR:
        lang = None if (language or "").lower() in ("", "auto") else language
        return OpenAIASR(api_key=api_key, model=model_name, language=lang)

    @lru_cache(maxsize=16)
    def _get_openai_tts_handle(
        api_key: str,
        model: str,
        voice: str,
        speed: float,
    ) -> tuple[OpenAITTS, threading.Lock]:
        return (
            OpenAITTS(api_key=api_key, model=model, voice=voice, speed=speed),
            threading.Lock(),
        )

    def _get_tts_synth_fn(bot: Bot, api_key: Optional[str]) -> Callable[[str], tuple[bytes, int]]:
        """
        Returns a thread-safe (per-handle lock) wav synthesizer for the bot's configured TTS vendor.
        """
        if not api_key:
            raise RuntimeError("No OpenAI API key configured for OpenAI TTS.")
        model = (getattr(bot, "openai_tts_model", None) or "gpt-4o-mini-tts").strip()
        voice = (getattr(bot, "openai_tts_voice", None) or "alloy").strip()
        speed_raw = getattr(bot, "openai_tts_speed", None)
        try:
            speed = float(speed_raw) if speed_raw is not None else 1.0
        except Exception:
            speed = 1.0

        tts, lock = _get_openai_tts_handle(api_key, model, voice, speed)

        def synth(text: str) -> tuple[bytes, int]:
            with lock:
                wav = tts.synthesize_wav_bytes(text)
            return wav, OpenAITTS.DEFAULT_SAMPLE_RATE

        return synth

    def _build_history(session: Session, bot: Bot, conversation_id: Optional[UUID]) -> list[Message]:
        def _system_prompt_with_runtime(*, prompt: str) -> str:
            ts = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            return f"Current Date Time(UTC): {ts}\n\n{prompt}"

        messages: list[Message] = [Message(role="system", content=_system_prompt_with_runtime(prompt=bot.system_prompt))]
        if not conversation_id:
            return messages
        conv = get_conversation(session, conversation_id)
        _assert_bot_in_conversation(conv, bot.id)
        meta = safe_json_loads(conv.metadata_json or "{}") or {}
        ctx = {"meta": meta}
        # Render system prompt with metadata variables (if any)
        messages = [
            Message(
                role="system",
                content=_system_prompt_with_runtime(prompt=render_template(bot.system_prompt, ctx=ctx)),
            )
        ]
        if meta:
            messages.append(
                Message(role="system", content=f"Conversation metadata (JSON): {json.dumps(meta, ensure_ascii=False)}")
            )
        if bool(conv.is_group):
            bots = _group_bots_from_conv(conv)
            slugs = ", ".join(f"@{b['slug']}" for b in bots)
            is_default = str(conv.bot_id or "") == str(bot.id)
            if is_default:
                routing = (
                    "Group routing: If the latest message includes @mentions, only respond when you are mentioned. "
                    "If there are no @mentions, you are the default responder and should reply. "
                    "If you have nothing to add, respond with <no_reply> (this will be hidden). "
                    f"Available: {slugs}"
                )
            else:
                routing = (
                    "Group routing: assistants should only respond when explicitly mentioned with @slug in the latest message. "
                    "If you are not mentioned, respond with <no_reply> (this will be hidden). "
                    f"Available: {slugs}"
                )
            messages.append(Message(role="system", content=routing))
        for m in list_messages(session, conversation_id=conversation_id):
            prefix = _format_group_message_prefix(
                conv=conv,
                sender_bot_id=m.sender_bot_id,
                sender_name=m.sender_name,
                fallback_role=m.role,
            )
            if m.role in ("user", "assistant"):
                messages.append(Message(role=m.role, content=render_template(f"{prefix}{m.content}", ctx=ctx)))
            elif m.role == "tool":
                # Store tool calls/results as system breadcrumbs to prevent repeated calls.
                try:
                    obj = json.loads(m.content or "")
                    if isinstance(obj, dict) and obj.get("tool") == "debug_llm_request":
                        continue
                except Exception:
                    pass
                messages.append(
                    Message(
                        role="system",
                        content=render_template(f"{prefix}Tool event: {m.content}", ctx=ctx),
                    )
                )
        return messages

    def _build_history_budgeted(
        *,
        session: Session,
        bot: Bot,
        conversation_id: Optional[UUID],
        llm_api_key: Optional[str],
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> list[Message]:
        """
        Builds an LLM history capped to a token budget by using:
        - rolling summary stored in conversation metadata
        - a sliding window of the most recent N user turns (configurable per bot)

        If summarization runs, status_cb("summarizing") is invoked (UI-only).
        """

        HISTORY_TOKEN_BUDGET = 400000
        SUMMARY_BATCH_MIN_MESSAGES = 8

        def _system_prompt_with_runtime(*, prompt: str) -> str:
            ts = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
            return f"Current Date Time(UTC): {ts}\n\n{prompt}"

        if not conversation_id:
            return [Message(role="system", content=_system_prompt_with_runtime(prompt=bot.system_prompt))]

        conv = get_conversation(session, conversation_id)
        _assert_bot_in_conversation(conv, bot.id)

        meta = safe_json_loads(conv.metadata_json or "{}") or {}
        if not isinstance(meta, dict):
            meta = {}

        memory = meta.get("memory") if isinstance(meta.get("memory"), dict) else {}
        if not isinstance(memory, dict):
            memory = {}
        memory_summary = str(memory.get("summary") or "").strip()
        pinned_facts = str(memory.get("pinned_facts") or "").strip()
        last_summarized_id = str(memory.get("last_summarized_message_id") or "").strip()

        ctx = {"meta": meta}
        system_prompt = _system_prompt_with_runtime(prompt=render_template(bot.system_prompt, ctx=ctx))

        db_msgs = list_messages(session, conversation_id=conversation_id)
        # Determine sliding window start index by last N user turns.
        try:
            n_turns = int(getattr(bot, "history_window_turns", 16) or 16)
        except Exception:
            n_turns = 16
        if n_turns < 1:
            n_turns = 1
        if n_turns > 64:
            n_turns = 64

        user_indices = [i for i, m in enumerate(db_msgs) if m.role == "user"]
        if len(user_indices) > n_turns:
            start_idx = user_indices[-n_turns]
        else:
            start_idx = 0

        def _format_for_summary(m) -> str:
            role = m.role
            content = (m.content or "").strip()
            if role == "tool":
                # Avoid huge tool payloads; keep a short breadcrumb.
                content = content[:2000]
            prefix = _format_group_message_prefix(
                conv=conv,
                sender_bot_id=m.sender_bot_id,
                sender_name=m.sender_name,
                fallback_role=role,
            )
            return f"{role.upper()}: {prefix}{content}"

        # Update rolling summary for messages that will be dropped from the prompt.
        old_msgs = db_msgs[:start_idx]
        new_old_msgs = old_msgs
        if last_summarized_id:
            # Keep only messages after last summarized id.
            found = False
            tmp = []
            for m in old_msgs:
                if found:
                    tmp.append(m)
                elif str(getattr(m, "id", "") or "") == last_summarized_id:
                    found = True
            new_old_msgs = tmp if found else old_msgs

        should_summarize = False
        if old_msgs and (not memory_summary):
            should_summarize = True
        elif len(new_old_msgs) >= SUMMARY_BATCH_MIN_MESSAGES:
            should_summarize = True

        if should_summarize and llm_api_key:
            if status_cb:
                status_cb("summarizing")
            chunk = "\n".join(_format_for_summary(m) for m in new_old_msgs)
            # Keep summarizer input bounded.
            chunk = chunk[:24000]

            summary_model = (getattr(bot, "summary_model", "") or "gpt-5-nano").strip() or "gpt-5-nano"
            summarizer = _build_llm_client(bot=bot, api_key=llm_api_key, model_override=summary_model)
            summary_prompt = (
                "You are a conversation summarizer.\n"
                "Return STRICT JSON with keys: summary, pinned_facts, open_tasks.\n"
                "- summary: concise running summary (<= 1200 words)\n"
                "- pinned_facts: stable facts/preferences (<= 400 words)\n"
                "- open_tasks: short list (<= 12 items)\n"
                "Do not include any extra keys.\n"
            )
            prior = memory_summary or ""
            summarizer_input = (
                f"PRIOR_SUMMARY:\n{prior}\n\n"
                f"NEW_MESSAGES_TO_ABSORB:\n{chunk}\n"
            )
            text = summarizer.complete_text(
                messages=[
                    Message(role="system", content=summary_prompt),
                    Message(role="user", content=summarizer_input),
                ]
            )
            new_summary = ""
            new_pinned = ""
            try:
                obj = json.loads(text)
                if isinstance(obj, dict):
                    new_summary = str(obj.get("summary") or "").strip()
                    new_pinned = str(obj.get("pinned_facts") or "").strip()
            except Exception:
                # Fail closed: keep existing summary.
                new_summary = memory_summary
                new_pinned = pinned_facts

            if new_summary:
                patch = {
                    "memory.summary": new_summary,
                    "memory.pinned_facts": new_pinned,
                    "memory.last_summarized_message_id": str(getattr(new_old_msgs[-1], "id", "")) if new_old_msgs else last_summarized_id,
                    "memory.updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
                meta = merge_conversation_metadata(session, conversation_id=conversation_id, patch=patch)
                if isinstance(meta, dict):
                    memory = meta.get("memory") if isinstance(meta.get("memory"), dict) else {}
                    if isinstance(memory, dict):
                        memory_summary = str(memory.get("summary") or "").strip()
                        pinned_facts = str(memory.get("pinned_facts") or "").strip()
                # Drop summarized messages from storage so they don't get re-summarized.
                if new_old_msgs:
                    try:
                        ids = [m.id for m in new_old_msgs if getattr(m, "id", None)]
                        if ids:
                            session.exec(delete(ConversationMessage).where(ConversationMessage.id.in_(ids)))
                            session.commit()
                    except Exception:
                        session.rollback()

        # Build prompt messages: system prompt + summary + recent window.
        messages: list[Message] = [Message(role="system", content=system_prompt)]
        if bool(conv.is_group):
            bots = _group_bots_from_conv(conv)
            slugs = ", ".join(f"@{b['slug']}" for b in bots)
            is_default = str(conv.bot_id or "") == str(bot.id)
            if is_default:
                routing = (
                    "Group routing: If the latest message includes @mentions, only respond when you are mentioned. "
                    "If there are no @mentions, you are the default responder and should reply. "
                    "If you have nothing to add, respond with <no_reply> (this will be hidden). "
                    f"Available: {slugs}"
                )
            else:
                routing = (
                    "Group routing: assistants should only respond when explicitly mentioned with @slug in the latest message. "
                    "If you are not mentioned, respond with <no_reply> (this will be hidden). "
                    f"Available: {slugs}"
                )
            messages.append(
                Message(
                    role="system",
                    content=routing,
                )
            )
        if memory_summary:
            messages.append(Message(role="system", content=f"Conversation summary:\n{memory_summary}"))
        if pinned_facts:
            messages.append(Message(role="system", content=f"Pinned facts:\n{pinned_facts}"))

        for m in db_msgs[start_idx:]:
            prefix = _format_group_message_prefix(
                conv=conv,
                sender_bot_id=m.sender_bot_id,
                sender_name=m.sender_name,
                fallback_role=m.role,
            )
            if m.role in ("user", "assistant"):
                messages.append(
                    Message(role=m.role, content=render_template(f"{prefix}{m.content}", ctx={"meta": meta}))
                )
            elif m.role == "tool":
                try:
                    obj = json.loads(m.content or "")
                    if isinstance(obj, dict) and obj.get("tool") == "debug_llm_request":
                        continue
                except Exception:
                    pass
                # Keep full tool breadcrumbs. (Integration tools already store filtered results; truncation can
                # hide items and confuse follow-up questions.)
                messages.append(
                    Message(
                        role="system",
                        content=render_template(f"{prefix}Tool event: {m.content or ''}", ctx={"meta": meta}),
                    )
                )

        # If still over budget, trim oldest messages (keeping system + last user turn).
        try:
            while estimate_messages_tokens(messages, bot.openai_model) > HISTORY_TOKEN_BUDGET and len(messages) > 4:
                # Drop the oldest non-system message after the initial system+summary blocks.
                del messages[3]
        except Exception:
            pass
        return messages

    def _build_history_budgeted_threadsafe(
        *,
        bot_id: UUID,
        conversation_id: Optional[UUID],
        llm_api_key: Optional[str],
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> list[Message]:
        # NOTE: SQLModel sessions are not thread-safe; callers can use this via asyncio.to_thread().
        with Session(engine) as session:
            bot = get_bot(session, bot_id)
            return _build_history_budgeted(
                session=session,
                bot=bot,
                conversation_id=conversation_id,
                llm_api_key=llm_api_key,
                status_cb=status_cb,
            )

    async def _build_history_budgeted_async(
        *,
        bot_id: UUID,
        conversation_id: Optional[UUID],
        llm_api_key: Optional[str],
        status_cb: Optional[Callable[[str], None]] = None,
    ) -> list[Message]:
        return await asyncio.to_thread(
            _build_history_budgeted_threadsafe,
            bot_id=bot_id,
            conversation_id=conversation_id,
            llm_api_key=llm_api_key,
            status_cb=status_cb,
        )

    def _get_conversation_meta(session: Session, *, conversation_id: UUID) -> dict:
        conv = get_conversation(session, conversation_id)
        meta = safe_json_loads(conv.metadata_json or "{}") or {}
        return meta if isinstance(meta, dict) else {}

    def _extract_group_mentions(text: str, conv: Conversation) -> list[UUID]:
        alias_map = _group_bot_aliases(conv)
        if not alias_map:
            return []
        hits: list[UUID] = []
        for m in re.finditer(r"@([a-zA-Z0-9][a-zA-Z0-9_-]{0,48})", text or ""):
            token = m.group(1).lower()
            candidates = {token}
            compact = re.sub(r"[-_]+", "", token)
            if compact:
                candidates.add(compact)
            for cand in candidates:
                for bot_id in alias_map.get(cand, []):
                    try:
                        bid = UUID(bot_id)
                    except Exception:
                        continue
                    if bid not in hits:
                        hits.append(bid)
        return hits

    async def _run_group_bot_turn(
        *,
        bot_id: UUID,
        conversation_id: UUID,
    ) -> str:
        req_id = f"group_{secrets.token_hex(6)}"
        with Session(engine) as session:
            bot = get_bot(session, bot_id)
            conv = get_conversation(session, conversation_id)
            provider, api_key, llm = _require_llm_client(session, bot=bot)
            history = await _build_history_budgeted_async(
                bot_id=bot.id,
                conversation_id=conversation_id,
                llm_api_key=api_key,
                status_cb=None,
            )
            tools_defs = _build_tools_for_bot(session, bot.id)

        t0 = time.time()
        first_token_ts: Optional[float] = None
        tool_calls: list[ToolCall] = []
        full_text_parts: list[str] = []
        citations: list[dict[str, Any]] = []
        dispatch_targets: list[UUID] = []

        async for ev in _aiter_from_blocking_iterator(
            lambda: llm.stream_text_or_tool(messages=history, tools=tools_defs)
        ):
            if isinstance(ev, ToolCall):
                tool_calls.append(ev)
                continue
            if isinstance(ev, CitationEvent):
                citations.extend(ev.citations)
                continue
            d = str(ev)
            if d:
                if first_token_ts is None:
                    first_token_ts = time.time()
                full_text_parts.append(d)

        llm_end_ts = time.time()
        rendered_reply = "".join(full_text_parts).strip()

        llm_ttfb_ms: Optional[int] = None
        if first_token_ts is not None:
            llm_ttfb_ms = int(round((first_token_ts - t0) * 1000.0))
        elif tool_calls and tool_calls[0].first_event_ts is not None:
            llm_ttfb_ms = int(round((tool_calls[0].first_event_ts - t0) * 1000.0))
        llm_total_ms = int(round((llm_end_ts - t0) * 1000.0))

        if tool_calls:
            with Session(engine) as session:
                bot = get_bot(session, bot_id)
                conv = get_conversation(session, conversation_id)
                meta_current = _get_conversation_meta(session, conversation_id=conversation_id)
                disabled_tools = _disabled_tool_names(bot)
                final = ""
                needs_followup_llm = False
                tool_failed = False
                followup_streamed = False

                for tc in tool_calls:
                    tool_name = tc.name
                    if tool_name == "set_variable":
                        tool_name = "set_metadata"

                    try:
                        tool_args = json.loads(tc.arguments_json or "{}")
                        if not isinstance(tool_args, dict):
                            tool_args = {}
                    except Exception:
                        tool_args = {}

                    tool_call_msg = add_message_with_metrics(
                        session,
                        conversation_id=conversation_id,
                        role="tool",
                        content=json.dumps({"tool": tool_name, "arguments": tool_args}, ensure_ascii=False),
                        sender_bot_id=bot.id,
                        sender_name=bot.name,
                    )
                    _mirror_group_message(session, conv=conv, msg=tool_call_msg)

                    next_reply = str(tool_args.get("next_reply") or "").strip()
                    wait_reply = str(tool_args.get("wait_reply") or "").strip()
                    follow_up = _parse_follow_up_flag(tool_args.get("follow_up")) or _parse_follow_up_flag(
                        tool_args.get("followup")
                    )
                    if (
                        tool_name in {"request_host_action", "capture_screenshot"}
                        and "follow_up" not in tool_args
                        and "followup" not in tool_args
                    ):
                        follow_up = True
                    if tool_name in {"request_host_action", "capture_screenshot"}:
                        next_reply = ""
                    raw_args = tool_args.get("args")
                    if isinstance(raw_args, dict):
                        patch = dict(raw_args)
                    else:
                        patch = dict(tool_args)
                        patch.pop("next_reply", None)
                        patch.pop("wait_reply", None)
                        patch.pop("follow_up", None)
                        patch.pop("followup", None)
                        patch.pop("args", None)

                    tool_cfg: IntegrationTool | None = None
                    response_json: Any | None = None
                    if tool_name in disabled_tools:
                        tool_result = {
                            "ok": False,
                            "error": {"message": f"Tool '{tool_name}' is disabled for this bot."},
                        }
                        tool_failed = True
                        needs_followup_llm = True
                        final = ""
                    elif tool_name == "set_metadata":
                        new_meta = merge_conversation_metadata(session, conversation_id=conversation_id, patch=patch)
                        tool_result = {"ok": True, "updated": patch, "metadata": new_meta}
                    elif tool_name == "web_search":
                        tool_result = {
                            "ok": False,
                            "error": {"message": "web_search runs inside the model; no server tool is available."},
                        }
                        tool_failed = True
                        needs_followup_llm = True
                        final = ""
                    elif tool_name == "http_request":
                        tool_result = await asyncio.to_thread(
                            _execute_http_request_tool, tool_args=patch, meta=meta_current
                        )
                        tool_failed = not bool(tool_result.get("ok", False))
                        needs_followup_llm = True
                        final = ""
                    elif tool_name == "capture_screenshot":
                        if not bool(getattr(bot, "enable_host_actions", False)):
                            tool_result = {"ok": False, "error": {"message": "Host actions are disabled for this bot."}}
                            tool_failed = True
                            needs_followup_llm = True
                            final = ""
                        elif not bool(getattr(bot, "enable_host_shell", False)):
                            tool_result = {"ok": False, "error": {"message": "Shell commands are disabled for this bot."}}
                            tool_failed = True
                            needs_followup_llm = True
                            final = ""
                        else:
                            try:
                                rel_path, target = _prepare_screenshot_target(conv)
                            except Exception as exc:
                                tool_result = {"ok": False, "error": {"message": str(exc) or "Invalid screenshot path"}}
                                tool_failed = True
                                needs_followup_llm = True
                                final = ""
                            else:
                                ok_cmd, cmd_or_err = _screencapture_command(target)
                                if not ok_cmd:
                                    tool_result = {"ok": False, "error": {"message": cmd_or_err}}
                                    tool_failed = True
                                    needs_followup_llm = True
                                    final = ""
                                else:
                                    action = _create_host_action(
                                        session,
                                        conv=conv,
                                        bot=bot,
                                        action_type="run_shell",
                                        payload={"command": cmd_or_err},
                                    )
                                if _host_action_requires_approval(bot):
                                    tool_result = _build_host_action_tool_result(action, ok=True)
                                    tool_result["path"] = rel_path
                                    needs_followup_llm = True
                                    final = ""
                                else:
                                    tool_result = await _execute_host_action_and_update_async(
                                        session, action=action
                                    )
                                    tool_failed = not bool(tool_result.get("ok", False))
                                    if tool_failed:
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        ok, summary_text = _summarize_image_file(
                                            session,
                                            bot=bot,
                                            image_path=target,
                                            prompt=str(patch.get("prompt") or "").strip(),
                                        )
                                        if not ok:
                                            tool_result["summary_error"] = summary_text
                                            tool_failed = True
                                            needs_followup_llm = True
                                            final = ""
                                        else:
                                            tool_result["summary"] = summary_text
                                            tool_result["path"] = rel_path
                                            needs_followup_llm = True
                                            final = ""
                    elif tool_name == "summarize_screenshot":
                        ok, summary_text, rel_path = _summarize_screenshot(
                            session,
                            conv=conv,
                            bot=bot,
                            path=str(patch.get("path") or "").strip(),
                            prompt=str(patch.get("prompt") or "").strip(),
                        )
                        if not ok:
                            tool_result = {"ok": False, "error": {"message": summary_text}}
                            tool_failed = True
                            needs_followup_llm = True
                            final = ""
                        else:
                            tool_result = {"ok": True, "summary": summary_text, "path": rel_path}
                            needs_followup_llm = True
                            final = ""
                    elif tool_name == "request_host_action":
                        if not bool(getattr(bot, "enable_host_actions", False)):
                            tool_result = {"ok": False, "error": {"message": "Host actions are disabled for this bot."}}
                            tool_failed = True
                            needs_followup_llm = True
                            final = ""
                        else:
                            try:
                                action_type, payload = _parse_host_action_args(patch)
                            except Exception as exc:
                                tool_result = {"ok": False, "error": {"message": str(exc) or "Invalid host action"}}
                                tool_failed = True
                                needs_followup_llm = True
                                final = ""
                            else:
                                if action_type == "run_shell" and not bool(getattr(bot, "enable_host_shell", False)):
                                    tool_result = {
                                        "ok": False,
                                        "error": {"message": "Shell commands are disabled for this bot."},
                                    }
                                    tool_failed = True
                                    needs_followup_llm = True
                                    final = ""
                                else:
                                    action = _create_host_action(
                                        session,
                                        conv=conv,
                                        bot=bot,
                                        action_type=action_type,
                                        payload=payload,
                                    )
                                    if _host_action_requires_approval(bot):
                                        tool_result = _build_host_action_tool_result(action, ok=True)
                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                        if candidate:
                                            final = candidate
                                            needs_followup_llm = False
                                        else:
                                            needs_followup_llm = True
                                            final = ""
                                    else:
                                        tool_result = await _execute_host_action_and_update_async(
                                            session, action=action
                                        )
                                        tool_failed = not bool(tool_result.get("ok", False))
                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                        if follow_up and not tool_failed:
                                            needs_followup_llm = True
                                            final = ""
                                        elif candidate and not tool_failed:
                                            final = candidate
                                            needs_followup_llm = False
                                        else:
                                            needs_followup_llm = True
                                            final = ""
                    elif tool_name == "give_command_to_data_agent":
                        if not bool(getattr(bot, "enable_data_agent", False)):
                            tool_result = {"ok": False, "error": {"message": "Isolated Workspace is disabled for this bot."}}
                            tool_failed = True
                            needs_followup_llm = True
                            final = ""
                        elif not docker_available():
                            tool_result = {
                                "ok": False,
                                "error": {"message": "Docker is not available. Install Docker to use Isolated Workspace."},
                            }
                            tool_failed = True
                            needs_followup_llm = True
                            final = ""
                        else:
                            what_to_do = str(patch.get("what_to_do") or "").strip()
                            if not what_to_do:
                                tool_result = {"ok": False, "error": {"message": "Missing required tool arg: what_to_do"}}
                                tool_failed = True
                                needs_followup_llm = True
                                final = ""
                            else:
                                try:
                                    logger.info(
                                        "Isolated Workspace tool: start conv=%s bot=%s what_to_do=%s",
                                        conversation_id,
                                        bot_id,
                                        (what_to_do[:200] + "…") if len(what_to_do) > 200 else what_to_do,
                                    )
                                    da = _data_agent_meta(meta_current)
                                    workspace_dir = (
                                        str(da.get("workspace_dir") or "").strip()
                                        or default_workspace_dir_for_conversation(conversation_id)
                                    )
                                    container_id = str(da.get("container_id") or "").strip()
                                    session_id = str(da.get("session_id") or "").strip()
                                    auth_json_raw = getattr(bot, "data_agent_auth_json", "") or "{}"
                                    git_token = (
                                        _get_git_token_plaintext(session, provider="github")
                                        if _git_auth_mode(auth_json_raw) == "token"
                                        else ""
                                    )

                                    if not container_id:
                                        container_id = await asyncio.to_thread(
                                            ensure_conversation_container,
                                            conversation_id=conversation_id,
                                            workspace_dir=workspace_dir,
                                            openai_api_key=api_key,
                                            git_token=git_token,
                                            auth_json=auth_json_raw,
                                        )
                                        meta_current = merge_conversation_metadata(
                                            session,
                                            conversation_id=conversation_id,
                                            patch={
                                                "data_agent.container_id": container_id,
                                                "data_agent.workspace_dir": workspace_dir,
                                            },
                                        )

                                    ctx = _build_data_agent_conversation_context(
                                        session,
                                        bot=bot,
                                        conversation_id=conversation_id,
                                        meta=meta_current,
                                    )
                                    api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
                                    auth_json = _merge_git_token_auth(auth_json_raw, git_token)
                                    sys_prompt = (
                                        (getattr(bot, "data_agent_system_prompt", "") or "").strip()
                                        or DEFAULT_DATA_AGENT_SYSTEM_PROMPT
                                    )

                                    da_res = await asyncio.to_thread(
                                        run_data_agent,
                                        conversation_id=conversation_id,
                                        container_id=container_id,
                                        session_id=session_id,
                                        workspace_dir=workspace_dir,
                                        api_spec_text=api_spec_text,
                                        auth_json=auth_json,
                                        system_prompt=sys_prompt,
                                        conversation_context=ctx,
                                        what_to_do=what_to_do,
                                        on_stream=lambda _t: None,
                                    )
                                    if da_res.session_id and da_res.session_id != session_id:
                                        meta_current = merge_conversation_metadata(
                                            session,
                                            conversation_id=conversation_id,
                                            patch={"data_agent.session_id": da_res.session_id},
                                        )
                                    tool_result = {
                                        "ok": bool(da_res.ok),
                                        "result_text": da_res.result_text,
                                        "data_agent_container_id": da_res.container_id,
                                        "data_agent_session_id": da_res.session_id,
                                        "data_agent_output_file": da_res.output_file,
                                        "data_agent_debug_file": da_res.debug_file,
                                        "error": da_res.error,
                                    }
                                    tool_failed = not bool(da_res.ok)
                                    if (
                                        bool(getattr(bot, "data_agent_return_result_directly", False))
                                        and bool(da_res.ok)
                                        and str(da_res.result_text or "").strip()
                                    ):
                                        needs_followup_llm = False
                                        final = str(da_res.result_text or "").strip()
                                    else:
                                        needs_followup_llm = True
                                        final = ""
                                except Exception as exc:
                                    logger.exception(
                                        "Isolated Workspace tool failed conv=%s bot=%s",
                                        conversation_id,
                                        bot_id,
                                    )
                                    tool_result = {"ok": False, "error": {"message": str(exc)}}
                                    tool_failed = True
                                    needs_followup_llm = True
                                    final = ""
                    else:
                        tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                        if not tool_cfg:
                            raise RuntimeError(f"Unknown tool: {tool_name}")
                        if not bool(getattr(tool_cfg, "enabled", True)):
                            response_json = {
                                "__tool_args_error__": {
                                    "missing": [],
                                    "message": f"Tool '{tool_name}' is disabled for this bot.",
                                }
                            }
                        else:
                            response_json = await asyncio.to_thread(
                                _execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                            )
                        if isinstance(response_json, dict) and "__tool_args_error__" in response_json:
                            err = response_json["__tool_args_error__"] or {}
                            tool_result = {"ok": False, "error": err}
                            tool_failed = True
                            needs_followup_llm = True
                            final = ""
                        elif isinstance(response_json, dict) and "__http_error__" in response_json:
                            err = response_json["__http_error__"] or {}
                            tool_result = {"ok": False, "error": err}
                            tool_failed = True
                            needs_followup_llm = True
                            final = ""
                        else:
                            pagination_info = None
                            if isinstance(response_json, dict) and "__igx_pagination__" in response_json:
                                pagination_info = response_json.pop("__igx_pagination__", None)
                            if bool(getattr(tool_cfg, "use_codex_response", False)):
                                tool_result = {"ok": True}
                                new_meta = meta_current
                            else:
                                mapped = _apply_response_mapper(
                                    mapper_json=tool_cfg.response_mapper_json,
                                    response_json=response_json,
                                    meta=meta_current,
                                    tool_args=patch,
                                )
                                new_meta = merge_conversation_metadata(
                                    session, conversation_id=conversation_id, patch=mapped
                                )
                                meta_current = new_meta
                                tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}
                            if pagination_info:
                                tool_result["pagination"] = pagination_info

                            static_preview = ""
                            if (tool_cfg.static_reply_template or "").strip():
                                try:
                                    static_preview = _render_static_reply(
                                        template_text=tool_cfg.static_reply_template,
                                        meta=new_meta or meta_current,
                                        response_json=response_json,
                                        tool_args=patch,
                                    ).strip()
                                except Exception:
                                    static_preview = ""
                            if (not static_preview) and bool(getattr(tool_cfg, "use_codex_response", False)):
                                fields_required = str(patch.get("fields_required") or "").strip()
                                why_api_was_called = str(patch.get("why_api_was_called") or "").strip()
                                what_to_search_for = str(patch.get("what_to_search_for") or "").strip()
                                if not fields_required:
                                    fields_required = what_to_search_for
                                if not fields_required or not why_api_was_called:
                                    tool_result["codex_error"] = "Missing fields_required / why_api_was_called."
                                else:
                                    fields_required_for_codex = fields_required
                                    if what_to_search_for and what_to_search_for not in fields_required_for_codex:
                                        fields_required_for_codex = (
                                            f"{fields_required_for_codex}\n\n(what_to_search_for) {what_to_search_for}"
                                        )
                                    did_postprocess = False
                                    if (tool_cfg.postprocess_python or "").strip():
                                        try:
                                            py_res = await asyncio.to_thread(
                                                run_python_postprocess,
                                                python_code=tool_cfg.postprocess_python,
                                                response_json=response_json,
                                                meta=new_meta or meta_current,
                                                args=patch,
                                                fields_required=fields_required_for_codex,
                                                why_api_was_called=why_api_was_called,
                                                timeout_s=60,
                                            )
                                            tool_result["python_ok"] = bool(getattr(py_res, "ok", False))
                                            tool_result["python_duration_ms"] = int(getattr(py_res, "duration_ms", 0) or 0)
                                            if getattr(py_res, "error", None):
                                                tool_result["python_error"] = str(getattr(py_res, "error"))
                                            if getattr(py_res, "stderr", None):
                                                tool_result["python_stderr"] = str(getattr(py_res, "stderr"))
                                            if py_res.ok:
                                                did_postprocess = True
                                                tool_result["postprocess_mode"] = "python"
                                                tool_result["codex_ok"] = True
                                                tool_result["codex_result_text"] = str(
                                                    getattr(py_res, "result_text", "") or ""
                                                )
                                                mp = getattr(py_res, "metadata_patch", None)
                                                if isinstance(mp, dict) and mp:
                                                    try:
                                                        meta_current = merge_conversation_metadata(
                                                            session,
                                                            conversation_id=conversation_id,
                                                            patch=mp,
                                                        )
                                                        tool_result["python_metadata_patch"] = mp
                                                    except Exception:
                                                        pass
                                        except Exception as exc:
                                            tool_result["python_ok"] = False
                                            tool_result["python_error"] = str(exc)

                                    if not did_postprocess:
                                        codex_model = (
                                            (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                            or "gpt-5.1-codex-mini"
                                        )
                                        try:
                                            agent_res = await asyncio.to_thread(
                                                run_codex_http_agent_one_shot,
                                                api_key=api_key or "",
                                                model=codex_model,
                                                response_json=response_json,
                                                fields_required=fields_required_for_codex,
                                                why_api_was_called=why_api_was_called,
                                                response_schema_json=getattr(tool_cfg, "response_schema_json", "") or "",
                                                conversation_id=str(conversation_id) if conversation_id is not None else None,
                                                req_id=req_id,
                                                tool_codex_prompt=getattr(tool_cfg, "codex_prompt", "") or "",
                                                progress_fn=lambda _p: None,
                                            )
                                            tool_result["postprocess_mode"] = "codex"
                                            tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                            tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                            tool_result["codex_output_file"] = getattr(agent_res, "result_text_path", "")
                                            tool_result["codex_debug_file"] = getattr(agent_res, "debug_json_path", "")
                                            tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                            tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                            tool_result["codex_continue_reason"] = getattr(agent_res, "continue_reason", "")
                                            tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                            err = getattr(agent_res, "error", None)
                                            if err:
                                                tool_result["codex_error"] = str(err)
                                        except Exception as exc:
                                            tool_result["codex_ok"] = False
                                            tool_result["codex_error"] = str(exc)

                    if tool_name == "capture_screenshot" and tool_failed:
                        msg = _tool_error_message(tool_result, fallback="Screenshot failed.")
                        final = f"Screenshot failed: {msg}"
                        needs_followup_llm = False
                    if tool_name == "request_host_action" and tool_failed:
                        msg = _tool_error_message(tool_result, fallback="Host action failed.")
                        final = f"Host action failed: {msg}"
                        needs_followup_llm = False

                    tool_result_msg = add_message_with_metrics(
                        session,
                        conversation_id=conversation_id,
                        role="tool",
                        content=json.dumps({"tool": tool_name, "result": tool_result}, ensure_ascii=False),
                        sender_bot_id=bot.id,
                        sender_name=bot.name,
                    )
                    _mirror_group_message(session, conv=conv, msg=tool_result_msg)
                    if isinstance(tool_result, dict):
                        meta_current = tool_result.get("metadata") or meta_current

                    if tool_failed:
                        break

                    candidate = ""
                    if tool_name != "set_metadata" and tool_cfg:
                        static_text = ""
                        if (tool_cfg.static_reply_template or "").strip():
                            static_text = _render_static_reply(
                                template_text=tool_cfg.static_reply_template,
                                meta=meta_current,
                                response_json=response_json,
                                tool_args=patch,
                            ).strip()
                        if static_text:
                            needs_followup_llm = False
                            final = static_text
                        else:
                            if bool(getattr(tool_cfg, "use_codex_response", False)):
                                if bool(getattr(tool_cfg, "return_result_directly", False)) and isinstance(tool_result, dict):
                                    direct = str(tool_result.get("codex_result_text") or "").strip()
                                    if direct:
                                        needs_followup_llm = False
                                        final = direct
                                    else:
                                        needs_followup_llm = True
                                        final = ""
                                else:
                                    needs_followup_llm = True
                                    final = ""
                            else:
                                needs_followup_llm = _should_followup_llm_for_tool(
                                    tool=tool_cfg, static_rendered=static_text
                                )
                                candidate = _render_with_meta(next_reply, meta_current).strip()
                                if candidate:
                                    final = candidate
                                    needs_followup_llm = False
                                else:
                                    final = ""
                    else:
                        candidate = _render_with_meta(next_reply, meta_current).strip()
                        final = candidate or final

                if needs_followup_llm:
                    followup_history = await _build_history_budgeted_async(
                        bot_id=bot.id,
                        conversation_id=conversation_id,
                        llm_api_key=api_key,
                        status_cb=None,
                    )
                    followup_history.append(
                        Message(
                            role="system",
                            content=(
                                ("The previous tool call failed. " if tool_failed else "")
                                + "Using the latest tool result(s) above, write the next assistant reply. "
                                "If the tool result contains codex_result_text, rephrase it for the user and do not mention file paths. "
                                "Do not call any tools."
                            ),
                        )
                    )
                    text2 = ""
                    async for d in _aiter_from_blocking_iterator(lambda: llm.stream_text(messages=followup_history)):
                        if d:
                            text2 += str(d)
                    rendered_reply = text2.strip()
                    followup_streamed = True
                    llm_ttfb_ms = None
                    llm_total_ms = None
                else:
                    rendered_reply = final

        if rendered_reply:
            with Session(engine) as session:
                conv = get_conversation(session, conversation_id)
            if bool(conv.is_group):
                rendered_reply = _sanitize_group_reply(rendered_reply, conv, bot_id)

        payload = None
        raw_reply = rendered_reply or ""
        if re.fullmatch(r"\s*<no_reply>\s*", raw_reply, flags=re.IGNORECASE):
            rendered_reply = ""
            suppress_reply = True
        else:
            rendered_reply = re.sub(r"\s*<no_reply>\s*$", "", raw_reply, flags=re.IGNORECASE).strip()
            suppress_reply = False
        with Session(engine) as session:
            in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                session=session,
                bot=bot,
                provider=provider,
                history=history,
                assistant_text=rendered_reply,
            )
            conv = get_conversation(session, conversation_id)
            assistant_msg = None
            if suppress_reply:
                if bool(conv.is_group):
                    mapping = _ensure_group_individual_conversations(session, conv)
                    target_id = mapping.get(str(bot.id))
                    if target_id:
                        add_message_with_metrics(
                            session,
                            conversation_id=UUID(target_id),
                            role="assistant",
                            content=rendered_reply,
                            sender_bot_id=bot.id,
                            sender_name=bot.name,
                            input_tokens_est=in_tok,
                            output_tokens_est=out_tok,
                            cost_usd_est=cost,
                            llm_ttfb_ms=llm_ttfb_ms,
                            llm_total_ms=llm_total_ms,
                            total_ms=llm_total_ms,
                            citations_json=json.dumps(citations, ensure_ascii=False),
                        )
            else:
                assistant_msg = add_message_with_metrics(
                    session,
                    conversation_id=conversation_id,
                    role="assistant",
                    content=rendered_reply,
                    sender_bot_id=bot.id,
                    sender_name=bot.name,
                    input_tokens_est=in_tok,
                    output_tokens_est=out_tok,
                    cost_usd_est=cost,
                    llm_ttfb_ms=llm_ttfb_ms,
                    llm_total_ms=llm_total_ms,
                    total_ms=llm_total_ms,
                    citations_json=json.dumps(citations, ensure_ascii=False),
                )
                _mirror_group_message(session, conv=conv, msg=assistant_msg)
            update_conversation_metrics(
                session,
                conversation_id=conversation_id,
                add_input_tokens_est=in_tok,
                add_output_tokens_est=out_tok,
                add_cost_usd_est=cost,
                last_asr_ms=None,
                last_llm_ttfb_ms=llm_ttfb_ms,
                last_llm_total_ms=llm_total_ms,
                last_tts_first_audio_ms=None,
                last_total_ms=llm_total_ms,
            )

            if assistant_msg is not None:
                payload = _group_message_payload(assistant_msg)
            if assistant_msg is not None:
                dispatch_targets = _extract_group_mentions(rendered_reply, conv)
                dispatch_targets = [bid for bid in dispatch_targets if str(bid) != str(bot.id)]

        if payload:
            await _group_ws_broadcast(conversation_id, {"type": "message", "message": payload})
        if dispatch_targets:
            _schedule_group_bots(conversation_id, dispatch_targets)

        return rendered_reply

    def _schedule_group_bots(conversation_id: UUID, targets: list[UUID]) -> None:
        if not targets:
            return
        with Session(engine) as session:
            conv = get_conversation(session, conversation_id)
            bot_map = {b["id"]: b for b in _group_bots_from_conv(conv)}

        async def _run_target(target_id: UUID) -> None:
            binfo = bot_map.get(str(target_id), {})
            await _group_ws_broadcast(
                conversation_id,
                {
                    "type": "status",
                    "bot_id": str(target_id),
                    "bot_name": binfo.get("name") or "assistant",
                    "state": "working",
                },
            )
            try:
                await _run_group_bot_turn(bot_id=target_id, conversation_id=conversation_id)
            finally:
                await _group_ws_broadcast(
                    conversation_id,
                    {
                        "type": "status",
                        "bot_id": str(target_id),
                        "bot_name": binfo.get("name") or "assistant",
                        "state": "idle",
                    },
                )

        try:
            loop = asyncio.get_running_loop()
            for bid in targets:
                loop.create_task(_run_target(bid))
        except Exception:
            pass

    def _normalize_llm_provider(provider: str) -> str:
        p = (provider or "").strip().lower()
        if p in ("openai", "openrouter", "local"):
            return p
        return "openai"

    def _provider_display_name(provider: str) -> str:
        if provider == "openrouter":
            return "OpenRouter"
        if provider == "local":
            return "Local"
        return "OpenAI"

    def _get_openai_api_key(session: Session) -> str:
        # Prefer env, fall back to the latest stored OpenAI key.
        key = os.environ.get("OPENAI_API_KEY") or ""
        if not key:
            try:
                crypto = require_crypto()
            except Exception:
                crypto = None
            if crypto is not None:
                try:
                    key = decrypt_provider_key(session, crypto=crypto, provider="openai") or ""
                except Exception:
                    key = ""
        if not key:
            key = _read_key_from_env_file("OPENAI_API_KEY")
        return (key or "").strip()

    def _get_openai_api_key_for_bot(session: Session, *, bot: Bot) -> str:
        _ = bot
        return _get_openai_api_key(session)

    def _get_openrouter_api_key(session: Session) -> str:
        key = os.environ.get("OPENROUTER_API_KEY") or ""
        if not key:
            try:
                crypto = require_crypto()
            except Exception:
                crypto = None
            if crypto is not None:
                try:
                    key = decrypt_provider_key(session, crypto=crypto, provider="openrouter") or ""
                except Exception:
                    key = ""
        if not key:
            key = _read_key_from_env_file("OPENROUTER_API_KEY")
        return (key or "").strip()

    def _get_openrouter_api_key_for_bot(session: Session, *, bot: Bot) -> str:
        _ = bot
        return _get_openrouter_api_key(session)

    def _llm_provider_for_bot(bot: Bot) -> str:
        return _normalize_llm_provider(getattr(bot, "llm_provider", "") or "openai")

    def _get_llm_api_key_for_bot(session: Session, *, bot: Bot) -> tuple[str, str]:
        provider = _llm_provider_for_bot(bot)
        if provider == "openrouter":
            return provider, _get_openrouter_api_key_for_bot(session, bot=bot)
        if provider == "local":
            return provider, ""
        return provider, _get_openai_api_key_for_bot(session, bot=bot)

    def _build_llm_client(*, bot: Bot, api_key: str, model_override: Optional[str] = None):
        provider = _llm_provider_for_bot(bot)
        model = (model_override or getattr(bot, "openai_model", "") or "o4-mini").strip() or "o4-mini"
        if provider == "openrouter":
            base_url = (os.environ.get("OPENROUTER_BASE_URL") or "").strip() or None
            referer = (os.environ.get("OPENROUTER_REFERER") or "").strip() or None
            title = (os.environ.get("OPENROUTER_TITLE") or "").strip() or None
            return OpenRouterLLM(
                model=model,
                api_key=api_key,
                base_url=base_url,
                referer=referer,
                title=title,
            )
        if provider == "local":
            base_url = (os.environ.get("IGX_LOCAL_LLM_BASE_URL") or "").strip()
            if not base_url:
                status = LOCAL_RUNTIME.status()
                port = int(status.get("server_port") or 0) or 0
                if port:
                    base_url = f"http://127.0.0.1:{port}"
            if not base_url:
                raise RuntimeError("Local runtime not ready.")
            return OpenAICompatLLM(model=model, base_url=base_url, api_key=None)
        return OpenAILLM(model=model, api_key=api_key)

    def _require_llm_client(
        session: Session,
        *,
        bot: Bot,
        model_override: Optional[str] = None,
    ) -> tuple[str, str, Any]:
        provider, api_key = _get_llm_api_key_for_bot(session, bot=bot)
        if provider != "local" and not api_key:
            raise HTTPException(
                status_code=400,
                detail=f"No {_provider_display_name(provider)} key configured for this bot.",
            )
        if provider == "local":
            base_url_env = (os.environ.get("IGX_LOCAL_LLM_BASE_URL") or "").strip()
            if not base_url_env and not LOCAL_RUNTIME.is_ready():
                raise HTTPException(status_code=400, detail="Local model not ready. Finish local setup first.")
        llm = _build_llm_client(bot=bot, api_key=api_key, model_override=model_override)
        return provider, api_key, llm

    def _normalize_git_provider(provider: str) -> str:
        p = (provider or "").strip().lower()
        if p in ("github", "gh"):
            return "github"
        raise HTTPException(status_code=400, detail="Unsupported provider")

    def _get_git_token_plaintext(session: Session, *, provider: str) -> str:
        try:
            crypto = require_crypto()
        except Exception:
            return ""
        rec = get_git_token(session, provider=provider)
        if not rec:
            return ""
        try:
            return crypto.decrypt_str(rec.token_ciphertext)
        except Exception:
            return ""

    def _parse_auth_json(auth_json: str) -> dict[str, Any]:
        try:
            obj = json.loads((auth_json or "").strip() or "{}")
        except Exception:
            return {}
        if not isinstance(obj, dict):
            return {}
        return obj

    def _git_auth_mode(auth_json: str) -> str:
        obj = _parse_auth_json(auth_json)
        method = str(obj.get("git_auth_method") or "").strip().lower()
        if method in ("ssh", "ssh-key", "ssh_key"):
            return "ssh"
        if method in ("token", "pat", "github_token", "github_pat"):
            return "token"
        for key in (
            "ssh_private_key",
            "ssh_private_key_path",
            "ssh_private_key_b64",
            "ssh_private_key_base64",
            "ssh_key",
            "ssh_key_path",
        ):
            if str(obj.get(key) or "").strip():
                return "ssh"
        return "token"

    def _merge_git_token_auth(auth_json: str, git_token: str) -> str:
        if not git_token or _git_auth_mode(auth_json) != "token":
            return (auth_json or "{}").strip() or "{}"
        obj = _parse_auth_json(auth_json)
        if not obj.get("github_token"):
            obj["github_token"] = git_token
        if not obj.get("GITHUB_TOKEN"):
            obj["GITHUB_TOKEN"] = git_token
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return (auth_json or "{}").strip() or "{}"

    async def _validate_github_token(token: str) -> tuple[bool, str | None]:
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(
                    "https://api.github.com/user",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github+json",
                        "User-Agent": "GravexStudio-VoiceBot",
                    },
                )
            if resp.status_code == 200:
                return True, None
            if resp.status_code in (401, 403):
                return False, "Invalid GitHub token"
            return False, f"GitHub validation error (status {resp.status_code})"
        except Exception as exc:
            return False, f"GitHub validation failed: {exc}"

    def _data_agent_meta(meta: dict) -> dict:
        da = meta.get("data_agent")
        return da if isinstance(da, dict) else {}

    def _ensure_data_agent_container(
        session: Session, *, bot: Bot, conversation_id: UUID, meta_current: dict
    ) -> tuple[str, str, str]:
        """
        Ensures the per-conversation Isolated Workspace runtime exists.

        Returns: (container_id, session_id, workspace_dir).
        """
        da = _data_agent_meta(meta_current)
        workspace_dir = str(da.get("workspace_dir") or "").strip() or default_workspace_dir_for_conversation(conversation_id)
        container_id = str(da.get("container_id") or "").strip()
        session_id = str(da.get("session_id") or "").strip()

        api_key = _get_openai_api_key_for_bot(session, bot=bot)
        if not api_key:
            raise RuntimeError("No OpenAI API key configured for this bot (needed for Isolated Workspace).")
        auth_json = getattr(bot, "data_agent_auth_json", "") or "{}"
        git_token = _get_git_token_plaintext(session, provider="github") if _git_auth_mode(auth_json) == "token" else ""

        if not container_id:
            container_id = ensure_conversation_container(
                conversation_id=conversation_id,
                workspace_dir=workspace_dir,
                openai_api_key=api_key,
                git_token=git_token,
                auth_json=auth_json,
            )
            meta_current = merge_conversation_metadata(
                session,
                conversation_id=conversation_id,
                patch={
                    "data_agent.container_id": container_id,
                    "data_agent.workspace_dir": workspace_dir,
                    "data_agent.session_id": session_id,
                },
            )
        return container_id, session_id, workspace_dir

    def _build_data_agent_conversation_context(session: Session, *, bot: Bot, conversation_id: UUID, meta: dict) -> dict[str, Any]:
        # Keep this small: reuse our existing summary (if any) + last N messages.
        summary = ""
        try:
            mem = meta.get("memory")
            if isinstance(mem, dict):
                summary = str(mem.get("summary") or "").strip()
        except Exception:
            summary = ""

        msgs = list_messages(session, conversation_id=conversation_id)
        n_turns = int(getattr(bot, "history_window_turns", 16) or 16)
        max_msgs = max(8, min(96, n_turns * 2))
        tail = msgs[-max_msgs:] if len(msgs) > max_msgs else msgs
        history = [{"role": m.role, "content": m.content} for m in tail]
        return {"summary": summary, "messages": history}

    def _initialize_data_agent_workspace(session: Session, *, bot: Bot, conversation_id: UUID, meta: dict) -> str:
        workspace_dir = _data_agent_workspace_dir_for_conversation(session, conversation_id=conversation_id)
        ws = Path(workspace_dir)
        ws.mkdir(parents=True, exist_ok=True)

        api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
        auth_json = (getattr(bot, "data_agent_auth_json", "") or "{}").strip() or "{}"
        sys_prompt = (getattr(bot, "data_agent_system_prompt", "") or "").strip() or DEFAULT_DATA_AGENT_SYSTEM_PROMPT
        ctx = _build_data_agent_conversation_context(session, bot=bot, conversation_id=conversation_id, meta=meta)

        try:
            (ws / "api_spec.json").write_text(api_spec_text, encoding="utf-8")
            (ws / "auth.json").write_text(auth_json, encoding="utf-8")
            (ws / "AGENTS.md").write_text(sys_prompt + "\n", encoding="utf-8")
            (ws / "conversation_context.json").write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return workspace_dir

    async def _kickoff_data_agent_container_if_enabled(*, bot_id: UUID, conversation_id: UUID) -> None:
        """
        Best-effort: start (and optionally prewarm) the per-conversation Isolated Workspace runtime at conversation start.

        NOTE: This uses Docker locally. For Kubernetes, this should be replaced with a Pod/Job-based runner.
        """
        lock = data_agent_kickoff_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            data_agent_kickoff_locks[conversation_id] = lock

        async with lock:
            try:
                with Session(engine) as session:
                    bot = get_bot(session, bot_id)
                    if not bool(getattr(bot, "enable_data_agent", False)):
                        return
                    if not docker_available():
                        logger.warning("Isolated Workspace kickoff: Docker not available conv=%s bot=%s", conversation_id, bot_id)
                        merge_conversation_metadata(
                            session,
                            conversation_id=conversation_id,
                            patch={
                                "data_agent.init_error": "Docker is not available. Install Docker to use Isolated Workspace.",
                                "data_agent.ready": False,
                            },
                        )
                        return

                    prewarm = bool(getattr(bot, "data_agent_prewarm_on_start", False))
                    meta = _get_conversation_meta(session, conversation_id=conversation_id)
                    da = _data_agent_meta(meta)
                    if prewarm and (bool(da.get("ready", False)) or bool(da.get("prewarm_in_progress", False))):
                        return
                    if (not prewarm) and str(da.get("container_id") or "").strip():
                        return

                    api_key = _get_openai_api_key_for_bot(session, bot=bot)
                    if not api_key:
                        logger.warning("Isolated Workspace kickoff: missing OpenAI key conv=%s bot=%s", conversation_id, bot_id)
                        merge_conversation_metadata(
                            session,
                            conversation_id=conversation_id,
                            patch={
                                "data_agent.init_error": "No OpenAI API key configured for this bot.",
                                "data_agent.ready": False,
                            },
                        )
                        return
                    auth_json = getattr(bot, "data_agent_auth_json", "") or "{}"
                    git_token = (
                        _get_git_token_plaintext(session, provider="github") if _git_auth_mode(auth_json) == "token" else ""
                    )

                    workspace_dir = (
                        str(da.get("workspace_dir") or "").strip()
                        or default_workspace_dir_for_conversation(conversation_id)
                    )
                    container_id = str(da.get("container_id") or "").strip()
                    session_id = str(da.get("session_id") or "").strip()

                if not container_id:
                    logger.info(
                        "Isolated Workspace kickoff: starting container conv=%s workspace=%s",
                        conversation_id,
                        workspace_dir,
                    )
                    container_id = await asyncio.to_thread(
                        ensure_conversation_container,
                        conversation_id=conversation_id,
                        workspace_dir=workspace_dir,
                        openai_api_key=api_key,
                        git_token=git_token,
                        auth_json=auth_json,
                    )
                    with Session(engine) as session:
                        merge_conversation_metadata(
                            session,
                            conversation_id=conversation_id,
                            patch={
                                "data_agent.container_id": container_id,
                                "data_agent.workspace_dir": workspace_dir,
                                "data_agent.session_id": session_id,
                            },
                        )

                if not prewarm:
                    return

                logger.info(
                    "Isolated Workspace prewarm: begin conv=%s container_id=%s session_id=%s",
                    conversation_id,
                    container_id,
                    session_id or "",
                )
                with Session(engine) as session:
                    bot = get_bot(session, bot_id)
                    meta = _get_conversation_meta(session, conversation_id=conversation_id)
                    da = _data_agent_meta(meta)
                    if bool(da.get("ready", False)) or bool(da.get("prewarm_in_progress", False)):
                        return

                    merge_conversation_metadata(
                        session,
                        conversation_id=conversation_id,
                        patch={
                            "data_agent.ready": False,
                            "data_agent.init_error": "",
                            "data_agent.prewarm_in_progress": True,
                            "data_agent.prewarm_started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        },
                    )

                    ctx = _build_data_agent_conversation_context(
                        session,
                        bot=bot,
                        conversation_id=conversation_id,
                        meta=meta,
                    )
                    api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
                    auth_json_raw = getattr(bot, "data_agent_auth_json", "") or "{}"
                    git_token_current = (
                        _get_git_token_plaintext(session, provider="github")
                        if _git_auth_mode(auth_json_raw) == "token"
                        else ""
                    )
                    auth_json = _merge_git_token_auth(auth_json_raw, git_token_current)
                    sys_prompt = (
                        (getattr(bot, "data_agent_system_prompt", "") or "").strip()
                        or DEFAULT_DATA_AGENT_SYSTEM_PROMPT
                    )

                init_task = (getattr(bot, "data_agent_prewarm_prompt", "") or "").strip()
                if not init_task:
                    init_task = (
                        "INIT / PREWARM:\n"
                        "- Open and read: api_spec.json, auth.json, conversation_context.json.\n"
                        "- Do NOT call external APIs.\n"
                        "- Output ok=true and result_text='READY'."
                    )
                try:
                    res = await asyncio.to_thread(
                        run_data_agent,
                        conversation_id=conversation_id,
                        container_id=container_id,
                        session_id=session_id,
                        workspace_dir=workspace_dir,
                        api_spec_text=api_spec_text,
                        auth_json=auth_json,
                        system_prompt=sys_prompt,
                        conversation_context=ctx,
                        what_to_do=init_task,
                        timeout_s=180.0,
                    )
                    with Session(engine) as session:
                        merge_conversation_metadata(
                            session,
                            conversation_id=conversation_id,
                            patch={
                                "data_agent.prewarm_in_progress": False,
                                "data_agent.prewarm_finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "data_agent.ready": bool(res.ok),
                                "data_agent.init_error": str(res.error or ""),
                                "data_agent.session_id": str(res.session_id or ""),
                                "data_agent.container_id": str(res.container_id or container_id),
                                "data_agent.workspace_dir": workspace_dir,
                            },
                        )
                    logger.info(
                        "Isolated Workspace prewarm: done conv=%s ok=%s ready=%s session_id=%s error=%s",
                        conversation_id,
                        bool(res.ok),
                        bool(res.ok),
                        str(res.session_id or ""),
                        str(res.error or ""),
                    )
                except Exception as exc:
                    with Session(engine) as session:
                        merge_conversation_metadata(
                            session,
                            conversation_id=conversation_id,
                            patch={
                                "data_agent.prewarm_in_progress": False,
                                "data_agent.prewarm_finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                                "data_agent.ready": False,
                                "data_agent.init_error": str(exc),
                            },
                        )
                    logger.info("Isolated Workspace prewarm: failed conv=%s error=%s", conversation_id, str(exc))

            except Exception:
                logger.exception("Isolated Workspace kickoff failed conv=%s bot=%s", conversation_id, bot_id)
                return

    def _set_metadata_tool_def() -> dict:
        return set_metadata_tool_def()

    def _set_variable_tool_def() -> dict:
        return set_variable_tool_def()

    def _web_search_tool_def() -> dict:
        return web_search_tool_def()

    def _http_request_tool_def() -> dict:
        return http_request_tool_def()

    def _host_action_tool_def() -> dict:
        return {
            "type": "function",
            "name": "request_host_action",
            "description": (
                "Queue a host action for user approval. Use this when a task requires running local shell commands or AppleScript. "
                "The action will appear in the Action Queue and only runs after user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["run_shell", "run_applescript"],
                        "description": "Type of host action.",
                    },
                    "command": {"type": "string", "description": "Shell command to run (for run_shell)."},
                    "script": {"type": "string", "description": "AppleScript to run (for run_applescript)."},
                    "follow_up": {
                        "type": "boolean",
                        "description": "If true, force a follow-up assistant reply after the action runs.",
                    },
                    "next_reply": {
                        "type": "string",
                        "description": "Optional reply to the user after queuing the action.",
                    },
                },
                "required": ["action"],
            },
            "strict": False,
        }

    def _disabled_tool_names(bot: Bot) -> set[str]:
        raw = (getattr(bot, "disabled_tools_json", "") or "[]").strip() or "[]"
        try:
            obj = json.loads(raw)
        except Exception:
            obj = []
        out: set[str] = set()
        if isinstance(obj, list):
            for x in obj:
                s = str(x or "").strip()
                if s:
                    out.add(s)
        # Never allow disabling set_metadata; conversations depend on it.
        out.discard("set_metadata")
        out.discard("set_variable")
        return out

    def _system_tools_defs(*, bot: Bot) -> list[dict[str, Any]]:
        # Tools that are always available for every bot (plus optional per-bot tools).
        #
        # Note: `set_variable` is kept as a runtime alias for backwards-compat, but we only expose
        # `set_metadata` to the model to avoid duplicate tools that do the same thing.
        tools = [
            _set_metadata_tool_def(),
            _http_request_tool_def(),
        ]
        if _llm_provider_for_bot(bot) == "openai":
            tools.insert(1, _web_search_tool_def())
        if bool(getattr(bot, "enable_data_agent", False)):
            tools.append(give_command_to_data_agent_tool_def())
            tools.append(_summarize_screenshot_tool_def())
        if bool(getattr(bot, "enable_host_actions", False)):
            tools.append(_host_action_tool_def())
            if bool(getattr(bot, "enable_host_shell", False)):
                tools.append(_capture_screenshot_tool_def())
        return tools

    def _system_tools_public_list(*, bot: Bot, disabled: set[str]) -> list[dict[str, Any]]:
        # UI-friendly list of built-in tools (do not include full JSON Schema).
        out: list[dict[str, Any]] = []
        for d in _system_tools_defs(bot=bot):
            name = str(d.get("name") or d.get("type") or "")
            if not name:
                continue
            desc = str(d.get("description") or "")
            if not desc and str(d.get("type") or "") == "web_search":
                desc = "Search the web for recent information with citations."
            can_disable = name not in ("set_metadata", "set_variable")
            out.append(
                {
                    "name": name,
                    "description": desc,
                    "enabled": name not in disabled,
                    "can_disable": can_disable,
                }
            )
        return [x for x in out if x.get("name")]

    def _integration_tool_def(t: IntegrationTool) -> dict[str, Any]:
        required_args = _parse_required_args_json(getattr(t, "args_required_json", "[]"))
        explicit_schema = _parse_parameters_schema_json(getattr(t, "parameters_schema_json", ""))
        if explicit_schema:
            args_schema = explicit_schema
        else:
            args_schema = {
                "type": "object",
                "properties": {k: {"type": "string"} for k in required_args},
                "required": required_args,
                "additionalProperties": True,
            }

        def _append_required_args_to_schema(schema: dict[str, Any], required: list[str]) -> dict[str, Any]:
            if not required:
                return schema
            props = schema.get("properties")
            if not isinstance(props, dict) and schema.get("type") in (None, "object"):
                props = {}
            if isinstance(props, dict):
                merged = dict(schema)
                merged["type"] = "object"
                merged_props = dict(props)
                for k in required:
                    merged_props.setdefault(k, {"type": "string"})
                merged["properties"] = merged_props
                req = merged.get("required")
                if not isinstance(req, list):
                    req = []
                for k in required:
                    if k not in req:
                        req.append(k)
                merged["required"] = req
                if "additionalProperties" not in merged:
                    merged["additionalProperties"] = True
                return merged
            return {
                "allOf": [
                    schema,
                    {
                        "type": "object",
                        "properties": {k: {"type": "string"} for k in required},
                        "required": required,
                        "additionalProperties": True,
                    },
                ]
            }

        # Always enforce required args (even if a custom args schema is provided).
        args_schema = _append_required_args_to_schema(args_schema, required_args)

        use_codex_response = bool(getattr(t, "use_codex_response", False))
        if use_codex_response:
            # Minimal schema extension: append intent fields so the executor model can
            # reliably understand what data is being fetched and why.
            intent_schema = {
                "type": "object",
                "properties": {
                    "fields_required": {
                        "type": "string",
                        "description": "Fields required from the HTTP response to craft the final user-facing response.",
                    },
                    "why_api_was_called": {
                        "type": "string",
                        "description": "User intent or business reason for calling this API.",
                    },
                    # Backwards-compat: accept older keys if already present in existing tool calls.
                    "what_to_search_for": {"type": "string"},
                    "why_to_search_for": {"type": "string"},
                },
                "required": ["fields_required", "why_api_was_called"],
                "additionalProperties": True,
            }

            args_schema = _append_required_args_to_schema(args_schema, list(intent_schema["required"]))
            props = args_schema.get("properties")
            if isinstance(props, dict):
                merged = dict(args_schema)
                merged_props = dict(props)
                for k, v in intent_schema["properties"].items():
                    merged_props.setdefault(k, v)
                merged["properties"] = merged_props
                args_schema = merged
            else:
                args_schema = {"allOf": [args_schema, intent_schema]}

        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "wait_reply": {
                    "type": "string",
                    "description": "Short filler message to say while the tool runs (not used for set_metadata).",
                },
                "args": {
                    "description": "Arguments used to call the integration (required).",
                    **args_schema,
                },
            },
            "additionalProperties": True,
        }
        if use_codex_response:
            schema["required"] = ["args", "wait_reply"]
        else:
            schema["properties"]["next_reply"] = {
                "type": "string",
                "description": (
                    "What the assistant should say next (no second LLM call). "
                    "Variables like {{.firstName}} are allowed."
                ),
            }
            schema["required"] = ["args", "next_reply", "wait_reply"]

        # Pagination: allow the model to optionally request fewer items than the API page size.
        try:
            pag = safe_json_loads(getattr(t, "pagination_json", "") or "") or None
        except Exception:
            pag = None
        if isinstance(pag, dict) and str(pag.get("items_path") or "").strip():
            max_items_cap = pag.get("max_items_cap")
            try:
                max_items_cap_i = int(max_items_cap) if max_items_cap is not None else 5000
            except Exception:
                max_items_cap_i = 5000
            if max_items_cap_i < 1:
                max_items_cap_i = 5000
            if max_items_cap_i > 50000:
                max_items_cap_i = 50000
            props = schema.get("properties")
            if isinstance(props, dict):
                args_props = props.get("args")
                if isinstance(args_props, dict):
                    # args_props is a schema object; add `max_items` as an optional arg.
                    args_props.setdefault("properties", {})
                    if isinstance(args_props.get("properties"), dict):
                        args_props["properties"].setdefault(
                            "max_items",
                            {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": max_items_cap_i,
                                "description": (
                                    f"Optional: stop pagination after collecting this many items (max {max_items_cap_i}). "
                                    "Backend will fetch multiple pages if needed."
                                ),
                            },
                        )
        return {
            "type": "function",
            "name": t.name,
            "description": (t.description or "").strip()
            + " This tool calls an external HTTP API and maps selected response fields into conversation metadata. "
            + (
                "Return your spoken/text response in next_reply (you can use metadata variables like {{.firstName}})."
                if not use_codex_response
                else "If Codex mode is enabled, the backend runs a Codex agent to generate a result string, and the main chat model will rephrase it."
            ),
            "parameters": schema,
            "strict": False,
        }

    def _build_tools_for_bot(session: Session, bot_id: UUID) -> list[dict[str, Any]]:
        bot = get_bot(session, bot_id)
        disabled = _disabled_tool_names(bot)
        tools: list[dict[str, Any]] = []
        for d in _system_tools_defs(bot=bot):
            tool_id = str(d.get("name") or d.get("type") or "")
            if tool_id and tool_id in disabled:
                continue
            tools.append(d)
        for t in list_integration_tools(session, bot_id=bot_id):
            if not bool(getattr(t, "enabled", True)):
                continue
            if str(getattr(t, "name", "") or "").strip() in disabled:
                continue
            try:
                tools.append(_integration_tool_def(t))
            except Exception:
                continue
        return tools

    @lru_cache(maxsize=1)
    def _get_openai_pricing() -> dict[str, ModelPrice]:
        raw = os.environ.get("OPENAI_PRICING_JSON") or ""
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except Exception:
            return {}
        out: dict[str, ModelPrice] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                try:
                    input_per_1m = v.get("input_per_1m")
                    output_per_1m = v.get("output_per_1m")
                    if input_per_1m is None or output_per_1m is None:
                        continue
                    out[k] = ModelPrice(input_per_1m=float(input_per_1m), output_per_1m=float(output_per_1m))
                except Exception:
                    continue
        return out

    def _refresh_openrouter_models_cache(session: Session) -> None:
        try:
            now = time.time()
            if (now - float(openrouter_models_cache.get("ts") or 0.0)) <= 3600.0:
                return
            openrouter_models_cache["ts"] = now
            openrouter_models_cache["models"] = []
            openrouter_models_cache["pricing"] = {}
            or_key = _get_openrouter_api_key(session)
            base_url = (os.environ.get("OPENROUTER_BASE_URL") or "").strip() or "https://openrouter.ai/api/v1"
            headers = {}
            if or_key:
                headers["Authorization"] = f"Bearer {or_key}"
            ref = (os.environ.get("OPENROUTER_REFERER") or "").strip()
            title = (os.environ.get("OPENROUTER_TITLE") or "").strip()
            if ref:
                headers["HTTP-Referer"] = ref
            if title:
                headers["X-Title"] = title
            resp = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=15.0)
            if resp.status_code >= 400:
                return
            data = resp.json()
            items = data.get("data") or []
            ids: list[str] = []
            pricing_map: dict[str, ModelPrice] = {}
            for m in items:
                if not isinstance(m, dict):
                    continue
                mid = m.get("id")
                if not isinstance(mid, str) or not mid.strip():
                    continue
                mid = mid.strip()
                ids.append(mid)
                price = m.get("pricing")
                if isinstance(price, dict):
                    try:
                        prompt = float(price.get("prompt"))
                        completion = float(price.get("completion"))
                    except Exception:
                        prompt = None
                        completion = None
                    if prompt is not None and completion is not None:
                        pricing_map[mid] = ModelPrice(
                            input_per_1m=prompt * 1_000_000.0,
                            output_per_1m=completion * 1_000_000.0,
                        )
            openrouter_models_cache["models"] = sorted(set(ids))
            openrouter_models_cache["pricing"] = pricing_map
        except Exception:
            return

    def _get_openrouter_pricing(session: Session) -> dict[str, ModelPrice]:
        _refresh_openrouter_models_cache(session)
        return dict(openrouter_models_cache.get("pricing") or {})

    def _get_model_price(session: Session, *, provider: str, model: str) -> Optional[ModelPrice]:
        if provider == "local":
            return None
        if provider == "openrouter":
            return _get_openrouter_pricing(session).get(model)
        return _get_openai_pricing().get(model)

    def _estimate_llm_cost_for_turn(
        *,
        session: Session,
        bot: Bot,
        provider: str,
        history: list[Message],
        assistant_text: str,
    ) -> tuple[int, int, float]:
        prompt_tokens = estimate_messages_tokens(history, bot.openai_model)
        output_tokens = estimate_text_tokens(assistant_text, bot.openai_model)
        price = _get_model_price(session, provider=provider, model=bot.openai_model)
        cost = estimate_cost_usd(model_price=price, input_tokens=prompt_tokens, output_tokens=output_tokens)
        return prompt_tokens, output_tokens, cost

    def _make_start_message_instruction(bot: Bot) -> str:
        # Keep this short and safe; system_prompt should drive tone/language.
        return (
            "Generate a short opening message to start a voice conversation. "
            "Keep it concise and end with a question."
        )

    async def _init_conversation_and_greet(
        *,
        bot_id: UUID,
        speak: bool,
        test_flag: bool,
        ws: WebSocket,
        req_id: str,
        debug: bool,
    ) -> UUID:
        init_start = time.time()
        # Create conversation + store first assistant message.
        with Session(engine) as session:
            bot = get_bot(session, bot_id)
            conv = create_conversation(session, bot_id=bot.id, test_flag=test_flag)
            conv_id = conv.id

            greeting_text = (bot.start_message_text or "").strip()
            llm_ttfb_ms: Optional[int] = None
            llm_total_ms: Optional[int] = None
            input_tokens_est: Optional[int] = None
            output_tokens_est: Optional[int] = None
            cost_usd_est: Optional[float] = None
            sent_greeting_delta = False
            llm_api_key: Optional[str] = None
            openai_api_key: Optional[str] = None
            provider = _llm_provider_for_bot(bot)

            needs_llm = not (bot.start_message_mode == "static" and greeting_text)
            if needs_llm:
                provider, llm_api_key, llm = _require_llm_client(session, bot=bot)
            else:
                llm = None
            if speak:
                openai_api_key = _get_openai_api_key_for_bot(session, bot=bot)
                if not openai_api_key:
                    raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot (needed for TTS).")

            if bot.start_message_mode == "static" and greeting_text:
                # Static greeting (no LLM).
                pass
            else:
                if llm is None or (provider != "local" and not llm_api_key):
                    raise HTTPException(
                        status_code=400,
                        detail=f"No {_provider_display_name(provider)} key configured for this bot.",
                    )
                ts = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
                sys_prompt = f"Current Date Time(UTC): {ts}\n\n{render_template(bot.system_prompt, ctx={'meta': {}})}"
                msgs = [
                    Message(role="system", content=sys_prompt),
                    Message(role="user", content=_make_start_message_instruction(bot)),
                ]
                if debug:
                    await _emit_llm_debug_payload(
                        ws=ws,
                        req_id=req_id,
                        conversation_id=conv_id,
                        phase="greeting_llm",
                        payload=llm.build_request_payload(messages=msgs, stream=True),
                    )
                t0 = time.time()
                first = None
                parts: list[str] = []
                async for d in _aiter_from_blocking_iterator(lambda: llm.stream_text(messages=msgs)):
                    d = str(d or "")
                    if first is None:
                        first = time.time()
                    if d:
                        parts.append(d)
                        await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
                        sent_greeting_delta = True
                t1 = time.time()
                greeting_text = "".join(parts).strip()
                if first is not None:
                    llm_ttfb_ms = int(round((first - t0) * 1000.0))
                llm_total_ms = int(round((t1 - t0) * 1000.0))

                # Estimate cost for this greeting turn.
                input_tokens_est, output_tokens_est, cost_usd_est = _estimate_llm_cost_for_turn(
                    session=session,
                    bot=bot,
                    provider=provider,
                    history=msgs,
                    assistant_text=greeting_text,
                )

            if not greeting_text:
                greeting_text = "Hi! How can I help you today?"

            # If this was static (or LLM produced no streamed deltas), still send text to UI.
            if not sent_greeting_delta:
                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": greeting_text})

            # Store assistant greeting as first message.
            add_message_with_metrics(
                session,
                conversation_id=conv_id,
                role="assistant",
                content=greeting_text,
                input_tokens_est=input_tokens_est,
                output_tokens_est=output_tokens_est,
                cost_usd_est=cost_usd_est,
                llm_ttfb_ms=llm_ttfb_ms,
                llm_total_ms=llm_total_ms,
            )
            if input_tokens_est is not None or output_tokens_est is not None or cost_usd_est is not None:
                update_conversation_metrics(
                    session,
                    conversation_id=conv_id,
                    add_input_tokens_est=int(input_tokens_est or 0),
                    add_output_tokens_est=int(output_tokens_est or 0),
                    add_cost_usd_est=float(cost_usd_est or 0.0),
                    last_asr_ms=None,
                    last_llm_ttfb_ms=llm_ttfb_ms,
                    last_llm_total_ms=llm_total_ms,
                    last_tts_first_audio_ms=None,
                    last_total_ms=None,
                )

            if speak:
                tts_synth = await asyncio.to_thread(_get_tts_synth_fn, bot, openai_api_key)
                # Synthesize whole greeting as one chunk for now.
                wav, sr = await asyncio.to_thread(tts_synth, greeting_text)
                await _ws_send_json(
                    ws,
                    {
                        "type": "audio_wav",
                        "req_id": req_id,
                        "wav_base64": base64.b64encode(wav).decode(),
                        "sr": sr,
                    },
                )

        timings: dict[str, int] = {"total": int(round((time.time() - init_start) * 1000.0))}
        if llm_ttfb_ms is not None:
            timings["llm_ttfb"] = llm_ttfb_ms
        if llm_total_ms is not None:
            timings["llm_total"] = llm_total_ms
        await _ws_send_json(ws, {"type": "metrics", "req_id": req_id, "timings_ms": timings})
        return conv_id

    def _apply_response_mapper(
        *,
        mapper_json: str,
        response_json: Any,
        meta: dict,
        tool_args: dict,
    ) -> dict:
        mapper = safe_json_loads(mapper_json or "{}") or {}
        if not isinstance(mapper, dict):
            return {}
        out: dict = {}
        ctx = {"response": response_json, "meta": meta, "args": tool_args, "params": tool_args}
        for k, tmpl in mapper.items():
            if not isinstance(k, str):
                continue
            if isinstance(tmpl, (dict, list)):
                out[k] = tmpl
                continue
            if tmpl is None:
                out[k] = None
                continue
            out[k] = eval_template_value(str(tmpl), ctx=ctx)
        return out

    def _render_with_meta(text: str, meta: dict) -> str:
        return render_template(text, ctx={"meta": meta})

    def _render_static_reply(
        *,
        template_text: str,
        meta: dict,
        response_json: Any,
        tool_args: dict,
    ) -> str:
        return render_jinja_template(
            template_text,
            ctx={"meta": meta, "response": response_json, "args": tool_args, "params": tool_args},
        )

    def _coerce_json_object(value: Any) -> dict:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return {}
            try:
                obj = json.loads(raw)
            except Exception:
                return {}
            return obj if isinstance(obj, dict) else {}
        return {}

    def _render_templates_in_obj(value: Any, *, ctx: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {k: _render_templates_in_obj(v, ctx=ctx) for k, v in value.items() if k is not None}
        if isinstance(value, list):
            return [_render_templates_in_obj(v, ctx=ctx) for v in value]
        if isinstance(value, str):
            return render_template(value, ctx=ctx)
        return value

    def _parse_query_params(value: Any, *, ctx: dict[str, Any]) -> dict[str, Any]:
        if isinstance(value, dict):
            return _render_templates_in_obj(value, ctx=ctx)
        if isinstance(value, str):
            rendered = render_template(value, ctx=ctx).strip()
            if not rendered:
                return {}
            if rendered.startswith("{") and rendered.endswith("}"):
                try:
                    obj = json.loads(rendered)
                except Exception:
                    obj = None
                if isinstance(obj, dict):
                    return _render_templates_in_obj(obj, ctx=ctx)
            pairs = parse_qsl(rendered, keep_blank_values=True)
            out: dict[str, Any] = {}
            for k, v in pairs:
                if not k:
                    continue
                out[k] = v
            return out
        return {}

    def _parse_fields_required(value: Any) -> list[str]:
        out: list[str] = []
        if isinstance(value, list):
            for item in value:
                s = str(item or "").strip()
                if s:
                    out.append(s)
            return out
        if isinstance(value, str):
            for part in re.split(r"[\\n,]+", value):
                s = part.strip()
                if s:
                    out.append(s)
        return out

    def _build_response_mapper_from_fields(fields_required: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in fields_required:
            s = str(item or "").strip()
            if not s:
                continue
            key = s
            expr = s
            for sep in ("=>", "=", ":"):
                if sep in s:
                    left, right = s.split(sep, 1)
                    if left.strip():
                        key = left.strip()
                        expr = right.strip() or expr
                    break
            if "{{" in expr and "}}" in expr:
                out[key] = expr
                continue
            if not (
                expr.startswith("response.")
                or expr.startswith("meta.")
                or expr.startswith("args.")
                or expr.startswith("params.")
            ):
                expr = f"response.{expr}"
            out[key] = f"{{{{{expr}}}}}"
        return out

    def _execute_http_request_tool(*, tool_args: dict, meta: dict) -> dict:
        url = str(tool_args.get("url") or "").strip()
        if not url:
            return {"ok": False, "error": {"message": "Missing required tool arg: url"}}

        ctx = {"meta": meta, "args": tool_args, "params": tool_args, "env": dict(os.environ)}
        url = render_template(url, ctx=ctx)
        method = str(tool_args.get("method") or "GET").strip().upper() or "GET"

        headers_raw = tool_args.get("headers")
        headers_obj = _coerce_json_object(headers_raw)
        if not headers_obj and isinstance(headers_raw, str):
            rendered_headers = render_template(headers_raw, ctx=ctx)
            headers_obj = _coerce_json_object(rendered_headers)
        headers_obj = _render_templates_in_obj(headers_obj, ctx=ctx)
        headers_obj = _normalize_headers_for_json(headers_obj)

        query_params = _parse_query_params(tool_args.get("query"), ctx=ctx)

        body_raw = tool_args.get("body")
        body_obj: Any = None
        if isinstance(body_raw, (dict, list)):
            body_obj = _render_templates_in_obj(body_raw, ctx=ctx)
        elif isinstance(body_raw, str):
            rendered_body = render_template(body_raw, ctx=ctx).strip()
            if rendered_body:
                if rendered_body.startswith("{") or rendered_body.startswith("["):
                    try:
                        body_obj = json.loads(rendered_body)
                    except Exception:
                        body_obj = rendered_body
                else:
                    body_obj = rendered_body
        else:
            body_obj = body_raw

        fields_required = _parse_fields_required(tool_args.get("fields_required"))
        mapper_obj = _coerce_json_object(tool_args.get("response_mapper_json"))
        if not mapper_obj and not fields_required:
            return {"ok": False, "error": {"message": "Missing required tool arg: fields_required"}}
        if not mapper_obj:
            mapper_obj = _build_response_mapper_from_fields(fields_required)
        mapper_json = json.dumps(mapper_obj, ensure_ascii=False)

        timeout = httpx.Timeout(60.0, connect=20.0)
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                req_kwargs = {"headers": headers_obj or None, "params": query_params or None}
                if method in ("GET", "HEAD"):
                    resp = client.request(method, url, **req_kwargs)
                else:
                    if isinstance(body_obj, (dict, list)):
                        resp = client.request(method, url, json=body_obj, **req_kwargs)
                    elif body_obj is None:
                        resp = client.request(method, url, **req_kwargs)
                    else:
                        resp = client.request(method, url, content=str(body_obj), **req_kwargs)
            if resp.status_code >= 400:
                err = _http_error_response(
                    url=str(resp.request.url),
                    status_code=resp.status_code,
                    body=(resp.text or None),
                    message=resp.reason_phrase,
                )
                return {"ok": False, "error": err.get("__http_error__") or {"message": "HTTP error"}}
            try:
                response_json = resp.json()
            except Exception:
                response_json = {"raw": resp.text or ""}
            mapped = _apply_response_mapper(
                mapper_json=mapper_json,
                response_json=response_json,
                meta=meta,
                tool_args=tool_args,
            )
            return {
                "ok": True,
                "status_code": int(resp.status_code),
                "url": str(resp.request.url),
                "data": mapped,
            }
        except httpx.RequestError as exc:
            err = _http_error_response(url=url, status_code=None, body=None, message=str(exc))
            return {"ok": False, "error": err.get("__http_error__")}

    def _execute_integration_http(
        *,
        tool: IntegrationTool,
        meta: dict,
        tool_args: dict,
    ) -> dict:
        pagination_raw = (getattr(tool, "pagination_json", "") or "").strip()
        pagination_cfg: dict[str, Any] | None = None
        pagination_cfg_error: str | None = None
        if pagination_raw:
            try:
                obj = json.loads(pagination_raw)
                if isinstance(obj, dict):
                    pagination_cfg = obj
                else:
                    pagination_cfg_error = "pagination_json must be a JSON object."
            except Exception as exc:
                pagination_cfg_error = f"invalid pagination_json: {exc}"

        # Apply JSON Schema defaults to tool args (helps URL/body templates that reference args.page/args.limit).
        args0 = dict(tool_args or {})
        try:
            schema_obj = json.loads(getattr(tool, "parameters_schema_json", "") or "null")
        except Exception:
            schema_obj = None
        if isinstance(schema_obj, dict):
            args0 = _apply_schema_defaults(schema_obj, args0)
        tool_args = args0

        required_args = _parse_required_args_json(getattr(tool, "args_required_json", "[]"))
        missing = _missing_required_args(required_args, tool_args or {})
        if bool(getattr(tool, "use_codex_response", False)):
            args0 = tool_args or {}
            # Prefer new keys, but accept old ones if present.
            if not str(args0.get("fields_required") or "").strip() and not str(args0.get("what_to_search_for") or "").strip():
                missing.append("fields_required")
            if not str(args0.get("why_api_was_called") or "").strip() and not str(args0.get("why_to_search_for") or "").strip():
                missing.append("why_api_was_called")
        if missing:
            return {
                "__tool_args_error__": {
                    "missing": sorted(set(missing)),
                    "message": f"Missing required tool args: {', '.join(sorted(set(missing)))}",
                }
            }

        def _single_request(*, loop_args: dict[str, Any]) -> tuple[Any, str]:
            # Render URL/body templates using current metadata + tool args.
            # Include env so integrations can reference server-provided secrets without storing them in metadata.
            ctx = {"meta": meta, "args": loop_args, "params": loop_args, "env": dict(os.environ)}
            url = render_template(tool.url, ctx=ctx)
            method = (tool.method or "GET").upper()

            headers_obj: dict[str, str] = {}
            headers_template = tool.headers_template_json or ""
            if headers_template.strip():
                rendered_headers = render_template(headers_template, ctx=ctx)
                try:
                    h = json.loads(rendered_headers)
                    if isinstance(h, dict):
                        for k, v in h.items():
                            if isinstance(k, str) and isinstance(v, str) and k.strip():
                                headers_obj[k] = v
                except Exception:
                    headers_obj = {}
            headers_obj = _normalize_headers_for_json(headers_obj)

            body_template = tool.request_body_template or ""
            body_obj = None
            if body_template.strip():
                rendered_body = render_template(body_template, ctx=ctx)
                try:
                    body_obj = json.loads(rendered_body)
                except Exception:
                    # If body isn't valid JSON, send as raw string.
                    body_obj = rendered_body

            loop_request_args = dict(loop_args)
            # Internal intent keys (LLM-only); never forward to the upstream HTTP API.
            for k in ("fields_required", "why_api_was_called", "what_to_search_for", "why_to_search_for", "max_items"):
                loop_request_args.pop(k, None)

            timeout = httpx.Timeout(60.0, connect=20.0)
            try:
                with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                    if method == "GET":
                        resp = client.request(method, url, headers=headers_obj or None)
                    else:
                        # Prefer JSON for objects/lists; otherwise raw data.
                        if isinstance(body_obj, (dict, list)):
                            # If pagination is enabled and the body is a dict, the caller may have inserted
                            # page/limit keys into loop_args and referenced them in the template; we keep
                            # the rendered body as the source of truth.
                            resp = client.request(method, url, json=body_obj, headers=headers_obj or None)
                        elif body_obj is None:
                            # Most APIs expect an object for JSON bodies. Send {} instead of null when args are empty.
                            resp = client.request(
                                method, url, json=(loop_request_args or {}), headers=headers_obj or None
                            )
                        else:
                            resp = client.request(method, url, content=str(body_obj), headers=headers_obj or None)
                if resp.status_code >= 400:
                    return (
                        _http_error_response(
                            url=str(resp.request.url),
                            status_code=resp.status_code,
                            body=(resp.text or None),
                            message=resp.reason_phrase,
                        ),
                        url,
                    )
                try:
                    return resp.json(), url
                except Exception:
                    return {"raw": resp.text}, url
            except httpx.RequestError as exc:
                return _http_error_response(url=url, status_code=None, body=None, message=str(exc)), url

        # If pagination is not configured, do a single request.
        if not isinstance(pagination_cfg, dict) or not pagination_cfg:
            resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
            if pagination_cfg_error and isinstance(resp_json, dict) and "__http_error__" not in resp_json:
                resp_json["__igx_pagination__"] = {"error": pagination_cfg_error}
            return resp_json

        mode = str(pagination_cfg.get("mode") or "page_limit").strip()
        if mode not in ("page_limit", "offset_limit"):
            resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
            if isinstance(resp_json, dict) and "__http_error__" not in resp_json:
                resp_json["__igx_pagination__"] = {"error": f"unsupported pagination mode: {mode}"}
            return resp_json

        items_path = str(pagination_cfg.get("items_path") or "").strip()
        if not items_path:
            resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
            if isinstance(resp_json, dict) and "__http_error__" not in resp_json:
                resp_json["__igx_pagination__"] = {"error": "pagination_json missing items_path."}
            return resp_json

        page_arg = str(pagination_cfg.get("page_arg") or "page").strip() or "page"
        limit_arg = str(pagination_cfg.get("limit_arg") or "limit").strip() or "limit"
        offset_arg = str(pagination_cfg.get("offset_arg") or "offset").strip() or "offset"
        max_pages = int(pagination_cfg.get("max_pages") or 5)
        if max_pages < 1:
            max_pages = 1
        if max_pages > 50:
            max_pages = 50

        # max_items is optional; if missing, we still cap total work via max_pages.
        max_items_cap = int(pagination_cfg.get("max_items_cap") or 5000)
        if max_items_cap < 1:
            max_items_cap = 5000
        if max_items_cap > 50000:
            max_items_cap = 50000

        requested_max_items = tool_args.get("max_items")
        if requested_max_items is None:
            requested_max_items = pagination_cfg.get("max_items_default")
        try:
            max_items = int(requested_max_items) if requested_max_items is not None else None
        except Exception:
            max_items = None
        if max_items is not None:
            if max_items < 1:
                max_items = None
            else:
                max_items = min(max_items, max_items_cap)

        def _read_int(v: Any, default: int) -> int:
            try:
                x = int(v)
                return x
            except Exception:
                return default

        limit_val = _read_int(tool_args.get(limit_arg), int(pagination_cfg.get("limit_default") or 100))
        if limit_val < 1:
            limit_val = 100

        start_page = _read_int(tool_args.get(page_arg), 1)
        if start_page < 1:
            start_page = 1

        start_offset = _read_int(tool_args.get(offset_arg), 0)
        if start_offset < 0:
            start_offset = 0

        base_resp: Any = None
        aggregated: list[Any] = []
        fetched = 0

        for i in range(max_pages):
            loop_args = dict(tool_args or {})
            loop_args[limit_arg] = limit_val
            if mode == "page_limit":
                loop_args[page_arg] = start_page + i
            else:
                loop_args[offset_arg] = start_offset + (i * limit_val)

            resp_json, _url = _single_request(loop_args=loop_args)
            # Pass-through errors / non-JSON bodies immediately.
            if isinstance(resp_json, dict) and resp_json.get("__http_error__"):
                return resp_json

            if base_resp is None:
                # Shallow copy so we can mutate the items list in-place without affecting downstream.
                base_resp = resp_json if not isinstance(resp_json, dict) else dict(resp_json)

            page_items = _get_json_path(resp_json, items_path)
            if not isinstance(page_items, list):
                # If we can't find a list at items_path, return first page + diagnostics.
                if isinstance(base_resp, dict):
                    base_resp["__igx_pagination__"] = {
                        "mode": mode,
                        "items_path": items_path,
                        "limit": limit_val,
                        "pages_fetched": fetched,
                        "items_returned": len(aggregated),
                        "max_items": max_items,
                        "max_pages": max_pages,
                        "error": f"items_path not a list: {items_path}",
                    }
                return base_resp

            fetched += 1
            aggregated.extend(page_items)

            if max_items is not None and len(aggregated) >= max_items:
                aggregated = aggregated[:max_items]
                break

            # Stop if the API returned fewer than the page size (common "last page" signal).
            if len(page_items) < limit_val:
                break

        if base_resp is None:
            resp_json, _ = _single_request(loop_args=dict(tool_args or {}))
            return resp_json

        # Replace the items list with the aggregated list (if possible).
        if not _set_json_path(base_resp, items_path, aggregated):
            # If we can't set, just return the first page response.
            if isinstance(base_resp, dict):
                base_resp["__igx_pagination__"] = {
                    "mode": mode,
                    "items_path": items_path,
                    "limit": limit_val,
                    "pages_fetched": fetched,
                    "items_returned": len(aggregated),
                    "max_items": max_items,
                    "max_pages": max_pages,
                    "error": f"failed to set items_path: {items_path}",
                }
            return base_resp

        # Minimal pagination diagnostics (kept out of the merged items list; caller can surface it in tool_result).
        if isinstance(base_resp, dict):
            base_resp["__igx_pagination__"] = {
                "mode": mode,
                "items_path": items_path,
                "limit": limit_val,
                "pages_fetched": fetched,
                "items_returned": len(aggregated),
                "max_items": max_items,
                "max_pages": max_pages,
            }

        return base_resp

    async def _ws_send_json(ws: WebSocket, obj: dict) -> None:
        try:
            await ws.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception:
            # Best-effort: if the client disconnects mid-generation, keep processing in the background.
            return

    _ASYNC_STREAM_DONE = object()

    async def _aiter_from_blocking_iterator(iterator_fn):
        """
        Runs a blocking iterator in a background thread and yields items asynchronously.

        This prevents the asyncio event loop from being blocked by network I/O (e.g. OpenAI SDK streaming).
        """
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _runner() -> None:
            try:
                for item in iterator_fn():
                    loop.call_soon_threadsafe(q.put_nowait, item)
            except BaseException as exc:
                loop.call_soon_threadsafe(q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, _ASYNC_STREAM_DONE)

        t = threading.Thread(target=_runner, daemon=True)
        t.start()

        while True:
            item = await q.get()
            if item is _ASYNC_STREAM_DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    async def _stream_llm_reply(
        *,
        ws: WebSocket,
        req_id: str,
        llm: OpenAILLM,
        messages: list[Message],
    ) -> tuple[str, Optional[int], int]:
        t0 = time.time()
        first: Optional[float] = None
        parts: list[str] = []
        async for d in _aiter_from_blocking_iterator(lambda: llm.stream_text(messages=messages)):
            d = str(d or "")
            if first is None:
                first = time.time()
            if d:
                parts.append(d)
                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
        t1 = time.time()
        text = "".join(parts).strip()
        ttfb_ms = int(round((first - t0) * 1000.0)) if first is not None else None
        total_ms = int(round((t1 - t0) * 1000.0))
        return text, ttfb_ms, total_ms

    def _estimate_wav_seconds(wav_bytes: bytes, sr: int) -> float:
        # Best-effort WAV duration extraction to avoid interrupting ongoing speech.
        # If parsing fails, fall back to a heuristic based on byte size.
        try:
            if len(wav_bytes) < 44 or sr <= 0:
                raise ValueError("bad wav")
            if wav_bytes[0:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
                raise ValueError("not wav")
            # Parse chunks.
            i = 12
            channels = 1
            bits_per_sample = 16
            data_size = None
            while i + 8 <= len(wav_bytes):
                cid = wav_bytes[i : i + 4]
                size = int.from_bytes(wav_bytes[i + 4 : i + 8], "little", signed=False)
                i += 8
                if cid == b"fmt " and i + 16 <= len(wav_bytes):
                    channels = int.from_bytes(wav_bytes[i + 2 : i + 4], "little", signed=False) or 1
                    bits_per_sample = int.from_bytes(wav_bytes[i + 14 : i + 16], "little", signed=False) or 16
                if cid == b"data":
                    data_size = size
                    break
                i += size + (size % 2)
            if data_size is None:
                data_size = max(0, len(wav_bytes) - 44)
            bytes_per_frame = max(1, int(channels) * max(1, int(bits_per_sample) // 8))
            frames = float(data_size) / float(bytes_per_frame)
            return max(0.0, frames / float(sr))
        except Exception:
            return max(0.5, min(12.0, float(len(wav_bytes)) / float(max(1, sr * 2))))

    def _iter_tts_chunks(delta_q: "queue.Queue[Optional[str]]") -> Generator[str, None, None]:
        chunker = SentenceChunker()
        while True:
            d = delta_q.get()
            if d is None:
                break
            if not d:
                continue
            for chunk in chunker.push(d):
                yield chunk
        tail = chunker.flush()
        if tail:
            yield tail

    def _record_llm_debug_payload(
        *,
        conversation_id: UUID,
        payload: dict[str, Any],
        phase: str,
    ) -> None:
        try:
            with Session(engine) as session:
                add_message_with_metrics(
                    session,
                    conversation_id=conversation_id,
                    role="tool",
                    content=json.dumps(
                        {"tool": "debug_llm_request", "arguments": {"phase": phase, "payload": payload}},
                        ensure_ascii=False,
                    ),
                )
        except Exception:
            # Debugging must never break the conversation.
            return

    async def _emit_llm_debug_payload(
        *,
        ws: WebSocket,
        req_id: str,
        conversation_id: UUID,
        payload: dict[str, Any],
        phase: str,
    ) -> None:
        # Best-effort: send to UI and persist to DB.
        try:
            await _ws_send_json(
                ws,
                {
                    "type": "tool_call",
                    "req_id": req_id,
                    "name": "debug_llm_request",
                    "arguments_json": json.dumps({"phase": phase, "payload": payload}, ensure_ascii=False),
                },
            )
        except Exception:
            pass
        _record_llm_debug_payload(conversation_id=conversation_id, payload=payload, phase=phase)

    @app.websocket("/ws/bots/{bot_id}/talk")
    async def talk_ws(bot_id: UUID, ws: WebSocket) -> None:  # pyright: ignore[reportGeneralTypeIssues]
        if not _basic_auth_ok(_ws_auth_header(ws)):
            await ws.accept()
            await _ws_send_json(ws, {"type": "error", "error": "Unauthorized"})
            await ws.close(code=4401)
            return
        await ws.accept()
        loop = asyncio.get_running_loop()

        def status(req_id: str, stage: str) -> None:
            asyncio.run_coroutine_threadsafe(
                _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": stage}), loop
            )

        active_req_id: Optional[str] = None
        audio_buf = bytearray()
        conv_id: Optional[UUID] = None
        speak = True
        test_flag = True
        debug_mode = False
        stop_ts: Optional[float] = None
        accepting_audio = False
        tts_synth: Optional[Callable[[str], tuple[bytes, int]]] = None

        try:
            while True:
                msg = await ws.receive()
                if "text" in msg and msg["text"] is not None:
                    try:
                        payload = json.loads(msg["text"])
                    except Exception:
                        await _ws_send_json(ws, {"type": "error", "error": "Invalid JSON"})
                        continue

                    msg_type = payload.get("type")
                    req_id = str(payload.get("req_id") or "")
                    if msg_type == "init":
                        if not req_id:
                            await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                            continue
                        if active_req_id is not None:
                            await _ws_send_json(
                                ws,
                                {
                                    "type": "error",
                                    "req_id": req_id,
                                    "error": "Another request is already in progress",
                                },
                            )
                            continue
                        active_req_id = req_id
                        speak = bool(payload.get("speak", True))
                        test_flag = bool(payload.get("test_flag", True))
                        debug_mode = bool(payload.get("debug", False))
                        accepting_audio = False
                        conversation_id_str = str(payload.get("conversation_id") or "").strip()
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "init"})
                        if conversation_id_str:
                            try:
                                with Session(engine) as session:
                                    bot = get_bot(session, bot_id)
                                    conv_id = UUID(conversation_id_str)
                                    conv = get_conversation(session, conv_id)
                                    if conv.bot_id != bot.id:
                                        raise HTTPException(status_code=400, detail="Conversation does not belong to bot")
                            except Exception as exc:
                                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                                active_req_id = None
                                conv_id = None
                                continue
                            await _ws_send_json(
                                ws,
                                {
                                    "type": "conversation",
                                    "req_id": req_id,
                                    "conversation_id": str(conv_id),
                                    "id": str(conv_id),
                                },
                            )
                            await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            accepting_audio = False
                            continue
                        try:
                            conv_id = await _init_conversation_and_greet(
                                bot_id=bot_id,
                                speak=speak,
                                test_flag=test_flag,
                                ws=ws,
                                req_id=req_id,
                                debug=debug_mode,
                            )
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue
                        await _ws_send_json(
                            ws,
                            {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)},
                        )
                        await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        accepting_audio = False
                        continue

                    if msg_type == "start":
                        if not req_id:
                            await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                            continue
                        if active_req_id is not None:
                            await _ws_send_json(
                                ws,
                                {
                                    "type": "error",
                                    "req_id": req_id,
                                    "error": "Another request is already in progress",
                                },
                            )
                            continue

                        active_req_id = req_id
                        audio_buf = bytearray()
                        debug_mode = bool(payload.get("debug", False))
                        speak = bool(payload.get("speak", True))
                        test_flag = bool(payload.get("test_flag", True))
                        accepting_audio = True

                        conversation_id_str = str(payload.get("conversation_id") or "").strip()

                        try:
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                if conversation_id_str:
                                    conv_id = UUID(conversation_id_str)
                                    conv = get_conversation(session, conv_id)
                                    if conv.bot_id != bot.id:
                                        raise HTTPException(
                                            status_code=400, detail="Conversation does not belong to bot"
                                        )
                                else:
                                    conv = create_conversation(session, bot_id=bot.id, test_flag=test_flag)
                                    conv_id = conv.id
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            active_req_id = None
                            conv_id = None
                            continue

                        await _ws_send_json(
                            ws,
                            {
                                "type": "conversation",
                                "req_id": req_id,
                                "conversation_id": str(conv_id),
                                "id": str(conv_id),
                            },
                        )
                        asyncio.create_task(
                            _kickoff_data_agent_container_if_enabled(bot_id=bot_id, conversation_id=conv_id)
                        )
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "recording"})

                    elif msg_type == "chat":
                        # Text-only chat turn (for when Speak is disabled).
                        if not req_id:
                            await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                            continue
                        if active_req_id is not None:
                            await _ws_send_json(
                                ws,
                                {
                                    "type": "error",
                                    "req_id": req_id,
                                    "error": "Another request is already in progress",
                                },
                            )
                            continue

                        active_req_id = req_id
                        speak = bool(payload.get("speak", True))
                        test_flag = bool(payload.get("test_flag", True))
                        debug_mode = bool(payload.get("debug", False))
                        user_text = str(payload.get("text") or "").strip()
                        conversation_id_str = str(payload.get("conversation_id") or "").strip()
                        accepting_audio = False
                        if not user_text:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": "Empty text"})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            continue

                        try:
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                if conversation_id_str:
                                    conv_id = UUID(conversation_id_str)
                                    conv = get_conversation(session, conv_id)
                                    if conv.bot_id != bot.id:
                                        raise HTTPException(status_code=400, detail="Conversation does not belong to bot")
                                else:
                                    conv = create_conversation(session, bot_id=bot.id, test_flag=test_flag)
                                    conv_id = conv.id
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue

                        await _ws_send_json(
                            ws,
                            {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)},
                        )
                        asyncio.create_task(_kickoff_data_agent_container_if_enabled(bot_id=bot_id, conversation_id=conv_id))

                        if user_text.lstrip().startswith("!"):
                            cmd = user_text.lstrip()[1:].strip()
                            if not cmd:
                                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": "Empty command"})
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                                active_req_id = None
                                continue
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                            try:
                                with Session(engine) as session:
                                    bot = get_bot(session, bot_id)
                                    add_message_with_metrics(session, conversation_id=conv_id, role="user", content=user_text)
                                    meta = _get_conversation_meta(session, conversation_id=conv_id)
                                    da = _data_agent_meta(meta)
                                    workspace_dir = (
                                        str(da.get("workspace_dir") or "").strip()
                                        or default_workspace_dir_for_conversation(conv_id)
                                    )
                                    container_id = str(da.get("container_id") or "").strip()
                                    if not container_id:
                                        api_key = _get_openai_api_key_for_bot(session, bot=bot)
                                        if not api_key:
                                            raise RuntimeError(
                                                "No OpenAI API key configured for this bot (needed to start Isolated Workspace container)."
                                            )
                                        auth_json_raw = getattr(bot, "data_agent_auth_json", "") or "{}"
                                        git_token = (
                                            _get_git_token_plaintext(session, provider="github")
                                            if _git_auth_mode(auth_json_raw) == "token"
                                            else ""
                                        )
                                        container_id = ensure_conversation_container(
                                            conversation_id=conv_id,
                                            workspace_dir=workspace_dir,
                                            openai_api_key=api_key,
                                            git_token=git_token,
                                            auth_json=auth_json_raw,
                                        )
                                        merge_conversation_metadata(
                                            session,
                                            conversation_id=conv_id,
                                            patch={
                                                "data_agent.container_id": container_id,
                                                "data_agent.workspace_dir": workspace_dir,
                                            },
                                        )
                                res = await asyncio.to_thread(run_container_command, container_id=container_id, command=cmd)
                                out = res.stdout
                                if res.stderr:
                                    out = (out + "\n" if out else "") + f"[stderr]\\n{res.stderr}"
                                if not out:
                                    out = f"(exit {res.exit_code})"
                                if len(out) > 8000:
                                    out = out[:8000] + "…"
                                with Session(engine) as session:
                                    add_message_with_metrics(session, conversation_id=conv_id, role="assistant", content=out)
                                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": out})
                                await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": out})
                            except Exception as exc:
                                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            continue

                        try:
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                provider, llm_api_key, llm = _require_llm_client(session, bot=bot)
                                openai_api_key: Optional[str] = None
                                if speak:
                                    openai_api_key = _get_openai_api_key_for_bot(session, bot=bot)
                                    if not openai_api_key:
                                        raise HTTPException(
                                            status_code=400,
                                            detail="No OpenAI key configured for this bot (needed for TTS).",
                                        )

                                add_message_with_metrics(
                                    session,
                                    conversation_id=conv_id,
                                    role="user",
                                    content=user_text,
                                )
                                loop = asyncio.get_running_loop()

                                def _status_cb(stage: str) -> None:
                                    asyncio.run_coroutine_threadsafe(
                                        _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": stage}), loop
                                    )

                                history = await _build_history_budgeted_async(
                                    bot_id=bot.id,
                                    conversation_id=conv_id,
                                    llm_api_key=llm_api_key,
                                    status_cb=_status_cb,
                                )
                                tools_defs = _build_tools_for_bot(session, bot.id)
                                if speak:
                                    tts_synth = _get_tts_synth_fn(bot, openai_api_key)
                                if debug_mode:
                                    await _emit_llm_debug_payload(
                                        ws=ws,
                                        req_id=req_id,
                                        conversation_id=conv_id,
                                        phase="chat_llm",
                                        payload=llm.build_request_payload(
                                            messages=history, tools=tools_defs, stream=True
                                        ),
                                    )
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                        llm_start_ts = time.time()

                        delta_q_client: "queue.Queue[Optional[str]]" = queue.Queue()
                        delta_q_tts: "queue.Queue[Optional[str]]" = queue.Queue()
                        audio_q: "queue.Queue[Optional[tuple[bytes, int]]]" = queue.Queue()
                        tool_calls: list[ToolCall] = []
                        error_q: "queue.Queue[Optional[str]]" = queue.Queue()
                        full_text_parts: list[str] = []
                        citations_collected: list[dict[str, Any]] = []
                        metrics_lock = threading.Lock()
                        citations_lock = threading.Lock()
                        first_token_ts: Optional[float] = None
                        tts_start_ts: Optional[float] = None
                        first_audio_ts: Optional[float] = None

                        def llm_thread() -> None:
                            try:
                                for ev in llm.stream_text_or_tool(messages=history, tools=tools_defs):
                                    if isinstance(ev, ToolCall):
                                        tool_calls.append(ev)
                                        continue
                                    if isinstance(ev, CitationEvent):
                                        with citations_lock:
                                            citations_collected.extend(ev.citations)
                                        continue
                                    d = ev
                                    full_text_parts.append(d)
                                    delta_q_client.put(d)
                                    if speak:
                                        delta_q_tts.put(d)
                            except Exception as exc:
                                error_q.put(str(exc))
                            finally:
                                delta_q_client.put(None)
                                if speak:
                                    delta_q_tts.put(None)

                        def tts_thread() -> None:
                            nonlocal tts_start_ts
                            if not speak:
                                audio_q.put(None)
                                return
                            try:
                                synth = tts_synth or _get_tts_synth_fn(bot, openai_api_key)
                                for text_to_speak in _iter_tts_chunks(delta_q_tts):
                                    if not text_to_speak:
                                        continue
                                    with metrics_lock:
                                        if tts_start_ts is None:
                                            tts_start_ts = time.time()
                                    status(req_id, "tts")
                                    wav, sr = synth(text_to_speak)
                                    audio_q.put((wav, sr))
                            except Exception as exc:
                                error_q.put(f"TTS failed: {exc}")
                            finally:
                                audio_q.put(None)

                        t1 = threading.Thread(target=llm_thread, daemon=True)
                        t2 = threading.Thread(target=tts_thread, daemon=True)
                        t1.start()
                        t2.start()

                        open_deltas = True
                        open_audio = True
                        while open_deltas or open_audio:
                            try:
                                err = error_q.get_nowait()
                                if err:
                                    await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": err})
                                    open_deltas = False
                                    open_audio = False
                                    break
                            except queue.Empty:
                                pass

                            try:
                                d = delta_q_client.get_nowait()
                                if d is None:
                                    open_deltas = False
                                else:
                                    if first_token_ts is None:
                                        first_token_ts = time.time()
                                    await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
                            except queue.Empty:
                                pass

                            if speak:
                                try:
                                    item = audio_q.get_nowait()
                                    if item is None:
                                        open_audio = False
                                    else:
                                        wav, sr = item
                                        if first_audio_ts is None:
                                            first_audio_ts = time.time()
                                        await _ws_send_json(
                                            ws,
                                            {
                                                "type": "audio_wav",
                                                "req_id": req_id,
                                                "wav_base64": base64.b64encode(wav).decode(),
                                                "sr": sr,
                                            },
                                        )
                                except queue.Empty:
                                    pass
                            else:
                                open_audio = False

                            if (open_deltas or open_audio) and first_token_ts is None:
                                await asyncio.sleep(0.01)
                            else:
                                await asyncio.sleep(0.005)

                        t1.join()
                        t2.join()

                        llm_end_ts = time.time()
                        final_text = "".join(full_text_parts).strip()
                        with citations_lock:
                            citations = list(citations_collected)
                        citations_json = json.dumps(citations, ensure_ascii=False) if citations else "[]"

                        timings: dict[str, int] = {"total": int(round((llm_end_ts - llm_start_ts) * 1000.0))}
                        if first_token_ts is not None:
                            timings["llm_ttfb"] = int(round((first_token_ts - llm_start_ts) * 1000.0))
                        elif tool_calls and tool_calls[0].first_event_ts is not None:
                            timings["llm_ttfb"] = int(round((tool_calls[0].first_event_ts - llm_start_ts) * 1000.0))
                        timings["llm_total"] = int(round((llm_end_ts - llm_start_ts) * 1000.0))
                        if first_audio_ts is not None and tts_start_ts is not None:
                            timings["tts_first_audio"] = int(round((first_audio_ts - tts_start_ts) * 1000.0))

                        await _ws_send_json(ws, {"type": "metrics", "req_id": req_id, "timings_ms": timings})

                        if tool_calls and conv_id is not None:
                            rendered_reply = ""
                            tool_error: Optional[str] = None
                            needs_followup_llm = False
                            tool_failed = False
                            followup_streamed = False
                            followup_persisted = False
                            tts_busy_until: float = 0.0
                            last_wait_text: str | None = None
                            last_wait_ts: float = 0.0
                            wait_repeat_s: float = 45.0

                            async def _send_interim(text: str, *, kind: str) -> None:
                                nonlocal tts_busy_until, last_wait_text, last_wait_ts
                                t = (text or "").strip()
                                if not t:
                                    return
                                if kind == "wait":
                                    now = time.time()
                                    if last_wait_text == t and (now - last_wait_ts) < wait_repeat_s:
                                        return
                                    last_wait_text = t
                                    last_wait_ts = now
                                await _ws_send_json(
                                    ws,
                                    {"type": "interim", "req_id": req_id, "kind": kind, "text": t},
                                )
                                if not speak:
                                    return
                                now = time.time()
                                if now < tts_busy_until:
                                    await asyncio.sleep(tts_busy_until - now)
                                status(req_id, "tts")
                                try:
                                    wav, sr = await asyncio.to_thread(tts_synth, t)
                                    tts_busy_until = time.time() + _estimate_wav_seconds(wav, sr) + 0.15
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "audio_wav",
                                            "req_id": req_id,
                                            "wav_base64": base64.b64encode(wav).decode(),
                                            "sr": sr,
                                        },
                                    )
                                except Exception:
                                    return
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                meta_current = _get_conversation_meta(session, conversation_id=conv_id)
                                disabled_tools = _disabled_tool_names(bot)

                                for tc in tool_calls:
                                    tool_name = tc.name
                                    if tool_name == "set_variable":
                                        tool_name = "set_metadata"

                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "tool_call",
                                            "req_id": req_id,
                                            "name": tool_name,
                                            "arguments_json": tc.arguments_json,
                                        },
                                    )
                                    tool_parse_error = False
                                    try:
                                        tool_args = json.loads(tc.arguments_json or "{}")
                                        if not isinstance(tool_args, dict):
                                            raise ValueError("tool args must be an object")
                                    except Exception as exc:
                                        tool_parse_error = True
                                        tool_error = str(exc) or "Invalid tool args"
                                        tool_args = {}

                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="tool",
                                        content=json.dumps({"tool": tool_name, "arguments": tool_args}, ensure_ascii=False),
                                    )

                                    next_reply = str(tool_args.get("next_reply") or "").strip()
                                    wait_reply = str(tool_args.get("wait_reply") or "").strip() or "Working on it…"
                                    follow_up = _parse_follow_up_flag(tool_args.get("follow_up")) or _parse_follow_up_flag(
                                        tool_args.get("followup")
                                    )
                                    if (
                                        tool_name in {"request_host_action", "capture_screenshot"}
                                        and "follow_up" not in tool_args
                                        and "followup" not in tool_args
                                    ):
                                        follow_up = True
                                    if tool_name in {"request_host_action", "capture_screenshot"}:
                                        next_reply = ""
                                    raw_args = tool_args.get("args")
                                    if isinstance(raw_args, dict):
                                        patch = dict(raw_args)
                                    else:
                                        patch = dict(tool_args)
                                        patch.pop("next_reply", None)
                                        patch.pop("wait_reply", None)
                                        patch.pop("follow_up", None)
                                        patch.pop("followup", None)
                                        patch.pop("args", None)

                                    tool_cfg: IntegrationTool | None = None
                                    response_json: Any | None = None

                                    if tool_parse_error:
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": tool_error, "status_code": 500},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        rendered_reply = ""
                                    elif tool_name in disabled_tools:
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": f"Tool '{tool_name}' is disabled for this bot."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        rendered_reply = ""
                                    elif tool_name == "set_metadata":
                                        new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=patch)
                                        tool_result = {"ok": True, "updated": patch, "metadata": new_meta}
                                    elif tool_name == "web_search":
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": "web_search runs inside the model; no server tool is available."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        rendered_reply = ""
                                    elif tool_name == "http_request":
                                        task = asyncio.create_task(
                                            asyncio.to_thread(
                                                _execute_http_request_tool, tool_args=patch, meta=meta_current
                                            )
                                        )
                                        if wait_reply:
                                            await _send_interim(wait_reply, kind="wait")
                                        while True:
                                            try:
                                                tool_result = await asyncio.wait_for(
                                                    asyncio.shield(task), timeout=60.0
                                                )
                                                break
                                            except asyncio.TimeoutError:
                                                if wait_reply:
                                                    await _send_interim(wait_reply, kind="wait")
                                                continue
                                        tool_failed = not bool(tool_result.get("ok", False))
                                        needs_followup_llm = True
                                        rendered_reply = ""
                                    elif tool_name == "capture_screenshot":
                                        if not bool(getattr(bot, "enable_host_actions", False)):
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Host actions are disabled for this bot."},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        elif not bool(getattr(bot, "enable_host_shell", False)):
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Shell commands are disabled for this bot."},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            try:
                                                rel_path, target = _prepare_screenshot_target(conv)
                                            except Exception as exc:
                                                tool_result = {
                                                    "ok": False,
                                                    "error": {"message": str(exc) or "Invalid screenshot path"},
                                                }
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                ok_cmd, cmd_or_err = _screencapture_command(target)
                                                if not ok_cmd:
                                                    tool_result = {"ok": False, "error": {"message": cmd_or_err}}
                                                    tool_failed = True
                                                    needs_followup_llm = True
                                                    rendered_reply = ""
                                                else:
                                                    action = _create_host_action(
                                                        session,
                                                        conv=conv,
                                                        bot=bot,
                                                        action_type="run_shell",
                                                        payload={"command": cmd_or_err},
                                                    )
                                                    if _host_action_requires_approval(bot):
                                                        tool_result = _build_host_action_tool_result(action, ok=True)
                                                        tool_result["path"] = rel_path
                                                        if follow_up:
                                                            rendered_reply = ""
                                                            needs_followup_llm = True
                                                        else:
                                                            candidate = _render_with_meta(next_reply, meta_current).strip()
                                                            if candidate:
                                                                rendered_reply = candidate
                                                                needs_followup_llm = False
                                                            else:
                                                                rendered_reply = (
                                                                    "Approve the screenshot capture in the Action Queue, then ask me to analyze it."
                                                                )
                                                                needs_followup_llm = False
                                                    else:
                                                        if wait_reply:
                                                            await _send_interim(wait_reply, kind="wait")
                                                        tool_result = await _execute_host_action_and_update_async(
                                                            session, action=action
                                                        )
                                                        tool_failed = not bool(tool_result.get("ok", False))
                                                        if tool_failed:
                                                            needs_followup_llm = True
                                                            rendered_reply = ""
                                                        else:
                                                            ok, summary_text = _summarize_image_file(
                                                                session,
                                                                bot=bot,
                                                                image_path=target,
                                                                prompt=str(patch.get("prompt") or "").strip(),
                                                            )
                                                            if not ok:
                                                                tool_result["summary_error"] = summary_text
                                                                tool_failed = True
                                                                needs_followup_llm = True
                                                                rendered_reply = ""
                                                            else:
                                                                tool_result["summary"] = summary_text
                                                                tool_result["path"] = rel_path
                                                                if follow_up:
                                                                    rendered_reply = ""
                                                                    needs_followup_llm = True
                                                                else:
                                                                    candidate = _render_with_meta(next_reply, meta_current).strip()
                                                                    if candidate:
                                                                        rendered_reply = candidate
                                                                        needs_followup_llm = False
                                                                    else:
                                                                        rendered_reply = summary_text
                                                                        needs_followup_llm = False
                                    elif tool_name == "summarize_screenshot":
                                        ok, summary_text, rel_path = _summarize_screenshot(
                                            session,
                                            conv=conv,
                                            bot=bot,
                                            path=str(patch.get("path") or "").strip(),
                                            prompt=str(patch.get("prompt") or "").strip(),
                                        )
                                        if not ok:
                                            tool_result = {"ok": False, "error": {"message": summary_text}}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            tool_result = {"ok": True, "summary": summary_text, "path": rel_path}
                                            candidate = _render_with_meta(next_reply, meta_current).strip()
                                            if candidate:
                                                rendered_reply = candidate
                                                needs_followup_llm = False
                                            else:
                                                rendered_reply = summary_text
                                                needs_followup_llm = False
                                    elif tool_name == "request_host_action":
                                        if not bool(getattr(bot, "enable_host_actions", False)):
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Host actions are disabled for this bot."},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            try:
                                                action_type, payload = _parse_host_action_args(patch)
                                            except Exception as exc:
                                                tool_result = {"ok": False, "error": {"message": str(exc) or "Invalid host action"}}
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                if action_type == "run_shell" and not bool(getattr(bot, "enable_host_shell", False)):
                                                    tool_result = {
                                                        "ok": False,
                                                        "error": {"message": "Shell commands are disabled for this bot."},
                                                    }
                                                    tool_failed = True
                                                    needs_followup_llm = True
                                                    rendered_reply = ""
                                                else:
                                                    action = _create_host_action(
                                                        session,
                                                        conv=conv,
                                                        bot=bot,
                                                        action_type=action_type,
                                                        payload=payload,
                                                    )
                                                    if _host_action_requires_approval(bot):
                                                        tool_result = _build_host_action_tool_result(action, ok=True)
                                                        if follow_up:
                                                            rendered_reply = ""
                                                            needs_followup_llm = True
                                                        else:
                                                            candidate = _render_with_meta(next_reply, meta_current).strip()
                                                            if candidate:
                                                                rendered_reply = candidate
                                                                needs_followup_llm = False
                                                            else:
                                                                needs_followup_llm = True
                                                                rendered_reply = ""
                                                    else:
                                                        if wait_reply:
                                                            await _send_interim(wait_reply, kind="wait")
                                                        tool_result = await _execute_host_action_and_update_async(
                                                            session, action=action
                                                        )
                                                        tool_failed = not bool(tool_result.get("ok", False))
                                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                                        if follow_up and not tool_failed:
                                                            needs_followup_llm = True
                                                            rendered_reply = ""
                                                        elif candidate and not tool_failed:
                                                            rendered_reply = candidate
                                                            needs_followup_llm = False
                                                        else:
                                                            needs_followup_llm = True
                                                            rendered_reply = ""
                                    elif tool_name == "give_command_to_data_agent":
                                        if not bool(getattr(bot, "enable_data_agent", False)):
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Isolated Workspace is disabled for this bot."},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        elif not docker_available():
                                            tool_result = {
                                                "ok": False,
                                                "error": {
                                                    "message": "Docker is not available. Install Docker to use Isolated Workspace.",
                                                },
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            what_to_do = str(patch.get("what_to_do") or "").strip()
                                            if not what_to_do:
                                                tool_result = {
                                                    "ok": False,
                                                    "error": {"message": "Missing required tool arg: what_to_do"},
                                                }
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                # Ensure the per-conversation runtime exists (Docker) and run Codex CLI.
                                                try:
                                                    logger.info(
                                                        "Isolated Workspace tool: start conv=%s bot=%s what_to_do=%s",
                                                        conv_id,
                                                        bot_id,
                                                        (what_to_do[:200] + "…") if len(what_to_do) > 200 else what_to_do,
                                                    )
                                                    da = _data_agent_meta(meta_current)
                                                    workspace_dir = (
                                                        str(da.get("workspace_dir") or "").strip()
                                                        or default_workspace_dir_for_conversation(conv_id)
                                                    )
                                                    container_id = str(da.get("container_id") or "").strip()
                                                    session_id = str(da.get("session_id") or "").strip()

                                                    api_key = _get_openai_api_key_for_bot(session, bot=bot)
                                                    if not api_key:
                                                        raise RuntimeError(
                                                            "No OpenAI API key configured for this bot (needed for Isolated Workspace)."
                                                        )
                                                    auth_json_raw = getattr(bot, "data_agent_auth_json", "") or "{}"
                                                    git_token = (
                                                        _get_git_token_plaintext(session, provider="github")
                                                        if _git_auth_mode(auth_json_raw) == "token"
                                                        else ""
                                                    )

                                                    # Ensure the container exists and is running even if metadata has a stale id.
                                                    ensured_container_id = await asyncio.to_thread(
                                                        ensure_conversation_container,
                                                        conversation_id=conv_id,
                                                        workspace_dir=workspace_dir,
                                                        openai_api_key=api_key,
                                                        git_token=git_token,
                                                        auth_json=auth_json_raw,
                                                    )
                                                    if ensured_container_id and ensured_container_id != container_id:
                                                        container_id = ensured_container_id
                                                        meta_current = merge_conversation_metadata(
                                                            session,
                                                            conversation_id=conv_id,
                                                            patch={
                                                                "data_agent.container_id": container_id,
                                                                "data_agent.workspace_dir": workspace_dir,
                                                            },
                                                        )

                                                    ctx = _build_data_agent_conversation_context(
                                                        session,
                                                        bot=bot,
                                                        conversation_id=conv_id,
                                                        meta=meta_current,
                                                    )
                                                    api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
                                                    auth_json = _merge_git_token_auth(auth_json_raw, git_token)
                                                    sys_prompt = (
                                                        (getattr(bot, "data_agent_system_prompt", "") or "").strip()
                                                        or DEFAULT_DATA_AGENT_SYSTEM_PROMPT
                                                    )
                                                    def _emit_tool_progress(text: str) -> None:
                                                        t = (text or "").strip()
                                                        if not t:
                                                            return
                                                        asyncio.run_coroutine_threadsafe(
                                                            _ws_send_json(
                                                                ws,
                                                                {
                                                                    "type": "tool_progress",
                                                                    "req_id": req_id,
                                                                    "name": tool_name,
                                                                    "text": t,
                                                                },
                                                            ),
                                                            loop,
                                                        )

                                                    task = asyncio.create_task(
                                                        asyncio.to_thread(
                                                            run_data_agent,
                                                            conversation_id=conv_id,
                                                            container_id=container_id,
                                                            session_id=session_id,
                                                            workspace_dir=workspace_dir,
                                                            api_spec_text=api_spec_text,
                                                            auth_json=auth_json,
                                                            system_prompt=sys_prompt,
                                                            conversation_context=ctx,
                                                            what_to_do=what_to_do,
                                                            on_stream=_emit_tool_progress,
                                                        )
                                                    )
                                                    if wait_reply:
                                                        await _send_interim(wait_reply, kind="wait")
                                                    last_wait = time.time()
                                                    while not task.done():
                                                        if wait_reply and (time.time() - last_wait) >= 10.0:
                                                            await _send_interim(wait_reply, kind="wait")
                                                            last_wait = time.time()
                                                        await asyncio.sleep(0.2)
                                                    da_res = await task

                                                    if da_res.session_id and da_res.session_id != session_id:
                                                        meta_current = merge_conversation_metadata(
                                                            session,
                                                            conversation_id=conv_id,
                                                            patch={"data_agent.session_id": da_res.session_id},
                                                        )
                                                    logger.info(
                                                        "Isolated Workspace tool: done conv=%s ok=%s container_id=%s session_id=%s output_file=%s error=%s",
                                                        conv_id,
                                                        bool(da_res.ok),
                                                        da_res.container_id,
                                                        da_res.session_id,
                                                        da_res.output_file,
                                                        da_res.error,
                                                    )
                                                    tool_result = {
                                                        "ok": bool(da_res.ok),
                                                        "result_text": da_res.result_text,
                                                        "data_agent_container_id": da_res.container_id,
                                                        "data_agent_session_id": da_res.session_id,
                                                        "data_agent_output_file": da_res.output_file,
                                                        "data_agent_debug_file": da_res.debug_file,
                                                        "error": da_res.error,
                                                    }
                                                    tool_failed = not bool(da_res.ok)
                                                    if (
                                                        bool(getattr(bot, "data_agent_return_result_directly", False))
                                                        and bool(da_res.ok)
                                                        and str(da_res.result_text or "").strip()
                                                    ):
                                                        needs_followup_llm = False
                                                        rendered_reply = str(da_res.result_text or "").strip()
                                                    else:
                                                        needs_followup_llm = True
                                                        rendered_reply = ""
                                                except Exception as exc:
                                                    logger.exception("Isolated Workspace tool failed conv=%s bot=%s", conv_id, bot_id)
                                                    tool_result = {
                                                        "ok": False,
                                                        "error": {"message": str(exc)},
                                                    }
                                                    tool_failed = True
                                                    needs_followup_llm = True
                                                    rendered_reply = ""
                                    else:
                                        tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                        if not tool_cfg:
                                            raise RuntimeError(f"Unknown tool: {tool_name}")
                                        if not bool(getattr(tool_cfg, "enabled", True)):
                                            response_json = {
                                                "__tool_args_error__": {
                                                    "missing": [],
                                                    "message": f"Tool '{tool_name}' is disabled for this bot.",
                                                }
                                            }
                                        else:
                                            task = asyncio.create_task(
                                                asyncio.to_thread(
                                                    _execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                                                )
                                            )
                                            if wait_reply:
                                                await _send_interim(wait_reply, kind="wait")
                                            while True:
                                                try:
                                                    response_json = await asyncio.wait_for(asyncio.shield(task), timeout=60.0)
                                                    break
                                                except asyncio.TimeoutError:
                                                    if wait_reply:
                                                        await _send_interim(wait_reply, kind="wait")
                                                    continue
                                        if speak:
                                            now = time.time()
                                            if now < tts_busy_until:
                                                await asyncio.sleep(tts_busy_until - now)
                                        if isinstance(response_json, dict) and "__tool_args_error__" in response_json:
                                            err = response_json["__tool_args_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        elif isinstance(response_json, dict) and "__http_error__" in response_json:
                                            err = response_json["__http_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            pagination_info = None
                                            if isinstance(response_json, dict) and "__igx_pagination__" in response_json:
                                                pagination_info = response_json.pop("__igx_pagination__", None)
                                            if bool(getattr(tool_cfg, "use_codex_response", False)):
                                                # Avoid bloating LLM-visible conversation metadata in Codex mode.
                                                tool_result = {"ok": True}
                                                new_meta = meta_current
                                            else:
                                                mapped = _apply_response_mapper(
                                                    mapper_json=tool_cfg.response_mapper_json,
                                                    response_json=response_json,
                                                    meta=meta_current,
                                                    tool_args=patch,
                                                )
                                                new_meta = merge_conversation_metadata(
                                                    session, conversation_id=conv_id, patch=mapped
                                                )
                                                meta_current = new_meta
                                                tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}
                                            if pagination_info:
                                                tool_result["pagination"] = pagination_info

                                            # Optional Codex HTTP agent (post-process the raw response JSON).
                                            # Static reply (if configured) takes priority.
                                            static_preview = ""
                                            if (tool_cfg.static_reply_template or "").strip():
                                                try:
                                                    static_preview = _render_static_reply(
                                                        template_text=tool_cfg.static_reply_template,
                                                        meta=new_meta or meta_current,
                                                        response_json=response_json,
                                                        tool_args=patch,
                                                    ).strip()
                                                except Exception:
                                                    static_preview = ""
                                            if (not static_preview) and bool(getattr(tool_cfg, "use_codex_response", False)):
                                                fields_required = str(patch.get("fields_required") or "").strip()
                                                what_to_search_for = str(patch.get("what_to_search_for") or "").strip()
                                                if not fields_required:
                                                    fields_required = what_to_search_for
                                                why_api_was_called = str(patch.get("why_api_was_called") or "").strip()
                                                if not why_api_was_called:
                                                    why_api_was_called = str(patch.get("why_to_search_for") or "").strip()
                                                if not fields_required or not why_api_was_called:
                                                    tool_result["codex_ok"] = False
                                                    tool_result["codex_error"] = "Missing fields_required / why_api_was_called."
                                                else:
                                                    fields_required_for_codex = fields_required
                                                    if what_to_search_for:
                                                        fields_required_for_codex = (
                                                            f"{fields_required_for_codex}\n\n(what_to_search_for) {what_to_search_for}"
                                                        )
                                                    did_postprocess = False
                                                    postprocess_python = str(
                                                        getattr(tool_cfg, "postprocess_python", "") or ""
                                                    ).strip()
                                                    if postprocess_python:
                                                        py_task = asyncio.create_task(
                                                            asyncio.to_thread(
                                                                run_python_postprocessor,
                                                                python_code=postprocess_python,
                                                                payload={
                                                                    "response_json": response_json,
                                                                    "meta": new_meta or meta_current,
                                                                    "args": patch,
                                                                    "fields_required": fields_required,
                                                                    "why_api_was_called": why_api_was_called,
                                                                },
                                                                timeout_s=60,
                                                            )
                                                        )
                                                        if wait_reply:
                                                            await _send_interim(wait_reply, kind="wait")
                                                        last_wait = time.time()
                                                        wait_interval_s = 15.0
                                                        while not py_task.done():
                                                            now = time.time()
                                                            if wait_reply and (now - last_wait) >= wait_interval_s:
                                                                await _send_interim(wait_reply, kind="wait")
                                                                last_wait = now
                                                            await asyncio.sleep(0.2)
                                                        try:
                                                            py_res = await py_task
                                                            tool_result["python_ok"] = bool(getattr(py_res, "ok", False))
                                                            tool_result["python_duration_ms"] = int(
                                                                getattr(py_res, "duration_ms", 0) or 0
                                                            )
                                                            if getattr(py_res, "error", None):
                                                                tool_result["python_error"] = str(getattr(py_res, "error"))
                                                            if getattr(py_res, "stderr", None):
                                                                tool_result["python_stderr"] = str(getattr(py_res, "stderr"))
                                                            if py_res.ok:
                                                                did_postprocess = True
                                                                tool_result["postprocess_mode"] = "python"
                                                                tool_result["codex_ok"] = True
                                                                tool_result["codex_result_text"] = str(
                                                                    getattr(py_res, "result_text", "") or ""
                                                                )
                                                                mp = getattr(py_res, "metadata_patch", None)
                                                                if isinstance(mp, dict) and mp:
                                                                    try:
                                                                        meta_current = merge_conversation_metadata(
                                                                            session,
                                                                            conversation_id=conv_id,
                                                                            patch=mp,
                                                                        )
                                                                        tool_result["python_metadata_patch"] = mp
                                                                    except Exception:
                                                                        pass
                                                                try:
                                                                    append_saved_run_index(
                                                                        conversation_id=str(conv_id),
                                                                        event={
                                                                            "kind": "integration_python_postprocess",
                                                                            "tool_name": tool_name,
                                                                            "req_id": req_id,
                                                                            "python_ok": tool_result.get("python_ok"),
                                                                            "python_duration_ms": tool_result.get(
                                                                                "python_duration_ms"
                                                                            ),
                                                                        },
                                                                    )
                                                                except Exception:
                                                                    pass
                                                        except Exception as exc:
                                                            tool_result["python_ok"] = False
                                                            tool_result["python_error"] = str(exc)

                                                    if not did_postprocess:
                                                        codex_model = (
                                                            (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                            or "gpt-5.1-codex-mini"
                                                        )
                                                        progress_q: "queue.Queue[str]" = queue.Queue()

                                                        def _progress(s: str) -> None:
                                                            try:
                                                                progress_q.put_nowait(str(s))
                                                            except Exception:
                                                                return

                                                        agent_task = asyncio.create_task(
                                                            asyncio.to_thread(
                                                                run_codex_http_agent_one_shot,
                                                                api_key=api_key or "",
                                                                model=codex_model,
                                                                response_json=response_json,
                                                                fields_required=fields_required_for_codex,
                                                                why_api_was_called=why_api_was_called,
                                                                response_schema_json=getattr(tool_cfg, "response_schema_json", "")
                                                                or "",
                                                                conversation_id=str(conv_id) if conv_id is not None else None,
                                                                req_id=req_id,
                                                                tool_codex_prompt=getattr(tool_cfg, "codex_prompt", "") or "",
                                                                progress_fn=_progress,
                                                            )
                                                        )
                                                        if wait_reply:
                                                            await _send_interim(wait_reply, kind="wait")
                                                        last_wait = time.time()
                                                        last_progress = last_wait
                                                        wait_interval_s = 15.0
                                                        while not agent_task.done():
                                                            try:
                                                                while True:
                                                                    p = progress_q.get_nowait()
                                                                    if p:
                                                                        await _send_interim(p, kind="progress")
                                                                        last_progress = time.time()
                                                            except queue.Empty:
                                                                pass
                                                            now = time.time()
                                                            if (
                                                                wait_reply
                                                                and (now - last_wait) >= wait_interval_s
                                                                and (now - last_progress) >= wait_interval_s
                                                            ):
                                                                await _send_interim(wait_reply, kind="wait")
                                                                last_wait = now
                                                            await asyncio.sleep(0.2)
                                                        try:
                                                            agent_res = await agent_task
                                                            tool_result["postprocess_mode"] = "codex"
                                                            tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                                            tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                                            tool_result["codex_output_file"] = getattr(
                                                                agent_res, "result_text_path", ""
                                                            )
                                                            tool_result["codex_debug_file"] = getattr(
                                                                agent_res, "debug_json_path", ""
                                                            )
                                                            tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                                            tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                                            tool_result["codex_continue_reason"] = getattr(
                                                                agent_res, "continue_reason", ""
                                                            )
                                                            tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                                            saved_input_json_path = getattr(agent_res, "input_json_path", "")
                                                            saved_schema_json_path = getattr(agent_res, "schema_json_path", "")
                                                            err = getattr(agent_res, "error", None)
                                                            if err:
                                                                tool_result["codex_error"] = str(err)
                                                        except Exception as exc:
                                                            tool_result["codex_ok"] = False
                                                            tool_result["codex_error"] = str(exc)
                                                            saved_input_json_path = ""
                                                            saved_schema_json_path = ""

                                                        try:
                                                            append_saved_run_index(
                                                                conversation_id=str(conv_id),
                                                                event={
                                                                    "kind": "integration_http",
                                                                    "tool_name": tool_name,
                                                                    "req_id": req_id,
                                                                    "input_json_path": saved_input_json_path,
                                                                    "schema_json_path": saved_schema_json_path,
                                                                    "fields_required": fields_required,
                                                                    "why_api_was_called": why_api_was_called,
                                                                    "codex_output_dir": tool_result.get("codex_output_dir"),
                                                                    "codex_ok": tool_result.get("codex_ok"),
                                                                },
                                                            )
                                                        except Exception:
                                                            pass
                                    if speak:
                                        now = time.time()
                                        if now < tts_busy_until:
                                            await asyncio.sleep(tts_busy_until - now)

                                    if tool_name == "capture_screenshot" and tool_failed:
                                        msg = _tool_error_message(tool_result, fallback="Screenshot failed.")
                                        rendered_reply = f"Screenshot failed: {msg}"
                                        needs_followup_llm = False
                                    if tool_name == "request_host_action" and tool_failed:
                                        msg = _tool_error_message(tool_result, fallback="Host action failed.")
                                        rendered_reply = f"Host action failed: {msg}"
                                        needs_followup_llm = False

                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="tool",
                                        content=json.dumps({"tool": tool_name, "result": tool_result}, ensure_ascii=False),
                                    )
                                    if isinstance(tool_result, dict):
                                        meta_current = tool_result.get("metadata") or meta_current

                                    await _ws_send_json(
                                        ws,
                                        {"type": "tool_result", "req_id": req_id, "name": tool_name, "result": tool_result},
                                    )

                                    if tool_failed:
                                        break

                                    candidate = ""
                                    if tool_name != "set_metadata" and tool_cfg:
                                        static_text = ""
                                        if (tool_cfg.static_reply_template or "").strip():
                                            static_text = _render_static_reply(
                                                template_text=tool_cfg.static_reply_template,
                                                meta=meta_current,
                                                response_json=response_json,
                                                tool_args=patch,
                                            ).strip()
                                        if static_text:
                                            needs_followup_llm = False
                                            rendered_reply = static_text
                                        else:
                                            if bool(getattr(tool_cfg, "use_codex_response", False)):
                                                if bool(getattr(tool_cfg, "return_result_directly", False)) and isinstance(
                                                    tool_result, dict
                                                ):
                                                    direct = str(tool_result.get("codex_result_text") or "").strip()
                                                    if direct:
                                                        needs_followup_llm = False
                                                        rendered_reply = direct
                                                    else:
                                                        needs_followup_llm = True
                                                        rendered_reply = ""
                                                else:
                                                    needs_followup_llm = True
                                                    rendered_reply = ""
                                            else:
                                                needs_followup_llm = _should_followup_llm_for_tool(
                                                    tool=tool_cfg, static_rendered=static_text
                                                )
                                                candidate = _render_with_meta(next_reply, meta_current).strip()
                                                if candidate:
                                                    rendered_reply = candidate
                                                    needs_followup_llm = False
                                                else:
                                                    rendered_reply = ""
                                    else:
                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                        rendered_reply = candidate or rendered_reply

                            # If static reply is missing/empty for an integration tool, ask the LLM again
                            # with tool call + tool result already persisted in history.
                            if needs_followup_llm and conv_id is not None:
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                                with Session(engine) as session:
                                    followup_bot = get_bot(session, bot_id)
                                    followup_history = await _build_history_budgeted_async(
                                        bot_id=followup_bot.id,
                                        conversation_id=conv_id,
                                        llm_api_key=llm_api_key,
                                        status_cb=None,
                                    )
                                followup_history.append(
                                    Message(
                                        role="system",
                                        content=(
                                            ("The previous tool call failed. " if tool_failed else "")
                                            + "Using the latest tool result(s) above, write the next assistant reply. "
                                            "If the tool result contains codex_result_text, rephrase it for the user and do not mention file paths. "
                                            "Do not call any tools."
                                        ),
                                    )
                                )
                                followup_model = followup_bot.openai_model
                                follow_llm = _build_llm_client(
                                    bot=followup_bot,
                                    api_key=llm_api_key,
                                    model_override=followup_model,
                                )
                                if debug_mode:
                                    await _emit_llm_debug_payload(
                                        ws=ws,
                                        req_id=req_id,
                                        conversation_id=conv_id,
                                        phase="tool_followup_llm",
                                        payload=follow_llm.build_request_payload(messages=followup_history, stream=True),
                                    )
                                text2, ttfb2, total2 = await _stream_llm_reply(
                                    ws=ws, req_id=req_id, llm=follow_llm, messages=followup_history
                                )
                                rendered_reply = text2.strip()
                                if rendered_reply:
                                    followup_streamed = True
                                    in_tok = int(estimate_messages_tokens(followup_history, followup_model) or 0)
                                    out_tok = int(estimate_text_tokens(rendered_reply, followup_model) or 0)
                                    with Session(engine) as session:
                                        price = _get_model_price(session, provider=provider, model=followup_model)
                                    cost = float(
                                        estimate_cost_usd(
                                            model_price=price,
                                            input_tokens=in_tok,
                                            output_tokens=out_tok,
                                        )
                                        or 0.0
                                    )
                                    with Session(engine) as session:
                                        add_message_with_metrics(
                                            session,
                                            conversation_id=conv_id,
                                            role="assistant",
                                            content=rendered_reply,
                                            input_tokens_est=in_tok or None,
                                            output_tokens_est=out_tok or None,
                                            cost_usd_est=cost or None,
                                            llm_ttfb_ms=ttfb2,
                                            llm_total_ms=total2,
                                            total_ms=total2,
                                        )
                                        update_conversation_metrics(
                                            session,
                                            conversation_id=conv_id,
                                            add_input_tokens_est=in_tok,
                                            add_output_tokens_est=out_tok,
                                            add_cost_usd_est=cost,
                                            last_asr_ms=None,
                                            last_llm_ttfb_ms=ttfb2,
                                            last_llm_total_ms=total2,
                                            last_tts_first_audio_ms=None,
                                            last_total_ms=total2,
                                        )
                                    followup_persisted = True
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "metrics",
                                            "req_id": req_id,
                                            "timings_ms": {"llm_ttfb": ttfb2, "llm_total": total2, "total": total2},
                                        },
                                    )

                            if tool_error:
                                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": tool_error})

                            if rendered_reply:
                                if not followup_streamed:
                                    await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": rendered_reply})
                                try:
                                    if not followup_persisted:
                                        with Session(engine) as session:
                                            in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                                                session=session,
                                                bot=bot,
                                                provider=provider,
                                                history=history,
                                                assistant_text=rendered_reply,
                                            )
                                            add_message_with_metrics(
                                                session,
                                                conversation_id=conv_id,
                                                role="assistant",
                                                content=rendered_reply,
                                                input_tokens_est=in_tok,
                                                output_tokens_est=out_tok,
                                                cost_usd_est=cost,
                                                llm_ttfb_ms=timings.get("llm_ttfb"),
                                                llm_total_ms=timings.get("llm_total"),
                                                total_ms=timings.get("total"),
                                            )
                                            update_conversation_metrics(
                                                session,
                                                conversation_id=conv_id,
                                                add_input_tokens_est=in_tok,
                                                add_output_tokens_est=out_tok,
                                                add_cost_usd_est=cost,
                                                last_asr_ms=None,
                                                last_llm_ttfb_ms=timings.get("llm_ttfb"),
                                                last_llm_total_ms=timings.get("llm_total"),
                                                last_tts_first_audio_ms=None,
                                                last_total_ms=timings.get("total"),
                                            )
                                except Exception:
                                    pass

                                if speak:
                                    status(req_id, "tts")
                                    if tts_synth is None:
                                        tts_synth = _get_tts_synth_fn(bot, openai_api_key)
                                    wav, sr = await asyncio.to_thread(tts_synth, rendered_reply)
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "audio_wav",
                                            "req_id": req_id,
                                            "wav_base64": base64.b64encode(wav).decode(),
                                            "sr": sr,
                                        },
                                    )

                            await _ws_send_json(
                                ws,
                                {"type": "done", "req_id": req_id, "text": rendered_reply, "citations": citations},
                            )
                        else:
                            # Store assistant response.
                            try:
                                with Session(engine) as session:
                                    in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                                        session=session,
                                        bot=bot,
                                        provider=provider,
                                        history=history,
                                        assistant_text=final_text,
                                    )
                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="assistant",
                                        content=final_text,
                                        input_tokens_est=in_tok,
                                        output_tokens_est=out_tok,
                                        cost_usd_est=cost,
                                        llm_ttfb_ms=timings.get("llm_ttfb"),
                                        llm_total_ms=timings.get("llm_total"),
                                        tts_first_audio_ms=timings.get("tts_first_audio"),
                                        total_ms=timings.get("total"),
                                        citations_json=citations_json,
                                    )
                                    update_conversation_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        add_input_tokens_est=in_tok,
                                        add_output_tokens_est=out_tok,
                                        add_cost_usd_est=cost,
                                        last_asr_ms=None,
                                        last_llm_ttfb_ms=timings.get("llm_ttfb"),
                                        last_llm_total_ms=timings.get("llm_total"),
                                        last_tts_first_audio_ms=timings.get("tts_first_audio"),
                                        last_total_ms=timings.get("total"),
                                    )
                            except Exception:
                                pass

                            await _ws_send_json(
                                ws,
                                {"type": "done", "req_id": req_id, "text": final_text, "citations": citations},
                            )

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                        active_req_id = None
                        conv_id = None
                        accepting_audio = False
                        continue

                    elif msg_type == "stop":
                        if not req_id or active_req_id != req_id:
                            await _ws_send_json(
                                ws, {"type": "error", "req_id": req_id or None, "error": "Unknown req_id"}
                            )
                            continue
                        accepting_audio = False
                        if not conv_id:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": "No conversation"})
                            active_req_id = None
                            continue

                        stop_ts = time.time()
                        if not audio_buf:
                            await _ws_send_json(ws, {"type": "asr", "req_id": req_id, "text": ""})
                            await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "asr"})

                        asr_start_ts = time.time()
                        try:
                            with Session(engine) as session:
                                bot = get_bot(session, bot_id)
                                openai_api_key = _get_openai_api_key_for_bot(session, bot=bot)
                                if not openai_api_key:
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "error",
                                            "req_id": req_id,
                                            "error": "No OpenAI key configured for this bot.",
                                        },
                                    )
                                    await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                                    active_req_id = None
                                    conv_id = None
                                    continue

                                pcm16 = bytes(audio_buf)

                                asr = await asyncio.to_thread(
                                    _get_asr(openai_api_key, bot.openai_asr_model, bot.language).transcribe_pcm16,
                                    pcm16=pcm16,
                                    sample_rate=16000,
                                )
                                asr_end_ts = time.time()

                            user_text = (asr.text or "").strip()
                            await _ws_send_json(ws, {"type": "asr", "req_id": req_id, "text": user_text})
                            if not user_text:
                                await _ws_send_json(
                                    ws,
                                    {
                                        "type": "metrics",
                                        "req_id": req_id,
                                        "timings_ms": {
                                            "asr": int(round((asr_end_ts - asr_start_ts) * 1000.0)),
                                            "total": int(
                                                round((time.time() - (stop_ts or asr_start_ts)) * 1000.0)
                                            ),
                                        },
                                    },
                                )
                                await _ws_send_json(ws, {"type": "done", "req_id": req_id, "text": ""})
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                                active_req_id = None
                                conv_id = None
                                continue

                            add_message_with_metrics(
                                session,
                                conversation_id=conv_id,
                                role="user",
                                content=user_text,
                                asr_ms=int(round((asr_end_ts - asr_start_ts) * 1000.0)),
                            )
                            loop = asyncio.get_running_loop()

                            def _status_cb(stage: str) -> None:
                                asyncio.run_coroutine_threadsafe(
                                    _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": stage}), loop
                                )

                            provider, llm_api_key, llm = _require_llm_client(session, bot=bot)
                            history = await _build_history_budgeted_async(
                                bot_id=bot.id,
                                conversation_id=conv_id,
                                llm_api_key=llm_api_key,
                                status_cb=_status_cb,
                            )
                            tools_defs = _build_tools_for_bot(session, bot.id)
                            if debug_mode:
                                await _emit_llm_debug_payload(
                                    ws=ws,
                                    req_id=req_id,
                                    conversation_id=conv_id,
                                    phase="asr_turn_llm",
                                    payload=llm.build_request_payload(messages=history, tools=tools_defs, stream=True),
                                )
                        except Exception as exc:
                            await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": str(exc)})
                            await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                            active_req_id = None
                            conv_id = None
                            continue

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                        llm_start_ts = time.time()

                        # Stream LLM deltas + TTS audio from background threads.
                        delta_q_client: "queue.Queue[Optional[str]]" = queue.Queue()
                        delta_q_tts: "queue.Queue[Optional[str]]" = queue.Queue()
                        audio_q: "queue.Queue[Optional[tuple[bytes, int]]]" = queue.Queue()
                        tool_calls: list[ToolCall] = []
                        error_q: "queue.Queue[Optional[str]]" = queue.Queue()
                        full_text_parts: list[str] = []
                        metrics_lock = threading.Lock()
                        citations_collected: list[dict[str, Any]] = []
                        citations_lock = threading.Lock()
                        first_token_ts: Optional[float] = None
                        tts_start_ts: Optional[float] = None
                        first_audio_ts: Optional[float] = None

                        def llm_thread() -> None:
                            try:
                                for ev in llm.stream_text_or_tool(messages=history, tools=tools_defs):
                                    if isinstance(ev, ToolCall):
                                        tool_calls.append(ev)
                                        continue
                                    if isinstance(ev, CitationEvent):
                                        with citations_lock:
                                            citations_collected.extend(ev.citations)
                                        continue
                                    d = ev
                                    full_text_parts.append(d)
                                    delta_q_client.put(d)
                                    if speak:
                                        delta_q_tts.put(d)
                            except Exception as exc:
                                error_q.put(str(exc))
                            finally:
                                delta_q_client.put(None)
                                if speak:
                                    delta_q_tts.put(None)

                        def tts_thread() -> None:
                            nonlocal tts_start_ts
                            if not speak:
                                audio_q.put(None)
                                return
                            try:
                                tts_synth = _get_tts_synth_fn(bot, openai_api_key)
                                for text_to_speak in _iter_tts_chunks(delta_q_tts):
                                    if not text_to_speak:
                                        continue
                                    with metrics_lock:
                                        if tts_start_ts is None:
                                            tts_start_ts = time.time()
                                    status(req_id, "tts")
                                    wav, sr = tts_synth(text_to_speak)
                                    audio_q.put((wav, sr))
                            except Exception as exc:
                                error_q.put(f"TTS failed: {exc}")
                            finally:
                                audio_q.put(None)

                        t1 = threading.Thread(target=llm_thread, daemon=True)
                        t2 = threading.Thread(target=tts_thread, daemon=True)
                        t1.start()
                        t2.start()

                        # Pump the queues to the websocket.
                        open_deltas = True
                        open_audio = True
                        while open_deltas or open_audio:
                            try:
                                err = error_q.get_nowait()
                                if err:
                                    await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": err})
                                    open_deltas = False
                                    open_audio = False
                                    break
                            except queue.Empty:
                                pass

                            sent_any = False
                            try:
                                d = delta_q_client.get_nowait()
                                if d is None:
                                    open_deltas = False
                                else:
                                    if first_token_ts is None:
                                        first_token_ts = time.time()
                                    await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
                                sent_any = True
                            except queue.Empty:
                                pass

                            if speak:
                                try:
                                    item = audio_q.get_nowait()
                                    if item is None:
                                        open_audio = False
                                    else:
                                        wav, sr = item
                                        if first_audio_ts is None:
                                            first_audio_ts = time.time()
                                        await _ws_send_json(
                                            ws,
                                            {
                                                "type": "audio_wav",
                                                "req_id": req_id,
                                                "wav_base64": base64.b64encode(wav).decode(),
                                                "sr": sr,
                                            },
                                        )
                                    sent_any = True
                                except queue.Empty:
                                    pass
                            else:
                                open_audio = False

                            if not sent_any:
                                await asyncio.sleep(0.01)

                        t1.join()
                        t2.join()
                        llm_end_ts = time.time()

                        final_text = "".join(full_text_parts).strip()
                        with citations_lock:
                            citations = list(citations_collected)
                        citations_json = json.dumps(citations, ensure_ascii=False) if citations else "[]"

                        timings: dict[str, int] = {
                            "asr": int(round((asr_end_ts - asr_start_ts) * 1000.0)),
                            "llm_total": int(round((llm_end_ts - llm_start_ts) * 1000.0)),
                            "total": int(round((time.time() - (stop_ts or llm_start_ts)) * 1000.0)),
                        }
                        if first_token_ts is not None:
                            timings["llm_ttfb"] = int(round((first_token_ts - llm_start_ts) * 1000.0))
                        elif tool_calls and tool_calls[0].first_event_ts is not None:
                            timings["llm_ttfb"] = int(round((tool_calls[0].first_event_ts - llm_start_ts) * 1000.0))
                        if speak and tts_start_ts is not None and first_audio_ts is not None:
                            timings["tts_first_audio"] = int(round((first_audio_ts - tts_start_ts) * 1000.0))
                            timings["tts_from_llm_start"] = int(round((first_audio_ts - llm_start_ts) * 1000.0))

                        # Persist aggregates + last latencies (best-effort).
                        try:
                            with Session(engine) as session:
                                in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                                    session=session,
                                    bot=bot,
                                    provider=provider,
                                    history=history,
                                    assistant_text=final_text,
                                )
                                if final_text and conv_id and not tool_calls:
                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="assistant",
                                        content=final_text,
                                        input_tokens_est=in_tok,
                                        output_tokens_est=out_tok,
                                        cost_usd_est=cost,
                                        asr_ms=timings.get("asr"),
                                        llm_ttfb_ms=timings.get("llm_ttfb"),
                                        llm_total_ms=timings.get("llm_total"),
                                        tts_first_audio_ms=timings.get("tts_first_audio"),
                                        total_ms=timings.get("total"),
                                        citations_json=citations_json,
                                    )
                                    update_conversation_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        add_input_tokens_est=in_tok,
                                        add_output_tokens_est=out_tok,
                                        add_cost_usd_est=cost,
                                        last_asr_ms=timings.get("asr"),
                                        last_llm_ttfb_ms=timings.get("llm_ttfb"),
                                        last_llm_total_ms=timings.get("llm_total"),
                                        last_tts_first_audio_ms=timings.get("tts_first_audio"),
                                        last_total_ms=timings.get("total"),
                                    )
                        except Exception:
                            pass

                        await _ws_send_json(ws, {"type": "metrics", "req_id": req_id, "timings_ms": timings})
                        if tool_calls and conv_id is not None:
                            rendered_reply = ""
                            tool_error: Optional[str] = None
                            needs_followup_llm = False
                            tool_failed = False
                            followup_streamed = False
                            followup_persisted = False
                            tts_busy_until: float = 0.0
                            last_wait_text: str | None = None
                            last_wait_ts: float = 0.0
                            wait_repeat_s: float = 45.0

                            async def _send_interim(text: str, *, kind: str) -> None:
                                nonlocal tts_busy_until, last_wait_text, last_wait_ts
                                t = (text or "").strip()
                                if not t:
                                    return
                                if kind == "wait":
                                    now = time.time()
                                    if last_wait_text == t and (now - last_wait_ts) < wait_repeat_s:
                                        return
                                    last_wait_text = t
                                    last_wait_ts = now
                                await _ws_send_json(
                                    ws,
                                    {"type": "interim", "req_id": req_id, "kind": kind, "text": t},
                                )
                                if not speak:
                                    return
                                now = time.time()
                                if now < tts_busy_until:
                                    await asyncio.sleep(tts_busy_until - now)
                                status(req_id, "tts")
                                try:
                                    wav, sr = await asyncio.to_thread(tts_synth, t)
                                    tts_busy_until = time.time() + _estimate_wav_seconds(wav, sr) + 0.15
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "audio_wav",
                                            "req_id": req_id,
                                            "wav_base64": base64.b64encode(wav).decode(),
                                            "sr": sr,
                                        },
                                    )
                                except Exception:
                                    return

                            with Session(engine) as session:
                                bot2 = get_bot(session, bot_id)
                                meta_current = _get_conversation_meta(session, conversation_id=conv_id)
                                disabled_tools = _disabled_tool_names(bot2)

                                for tc in tool_calls:
                                    tool_name = tc.name
                                    if tool_name == "set_variable":
                                        tool_name = "set_metadata"

                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "tool_call",
                                            "req_id": req_id,
                                            "name": tool_name,
                                            "arguments_json": tc.arguments_json,
                                        },
                                    )
                                    tool_parse_error = False
                                    try:
                                        tool_args = json.loads(tc.arguments_json or "{}")
                                        if not isinstance(tool_args, dict):
                                            raise ValueError("tool args must be an object")
                                    except Exception as exc:
                                        tool_parse_error = True
                                        tool_error = str(exc) or "Invalid tool args"
                                        tool_args = {}

                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="tool",
                                        content=json.dumps({"tool": tool_name, "arguments": tool_args}, ensure_ascii=False),
                                    )

                                    next_reply = str(tool_args.get("next_reply") or "").strip()
                                    wait_reply = str(tool_args.get("wait_reply") or "").strip() or "Working on it…"
                                    follow_up = _parse_follow_up_flag(tool_args.get("follow_up")) or _parse_follow_up_flag(
                                        tool_args.get("followup")
                                    )
                                    if (
                                        tool_name in {"request_host_action", "capture_screenshot"}
                                        and "follow_up" not in tool_args
                                        and "followup" not in tool_args
                                    ):
                                        follow_up = True
                                    if tool_name in {"request_host_action", "capture_screenshot"}:
                                        next_reply = ""
                                    raw_args = tool_args.get("args")
                                    if isinstance(raw_args, dict):
                                        patch = dict(raw_args)
                                    else:
                                        patch = dict(tool_args)
                                        patch.pop("next_reply", None)
                                        patch.pop("wait_reply", None)
                                        patch.pop("follow_up", None)
                                        patch.pop("followup", None)
                                        patch.pop("args", None)

                                    tool_cfg: IntegrationTool | None = None
                                    response_json: Any | None = None

                                    if tool_parse_error:
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": tool_error, "status_code": 500},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        rendered_reply = ""
                                    elif tool_name in disabled_tools:
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": f"Tool '{tool_name}' is disabled for this bot."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        rendered_reply = ""
                                    elif tool_name == "set_metadata":
                                        new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=patch)
                                        tool_result = {"ok": True, "updated": patch, "metadata": new_meta}
                                    elif tool_name == "web_search":
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": "web_search runs inside the model; no server tool is available."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        rendered_reply = ""
                                    elif tool_name == "http_request":
                                        task = asyncio.create_task(
                                            asyncio.to_thread(
                                                _execute_http_request_tool, tool_args=patch, meta=meta_current
                                            )
                                        )
                                        if wait_reply:
                                            await _send_interim(wait_reply, kind="wait")
                                        while True:
                                            try:
                                                tool_result = await asyncio.wait_for(
                                                    asyncio.shield(task), timeout=60.0
                                                )
                                                break
                                            except asyncio.TimeoutError:
                                                if wait_reply:
                                                    await _send_interim(wait_reply, kind="wait")
                                                continue
                                        tool_failed = not bool(tool_result.get("ok", False))
                                        needs_followup_llm = True
                                        rendered_reply = ""
                                    elif tool_name == "capture_screenshot":
                                        if not bool(getattr(bot2, "enable_host_actions", False)):
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Host actions are disabled for this bot."},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        elif not bool(getattr(bot2, "enable_host_shell", False)):
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Shell commands are disabled for this bot."},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            try:
                                                rel_path, target = _prepare_screenshot_target(conv)
                                            except Exception as exc:
                                                tool_result = {
                                                    "ok": False,
                                                    "error": {"message": str(exc) or "Invalid screenshot path"},
                                                }
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                ok_cmd, cmd_or_err = _screencapture_command(target)
                                                if not ok_cmd:
                                                    tool_result = {"ok": False, "error": {"message": cmd_or_err}}
                                                    tool_failed = True
                                                    needs_followup_llm = True
                                                    rendered_reply = ""
                                                else:
                                                    action = _create_host_action(
                                                        session,
                                                        conv=conv,
                                                        bot=bot2,
                                                        action_type="run_shell",
                                                        payload={"command": cmd_or_err},
                                                    )
                                                    if _host_action_requires_approval(bot2):
                                                        tool_result = _build_host_action_tool_result(action, ok=True)
                                                        tool_result["path"] = rel_path
                                                        if follow_up:
                                                            rendered_reply = ""
                                                            needs_followup_llm = True
                                                        else:
                                                            candidate = _render_with_meta(next_reply, meta_current).strip()
                                                            if candidate:
                                                                rendered_reply = candidate
                                                                needs_followup_llm = False
                                                            else:
                                                                rendered_reply = (
                                                                    "Approve the screenshot capture in the Action Queue, then ask me to analyze it."
                                                                )
                                                                needs_followup_llm = False
                                                    else:
                                                        if wait_reply:
                                                            await _send_interim(wait_reply, kind="wait")
                                                        tool_result = await _execute_host_action_and_update_async(
                                                            session, action=action
                                                        )
                                                        tool_failed = not bool(tool_result.get("ok", False))
                                                        if tool_failed:
                                                            needs_followup_llm = True
                                                            rendered_reply = ""
                                                        else:
                                                            ok, summary_text = _summarize_image_file(
                                                                session,
                                                                bot=bot2,
                                                                image_path=target,
                                                                prompt=str(patch.get("prompt") or "").strip(),
                                                            )
                                                            if not ok:
                                                                tool_result["summary_error"] = summary_text
                                                                tool_failed = True
                                                                needs_followup_llm = True
                                                                rendered_reply = ""
                                                            else:
                                                                tool_result["summary"] = summary_text
                                                                tool_result["path"] = rel_path
                                                                if follow_up:
                                                                    rendered_reply = ""
                                                                    needs_followup_llm = True
                                                                else:
                                                                    candidate = _render_with_meta(next_reply, meta_current).strip()
                                                                    if candidate:
                                                                        rendered_reply = candidate
                                                                        needs_followup_llm = False
                                                                    else:
                                                                        rendered_reply = summary_text
                                                                        needs_followup_llm = False
                                    elif tool_name == "summarize_screenshot":
                                        ok, summary_text, rel_path = _summarize_screenshot(
                                            session,
                                            conv=conv,
                                            bot=bot2,
                                            path=str(patch.get("path") or "").strip(),
                                            prompt=str(patch.get("prompt") or "").strip(),
                                        )
                                        if not ok:
                                            tool_result = {"ok": False, "error": {"message": summary_text}}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            tool_result = {"ok": True, "summary": summary_text, "path": rel_path}
                                            candidate = _render_with_meta(next_reply, meta_current).strip()
                                            if candidate:
                                                rendered_reply = candidate
                                                needs_followup_llm = False
                                            else:
                                                rendered_reply = summary_text
                                                needs_followup_llm = False
                                    elif tool_name == "request_host_action":
                                        if not bool(getattr(bot2, "enable_host_actions", False)):
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Host actions are disabled for this bot."},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            try:
                                                action_type, payload = _parse_host_action_args(patch)
                                            except Exception as exc:
                                                tool_result = {"ok": False, "error": {"message": str(exc) or "Invalid host action"}}
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                if action_type == "run_shell" and not bool(getattr(bot2, "enable_host_shell", False)):
                                                    tool_result = {
                                                        "ok": False,
                                                        "error": {"message": "Shell commands are disabled for this bot."},
                                                    }
                                                    tool_failed = True
                                                    needs_followup_llm = True
                                                    rendered_reply = ""
                                                else:
                                                    action = _create_host_action(
                                                        session,
                                                        conv=conv,
                                                        bot=bot2,
                                                        action_type=action_type,
                                                        payload=payload,
                                                    )
                                                    if _host_action_requires_approval(bot2):
                                                        tool_result = _build_host_action_tool_result(action, ok=True)
                                                        if follow_up:
                                                            rendered_reply = ""
                                                            needs_followup_llm = True
                                                        else:
                                                            candidate = _render_with_meta(next_reply, meta_current).strip()
                                                            if candidate:
                                                                rendered_reply = candidate
                                                                needs_followup_llm = False
                                                            else:
                                                                needs_followup_llm = True
                                                                rendered_reply = ""
                                                    else:
                                                        if wait_reply:
                                                            await _send_interim(wait_reply, kind="wait")
                                                        tool_result = await _execute_host_action_and_update_async(
                                                            session, action=action
                                                        )
                                                        tool_failed = not bool(tool_result.get("ok", False))
                                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                                        if follow_up and not tool_failed:
                                                            needs_followup_llm = True
                                                            rendered_reply = ""
                                                        elif candidate and not tool_failed:
                                                            rendered_reply = candidate
                                                            needs_followup_llm = False
                                                        else:
                                                            needs_followup_llm = True
                                                            rendered_reply = ""
                                    elif tool_name == "give_command_to_data_agent":
                                        if not bool(getattr(bot2, "enable_data_agent", False)):
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Isolated Workspace is disabled for this bot."},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        elif not docker_available():
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Docker is not available. Install Docker to use Isolated Workspace."},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            what_to_do = str(patch.get("what_to_do") or "").strip()
                                            if not what_to_do:
                                                tool_result = {
                                                    "ok": False,
                                                    "error": {"message": "Missing required tool arg: what_to_do"},
                                                }
                                                tool_failed = True
                                                needs_followup_llm = True
                                                rendered_reply = ""
                                            else:
                                                try:
                                                    logger.info(
                                                        "Isolated Workspace tool: start conv=%s bot=%s what_to_do=%s",
                                                        conv_id,
                                                        bot_id,
                                                        (what_to_do[:200] + "…") if len(what_to_do) > 200 else what_to_do,
                                                    )
                                                    da = _data_agent_meta(meta_current)
                                                    workspace_dir = (
                                                        str(da.get("workspace_dir") or "").strip()
                                                        or default_workspace_dir_for_conversation(conv_id)
                                                    )
                                                    container_id = str(da.get("container_id") or "").strip()
                                                    session_id = str(da.get("session_id") or "").strip()

                                                    api_key = _get_openai_api_key_for_bot(session, bot=bot2)
                                                    if not api_key:
                                                        raise RuntimeError(
                                                            "No OpenAI API key configured for this bot (needed for Isolated Workspace)."
                                                        )
                                                    auth_json_raw = getattr(bot2, "data_agent_auth_json", "") or "{}"
                                                    git_token = (
                                                        _get_git_token_plaintext(session, provider="github")
                                                        if _git_auth_mode(auth_json_raw) == "token"
                                                        else ""
                                                    )

                                                    if not container_id:
                                                        container_id = await asyncio.to_thread(
                                                            ensure_conversation_container,
                                                            conversation_id=conv_id,
                                                            workspace_dir=workspace_dir,
                                                            openai_api_key=api_key,
                                                            git_token=git_token,
                                                            auth_json=auth_json_raw,
                                                        )
                                                        meta_current = merge_conversation_metadata(
                                                            session,
                                                            conversation_id=conv_id,
                                                            patch={
                                                                "data_agent.container_id": container_id,
                                                                "data_agent.workspace_dir": workspace_dir,
                                                            },
                                                        )

                                                    ctx = _build_data_agent_conversation_context(
                                                        session,
                                                        bot=bot2,
                                                        conversation_id=conv_id,
                                                        meta=meta_current,
                                                    )
                                                    api_spec_text = getattr(bot2, "data_agent_api_spec_text", "") or ""
                                                    auth_json = _merge_git_token_auth(auth_json_raw, git_token)
                                                    sys_prompt = (
                                                        (getattr(bot2, "data_agent_system_prompt", "") or "").strip()
                                                        or DEFAULT_DATA_AGENT_SYSTEM_PROMPT
                                                    )
                                                    def _emit_tool_progress(text: str) -> None:
                                                        t = (text or "").strip()
                                                        if not t:
                                                            return
                                                        asyncio.run_coroutine_threadsafe(
                                                            _ws_send_json(
                                                                ws,
                                                                {
                                                                    "type": "tool_progress",
                                                                    "req_id": req_id,
                                                                    "name": tool_name,
                                                                    "text": t,
                                                                },
                                                            ),
                                                            loop,
                                                        )

                                                    task = asyncio.create_task(
                                                        asyncio.to_thread(
                                                            run_data_agent,
                                                            conversation_id=conv_id,
                                                            container_id=container_id,
                                                            session_id=session_id,
                                                            workspace_dir=workspace_dir,
                                                            api_spec_text=api_spec_text,
                                                            auth_json=auth_json,
                                                            system_prompt=sys_prompt,
                                                            conversation_context=ctx,
                                                            what_to_do=what_to_do,
                                                            on_stream=_emit_tool_progress,
                                                        )
                                                    )
                                                    if wait_reply:
                                                        await _send_interim(wait_reply, kind="wait")
                                                    last_wait = time.time()
                                                    while not task.done():
                                                        if wait_reply and (time.time() - last_wait) >= 10.0:
                                                            await _send_interim(wait_reply, kind="wait")
                                                            last_wait = time.time()
                                                        await asyncio.sleep(0.2)
                                                    da_res = await task

                                                    if da_res.session_id and da_res.session_id != session_id:
                                                        meta_current = merge_conversation_metadata(
                                                            session,
                                                            conversation_id=conv_id,
                                                            patch={"data_agent.session_id": da_res.session_id},
                                                        )
                                                    logger.info(
                                                        "Isolated Workspace tool: done conv=%s ok=%s container_id=%s session_id=%s output_file=%s error=%s",
                                                        conv_id,
                                                        bool(da_res.ok),
                                                        da_res.container_id,
                                                        da_res.session_id,
                                                        da_res.output_file,
                                                        da_res.error,
                                                    )
                                                    tool_result = {
                                                        "ok": bool(da_res.ok),
                                                        "result_text": da_res.result_text,
                                                        "data_agent_container_id": da_res.container_id,
                                                        "data_agent_session_id": da_res.session_id,
                                                        "data_agent_output_file": da_res.output_file,
                                                        "data_agent_debug_file": da_res.debug_file,
                                                        "error": da_res.error,
                                                    }
                                                    tool_failed = not bool(da_res.ok)
                                                    if (
                                                        bool(getattr(bot2, "data_agent_return_result_directly", False))
                                                        and bool(da_res.ok)
                                                        and str(da_res.result_text or "").strip()
                                                    ):
                                                        needs_followup_llm = False
                                                        rendered_reply = str(da_res.result_text or "").strip()
                                                    else:
                                                        needs_followup_llm = True
                                                        rendered_reply = ""
                                                except Exception as exc:
                                                    logger.exception("Isolated Workspace tool failed conv=%s bot=%s", conv_id, bot_id)
                                                    tool_result = {
                                                        "ok": False,
                                                        "error": {"message": str(exc)},
                                                    }
                                                    tool_failed = True
                                                    needs_followup_llm = True
                                                    rendered_reply = ""
                                    else:
                                        tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                        if not tool_cfg:
                                            raise RuntimeError(f"Unknown tool: {tool_name}")
                                        if not bool(getattr(tool_cfg, "enabled", True)):
                                            response_json = {
                                                "__tool_args_error__": {
                                                    "missing": [],
                                                    "message": f"Tool '{tool_name}' is disabled for this bot.",
                                                }
                                            }
                                        else:
                                            task = asyncio.create_task(
                                                asyncio.to_thread(
                                                    _execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                                                )
                                            )
                                            if wait_reply:
                                                await _send_interim(wait_reply, kind="wait")
                                            while True:
                                                try:
                                                    response_json = await asyncio.wait_for(asyncio.shield(task), timeout=60.0)
                                                    break
                                                except asyncio.TimeoutError:
                                                    if wait_reply:
                                                        await _send_interim(wait_reply, kind="wait")
                                                    continue
                                        if speak:
                                            now = time.time()
                                            if now < tts_busy_until:
                                                await asyncio.sleep(tts_busy_until - now)
                                        if isinstance(response_json, dict) and "__tool_args_error__" in response_json:
                                            err = response_json["__tool_args_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        elif isinstance(response_json, dict) and "__http_error__" in response_json:
                                            err = response_json["__http_error__"] or {}
                                            tool_result = {"ok": False, "error": err}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            rendered_reply = ""
                                        else:
                                            pagination_info = None
                                            if isinstance(response_json, dict) and "__igx_pagination__" in response_json:
                                                pagination_info = response_json.pop("__igx_pagination__", None)
                                            if bool(getattr(tool_cfg, "use_codex_response", False)):
                                                # Avoid bloating LLM-visible conversation metadata in Codex mode.
                                                tool_result = {"ok": True}
                                                new_meta = meta_current
                                            else:
                                                mapped = _apply_response_mapper(
                                                    mapper_json=tool_cfg.response_mapper_json,
                                                    response_json=response_json,
                                                    meta=meta_current,
                                                    tool_args=patch,
                                                )
                                                new_meta = merge_conversation_metadata(
                                                    session, conversation_id=conv_id, patch=mapped
                                                )
                                                meta_current = new_meta
                                                tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}
                                            if pagination_info:
                                                tool_result["pagination"] = pagination_info

                                            # Optional Codex HTTP agent (post-process the raw response JSON).
                                            # Static reply (if configured) takes priority.
                                            static_preview = ""
                                            if (tool_cfg.static_reply_template or "").strip():
                                                try:
                                                    static_preview = _render_static_reply(
                                                        template_text=tool_cfg.static_reply_template,
                                                        meta=new_meta or meta_current,
                                                        response_json=response_json,
                                                        tool_args=patch,
                                                    ).strip()
                                                except Exception:
                                                    static_preview = ""
                                            if (not static_preview) and bool(getattr(tool_cfg, "use_codex_response", False)):
                                                fields_required = str(patch.get("fields_required") or "").strip()
                                                what_to_search_for = str(patch.get("what_to_search_for") or "").strip()
                                                if not fields_required:
                                                    fields_required = what_to_search_for
                                                why_api_was_called = str(patch.get("why_api_was_called") or "").strip()
                                                if not why_api_was_called:
                                                    why_api_was_called = str(patch.get("why_to_search_for") or "").strip()
                                                if not fields_required or not why_api_was_called:
                                                    tool_result["codex_ok"] = False
                                                    tool_result["codex_error"] = "Missing fields_required / why_api_was_called."
                                                else:
                                                    fields_required_for_codex = fields_required
                                                    if what_to_search_for:
                                                        fields_required_for_codex = (
                                                            f"{fields_required_for_codex}\n\n(what_to_search_for) {what_to_search_for}"
                                                        )
                                                    did_postprocess = False
                                                    postprocess_python = str(
                                                        getattr(tool_cfg, "postprocess_python", "") or ""
                                                    ).strip()
                                                    if postprocess_python:
                                                        py_task = asyncio.create_task(
                                                            asyncio.to_thread(
                                                                run_python_postprocessor,
                                                                python_code=postprocess_python,
                                                                payload={
                                                                    "response_json": response_json,
                                                                    "meta": new_meta or meta_current,
                                                                    "args": patch,
                                                                    "fields_required": fields_required,
                                                                    "why_api_was_called": why_api_was_called,
                                                                },
                                                                timeout_s=60,
                                                            )
                                                        )
                                                        if wait_reply:
                                                            await _public_send_interim(
                                                                ws, req_id=req_id, kind="wait", text=wait_reply
                                                            )
                                                        last_wait = time.time()
                                                        wait_interval_s = 15.0
                                                        while not py_task.done():
                                                            now = time.time()
                                                            if wait_reply and (now - last_wait) >= wait_interval_s:
                                                                await _public_send_interim(
                                                                    ws, req_id=req_id, kind="wait", text=wait_reply
                                                                )
                                                                last_wait = now
                                                            await asyncio.sleep(0.2)
                                                        try:
                                                            py_res = await py_task
                                                            tool_result["python_ok"] = bool(getattr(py_res, "ok", False))
                                                            tool_result["python_duration_ms"] = int(
                                                                getattr(py_res, "duration_ms", 0) or 0
                                                            )
                                                            if getattr(py_res, "error", None):
                                                                tool_result["python_error"] = str(getattr(py_res, "error"))
                                                            if getattr(py_res, "stderr", None):
                                                                tool_result["python_stderr"] = str(getattr(py_res, "stderr"))
                                                            if py_res.ok:
                                                                did_postprocess = True
                                                                tool_result["postprocess_mode"] = "python"
                                                                tool_result["codex_ok"] = True
                                                                tool_result["codex_result_text"] = str(
                                                                    getattr(py_res, "result_text", "") or ""
                                                                )
                                                                mp = getattr(py_res, "metadata_patch", None)
                                                                if isinstance(mp, dict) and mp:
                                                                    try:
                                                                        meta_current = merge_conversation_metadata(
                                                                            session,
                                                                            conversation_id=conv_id,
                                                                            patch=mp,
                                                                        )
                                                                        tool_result["python_metadata_patch"] = mp
                                                                    except Exception:
                                                                        pass
                                                                try:
                                                                    append_saved_run_index(
                                                                        conversation_id=str(conv_id),
                                                                        event={
                                                                            "kind": "integration_python_postprocess",
                                                                            "tool_name": tool_name,
                                                                            "req_id": req_id,
                                                                            "python_ok": tool_result.get("python_ok"),
                                                                            "python_duration_ms": tool_result.get(
                                                                                "python_duration_ms"
                                                                            ),
                                                                        },
                                                                    )
                                                                except Exception:
                                                                    pass
                                                        except Exception as exc:
                                                            tool_result["python_ok"] = False
                                                            tool_result["python_error"] = str(exc)

                                                    if not did_postprocess:
                                                        codex_model = (
                                                            (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                            or "gpt-5.1-codex-mini"
                                                        )
                                                        progress_q: "queue.Queue[str]" = queue.Queue()

                                                        def _progress(s: str) -> None:
                                                            try:
                                                                progress_q.put_nowait(str(s))
                                                            except Exception:
                                                                return

                                                        agent_task = asyncio.create_task(
                                                            asyncio.to_thread(
                                                                run_codex_http_agent_one_shot,
                                                                api_key=api_key or "",
                                                                model=codex_model,
                                                                response_json=response_json,
                                                                fields_required=fields_required_for_codex,
                                                                why_api_was_called=why_api_was_called,
                                                                response_schema_json=getattr(tool_cfg, "response_schema_json", "")
                                                                or "",
                                                                conversation_id=str(conv_id) if conv_id is not None else None,
                                                                req_id=req_id,
                                                                tool_codex_prompt=getattr(tool_cfg, "codex_prompt", "") or "",
                                                                progress_fn=_progress,
                                                            )
                                                        )
                                                        if wait_reply:
                                                            await _public_send_interim(
                                                                ws, req_id=req_id, kind="wait", text=wait_reply
                                                            )
                                                        last_wait = time.time()
                                                        last_progress = last_wait
                                                        wait_interval_s = 15.0
                                                        while not agent_task.done():
                                                            try:
                                                                while True:
                                                                    p = progress_q.get_nowait()
                                                                    if p:
                                                                        await _public_send_interim(
                                                                            ws,
                                                                            req_id=req_id,
                                                                            kind="progress",
                                                                            text=p,
                                                                        )
                                                                        last_progress = time.time()
                                                            except queue.Empty:
                                                                pass
                                                            now = time.time()
                                                            if (
                                                                wait_reply
                                                                and (now - last_wait) >= wait_interval_s
                                                                and (now - last_progress) >= wait_interval_s
                                                            ):
                                                                await _public_send_interim(
                                                                    ws, req_id=req_id, kind="wait", text=wait_reply
                                                                )
                                                                last_wait = now
                                                            await asyncio.sleep(0.2)
                                                        try:
                                                            agent_res = await agent_task
                                                            tool_result["postprocess_mode"] = "codex"
                                                            tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                                            tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                                            tool_result["codex_output_file"] = getattr(agent_res, "result_text_path", "")
                                                            tool_result["codex_debug_file"] = getattr(agent_res, "debug_json_path", "")
                                                            tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                                            tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                                            tool_result["codex_continue_reason"] = getattr(
                                                                agent_res, "continue_reason", ""
                                                            )
                                                            tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                                            saved_input_json_path = getattr(agent_res, "input_json_path", "")
                                                            saved_schema_json_path = getattr(agent_res, "schema_json_path", "")
                                                            err = getattr(agent_res, "error", None)
                                                            if err:
                                                                tool_result["codex_error"] = str(err)
                                                        except Exception as exc:
                                                            tool_result["codex_ok"] = False
                                                            tool_result["codex_error"] = str(exc)
                                                            saved_input_json_path = ""
                                                            saved_schema_json_path = ""

                                                        try:
                                                            append_saved_run_index(
                                                                conversation_id=str(conv_id),
                                                                event={
                                                                    "kind": "integration_http",
                                                                    "tool_name": tool_name,
                                                                    "req_id": req_id,
                                                                    "input_json_path": saved_input_json_path,
                                                                    "schema_json_path": saved_schema_json_path,
                                                                    "fields_required": fields_required,
                                                                    "why_api_was_called": why_api_was_called,
                                                                    "codex_output_dir": tool_result.get("codex_output_dir"),
                                                                    "codex_ok": tool_result.get("codex_ok"),
                                                                },
                                                            )
                                                        except Exception:
                                                            pass

                                    if tool_name == "capture_screenshot" and tool_failed:
                                        msg = _tool_error_message(tool_result, fallback="Screenshot failed.")
                                        rendered_reply = f"Screenshot failed: {msg}"
                                        needs_followup_llm = False
                                    if tool_name == "request_host_action" and tool_failed:
                                        msg = _tool_error_message(tool_result, fallback="Host action failed.")
                                        rendered_reply = f"Host action failed: {msg}"
                                        needs_followup_llm = False

                                    add_message_with_metrics(
                                        session,
                                        conversation_id=conv_id,
                                        role="tool",
                                        content=json.dumps({"tool": tool_name, "result": tool_result}, ensure_ascii=False),
                                    )
                                    if isinstance(tool_result, dict):
                                        meta_current = tool_result.get("metadata") or meta_current

                                    await _ws_send_json(
                                        ws,
                                        {"type": "tool_result", "req_id": req_id, "name": tool_name, "result": tool_result},
                                    )

                                    if tool_failed:
                                        break

                                    if tool_name != "set_metadata" and tool_cfg:
                                        static_text = ""
                                        if (tool_cfg.static_reply_template or "").strip():
                                            static_text = _render_static_reply(
                                                template_text=tool_cfg.static_reply_template,
                                                meta=meta_current,
                                                response_json=response_json,
                                                tool_args=patch,
                                            ).strip()
                                        if static_text:
                                            needs_followup_llm = False
                                            rendered_reply = static_text
                                        else:
                                            needs_followup_llm = _should_followup_llm_for_tool(
                                                tool=tool_cfg, static_rendered=static_text
                                            )
                                            rendered_reply = ""
                                    else:
                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                        rendered_reply = candidate or rendered_reply

                            # If static reply is missing/empty for an integration tool, ask the LLM again
                            # with tool call + tool result already persisted in history.
                            if needs_followup_llm and conv_id is not None:
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                                with Session(engine) as session:
                                    bot2 = get_bot(session, bot_id)
                                    followup_history = await _build_history_budgeted_async(
                                        bot_id=bot2.id,
                                        conversation_id=conv_id,
                                        llm_api_key=llm_api_key,
                                        status_cb=None,
                                    )
                                    followup_history.append(
                                        Message(
                                            role="system",
                                            content=(
                                                "The previous tool call failed. "
                                                if tool_failed
                                                else ""
                                            )
                                            + "Using the latest tool result(s) above, write the next assistant reply. "
                                            "If the tool result contains codex_result_text, rephrase it for the user and do not mention file paths. "
                                            "Do not call any tools.",
                                        )
                                    )
                                follow_llm = _build_llm_client(
                                    bot=bot,
                                    api_key=llm_api_key,
                                    model_override=bot.openai_model,
                                )
                                if debug_mode:
                                    await _emit_llm_debug_payload(
                                        ws=ws,
                                        req_id=req_id,
                                        conversation_id=conv_id,
                                        phase="tool_followup_llm",
                                        payload=follow_llm.build_request_payload(
                                            messages=followup_history, stream=True
                                        ),
                                    )
                                text2, ttfb2, total2 = await _stream_llm_reply(
                                    ws=ws, req_id=req_id, llm=follow_llm, messages=followup_history
                                )
                                rendered_reply = text2.strip()
                                if rendered_reply:
                                    followup_streamed = True
                                    in_tok = int(estimate_messages_tokens(followup_history, bot.openai_model) or 0)
                                    out_tok = int(estimate_text_tokens(rendered_reply, bot.openai_model) or 0)
                                    with Session(engine) as session:
                                        price = _get_model_price(session, provider=provider, model=bot.openai_model)
                                    cost = float(
                                        estimate_cost_usd(model_price=price, input_tokens=in_tok, output_tokens=out_tok) or 0.0
                                    )
                                    with Session(engine) as session:
                                        add_message_with_metrics(
                                            session,
                                            conversation_id=conv_id,
                                            role="assistant",
                                            content=rendered_reply,
                                            input_tokens_est=in_tok or None,
                                            output_tokens_est=out_tok or None,
                                            cost_usd_est=cost or None,
                                            asr_ms=timings.get("asr"),
                                            llm_ttfb_ms=ttfb2,
                                            llm_total_ms=total2,
                                            total_ms=total2,
                                        )
                                        update_conversation_metrics(
                                            session,
                                            conversation_id=conv_id,
                                            add_input_tokens_est=in_tok,
                                            add_output_tokens_est=out_tok,
                                            add_cost_usd_est=cost,
                                            last_asr_ms=timings.get("asr"),
                                            last_llm_ttfb_ms=ttfb2,
                                            last_llm_total_ms=total2,
                                            last_tts_first_audio_ms=None,
                                            last_total_ms=total2,
                                        )
                                    followup_persisted = True
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "metrics",
                                            "req_id": req_id,
                                            "timings_ms": {"llm_ttfb": ttfb2, "llm_total": total2, "total": total2},
                                        },
                                    )

                            if tool_error:
                                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": tool_error})

                            if rendered_reply:
                                if not followup_streamed:
                                    await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": rendered_reply})
                                try:
                                    if not followup_persisted:
                                        with Session(engine) as session:
                                            in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                                                session=session,
                                                bot=bot,
                                                provider=provider,
                                                history=history,
                                                assistant_text=rendered_reply,
                                            )
                                            add_message_with_metrics(
                                                session,
                                                conversation_id=conv_id,
                                                role="assistant",
                                                content=rendered_reply,
                                                input_tokens_est=in_tok,
                                                output_tokens_est=out_tok,
                                                cost_usd_est=cost,
                                                asr_ms=timings.get("asr"),
                                                llm_ttfb_ms=timings.get("llm_ttfb"),
                                                llm_total_ms=timings.get("llm_total"),
                                                total_ms=timings.get("total"),
                                                citations_json=citations_json,
                                            )
                                            update_conversation_metrics(
                                                session,
                                                conversation_id=conv_id,
                                                add_input_tokens_est=in_tok,
                                                add_output_tokens_est=out_tok,
                                                add_cost_usd_est=cost,
                                                last_asr_ms=timings.get("asr"),
                                                last_llm_ttfb_ms=timings.get("llm_ttfb"),
                                                last_llm_total_ms=timings.get("llm_total"),
                                                last_tts_first_audio_ms=None,
                                                last_total_ms=timings.get("total"),
                                            )
                                except Exception:
                                    pass

                                if speak:
                                    status(req_id, "tts")
                                    if tts_synth is None:
                                        tts_synth = _get_tts_synth_fn(bot, openai_api_key)
                                    wav, sr = await asyncio.to_thread(tts_synth, rendered_reply)
                                    await _ws_send_json(
                                        ws,
                                        {
                                            "type": "audio_wav",
                                            "req_id": req_id,
                                            "wav_base64": base64.b64encode(wav).decode(),
                                            "sr": sr,
                                        },
                                    )

                            await _ws_send_json(
                                ws,
                                {"type": "done", "req_id": req_id, "text": rendered_reply, "citations": citations},
                            )
                        else:
                            await _ws_send_json(
                                ws,
                                {"type": "done", "req_id": req_id, "text": final_text, "citations": citations},
                            )

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})

                        active_req_id = None
                        conv_id = None
                        audio_buf = bytearray()
                        accepting_audio = False

                    else:
                        await _ws_send_json(
                            ws,
                            {"type": "error", "req_id": req_id or None, "error": f"Unknown message type: {msg_type}"},
                        )

                elif "bytes" in msg and msg["bytes"] is not None:
                    # Be tolerant to stray/late audio frames (browser worklet flush, etc.)
                    if active_req_id is None or not accepting_audio:
                        continue
                    audio_buf.extend(msg["bytes"])
                else:
                    # ignore
                    pass

        except WebSocketDisconnect:
            return
        except RuntimeError:
            # Starlette can raise RuntimeError if receive() is called after disconnect was already processed.
            return
        except Exception as exc:
            try:
                await _ws_send_json(ws, {"type": "error", "error": f"Server error: {exc}"})
            except Exception:
                pass
            return

    def _parse_allowed_bot_ids(k) -> set[str]:
        try:
            ids = json.loads(getattr(k, "allowed_bot_ids_json", "") or "[]")
            if not isinstance(ids, list):
                return set()
            return {str(x) for x in ids if isinstance(x, str) and str(x).strip()}
        except Exception:
            return set()

    def _origin_allowed(k, origin: Optional[str]) -> bool:
        allowed = (getattr(k, "allowed_origins", "") or "").strip()
        if not allowed:
            return True
        origin_val = (origin or "").strip()
        allowset = {o.strip() for o in allowed.split(",") if o.strip()}
        return origin_val in allowset

    def _bot_allowed(k, bot_id: UUID) -> bool:
        allowset = _parse_allowed_bot_ids(k)
        if not allowset:
            return True
        return str(bot_id) in allowset

    def _require_public_conversation_access(
        *,
        session: Session,
        request: Request,
        conversation_id: UUID,
        key: str,
    ) -> tuple[Any, Any, Any]:
        """
        Returns (client_key, conversation, bot) for a public request.
        Enforces:
        - key is valid
        - origin is allowed (if present)
        - conversation is owned by this key
        - bot is allowed for this key
        """
        key_secret = (key or "").strip()
        if not key_secret:
            raise HTTPException(status_code=401, detail="Missing key")
        ck = verify_client_key(session, secret=key_secret)
        if not ck:
            raise HTTPException(status_code=401, detail="Invalid key")
        origin = request.headers.get("origin")
        if origin and (not _origin_allowed(ck, origin)):
            raise HTTPException(status_code=403, detail="Origin not allowed")
        try:
            conv = get_conversation(session, conversation_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conv.client_key_id != ck.id:
            raise HTTPException(status_code=403, detail="Conversation not accessible with this key")
        if not _bot_allowed(ck, conv.bot_id):
            raise HTTPException(status_code=403, detail="Bot not allowed for this key")
        bot = get_bot(session, conv.bot_id)
        return ck, conv, bot

    def _conversation_messages_payload(
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

    def _render_conversation_html(
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
            # Minimal query builder; values are pre-escaped for URL context.
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

    @app.get("/conversations/{conversation_id}")
    def public_conversation_transcript(
        conversation_id: UUID,
        request: Request,
        key: str = Query("", description="Client key secret (igx_...)"),
        include_tools: bool = Query(False, description="Include tool call/result messages"),
        include_system: bool = Query(False, description="Include system messages"),
        format: str = Query("html", description="html|json"),
        session: Session = Depends(get_session),
    ) -> Response:
        if not key.strip():
            # Allow SPA reloads on /conversations/{id} without a client key.
            if ui_index.exists() and _accepts_html(request.headers.get("accept") or ""):
                return FileResponse(str(ui_index))
            raise HTTPException(status_code=401, detail="Missing key")
        ck = verify_client_key(session, secret=key.strip())
        if not ck:
            raise HTTPException(status_code=401, detail="Invalid key")
        origin = request.headers.get("origin") if request else None
        if origin and (not _origin_allowed(ck, origin)):
            raise HTTPException(status_code=403, detail="Origin not allowed")

        try:
            conv = get_conversation(session, conversation_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conv.client_key_id != ck.id:
            raise HTTPException(status_code=403, detail="Conversation not accessible with this key")
        if not _bot_allowed(ck, conv.bot_id):
            raise HTTPException(status_code=403, detail="Bot not allowed for this key")

        payload = _conversation_messages_payload(
            session=session,
            conversation_id=conversation_id,
            include_tools=bool(include_tools),
            include_system=bool(include_system),
        )

        fmt = (format or "").strip().lower()
        if fmt == "json":
            return Response(content=json.dumps(payload, ensure_ascii=False), media_type="application/json")

        title = "Conversation Transcript"
        page = _render_conversation_html(
            title=title,
            conversation_id=conversation_id,
            key=key.strip(),
            include_tools=bool(include_tools),
            include_system=bool(include_system),
            payload=payload,
        )
        return HTMLResponse(content=page)

    def _is_path_within_root(root: Path, child: Path) -> bool:
        root_abs = root.resolve()
        child_abs = child.resolve()
        return child_abs == root_abs or root_abs in child_abs.parents

    def _data_agent_workspace_dir_for_conversation(session: Session, *, conversation_id: UUID) -> str:
        meta = _get_conversation_meta(session, conversation_id=conversation_id)
        da = _data_agent_meta(meta)
        return str(da.get("workspace_dir") or "").strip() or default_workspace_dir_for_conversation(conversation_id)

    def _should_hide_data_agent_path(rel: str, *, include_hidden: bool) -> bool:
        # Hide secrets/internal state from public clients.
        r = (rel or "").lstrip("/").strip()
        if not r:
            return False
        parts = [p for p in r.split("/") if p]
        if not parts:
            return False
        # Never expose Codex runtime directory.
        if parts[0] == ".codex":
            return True
        # Hide dotfiles/dirs unless explicitly requested.
        if (not include_hidden) and any(p.startswith(".") for p in parts):
            return True
        # Never expose auth or potential credential/config files.
        deny = {
            "auth.json",
            "AGENTS.md",
            "api_spec.json",
            "output_schema.json",
        }
        if parts[-1] in deny:
            return True
        return False

    def _resolve_data_agent_target(
        session: Session,
        *,
        conversation_id: UUID,
        path: str,
        include_hidden: bool,
    ) -> tuple[Path, str, Path]:
        workspace_dir = _data_agent_workspace_dir_for_conversation(session, conversation_id=conversation_id)
        root = Path(workspace_dir).resolve()
        req_rel = (path or "").lstrip("/").strip()
        target = (root / req_rel).resolve()
        if not _is_path_within_root(root, target):
            raise HTTPException(status_code=400, detail="Invalid path")
        if req_rel and _should_hide_data_agent_path(req_rel, include_hidden=bool(include_hidden)):
            raise HTTPException(status_code=403, detail="Path not allowed")
        return root, req_rel, target

    def _conversation_files_payload(
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
        root, req_rel, target = _resolve_data_agent_target(
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
            if _should_hide_data_agent_path(rel, include_hidden=bool(include_hidden)):
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

    @app.get("/conversations/{conversation_id}/files")
    def public_conversation_files(
        conversation_id: UUID,
        request: Request,
        key: str = Query("", description="Client key secret (igx_...)"),
        path: str = Query("", description="Directory path relative to the data-agent workspace"),
        recursive: bool = Query(False, description="List files recursively"),
        include_hidden: bool = Query(False, description="Include dotfiles (still blocks secrets like auth.json)"),
        format: str = Query("html", description="html|json"),
        session: Session = Depends(get_session),
    ) -> Response:
        ck, conv, bot = _require_public_conversation_access(
            session=session, request=request, conversation_id=conversation_id, key=key
        )
        base_key = (key or "").strip()
        payload = _conversation_files_payload(
            session=session,
            conversation_id=conversation_id,
            conv=conv,
            bot=bot,
            path=path,
            recursive=recursive,
            include_hidden=include_hidden,
            download_url_for=lambda rel: (
                f"/conversations/{conversation_id}/files/download?key={_url_quote(base_key)}&path={_url_quote(rel)}"
            ),
        )
        items = payload.get("items") or []
        max_items = int(payload.get("max_items") or 0)
        req_rel = str(payload.get("path") or "")

        fmt = (format or "").strip().lower()
        if fmt == "json":
            return Response(content=json.dumps(payload, ensure_ascii=False), media_type="application/json")

        # HTML table UI
        def _fmt_size(sz: Optional[int]) -> str:
            if sz is None:
                return ""
            n = float(sz)
            for unit in ["B", "KB", "MB", "GB"]:
                if n < 1024.0:
                    return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
                n /= 1024.0
            return f"{n:.1f} TB"

        def _q(**params: str) -> str:
            parts = []
            for k, v in params.items():
                parts.append(f"{k}={v}")
            return "&".join(parts)

        base_params = {"key": _url_quote((key or "").strip())}
        cur_path_q = _url_quote(req_rel)
        json_href = f"/conversations/{conversation_id}/files?{_q(**base_params, path=cur_path_q, recursive=('1' if recursive else '0'), include_hidden=('1' if include_hidden else '0'), format='json')}"
        rec_on = f"/conversations/{conversation_id}/files?{_q(**base_params, path=cur_path_q, recursive='1', include_hidden=('1' if include_hidden else '0'))}"
        rec_off = f"/conversations/{conversation_id}/files?{_q(**base_params, path=cur_path_q, recursive='0', include_hidden=('1' if include_hidden else '0'))}"
        hidden_on = f"/conversations/{conversation_id}/files?{_q(**base_params, path=cur_path_q, recursive=('1' if recursive else '0'), include_hidden='1')}"
        hidden_off = f"/conversations/{conversation_id}/files?{_q(**base_params, path=cur_path_q, recursive=('1' if recursive else '0'), include_hidden='0')}"
        transcript_href = f"/conversations/{conversation_id}?{_q(**base_params)}"

        rows: list[str] = []
        for it in items:
            rel = str(it.get("path") or "")
            is_dir = bool(it.get("is_dir"))
            mtime = str(it.get("mtime") or "")
            size = _fmt_size(it.get("size_bytes"))  # type: ignore[arg-type]
            dl = str(it.get("download_url") or "")
            name = rel
            href = ""
            if is_dir:
                href = f"/conversations/{conversation_id}/files?{_q(**base_params, path=_url_quote(rel), recursive='0', include_hidden=('1' if include_hidden else '0'))}"
            elif dl:
                href = dl
            link = f"<a href='{html.escape(href)}'>{html.escape(name)}</a>" if href else html.escape(name)
            kind = "dir" if is_dir else "file"
            rows.append(
                "<tr>"
                f"<td class='c-kind'>{html.escape(kind)}</td>"
                f"<td class='c-path'>{link}</td>"
                f"<td class='c-size'>{html.escape(size)}</td>"
                f"<td class='c-time'>{html.escape(mtime)}</td>"
                "</tr>"
            )

        page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Conversation Files</title>
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
    .c-kind {{ width: 80px; white-space: nowrap; color: var(--muted); font-size: 12px; }}
    .c-size {{ width: 110px; white-space: nowrap; color: var(--muted); font-size: 12px; }}
    .c-time {{ width: 260px; white-space: nowrap; color: var(--muted); font-size: 12px; }}
    .c-path a {{ color: var(--link); text-decoration: none; }}
    .c-path a:hover {{ text-decoration: underline; }}
    .footer {{ margin-top: 12px; font-size: 12px; color: var(--muted); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <div>
        <h1>Files for conversation {html.escape(str(conversation_id))}</h1>
        <div class="sub">Path: /{html.escape(req_rel)} • Items: {len(items)} (max {max_items})</div>
      </div>
      <div class="actions">
        <a class="btn" href="{html.escape(transcript_href)}">Transcript</a>
        <a class="btn primary" href="{html.escape(rec_on)}">Recursive on</a>
        <a class="btn" href="{html.escape(rec_off)}">Recursive off</a>
        <a class="btn" href="{html.escape(hidden_on)}">Show hidden</a>
        <a class="btn" href="{html.escape(hidden_off)}">Hide hidden</a>
        <a class="btn" href="{html.escape(json_href)}">JSON</a>
      </div>
    </div>
    <div class="panel">
      <table>
        <thead>
          <tr>
            <th>Type</th>
            <th>Path</th>
            <th>Size</th>
            <th>Modified (UTC)</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    <div class="footer">
      Hidden/secrets are filtered (e.g. <code>auth.json</code>, <code>AGENTS.md</code>, <code>.codex/</code>).
    </div>
  </div>
</body>
</html>
"""
        return HTMLResponse(content=page)

    @app.get("/conversations/{conversation_id}/files/download")
    def public_conversation_file_download(
        conversation_id: UUID,
        request: Request,
        key: str = Query("", description="Client key secret (igx_...)"),
        path: str = Query(..., description="File path relative to the data-agent workspace"),
        session: Session = Depends(get_session),
    ) -> FileResponse:
        _ck, _conv, _bot = _require_public_conversation_access(
            session=session, request=request, conversation_id=conversation_id, key=key
        )
        req_rel = (path or "").lstrip("/").strip()
        if not req_rel:
            raise HTTPException(status_code=400, detail="Missing path")
        root, req_rel, target = _resolve_data_agent_target(
            session,
            conversation_id=conversation_id,
            path=req_rel,
            include_hidden=False,
        )
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        mt, _ = mimetypes.guess_type(str(target))
        return FileResponse(
            path=str(target),
            media_type=mt or "application/octet-stream",
            filename=target.name,
        )

    @app.get("/api/conversations/{conversation_id}/files")
    def api_conversation_files(
        conversation_id: UUID,
        path: str = Query("", description="Directory path relative to the data-agent workspace"),
        recursive: bool = Query(False, description="List files recursively"),
        include_hidden: bool = Query(False, description="Include dotfiles (still blocks secrets like auth.json)"),
        session: Session = Depends(get_session),
    ) -> dict:
        conv = get_conversation(session, conversation_id)
        bot = get_bot(session, conv.bot_id)
        return _conversation_files_payload(
            session=session,
            conversation_id=conversation_id,
            conv=conv,
            bot=bot,
            path=path,
            recursive=recursive,
            include_hidden=include_hidden,
            download_url_for=lambda rel: f"/api/conversations/{conversation_id}/files/download?path={_url_quote(rel)}",
        )

    @app.get("/api/conversations/{conversation_id}/files/download")
    def api_conversation_file_download(
        conversation_id: UUID,
        path: str = Query(..., description="File path relative to the data-agent workspace"),
        session: Session = Depends(get_session),
    ) -> FileResponse:
        _ = get_conversation(session, conversation_id)
        req_rel = (path or "").lstrip("/").strip()
        if not req_rel:
            raise HTTPException(status_code=400, detail="Missing path")
        _root, _req_rel, target = _resolve_data_agent_target(
            session,
            conversation_id=conversation_id,
            path=req_rel,
            include_hidden=False,
        )
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        mt, _ = mimetypes.guess_type(str(target))
        return FileResponse(
            path=str(target),
            media_type=mt or "application/octet-stream",
            filename=target.name,
        )

    @app.post("/api/conversations/{conversation_id}/files/upload")
    async def api_conversation_files_upload(
        conversation_id: UUID,
        files: list[UploadFile] = File(...),
        session: Session = Depends(get_session),
    ) -> dict:
        conv = get_conversation(session, conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
        bot = get_bot(session, conv.bot_id)
        if not bot or not bool(getattr(bot, "enable_data_agent", False)):
            raise HTTPException(status_code=400, detail="Enable Isolated Workspace to upload files.")

        meta = _get_conversation_meta(session, conversation_id=conversation_id)
        try:
            _ensure_data_agent_container(session, bot=bot, conversation_id=conversation_id, meta_current=meta)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        workspace_dir = _initialize_data_agent_workspace(session, bot=bot, conversation_id=conversation_id, meta=meta)
        root = Path(workspace_dir).resolve()

        saved: list[str] = []
        for f in files:
            rel = _sanitize_upload_path(f.filename or "")
            if not rel:
                continue
            target = (root / rel).resolve()
            if not _is_path_within_root(root, target):
                raise HTTPException(status_code=400, detail="Invalid filename")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(rel)

        return {"ok": True, "files": saved, "workspace_dir": str(root)}

    @app.get("/api/conversations/{conversation_id}/host-actions")
    def api_conversation_host_actions(
        conversation_id: UUID,
        session: Session = Depends(get_session),
    ) -> dict:
        _ = get_conversation(session, conversation_id)
        stmt = select(HostAction).where(HostAction.conversation_id == conversation_id).order_by(HostAction.created_at.desc())
        items = list(session.exec(stmt))
        return {"items": [_host_action_payload(a) for a in items]}

    @app.post("/api/host-actions/{action_id}/run")
    def api_run_host_action(
        action_id: UUID,
        session: Session = Depends(get_session),
    ) -> dict:
        action = session.get(HostAction, action_id)
        if not action:
            raise HTTPException(status_code=404, detail="Host action not found")
        conv = get_conversation(session, action.conversation_id)
        req_bot_id = action.requested_by_bot_id or conv.bot_id
        bot = get_bot(session, req_bot_id)
        if not bool(getattr(bot, "enable_host_actions", False)):
            raise HTTPException(status_code=400, detail="Host actions are disabled for this assistant.")
        if action.action_type == "run_shell" and not bool(getattr(bot, "enable_host_shell", False)):
            raise HTTPException(status_code=400, detail="Shell commands are disabled for this assistant.")

        tool_result = _execute_host_action_and_update(session, action=action)
        tool_result_msg = add_message_with_metrics(
            session,
            conversation_id=action.conversation_id,
            role="tool",
            content=json.dumps({"tool": "request_host_action", "result": tool_result}, ensure_ascii=False),
            sender_bot_id=bot.id,
            sender_name=bot.name,
        )
        _mirror_group_message(session, conv=conv, msg=tool_result_msg)

        return _host_action_payload(action)

    async def _public_send_done(
        ws: WebSocket, *, req_id: str, text: str, metrics: dict, citations: Optional[list[dict]] = None
    ) -> None:
        payload = {"type": "done", "req_id": req_id, "text": text, "metrics": metrics}
        if citations:
            payload["citations"] = citations
        await _ws_send_json(ws, payload)

    async def _public_send_interim(ws: WebSocket, *, req_id: str, kind: str, text: str) -> None:
        t = (text or "").strip()
        if not t:
            return
        if kind == "wait":
            now = time.time()
            cache = getattr(_public_send_interim, "_wait_cache", None)
            if cache is None:
                cache = {}
                setattr(_public_send_interim, "_wait_cache", cache)
            key = f"{id(ws)}:{req_id}:{t}"
            last_ts = cache.get(key)
            if isinstance(last_ts, float) and (now - last_ts) < 45.0:
                return
            cache[key] = now
            if len(cache) > 1024:
                for k, ts in list(cache.items()):
                    if not isinstance(ts, float) or (now - ts) > 300.0:
                        cache.pop(k, None)
        await _ws_send_json(ws, {"type": "interim", "req_id": req_id, "kind": kind, "text": t})

    async def _public_send_greeting(
        *,
        ws: WebSocket,
        req_id: str,
        bot: Bot,
        conv_id: UUID,
        provider: str,
        llm_api_key: str,
    ) -> tuple[str, dict]:
        greeting_text = (bot.start_message_text or "").strip()
        llm_ttfb_ms: Optional[int] = None
        llm_total_ms: Optional[int] = None
        input_tokens_est: int = 0
        output_tokens_est: int = 0
        cost_usd_est: float = 0.0
        sent_delta = False

        if bot.start_message_mode == "static" and greeting_text:
            pass
        else:
            llm = _build_llm_client(bot=bot, api_key=llm_api_key)
            msgs = [
                Message(role="system", content=render_template(bot.system_prompt, ctx={"meta": {}})),
                Message(role="user", content=_make_start_message_instruction(bot)),
            ]
            t0 = time.time()
            first = None
            parts: list[str] = []
            async for d in _aiter_from_blocking_iterator(lambda: llm.stream_text(messages=msgs)):
                d = str(d or "")
                if first is None:
                    first = time.time()
                if d:
                    parts.append(d)
                    await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})
                    sent_delta = True
            t1 = time.time()
            greeting_text = "".join(parts).strip() or greeting_text
            if first is not None:
                llm_ttfb_ms = int(round((first - t0) * 1000.0))
            llm_total_ms = int(round((t1 - t0) * 1000.0))

            input_tokens_est = int(estimate_messages_tokens(msgs, bot.openai_model) or 0)
            output_tokens_est = int(estimate_text_tokens(greeting_text, bot.openai_model) or 0)
            with Session(engine) as session:
                price = _get_model_price(session, provider=provider, model=bot.openai_model)
            cost_usd_est = float(
                estimate_cost_usd(model_price=price, input_tokens=input_tokens_est, output_tokens=output_tokens_est) or 0.0
            )

        if not greeting_text:
            greeting_text = "Hi! How can I help you today?"

        if not sent_delta:
            await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": greeting_text})

        with Session(engine) as session:
            add_message_with_metrics(
                session,
                conversation_id=conv_id,
                role="assistant",
                content=greeting_text,
                input_tokens_est=input_tokens_est or None,
                output_tokens_est=output_tokens_est or None,
                cost_usd_est=cost_usd_est or None,
                llm_ttfb_ms=llm_ttfb_ms,
                llm_total_ms=llm_total_ms,
            )
            update_conversation_metrics(
                session,
                conversation_id=conv_id,
                add_input_tokens_est=input_tokens_est,
                add_output_tokens_est=output_tokens_est,
                add_cost_usd_est=cost_usd_est,
                last_asr_ms=None,
                last_llm_ttfb_ms=llm_ttfb_ms,
                last_llm_total_ms=llm_total_ms,
                last_tts_first_audio_ms=None,
                last_total_ms=None,
            )

        metrics = {
            "model": bot.openai_model,
            "input_tokens_est": input_tokens_est,
            "output_tokens_est": output_tokens_est,
            "cost_usd_est": cost_usd_est,
            "llm_ttfb_ms": llm_ttfb_ms,
            "llm_total_ms": llm_total_ms,
        }
        return greeting_text, metrics

    @app.websocket("/public/v1/ws/bots/{bot_id}/chat")
    async def public_chat_ws(bot_id: UUID, ws: WebSocket) -> None:
        if not _basic_auth_ok(_ws_auth_header(ws)):
            await ws.accept()
            await _ws_send_json(ws, {"type": "error", "error": "Unauthorized"})
            await ws.close(code=4401)
            return
        await ws.accept()

        key_secret = (ws.query_params.get("key") or "").strip()
        external_id = (ws.query_params.get("user_conversation_id") or "").strip()
        if not key_secret or not external_id:
            await _ws_send_json(ws, {"type": "error", "error": "Missing key or user_conversation_id"})
            await ws.close(code=4400)
            return

        origin = ws.headers.get("origin")
        conv_id: Optional[UUID] = None
        with Session(engine) as session:
            ck = verify_client_key(session, secret=key_secret)
            if not ck:
                await _ws_send_json(ws, {"type": "error", "error": "Invalid client key"})
                await ws.close(code=4401)
                return
            if not _origin_allowed(ck, origin):
                await _ws_send_json(ws, {"type": "error", "error": "Origin not allowed"})
                await ws.close(code=4403)
                return
            if not _bot_allowed(ck, bot_id):
                await _ws_send_json(ws, {"type": "error", "error": "Bot not allowed for this key"})
                await ws.close(code=4403)
                return
            # Create (or load) the conversation immediately on connect so we can prewarm the Isolated Workspace
            # as soon as the conversation exists (before the first user message).
            try:
                bot = get_bot(session, bot_id)
                conv = get_or_create_conversation_by_external_id(
                    session,
                    bot_id=bot.id,
                    test_flag=False,
                    client_key_id=ck.id,
                    external_id=external_id,
                )
                conv_id = conv.id
            except Exception:
                conv_id = None

        if conv_id is not None:
            asyncio.create_task(_kickoff_data_agent_container_if_enabled(bot_id=bot_id, conversation_id=conv_id))

        try:
            while True:
                raw = await ws.receive_text()
                try:
                    payload = json.loads(raw)
                except Exception:
                    await _ws_send_json(ws, {"type": "error", "error": "Invalid JSON"})
                    continue

                msg_type = str(payload.get("type") or "")
                req_id = str(payload.get("req_id") or "")
                if not req_id:
                    await _ws_send_json(ws, {"type": "error", "error": "Missing req_id"})
                    continue

                if msg_type == "start":
                    with Session(engine) as session:
                        bot = get_bot(session, bot_id)
                        ck = verify_client_key(session, secret=key_secret)
                        if not ck:
                            raise HTTPException(status_code=401, detail="Invalid client key")
                        conv = get_or_create_conversation_by_external_id(
                            session, bot_id=bot.id, test_flag=False, client_key_id=ck.id, external_id=external_id
                        )
                        conv_id = conv.id

                        await _ws_send_json(
                            ws,
                            {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)},
                        )
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})

                        if len(list_messages(session, conversation_id=conv_id)) == 0:
                            need_llm = not (
                                bot.start_message_mode == "static" and (bot.start_message_text or "").strip()
                            )
                            provider = _llm_provider_for_bot(bot)
                            llm_api_key = ""
                            if need_llm:
                                provider, llm_api_key, _ = _require_llm_client(session, bot=bot)
                            text, metrics = await _public_send_greeting(
                                ws=ws,
                                req_id=req_id,
                                bot=bot,
                                conv_id=conv_id,
                                provider=provider,
                                llm_api_key=llm_api_key,
                            )
                            await _public_send_done(ws, req_id=req_id, text=text, metrics=metrics)
                        else:
                            await _public_send_done(ws, req_id=req_id, text="", metrics={})

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                    continue

                if msg_type == "chat":
                    user_text = str(payload.get("text") or "").strip()
                    if not user_text:
                        await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": "Empty text"})
                        continue

                    with Session(engine) as session:
                        bot = get_bot(session, bot_id)
                        ck = verify_client_key(session, secret=key_secret)
                        if not ck:
                            raise HTTPException(status_code=401, detail="Invalid client key")
                        conv = get_or_create_conversation_by_external_id(
                            session, bot_id=bot.id, test_flag=False, client_key_id=ck.id, external_id=external_id
                        )
                        conv_id = conv.id
                        await _ws_send_json(
                            ws, {"type": "conversation", "req_id": req_id, "conversation_id": str(conv_id), "id": str(conv_id)}
                        )

                        provider, llm_api_key, llm = _require_llm_client(session, bot=bot)

                        add_message_with_metrics(session, conversation_id=conv_id, role="user", content=user_text)
                        loop = asyncio.get_running_loop()

                        def _status_cb(stage: str) -> None:
                            asyncio.run_coroutine_threadsafe(
                                _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": stage}), loop
                            )

                        history = await _build_history_budgeted_async(
                            bot_id=bot.id,
                            conversation_id=conv_id,
                            llm_api_key=llm_api_key,
                            status_cb=_status_cb,
                        )
                        tools_defs = _build_tools_for_bot(session, bot.id)

                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                        t0 = time.time()
                        first_token_ts: Optional[float] = None
                        full_text_parts: list[str] = []
                        tool_calls: list[ToolCall] = []
                        citations: list[dict[str, Any]] = []

                        async for ev in _aiter_from_blocking_iterator(
                            lambda: llm.stream_text_or_tool(messages=history, tools=tools_defs)
                        ):
                            if isinstance(ev, ToolCall):
                                tool_calls.append(ev)
                                continue
                            if isinstance(ev, CitationEvent):
                                citations.extend(ev.citations)
                                continue
                            d = str(ev)
                            if d:
                                if first_token_ts is None:
                                    first_token_ts = time.time()
                                full_text_parts.append(d)
                                await _ws_send_json(ws, {"type": "text_delta", "req_id": req_id, "delta": d})

                        llm_end_ts = time.time()
                        rendered_reply = "".join(full_text_parts).strip()
                        citations_json = json.dumps(citations, ensure_ascii=False) if citations else "[]"

                        llm_ttfb_ms: Optional[int] = None
                        if first_token_ts is not None:
                            llm_ttfb_ms = int(round((first_token_ts - t0) * 1000.0))
                        elif tool_calls and tool_calls[0].first_event_ts is not None:
                            llm_ttfb_ms = int(round((tool_calls[0].first_event_ts - t0) * 1000.0))
                        llm_total_ms = int(round((llm_end_ts - t0) * 1000.0))

                        if tool_calls:
                            meta_current = _get_conversation_meta(session, conversation_id=conv_id)
                            disabled_tools = _disabled_tool_names(bot)
                            final = ""
                            needs_followup_llm = False
                            tool_failed = False
                            followup_streamed = False

                            for tc in tool_calls:
                                tool_name = tc.name
                                if tool_name == "set_variable":
                                    tool_name = "set_metadata"

                                tool_args = json.loads(tc.arguments_json or "{}")
                                if not isinstance(tool_args, dict):
                                    tool_args = {}

                                add_message_with_metrics(
                                    session,
                                    conversation_id=conv_id,
                                    role="tool",
                                    content=json.dumps({"tool": tool_name, "arguments": tool_args}, ensure_ascii=False),
                                )

                                next_reply = str(tool_args.get("next_reply") or "").strip()
                                wait_reply = str(tool_args.get("wait_reply") or "").strip()
                                follow_up = _parse_follow_up_flag(tool_args.get("follow_up")) or _parse_follow_up_flag(
                                    tool_args.get("followup")
                                )
                                if (
                                    tool_name in {"request_host_action", "capture_screenshot"}
                                    and "follow_up" not in tool_args
                                    and "followup" not in tool_args
                                ):
                                    follow_up = True
                                if tool_name in {"request_host_action", "capture_screenshot"}:
                                    next_reply = ""
                                raw_args = tool_args.get("args")
                                if isinstance(raw_args, dict):
                                    patch = dict(raw_args)
                                else:
                                    patch = dict(tool_args)
                                    patch.pop("next_reply", None)
                                    patch.pop("wait_reply", None)
                                    patch.pop("follow_up", None)
                                    patch.pop("followup", None)
                                    patch.pop("args", None)

                                tool_cfg: IntegrationTool | None = None
                                response_json: Any | None = None
                                if tool_name in disabled_tools:
                                    tool_result = {
                                        "ok": False,
                                        "error": {"message": f"Tool '{tool_name}' is disabled for this bot."},
                                    }
                                    tool_failed = True
                                    needs_followup_llm = True
                                    final = ""
                                elif tool_name == "set_metadata":
                                    new_meta = merge_conversation_metadata(session, conversation_id=conv_id, patch=patch)
                                    tool_result = {"ok": True, "updated": patch, "metadata": new_meta}
                                elif tool_name == "web_search":
                                    tool_result = {
                                        "ok": False,
                                        "error": {"message": "web_search runs inside the model; no server tool is available."},
                                    }
                                    tool_failed = True
                                    needs_followup_llm = True
                                    final = ""
                                elif tool_name == "http_request":
                                    tool_result = await asyncio.to_thread(
                                        _execute_http_request_tool, tool_args=patch, meta=meta_current
                                    )
                                    tool_failed = not bool(tool_result.get("ok", False))
                                    needs_followup_llm = True
                                    final = ""
                                elif tool_name == "capture_screenshot":
                                    if not bool(getattr(bot, "enable_host_actions", False)):
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": "Host actions are disabled for this bot."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    elif not bool(getattr(bot, "enable_host_shell", False)):
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": "Shell commands are disabled for this bot."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        try:
                                            rel_path, target = _prepare_screenshot_target(conv)
                                        except Exception as exc:
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": str(exc) or "Invalid screenshot path"},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            final = ""
                                        else:
                                            ok_cmd, cmd_or_err = _screencapture_command(target)
                                            if not ok_cmd:
                                                tool_result = {"ok": False, "error": {"message": cmd_or_err}}
                                                tool_failed = True
                                                needs_followup_llm = True
                                                final = ""
                                            else:
                                                action = _create_host_action(
                                                    session,
                                                    conv=conv,
                                                    bot=bot,
                                                    action_type="run_shell",
                                                    payload={"command": cmd_or_err},
                                                )
                                                if _host_action_requires_approval(bot):
                                                    tool_result = _build_host_action_tool_result(action, ok=True)
                                                    tool_result["path"] = rel_path
                                                    if follow_up:
                                                        final = ""
                                                        needs_followup_llm = True
                                                    else:
                                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                                        if candidate:
                                                            final = candidate
                                                            needs_followup_llm = False
                                                        else:
                                                            final = (
                                                                "Approve the screenshot capture in the Action Queue, then ask me to analyze it."
                                                            )
                                                            needs_followup_llm = False
                                                else:
                                                    tool_result = await _execute_host_action_and_update_async(
                                                        session, action=action
                                                    )
                                                    tool_failed = not bool(tool_result.get("ok", False))
                                                    if tool_failed:
                                                        needs_followup_llm = True
                                                        final = ""
                                                    else:
                                                        ok, summary_text = _summarize_image_file(
                                                            session,
                                                            bot=bot,
                                                            image_path=target,
                                                            prompt=str(patch.get("prompt") or "").strip(),
                                                        )
                                                        if not ok:
                                                            tool_result["summary_error"] = summary_text
                                                            tool_failed = True
                                                            needs_followup_llm = True
                                                            final = ""
                                                        else:
                                                            tool_result["summary"] = summary_text
                                                            tool_result["path"] = rel_path
                                                            if follow_up:
                                                                final = ""
                                                                needs_followup_llm = True
                                                            else:
                                                                candidate = _render_with_meta(next_reply, meta_current).strip()
                                                                if candidate:
                                                                    final = candidate
                                                                    needs_followup_llm = False
                                                                else:
                                                                    final = summary_text
                                                                    needs_followup_llm = False
                                elif tool_name == "summarize_screenshot":
                                    ok, summary_text, rel_path = _summarize_screenshot(
                                        session,
                                        conv=conv,
                                        bot=bot,
                                        path=str(patch.get("path") or "").strip(),
                                        prompt=str(patch.get("prompt") or "").strip(),
                                    )
                                    if not ok:
                                        tool_result = {"ok": False, "error": {"message": summary_text}}
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        tool_result = {"ok": True, "summary": summary_text, "path": rel_path}
                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                        if candidate:
                                            final = candidate
                                            needs_followup_llm = False
                                        else:
                                            final = summary_text
                                            needs_followup_llm = False
                                elif tool_name == "request_host_action":
                                    if not bool(getattr(bot, "enable_host_actions", False)):
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": "Host actions are disabled for this bot."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        try:
                                            action_type, payload = _parse_host_action_args(patch)
                                        except Exception as exc:
                                            tool_result = {"ok": False, "error": {"message": str(exc) or "Invalid host action"}}
                                            tool_failed = True
                                            needs_followup_llm = True
                                            final = ""
                                        else:
                                            if action_type == "run_shell" and not bool(getattr(bot, "enable_host_shell", False)):
                                                tool_result = {
                                                    "ok": False,
                                                    "error": {"message": "Shell commands are disabled for this bot."},
                                                }
                                                tool_failed = True
                                                needs_followup_llm = True
                                                final = ""
                                            else:
                                                action = _create_host_action(
                                                    session,
                                                    conv=conv,
                                                    bot=bot,
                                                    action_type=action_type,
                                                    payload=payload,
                                                )
                                                if _host_action_requires_approval(bot):
                                                    tool_result = _build_host_action_tool_result(action, ok=True)
                                                    if follow_up:
                                                        final = ""
                                                        needs_followup_llm = True
                                                    else:
                                                        candidate = _render_with_meta(next_reply, meta_current).strip()
                                                        if candidate:
                                                            final = candidate
                                                            needs_followup_llm = False
                                                        else:
                                                            needs_followup_llm = True
                                                            final = ""
                                                else:
                                                    tool_result = await _execute_host_action_and_update_async(
                                                        session, action=action
                                                    )
                                                    tool_failed = not bool(tool_result.get("ok", False))
                                                    candidate = _render_with_meta(next_reply, meta_current).strip()
                                                    if follow_up and not tool_failed:
                                                        needs_followup_llm = True
                                                        final = ""
                                                    elif candidate and not tool_failed:
                                                        final = candidate
                                                        needs_followup_llm = False
                                                    else:
                                                        needs_followup_llm = True
                                                        final = ""
                                elif tool_name == "give_command_to_data_agent":
                                    if not bool(getattr(bot, "enable_data_agent", False)):
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": "Isolated Workspace is disabled for this bot."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    elif not docker_available():
                                        tool_result = {
                                            "ok": False,
                                            "error": {"message": "Docker is not available. Install Docker to use Isolated Workspace."},
                                        }
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        what_to_do = str(patch.get("what_to_do") or "").strip()
                                        if not what_to_do:
                                            tool_result = {
                                                "ok": False,
                                                "error": {"message": "Missing required tool arg: what_to_do"},
                                            }
                                            tool_failed = True
                                            needs_followup_llm = True
                                            final = ""
                                        else:
                                            try:
                                                logger.info(
                                                    "Isolated Workspace tool: start conv=%s bot=%s what_to_do=%s",
                                                    conv_id,
                                                    bot_id,
                                                    (what_to_do[:200] + "…") if len(what_to_do) > 200 else what_to_do,
                                                )
                                                da = _data_agent_meta(meta_current)
                                                workspace_dir = (
                                                    str(da.get("workspace_dir") or "").strip()
                                                    or default_workspace_dir_for_conversation(conv_id)
                                                )
                                                container_id = str(da.get("container_id") or "").strip()
                                                session_id = str(da.get("session_id") or "").strip()
                                                auth_json_raw = getattr(bot, "data_agent_auth_json", "") or "{}"
                                                git_token = (
                                                    _get_git_token_plaintext(session, provider="github")
                                                    if _git_auth_mode(auth_json_raw) == "token"
                                                    else ""
                                                )

                                                if not container_id:
                                                    api_key = _get_openai_api_key_for_bot(session, bot=bot)
                                                    if not api_key:
                                                        raise RuntimeError(
                                                            "No OpenAI API key configured for this bot (needed for Isolated Workspace)."
                                                        )
                                                    container_id = await asyncio.to_thread(
                                                        ensure_conversation_container,
                                                        conversation_id=conv_id,
                                                        workspace_dir=workspace_dir,
                                                        openai_api_key=api_key,
                                                        git_token=git_token,
                                                        auth_json=auth_json_raw,
                                                    )
                                                    meta_current = merge_conversation_metadata(
                                                        session,
                                                        conversation_id=conv_id,
                                                        patch={
                                                            "data_agent.container_id": container_id,
                                                            "data_agent.workspace_dir": workspace_dir,
                                                        },
                                                    )

                                                ctx = _build_data_agent_conversation_context(
                                                    session,
                                                    bot=bot,
                                                    conversation_id=conv_id,
                                                    meta=meta_current,
                                                )
                                                api_spec_text = getattr(bot, "data_agent_api_spec_text", "") or ""
                                                auth_json = _merge_git_token_auth(auth_json_raw, git_token)
                                                sys_prompt = (
                                                    (getattr(bot, "data_agent_system_prompt", "") or "").strip()
                                                    or DEFAULT_DATA_AGENT_SYSTEM_PROMPT
                                                )
                                                def _emit_tool_progress(text: str) -> None:
                                                    t = (text or "").strip()
                                                    if not t:
                                                        return
                                                    asyncio.run_coroutine_threadsafe(
                                                        _public_send_interim(ws, req_id=req_id, kind="tool", text=t),
                                                        loop,
                                                    )

                                                task = asyncio.create_task(
                                                    asyncio.to_thread(
                                                        run_data_agent,
                                                        conversation_id=conv_id,
                                                        container_id=container_id,
                                                        session_id=session_id,
                                                        workspace_dir=workspace_dir,
                                                        api_spec_text=api_spec_text,
                                                        auth_json=auth_json,
                                                        system_prompt=sys_prompt,
                                                        conversation_context=ctx,
                                                        what_to_do=what_to_do,
                                                        on_stream=_emit_tool_progress,
                                                    )
                                                )
                                                if wait_reply:
                                                    await _public_send_interim(
                                                        ws, req_id=req_id, kind="wait", text=wait_reply
                                                    )
                                                last_wait = time.time()
                                                while not task.done():
                                                    if wait_reply and (time.time() - last_wait) >= 10.0:
                                                        await _public_send_interim(
                                                            ws, req_id=req_id, kind="wait", text=wait_reply
                                                        )
                                                        last_wait = time.time()
                                                    await asyncio.sleep(0.2)
                                                da_res = await task

                                                if da_res.session_id and da_res.session_id != session_id:
                                                    meta_current = merge_conversation_metadata(
                                                        session,
                                                        conversation_id=conv_id,
                                                        patch={"data_agent.session_id": da_res.session_id},
                                                    )
                                                logger.info(
                                                    "Isolated Workspace tool: done conv=%s ok=%s container_id=%s session_id=%s output_file=%s error=%s",
                                                    conv_id,
                                                    bool(da_res.ok),
                                                    da_res.container_id,
                                                    da_res.session_id,
                                                    da_res.output_file,
                                                    da_res.error,
                                                )
                                                tool_result = {
                                                    "ok": bool(da_res.ok),
                                                    "result_text": da_res.result_text,
                                                    "data_agent_container_id": da_res.container_id,
                                                    "data_agent_session_id": da_res.session_id,
                                                    "data_agent_output_file": da_res.output_file,
                                                    "data_agent_debug_file": da_res.debug_file,
                                                    "error": da_res.error,
                                                }
                                                tool_failed = not bool(da_res.ok)
                                                if (
                                                    bool(getattr(bot, "data_agent_return_result_directly", False))
                                                    and bool(da_res.ok)
                                                    and str(da_res.result_text or "").strip()
                                                ):
                                                    needs_followup_llm = False
                                                    final = str(da_res.result_text or "").strip()
                                                else:
                                                    needs_followup_llm = True
                                                    final = ""
                                            except Exception as exc:
                                                logger.exception("Isolated Workspace tool failed conv=%s bot=%s", conv_id, bot_id)
                                                tool_result = {"ok": False, "error": {"message": str(exc)}}
                                                tool_failed = True
                                                needs_followup_llm = True
                                                final = ""
                                else:
                                    tool_cfg = get_integration_tool_by_name(session, bot_id=bot.id, name=tool_name)
                                    if not tool_cfg:
                                        raise RuntimeError(f"Unknown tool: {tool_name}")
                                    if not bool(getattr(tool_cfg, "enabled", True)):
                                        response_json = {
                                            "__tool_args_error__": {
                                                "missing": [],
                                                "message": f"Tool '{tool_name}' is disabled for this bot.",
                                            }
                                        }
                                    else:
                                        task = asyncio.create_task(
                                            asyncio.to_thread(
                                                _execute_integration_http, tool=tool_cfg, meta=meta_current, tool_args=patch
                                            )
                                        )
                                        if wait_reply:
                                            await _public_send_interim(ws, req_id=req_id, kind="wait", text=wait_reply)
                                        while True:
                                            try:
                                                response_json = await asyncio.wait_for(asyncio.shield(task), timeout=60.0)
                                                break
                                            except asyncio.TimeoutError:
                                                if wait_reply:
                                                    await _public_send_interim(
                                                        ws, req_id=req_id, kind="wait", text=wait_reply
                                                    )
                                                continue
                                    if isinstance(response_json, dict) and "__tool_args_error__" in response_json:
                                        err = response_json["__tool_args_error__"] or {}
                                        tool_result = {"ok": False, "error": err}
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    elif isinstance(response_json, dict) and "__http_error__" in response_json:
                                        err = response_json["__http_error__"] or {}
                                        tool_result = {"ok": False, "error": err}
                                        tool_failed = True
                                        needs_followup_llm = True
                                        final = ""
                                    else:
                                        pagination_info = None
                                        if isinstance(response_json, dict) and "__igx_pagination__" in response_json:
                                            pagination_info = response_json.pop("__igx_pagination__", None)
                                        if bool(getattr(tool_cfg, "use_codex_response", False)):
                                            # Avoid bloating LLM-visible conversation metadata in Codex mode.
                                            tool_result = {"ok": True}
                                            new_meta = meta_current
                                        else:
                                            mapped = _apply_response_mapper(
                                                mapper_json=tool_cfg.response_mapper_json,
                                                response_json=response_json,
                                                meta=meta_current,
                                                tool_args=patch,
                                            )
                                            new_meta = merge_conversation_metadata(
                                                session, conversation_id=conv_id, patch=mapped
                                            )
                                            meta_current = new_meta
                                            tool_result = {"ok": True, "updated": mapped, "metadata": new_meta}
                                        if pagination_info:
                                            tool_result["pagination"] = pagination_info

                                        # Optional Codex HTTP agent (post-process the raw response JSON).
                                        # Static reply (if configured) takes priority.
                                        static_preview = ""
                                        if (tool_cfg.static_reply_template or "").strip():
                                            try:
                                                static_preview = _render_static_reply(
                                                    template_text=tool_cfg.static_reply_template,
                                                    meta=new_meta or meta_current,
                                                    response_json=response_json,
                                                    tool_args=patch,
                                                ).strip()
                                            except Exception:
                                                static_preview = ""
                                        if (not static_preview) and bool(getattr(tool_cfg, "use_codex_response", False)):
                                            fields_required = str(patch.get("fields_required") or "").strip()
                                            what_to_search_for = str(patch.get("what_to_search_for") or "").strip()
                                            if not fields_required:
                                                fields_required = what_to_search_for
                                            why_api_was_called = str(patch.get("why_api_was_called") or "").strip()
                                            if not why_api_was_called:
                                                why_api_was_called = str(patch.get("why_to_search_for") or "").strip()
                                            if not fields_required or not why_api_was_called:
                                                tool_result["codex_ok"] = False
                                                tool_result["codex_error"] = "Missing fields_required / why_api_was_called."
                                            else:
                                                fields_required_for_codex = fields_required
                                                if what_to_search_for:
                                                    fields_required_for_codex = (
                                                        f"{fields_required_for_codex}\n\n(what_to_search_for) {what_to_search_for}"
                                                    )
                                                did_postprocess = False
                                                postprocess_python = str(getattr(tool_cfg, "postprocess_python", "") or "").strip()
                                                if postprocess_python:
                                                    py_task = asyncio.create_task(
                                                        asyncio.to_thread(
                                                            run_python_postprocessor,
                                                            python_code=postprocess_python,
                                                            payload={
                                                                "response_json": response_json,
                                                                "meta": new_meta or meta_current,
                                                                "args": patch,
                                                                "fields_required": fields_required,
                                                                "why_api_was_called": why_api_was_called,
                                                            },
                                                            timeout_s=60,
                                                        )
                                                    )
                                                    if wait_reply:
                                                        await _public_send_interim(
                                                            ws, req_id=req_id, kind="wait", text=wait_reply
                                                        )
                                                    last_wait = time.time()
                                                    wait_interval_s = 15.0
                                                    while not py_task.done():
                                                        now = time.time()
                                                        if wait_reply and (now - last_wait) >= wait_interval_s:
                                                            await _public_send_interim(
                                                                ws, req_id=req_id, kind="wait", text=wait_reply
                                                            )
                                                            last_wait = now
                                                        await asyncio.sleep(0.2)
                                                    try:
                                                        py_res = await py_task
                                                        tool_result["python_ok"] = bool(getattr(py_res, "ok", False))
                                                        tool_result["python_duration_ms"] = int(
                                                            getattr(py_res, "duration_ms", 0) or 0
                                                        )
                                                        if getattr(py_res, "error", None):
                                                            tool_result["python_error"] = str(getattr(py_res, "error"))
                                                        if getattr(py_res, "stderr", None):
                                                            tool_result["python_stderr"] = str(getattr(py_res, "stderr"))
                                                        if py_res.ok:
                                                            did_postprocess = True
                                                            tool_result["postprocess_mode"] = "python"
                                                            tool_result["codex_ok"] = True
                                                            tool_result["codex_result_text"] = str(
                                                                getattr(py_res, "result_text", "") or ""
                                                            )
                                                            mp = getattr(py_res, "metadata_patch", None)
                                                            if isinstance(mp, dict) and mp:
                                                                try:
                                                                    meta_current = merge_conversation_metadata(
                                                                        session,
                                                                        conversation_id=conv_id,
                                                                        patch=mp,
                                                                    )
                                                                    tool_result["python_metadata_patch"] = mp
                                                                except Exception:
                                                                    pass
                                                            try:
                                                                append_saved_run_index(
                                                                    conversation_id=str(conv_id),
                                                                    event={
                                                                        "kind": "integration_python_postprocess",
                                                                        "tool_name": tool_name,
                                                                        "req_id": req_id,
                                                                        "python_ok": tool_result.get("python_ok"),
                                                                        "python_duration_ms": tool_result.get(
                                                                            "python_duration_ms"
                                                                        ),
                                                                    },
                                                                )
                                                            except Exception:
                                                                pass
                                                    except Exception as exc:
                                                        tool_result["python_ok"] = False
                                                        tool_result["python_error"] = str(exc)

                                                if not did_postprocess:
                                                    codex_model = (
                                                        (getattr(bot, "codex_model", "") or "gpt-5.1-codex-mini").strip()
                                                        or "gpt-5.1-codex-mini"
                                                    )
                                                    progress_q: "queue.Queue[str]" = queue.Queue()

                                                    def _progress(s: str) -> None:
                                                        try:
                                                            progress_q.put_nowait(str(s))
                                                        except Exception:
                                                            return

                                                    agent_task = asyncio.create_task(
                                                        asyncio.to_thread(
                                                            run_codex_http_agent_one_shot,
                                                            api_key=api_key or "",
                                                            model=codex_model,
                                                            response_json=response_json,
                                                            fields_required=fields_required_for_codex,
                                                            why_api_was_called=why_api_was_called,
                                                            response_schema_json=getattr(tool_cfg, "response_schema_json", "")
                                                            or "",
                                                            conversation_id=str(conv_id) if conv_id is not None else None,
                                                            req_id=req_id,
                                                            tool_codex_prompt=getattr(tool_cfg, "codex_prompt", "") or "",
                                                            progress_fn=_progress,
                                                        )
                                                    )
                                                    if wait_reply:
                                                        await _public_send_interim(
                                                            ws, req_id=req_id, kind="wait", text=wait_reply
                                                        )
                                                    last_wait = time.time()
                                                    last_progress = last_wait
                                                    wait_interval_s = 15.0
                                                    while not agent_task.done():
                                                        try:
                                                            while True:
                                                                p = progress_q.get_nowait()
                                                                if p:
                                                                    await _public_send_interim(
                                                                        ws, req_id=req_id, kind="progress", text=p
                                                                    )
                                                                    last_progress = time.time()
                                                        except queue.Empty:
                                                            pass
                                                        now = time.time()
                                                        if (
                                                            wait_reply
                                                            and (now - last_wait) >= wait_interval_s
                                                            and (now - last_progress) >= wait_interval_s
                                                        ):
                                                            await _public_send_interim(
                                                                ws, req_id=req_id, kind="wait", text=wait_reply
                                                            )
                                                            last_wait = now
                                                        await asyncio.sleep(0.2)
                                                    try:
                                                        agent_res = await agent_task
                                                        tool_result["postprocess_mode"] = "codex"
                                                        tool_result["codex_ok"] = bool(getattr(agent_res, "ok", False))
                                                        tool_result["codex_output_dir"] = getattr(agent_res, "output_dir", "")
                                                        tool_result["codex_output_file"] = getattr(
                                                            agent_res, "result_text_path", ""
                                                        )
                                                        tool_result["codex_debug_file"] = getattr(agent_res, "debug_json_path", "")
                                                        tool_result["codex_result_text"] = getattr(agent_res, "result_text", "")
                                                        tool_result["codex_stop_reason"] = getattr(agent_res, "stop_reason", "")
                                                        tool_result["codex_continue_reason"] = getattr(
                                                            agent_res, "continue_reason", ""
                                                        )
                                                        tool_result["codex_next_step"] = getattr(agent_res, "next_step", "")
                                                        saved_input_json_path = getattr(agent_res, "input_json_path", "")
                                                        saved_schema_json_path = getattr(agent_res, "schema_json_path", "")
                                                        err = getattr(agent_res, "error", None)
                                                        if err:
                                                            tool_result["codex_error"] = str(err)
                                                    except Exception as exc:
                                                        tool_result["codex_ok"] = False
                                                        tool_result["codex_error"] = str(exc)
                                                        saved_input_json_path = ""
                                                        saved_schema_json_path = ""

                                                    try:
                                                        append_saved_run_index(
                                                            conversation_id=str(conv_id),
                                                            event={
                                                                "kind": "integration_http",
                                                                "tool_name": tool_name,
                                                                "req_id": req_id,
                                                                "input_json_path": saved_input_json_path,
                                                                "schema_json_path": saved_schema_json_path,
                                                                "fields_required": fields_required,
                                                                "why_api_was_called": why_api_was_called,
                                                                "codex_output_dir": tool_result.get("codex_output_dir"),
                                                                "codex_ok": tool_result.get("codex_ok"),
                                                            },
                                                        )
                                                    except Exception:
                                                        pass

                                add_message_with_metrics(
                                    session,
                                    conversation_id=conv_id,
                                    role="tool",
                                    content=json.dumps({"tool": tool_name, "result": tool_result}, ensure_ascii=False),
                                )
                                if isinstance(tool_result, dict):
                                    meta_current = tool_result.get("metadata") or meta_current

                                if tool_failed:
                                    break

                                candidate = ""
                                if tool_name != "set_metadata" and tool_cfg:
                                    static_text = ""
                                    if (tool_cfg.static_reply_template or "").strip():
                                        static_text = _render_static_reply(
                                            template_text=tool_cfg.static_reply_template,
                                            meta=meta_current,
                                            response_json=response_json,
                                            tool_args=patch,
                                        ).strip()
                                    if static_text:
                                        needs_followup_llm = False
                                        final = static_text
                                    else:
                                        if bool(getattr(tool_cfg, "use_codex_response", False)):
                                            if bool(getattr(tool_cfg, "return_result_directly", False)) and isinstance(
                                                tool_result, dict
                                            ):
                                                direct = str(tool_result.get("codex_result_text") or "").strip()
                                                if direct:
                                                    needs_followup_llm = False
                                                    final = direct
                                                else:
                                                    needs_followup_llm = True
                                                    final = ""
                                            else:
                                                needs_followup_llm = True
                                                final = ""
                                        else:
                                            needs_followup_llm = _should_followup_llm_for_tool(
                                                tool=tool_cfg, static_rendered=static_text
                                            )
                                            candidate = _render_with_meta(next_reply, meta_current).strip()
                                            if candidate:
                                                final = candidate
                                                needs_followup_llm = False
                                            else:
                                                final = ""
                                else:
                                    candidate = _render_with_meta(next_reply, meta_current).strip()
                                    final = candidate or final

                            if needs_followup_llm:
                                followup_history = await _build_history_budgeted_async(
                                    bot_id=bot.id,
                                    conversation_id=conv_id,
                                    llm_api_key=llm_api_key,
                                    status_cb=None,
                                )
                                followup_history.append(
                                    Message(
                                        role="system",
                                        content=(
                                            ("The previous tool call failed. " if tool_failed else "")
                                            + "Using the latest tool result(s) above, write the next assistant reply. "
                                            "If the tool result contains codex_result_text, rephrase it for the user and do not mention file paths. "
                                            "Do not call any tools."
                                        ),
                                    )
                                )
                                await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "llm"})
                                text2, ttfb2, total2 = await _stream_llm_reply(
                                    ws=ws, req_id=req_id, llm=llm, messages=followup_history
                                )
                                rendered_reply = text2.strip()
                                followup_streamed = True
                                llm_ttfb_ms = ttfb2
                                llm_total_ms = total2
                            else:
                                rendered_reply = final

                            if rendered_reply and not followup_streamed:
                                await _ws_send_json(
                                    ws, {"type": "text_delta", "req_id": req_id, "delta": rendered_reply}
                                )

                        in_tok, out_tok, cost = _estimate_llm_cost_for_turn(
                            session=session,
                            bot=bot,
                            provider=provider,
                            history=history,
                            assistant_text=rendered_reply,
                        )
                        add_message_with_metrics(
                            session,
                            conversation_id=conv_id,
                            role="assistant",
                            content=rendered_reply,
                            input_tokens_est=in_tok,
                            output_tokens_est=out_tok,
                            cost_usd_est=cost,
                            llm_ttfb_ms=llm_ttfb_ms,
                            llm_total_ms=llm_total_ms,
                            total_ms=llm_total_ms,
                            citations_json=citations_json,
                        )
                        update_conversation_metrics(
                            session,
                            conversation_id=conv_id,
                            add_input_tokens_est=in_tok,
                            add_output_tokens_est=out_tok,
                            add_cost_usd_est=cost,
                            last_asr_ms=None,
                            last_llm_ttfb_ms=llm_ttfb_ms,
                            last_llm_total_ms=llm_total_ms,
                            last_tts_first_audio_ms=None,
                            last_total_ms=llm_total_ms,
                        )

                        metrics = {
                            "model": bot.openai_model,
                            "input_tokens_est": in_tok,
                            "output_tokens_est": out_tok,
                            "cost_usd_est": cost,
                            "llm_ttfb_ms": llm_ttfb_ms,
                            "llm_total_ms": llm_total_ms,
                        }
                        await _public_send_done(
                            ws, req_id=req_id, text=rendered_reply, metrics=metrics, citations=citations
                        )
                        await _ws_send_json(ws, {"type": "status", "req_id": req_id, "stage": "idle"})
                    continue

                await _ws_send_json(ws, {"type": "error", "req_id": req_id, "error": f"Unknown message type: {msg_type}"})

        except WebSocketDisconnect:
            return
        except RuntimeError:
            return

    @app.get("/public/widget.js")
    def public_widget_js() -> Response:
        p = Path(__file__).parent / "static" / "embed-widget.js"
        if not p.exists():
            raise HTTPException(status_code=404, detail="widget.js not found")
        return Response(content=p.read_text("utf-8"), media_type="application/javascript")

    @app.get("/api/options")
    def api_options(session: Session = Depends(get_session)) -> dict:
        pricing = _get_openai_pricing()
        dynamic_models: list[str] = []
        try:
            # Only fetch occasionally to avoid slow UI loads.
            now = time.time()
            if (now - float(openai_models_cache.get("ts") or 0.0)) > 3600.0:
                openai_models_cache["ts"] = now
                openai_models_cache["models"] = []
                api_key = (os.environ.get("OPENAI_API_KEY") or settings.openai_api_key or "").strip()
                if api_key:
                    try:
                        from openai import OpenAI  # type: ignore

                        client = OpenAI(api_key=api_key)
                        resp = client.models.list()
                        data = getattr(resp, "data", None) or []
                        ids: list[str] = []
                        for m in data:
                            mid = getattr(m, "id", None)
                            if not isinstance(mid, str) or not mid.strip():
                                continue
                            mid = mid.strip()
                            # Keep LLM-ish models; drop embeddings/audio/moderation/image models.
                            if not (mid.startswith("gpt-") or mid.startswith("o")):
                                continue
                            if mid.startswith(("tts-", "whisper-", "text-embedding-", "omni-moderation", "gpt-4o-mini-tts")):
                                continue
                            ids.append(mid)
                        openai_models_cache["models"] = sorted(set(ids))
                    except Exception:
                        # Ignore fetch failures; fall back to the curated list.
                        openai_models_cache["models"] = []
            dynamic_models = list(openai_models_cache.get("models") or [])
        except Exception:
            dynamic_models = []

        openai_models = sorted(set(ui_options.get("openai_models", []) + list(pricing.keys()) + dynamic_models))
        _refresh_openrouter_models_cache(session)
        openrouter_models = list(openrouter_models_cache.get("models") or [])
        openrouter_pricing = dict(openrouter_models_cache.get("pricing") or {})
        local_models = LOCAL_RUNTIME.list_models()
        default_provider = (_get_app_setting(session, "default_llm_provider") or "").strip().lower() or "openai"
        default_model = (_get_app_setting(session, "default_llm_model") or "").strip()
        return {
            "openai_models": openai_models,
            "openai_pricing": {k: {"input_per_1m": v.input_per_1m, "output_per_1m": v.output_per_1m} for k, v in pricing.items()},
            "openrouter_models": openrouter_models,
            "openrouter_pricing": {
                k: {"input_per_1m": v.input_per_1m, "output_per_1m": v.output_per_1m}
                for k, v in openrouter_pricing.items()
            },
            "local_models": local_models,
            "default_llm_provider": default_provider,
            "default_llm_model": default_model,
            "llm_providers": ["openai", "openrouter", "local"],
            "openai_asr_models": ui_options.get("openai_asr_models", []),
            "languages": ui_options.get("languages", []),
            "openai_tts_models": ui_options.get("openai_tts_models", []),
            "openai_tts_voices": ui_options.get("openai_tts_voices", []),
            "start_message_modes": ["llm", "static"],
            "http_methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],
        }

    @app.get("/api/status")
    def api_status(session: Session = Depends(get_session)) -> dict:
        openai_key = bool(_get_openai_api_key(session))
        openrouter_key = bool(_get_openrouter_api_key(session))
        local_ready = LOCAL_RUNTIME.is_ready()
        return {
            "openai_key_configured": openai_key,
            "openrouter_key_configured": openrouter_key,
            "local_ready": local_ready,
            "local_status": LOCAL_RUNTIME.status(),
            "llm_key_configured": openai_key or openrouter_key or local_ready,
            "docker_available": docker_available(),
        }

    @app.get("/api/local/status")
    def api_local_status() -> dict:
        return LOCAL_RUNTIME.status()

    @app.get("/api/local/models")
    def api_local_models() -> dict:
        return {"items": LOCAL_RUNTIME.list_models()}

    @app.post("/api/local/setup")
    def api_local_setup(payload: LocalSetupRequest, session: Session = Depends(get_session)) -> dict:
        model_id = (payload.model_id or "").strip()
        custom_url = (payload.custom_url or "").strip()
        custom_name = (payload.custom_name or "").strip()
        try:
            status = LOCAL_RUNTIME.start(model_id=model_id, custom_url=custom_url, custom_name=custom_name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        default_model = model_id or custom_name
        if not default_model and custom_url:
            default_model = custom_url.rsplit("/", 1)[-1].split("?", 1)[0].strip()
        if default_model:
            _set_app_setting(session, "default_llm_provider", "local")
            _set_app_setting(session, "default_llm_model", default_model)
        try:
            bot = _get_or_create_system_bot(session)
            bot.llm_provider = "local"
            if default_model:
                bot.openai_model = default_model
            bot.updated_at = dt.datetime.now(dt.timezone.utc)
            session.add(bot)
            session.commit()
        except Exception:
            pass
        return status

    @app.post("/api/data-agent/pull-image")
    def api_pull_data_agent_image() -> dict:
        if not docker_available():
            raise HTTPException(status_code=400, detail="Docker not available")
        try:
            image = ensure_image_pulled()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return {"ok": True, "image": image}

    @app.get("/api/data-agent/containers")
    def api_data_agent_containers() -> dict:
        return list_data_agent_containers()

    @app.post("/api/data-agent/containers/{container_id}/stop")
    def api_data_agent_container_stop(container_id: str) -> dict:
        res = stop_data_agent_container(container_id)
        if not res.get("docker_available", True):
            raise HTTPException(status_code=400, detail="Docker not available")
        if not res.get("stopped", False):
            raise HTTPException(status_code=500, detail=str(res.get("error") or "Failed to stop container"))
        return res

    @app.get("/api/downloads/{token}")
    def download_file(token: str) -> FileResponse:
        """
        Download a previously exported file by token.

        NOTE: This uses an unguessable token and a strict allowlist of temp roots.
        TODO(cleanup): add TTL + authentication/authorization as needed for production.
        """
        obj = load_download_token(token=token)
        if not obj:
            raise HTTPException(status_code=404, detail="Download token not found")
        file_path = str(obj.get("file_path") or "").strip()
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        if not is_allowed_download_path(file_path):
            raise HTTPException(status_code=403, detail="File path not allowed")
        filename = str(obj.get("filename") or os.path.basename(file_path) or "download").strip()
        mime_type = str(obj.get("mime_type") or "application/octet-stream").strip()
        return FileResponse(path=file_path, media_type=mime_type, filename=filename)

    def _bot_to_dict(bot: Bot) -> dict:
        disabled = _disabled_tool_names(bot)
        return {
            "id": str(bot.id),
            "name": bot.name,
            "llm_provider": _llm_provider_for_bot(bot),
            "openai_model": bot.openai_model,
            "web_search_model": getattr(bot, "web_search_model", bot.openai_model),
            "codex_model": getattr(bot, "codex_model", "gpt-5.1-codex-mini"),
            "summary_model": getattr(bot, "summary_model", "gpt-5-nano"),
            "history_window_turns": int(getattr(bot, "history_window_turns", 16) or 16),
            "enable_data_agent": bool(getattr(bot, "enable_data_agent", False)),
            "data_agent_api_spec_text": getattr(bot, "data_agent_api_spec_text", "") or "",
            "data_agent_auth_json": getattr(bot, "data_agent_auth_json", "") or "{}",
            "data_agent_system_prompt": getattr(bot, "data_agent_system_prompt", "") or "",
            "data_agent_return_result_directly": bool(getattr(bot, "data_agent_return_result_directly", False)),
            "data_agent_prewarm_on_start": bool(getattr(bot, "data_agent_prewarm_on_start", False)),
            "data_agent_prewarm_prompt": getattr(bot, "data_agent_prewarm_prompt", "") or "",
            "enable_host_actions": bool(getattr(bot, "enable_host_actions", False)),
            "enable_host_shell": bool(getattr(bot, "enable_host_shell", False)),
            "require_host_action_approval": bool(getattr(bot, "require_host_action_approval", False)),
            "disabled_tools": sorted(disabled),
            "system_prompt": bot.system_prompt,
            "language": bot.language,
            "openai_asr_model": getattr(bot, "openai_asr_model", "gpt-4o-mini-transcribe"),
            "openai_tts_model": bot.openai_tts_model,
            "openai_tts_voice": bot.openai_tts_voice,
            "openai_tts_speed": float(bot.openai_tts_speed),
            "start_message_mode": bot.start_message_mode,
            "start_message_text": bot.start_message_text,
            "created_at": bot.created_at.isoformat(),
            "updated_at": bot.updated_at.isoformat(),
        }

    def _tool_to_dict(t: IntegrationTool) -> dict:
        return {
            "id": str(t.id),
            "bot_id": str(t.bot_id),
            "name": t.name,
            "description": t.description,
            "url": t.url,
            "method": t.method,
            "enabled": bool(getattr(t, "enabled", True)),
            "args_required": _parse_required_args_json(getattr(t, "args_required_json", "[]")),
            # Never expose secret headers (write-only). Return masked preview for UI.
            "headers_template_json": "{}",
            "headers_template_json_masked": _mask_headers_json(t.headers_template_json),
            "headers_configured": _headers_configured(t.headers_template_json),
            "request_body_template": t.request_body_template,
            "parameters_schema_json": t.parameters_schema_json,
            "response_schema_json": getattr(t, "response_schema_json", "") or "",
            "codex_prompt": getattr(t, "codex_prompt", "") or "",
            "postprocess_python": getattr(t, "postprocess_python", "") or "",
            "return_result_directly": bool(getattr(t, "return_result_directly", False)),
            "response_mapper_json": t.response_mapper_json,
            "pagination_json": getattr(t, "pagination_json", "") or "",
            "static_reply_template": t.static_reply_template,
            "use_codex_response": bool(getattr(t, "use_codex_response", False)),
            "created_at": t.created_at.isoformat(),
            "updated_at": t.updated_at.isoformat(),
        }

    @app.get("/api/bots")
    def api_list_bots(session: Session = Depends(get_session)) -> dict:
        bots = list_bots(session)
        stats = bots_aggregate_metrics(session)
        items = []
        for b in bots:
            d = _bot_to_dict(b)
            d["stats"] = stats.get(
                b.id,
                {
                    "conversations": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "avg_llm_ttfb_ms": None,
                    "avg_llm_total_ms": None,
                    "avg_total_ms": None,
                },
            )
            items.append(d)
        return {"items": items}

    @app.post("/api/bots")
    def api_create_bot(payload: BotCreateRequest, session: Session = Depends(get_session)) -> dict:
        default_provider = (_get_app_setting(session, "default_llm_provider") or "").strip().lower()
        default_model = (_get_app_setting(session, "default_llm_model") or "").strip()
        provider = _normalize_llm_provider(payload.llm_provider or default_provider or "openai")
        openai_model = (payload.openai_model or default_model or "o4-mini").strip() or "o4-mini"
        bot = Bot(
            name=payload.name,
            llm_provider=provider,
            openai_model=openai_model,
            openai_asr_model=(payload.openai_asr_model or "gpt-4o-mini-transcribe").strip() or "gpt-4o-mini-transcribe",
            web_search_model=(payload.web_search_model or openai_model).strip() or openai_model,
            codex_model=(payload.codex_model or "gpt-5.1-codex-mini").strip() or "gpt-5.1-codex-mini",
            summary_model=(payload.summary_model or openai_model or "gpt-5-nano").strip() or (openai_model or "gpt-5-nano"),
            history_window_turns=int(payload.history_window_turns or 16),
            enable_data_agent=bool(getattr(payload, "enable_data_agent", False)),
            data_agent_api_spec_text=(payload.data_agent_api_spec_text or ""),
            data_agent_auth_json=(payload.data_agent_auth_json or "{}"),
            data_agent_system_prompt=(payload.data_agent_system_prompt or ""),
            data_agent_return_result_directly=bool(getattr(payload, "data_agent_return_result_directly", False)),
            data_agent_prewarm_on_start=bool(getattr(payload, "data_agent_prewarm_on_start", False)),
            data_agent_prewarm_prompt=(payload.data_agent_prewarm_prompt or ""),
            enable_host_actions=bool(getattr(payload, "enable_host_actions", False)),
            enable_host_shell=bool(getattr(payload, "enable_host_shell", False)),
            require_host_action_approval=bool(getattr(payload, "require_host_action_approval", False)),
            system_prompt=payload.system_prompt,
            language=payload.language,
            openai_tts_model=(payload.openai_tts_model or "gpt-4o-mini-tts").strip() or "gpt-4o-mini-tts",
            openai_tts_voice=(payload.openai_tts_voice or "alloy").strip() or "alloy",
            openai_tts_speed=float(payload.openai_tts_speed or 1.0),
            start_message_mode=(payload.start_message_mode or "llm").strip() or "llm",
            start_message_text=payload.start_message_text or "",
        )
        create_bot(session, bot)
        return _bot_to_dict(bot)

    @app.get("/api/bots/{bot_id}")
    def api_get_bot(bot_id: UUID, session: Session = Depends(get_session)) -> dict:
        bot = get_bot(session, bot_id)
        return _bot_to_dict(bot)

    @app.put("/api/bots/{bot_id}")
    def api_update_bot(bot_id: UUID, payload: BotUpdateRequest, session: Session = Depends(get_session)) -> dict:
        patch = {}
        for k, v in payload.model_dump(exclude_unset=True).items():
            if k == "llm_provider":
                raw = (v or "").strip().lower()
                if raw and raw not in ("openai", "openrouter", "local"):
                    raise HTTPException(status_code=400, detail="Unsupported LLM provider.")
                patch[k] = raw or "openai"
            elif k in (
                "openai_tts_model",
                "openai_tts_voice",
                "web_search_model",
                "codex_model",
                "summary_model",
                "openai_asr_model",
            ):
                patch[k] = (v or "").strip()
            elif k == "openai_tts_speed":
                patch[k] = float(v) if v is not None else 1.0
            elif k == "history_window_turns":
                try:
                    n = int(v) if v is not None else 16
                except Exception:
                    n = 16
                if n < 1:
                    n = 1
                if n > 64:
                    n = 64
                patch[k] = n
            elif k == "disabled_tools":
                vals = v or []
                if not isinstance(vals, list):
                    vals = []
                cleaned: list[str] = []
                for x in vals:
                    s = str(x or "").strip()
                    if not s:
                        continue
                    if s in ("set_metadata", "set_variable"):
                        continue
                    if s not in cleaned:
                        cleaned.append(s)
                patch["disabled_tools_json"] = json.dumps(cleaned, ensure_ascii=False)
            else:
                patch[k] = v
        bot = update_bot(session, bot_id, patch)
        return _bot_to_dict(bot)

    @app.delete("/api/bots/{bot_id}")
    def api_delete_bot(bot_id: UUID, session: Session = Depends(get_session)) -> dict:
        delete_bot(session, bot_id)
        return {"ok": True}

    @app.get("/api/bots/{bot_id}/tools")
    def api_list_tools(bot_id: UUID, session: Session = Depends(get_session)) -> dict:
        bot = get_bot(session, bot_id)
        tools = list_integration_tools(session, bot_id=bot_id)
        disabled = _disabled_tool_names(bot)
        return {
            "items": [_tool_to_dict(t) for t in tools],
            "system_tools": _system_tools_public_list(bot=bot, disabled=disabled),
            "disabled_tools": sorted(disabled),
        }

    @app.post("/api/bots/{bot_id}/tools")
    def api_create_tool(bot_id: UUID, payload: IntegrationToolCreateRequest, session: Session = Depends(get_session)) -> dict:
        _ = get_bot(session, bot_id)
        name = (payload.name or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Tool name is required")
        if get_integration_tool_by_name(session, bot_id=bot_id, name=name):
            raise HTTPException(status_code=400, detail="Tool name already exists for this bot")
        tool = IntegrationTool(
            bot_id=bot_id,
            name=name,
            description=payload.description or "",
            url=payload.url,
            method=(payload.method or "GET").upper(),
            use_codex_response=bool(payload.use_codex_response),
            enabled=bool(payload.enabled),
            args_required_json=json.dumps(payload.args_required or [], ensure_ascii=False),
            headers_template_json=payload.headers_template_json or "{}",
            request_body_template=payload.request_body_template or "{}",
            parameters_schema_json=payload.parameters_schema_json or "",
            response_schema_json=payload.response_schema_json or "",
            codex_prompt=(payload.codex_prompt or ""),
            postprocess_python=(payload.postprocess_python or ""),
            return_result_directly=bool(payload.return_result_directly),
            response_mapper_json=payload.response_mapper_json or "{}",
            pagination_json=payload.pagination_json or "",
            static_reply_template=payload.static_reply_template or "",
        )
        create_integration_tool(session, tool)
        return _tool_to_dict(tool)

    @app.put("/api/bots/{bot_id}/tools/{tool_id}")
    def api_update_tool(
        bot_id: UUID, tool_id: UUID, payload: IntegrationToolUpdateRequest, session: Session = Depends(get_session)
    ) -> dict:
        _ = get_bot(session, bot_id)
        tool = get_integration_tool(session, tool_id)
        if tool.bot_id != bot_id:
            raise HTTPException(status_code=400, detail="Tool does not belong to bot")
        patch = payload.model_dump(exclude_unset=True)
        if "name" in patch:
            name = str(patch["name"] or "").strip()
            if not name:
                raise HTTPException(status_code=400, detail="Tool name is required")
            existing = get_integration_tool_by_name(session, bot_id=bot_id, name=name)
            if existing and existing.id != tool_id:
                raise HTTPException(status_code=400, detail="Tool name already exists for this bot")
            patch["name"] = name
        if "method" in patch and patch["method"] is not None:
            patch["method"] = str(patch["method"]).upper()
        if "args_required" in patch:
            patch["args_required_json"] = json.dumps(patch.pop("args_required") or [], ensure_ascii=False)
        if "headers_template_json" in patch:
            patch["headers_template_json"] = patch["headers_template_json"] or "{}"
        if "parameters_schema_json" in patch:
            patch["parameters_schema_json"] = patch["parameters_schema_json"] or ""
        if "response_schema_json" in patch:
            patch["response_schema_json"] = patch["response_schema_json"] or ""
        if "codex_prompt" in patch:
            patch["codex_prompt"] = patch["codex_prompt"] or ""
        if "postprocess_python" in patch:
            patch["postprocess_python"] = patch["postprocess_python"] or ""
        if "return_result_directly" in patch and patch["return_result_directly"] is not None:
            patch["return_result_directly"] = bool(patch["return_result_directly"])
        if "pagination_json" in patch:
            patch["pagination_json"] = patch["pagination_json"] or ""
        tool = update_integration_tool(session, tool_id, patch)
        return _tool_to_dict(tool)

    @app.delete("/api/bots/{bot_id}/tools/{tool_id}")
    def api_delete_tool(bot_id: UUID, tool_id: UUID, session: Session = Depends(get_session)) -> dict:
        _ = get_bot(session, bot_id)
        tool = get_integration_tool(session, tool_id)
        if tool.bot_id != bot_id:
            raise HTTPException(status_code=400, detail="Tool does not belong to bot")
        delete_integration_tool(session, tool_id)
        return {"ok": True}

    @app.get("/api/keys")
    def api_list_keys(provider: Optional[str] = None, session: Session = Depends(get_session)) -> dict:
        keys = list_keys(session, provider=provider)
        return {
            "items": [
                {
                    "id": str(k.id),
                    "provider": k.provider,
                    "name": k.name,
                    "hint": k.hint,
                    "created_at": k.created_at.isoformat(),
                }
                for k in keys
            ]
        }

    @app.post("/api/keys")
    def api_create_key(payload: ApiKeyCreateRequest, session: Session = Depends(get_session)) -> dict:
        provider = (payload.provider or "").strip().lower() or "openai"
        if provider not in ("openai", "openrouter"):
            raise HTTPException(status_code=400, detail="Unsupported provider.")
        crypto = require_crypto()
        k = create_key(session, crypto=crypto, provider=provider, name=payload.name, secret=payload.secret)
        return {
            "id": str(k.id),
            "provider": k.provider,
            "name": k.name,
            "hint": k.hint,
            "created_at": k.created_at.isoformat(),
        }

    @app.delete("/api/keys/{key_id}")
    def api_delete_key(key_id: UUID, session: Session = Depends(get_session)) -> dict:
        delete_key(session, key_id)
        return {"ok": True}

    @app.get("/api/client-keys")
    def api_list_client_keys(session: Session = Depends(get_session)) -> dict:
        items = []
        for k in list_client_keys(session):
            try:
                allowed_bot_ids = json.loads(k.allowed_bot_ids_json or "[]")
                if not isinstance(allowed_bot_ids, list):
                    allowed_bot_ids = []
            except Exception:
                allowed_bot_ids = []
            items.append(
                {
                    "id": str(k.id),
                    "name": k.name,
                    "hint": k.hint,
                    "allowed_origins": k.allowed_origins,
                    "allowed_bot_ids": [str(x) for x in allowed_bot_ids if isinstance(x, str)],
                    "created_at": k.created_at.isoformat(),
                }
            )
        return {"items": items}

    @app.post("/api/client-keys")
    def api_create_client_key(payload: ClientKeyCreateRequest, session: Session = Depends(get_session)) -> dict:
        name = (payload.name or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Name is required")
        secret = (payload.secret or "").strip()
        generated = False
        if not secret:
            generated = True
            secret = "igx_" + secrets.token_urlsafe(24)
        allowed_bot_ids = [str(x) for x in (payload.allowed_bot_ids or []) if str(x).strip()]
        k = create_client_key(
            session,
            name=name,
            secret=secret,
            allowed_origins=(payload.allowed_origins or "").strip(),
            allowed_bot_ids=allowed_bot_ids,
        )
        out = {
            "id": str(k.id),
            "name": k.name,
            "hint": k.hint,
            "allowed_origins": k.allowed_origins,
            "allowed_bot_ids": allowed_bot_ids,
            "created_at": k.created_at.isoformat(),
        }
        if generated:
            out["secret"] = secret
        return out

    @app.delete("/api/client-keys/{key_id}")
    def api_delete_client_key(key_id: UUID, session: Session = Depends(get_session)) -> dict:
        _ = get_client_key(session, key_id)
        delete_client_key(session, key_id)
        return {"ok": True}

    @app.get("/api/user/git-token")
    def api_get_git_token(session: Session = Depends(get_session)) -> dict:
        provider = "github"
        rec = get_git_token(session, provider=provider)
        if not rec:
            return {"provider": provider, "configured": False}
        return {
            "provider": provider,
            "configured": True,
            "hint": rec.hint,
            "created_at": rec.created_at.isoformat(),
            "updated_at": rec.updated_at.isoformat(),
        }

    @app.post("/api/user/git-token")
    async def api_set_git_token(payload: GitTokenRequest, session: Session = Depends(get_session)) -> dict:
        provider = _normalize_git_provider(payload.provider)
        token = (payload.token or "").strip()
        if not token:
            raise HTTPException(status_code=400, detail="Token is required")
        crypto = require_crypto()

        validated = None
        warning = None
        if provider == "github":
            ok, msg = await _validate_github_token(token)
            if ok:
                validated = True
            else:
                if msg == "Invalid GitHub token":
                    raise HTTPException(status_code=400, detail=msg)
                validated = False
                warning = msg

        encrypted = crypto.encrypt_str(token)
        hint = build_hint(token)
        rec = upsert_git_token(session, provider=provider, token_ciphertext=encrypted, hint=hint)
        out = {
            "provider": provider,
            "hint": rec.hint,
            "created_at": rec.created_at.isoformat(),
            "updated_at": rec.updated_at.isoformat(),
        }
        if validated is not None:
            out["validated"] = validated
        if warning:
            out["warning"] = warning
        return out

    @app.get("/api/conversations")
    def api_list_conversations(
        page: int = 1,
        page_size: int = 50,
        bot_id: Optional[UUID] = None,
        test_flag: Optional[bool] = None,
        include_groups: bool = False,
        session: Session = Depends(get_session),
    ) -> dict:
        page = max(1, int(page))
        page_size = min(200, max(10, int(page_size)))
        offset = (page - 1) * page_size
        total = count_conversations(session, bot_id=bot_id, test_flag=test_flag, include_groups=include_groups)
        convs = list_conversations(
            session,
            bot_id=bot_id,
            test_flag=test_flag,
            include_groups=include_groups,
            limit=page_size,
            offset=offset,
        )
        bots_by_id = {b.id: b for b in list_bots(session)}
        items = []
        for c in convs:
            b = bots_by_id.get(c.bot_id)
            items.append(
                {
                    "id": str(c.id),
                    "bot_id": str(c.bot_id),
                    "bot_name": b.name if b else None,
                    "test_flag": bool(c.test_flag),
                    "metadata_json": c.metadata_json or "{}",
                    "llm_input_tokens_est": int(c.llm_input_tokens_est or 0),
                    "llm_output_tokens_est": int(c.llm_output_tokens_est or 0),
                    "cost_usd_est": float(c.cost_usd_est or 0.0),
                    "last_asr_ms": c.last_asr_ms,
                    "last_llm_ttfb_ms": c.last_llm_ttfb_ms,
                    "last_llm_total_ms": c.last_llm_total_ms,
                    "last_tts_first_audio_ms": c.last_tts_first_audio_ms,
                    "last_total_ms": c.last_total_ms,
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                }
            )
        return {"items": items, "page": page, "page_size": page_size, "total": total}

    @app.get("/api/conversations/{conversation_id}")
    def api_conversation_detail(conversation_id: UUID, session: Session = Depends(get_session)) -> dict:
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
            if m.role == "tool":
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
                    "metrics": {
                        "in": m.input_tokens_est,
                        "out": m.output_tokens_est,
                        "cost": m.cost_usd_est,
                        "asr": m.asr_ms,
                        "llm1": m.llm_ttfb_ms,
                        "llm": m.llm_total_ms,
                        "tts1": m.tts_first_audio_ms,
                        "total": m.total_ms,
                    },
                }
            )

        return {
            "conversation": {
                "id": str(conv.id),
                "bot_id": str(conv.bot_id),
                "bot_name": bot.name,
                "test_flag": bool(conv.test_flag),
                "metadata_json": conv.metadata_json or "{}",
                "is_group": bool(conv.is_group),
                "group_title": conv.group_title or "",
                "group_bots_json": conv.group_bots_json or "[]",
                "llm_input_tokens_est": int(conv.llm_input_tokens_est or 0),
                "llm_output_tokens_est": int(conv.llm_output_tokens_est or 0),
                "cost_usd_est": float(conv.cost_usd_est or 0.0),
                "last_asr_ms": conv.last_asr_ms,
                "last_llm_ttfb_ms": conv.last_llm_ttfb_ms,
                "last_llm_total_ms": conv.last_llm_total_ms,
                "last_tts_first_audio_ms": conv.last_tts_first_audio_ms,
                "last_total_ms": conv.last_total_ms,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            },
            "bot": _bot_to_dict(bot),
            "messages": messages,
        }

    @app.get("/api/conversations/{conversation_id}/messages")
    def api_conversation_messages(
        conversation_id: UUID,
        since: Optional[str] = None,
        before: Optional[str] = None,
        before_id: Optional[str] = None,
        limit: int = 200,
        order: str = "asc",
        include_tools: bool = False,
        session: Session = Depends(get_session),
    ) -> dict:
        _ = get_conversation(session, conversation_id)
        since_dt: dt.datetime | None = None
        if since:
            try:
                since_dt = dt.datetime.fromisoformat(str(since))
            except Exception:
                since_dt = None
        before_dt: dt.datetime | None = None
        if before:
            try:
                before_dt = dt.datetime.fromisoformat(str(before))
            except Exception:
                before_dt = None
        before_uuid: UUID | None = None
        if before_id:
            try:
                before_uuid = UUID(str(before_id))
            except Exception:
                before_uuid = None
        stmt = select(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id)
        if not include_tools:
            stmt = stmt.where(ConversationMessage.role != "tool")
        if since_dt is not None:
            stmt = stmt.where(ConversationMessage.created_at > since_dt)
        if before_dt is not None:
            if before_uuid is not None:
                stmt = stmt.where(
                    or_(
                        ConversationMessage.created_at < before_dt,
                        and_(
                            ConversationMessage.created_at == before_dt,
                            ConversationMessage.id < before_uuid,
                        ),
                    )
                )
            else:
                stmt = stmt.where(ConversationMessage.created_at < before_dt)
        if str(order).lower() == "desc":
            stmt = stmt.order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc())
        else:
            stmt = stmt.order_by(ConversationMessage.created_at.asc(), ConversationMessage.id.asc())
        stmt = stmt.limit(min(500, max(1, int(limit))))
        msgs_raw = list(session.exec(stmt))
        messages = []
        for m in msgs_raw:
            if m.role == "tool" and not include_tools:
                continue
            tool_obj = None
            tool_name = None
            tool_kind = None
            if m.role == "tool":
                tool_obj = safe_json_loads(m.content or "{}")
                tool_name = tool_obj.get("tool") if tool_obj and isinstance(tool_obj.get("tool"), str) else None
                if tool_obj:
                    if "arguments" in tool_obj:
                        tool_kind = "call"
                    elif "result" in tool_obj:
                        tool_kind = "result"
            try:
                citations = json.loads(getattr(m, "citations_json", "") or "[]")
                if not isinstance(citations, list):
                    citations = []
            except Exception:
                citations = []
            messages.append(
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                    "tool": tool_obj,
                    "tool_name": tool_name,
                    "tool_kind": tool_kind,
                    "citations": citations,
                    "sender_bot_id": str(m.sender_bot_id) if m.sender_bot_id else None,
                    "sender_name": m.sender_name,
                    "metrics": {
                        "in": m.input_tokens_est,
                        "out": m.output_tokens_est,
                        "cost": m.cost_usd_est,
                        "asr": m.asr_ms,
                        "llm1": m.llm_ttfb_ms,
                        "llm": m.llm_total_ms,
                        "tts1": m.tts_first_audio_ms,
                        "total": m.total_ms,
                    },
                }
            )
        return {"conversation_id": str(conversation_id), "messages": messages}

    def _group_message_payload(m: ConversationMessage) -> dict | None:
        if m.role == "tool":
            return None
        tool_obj = safe_json_loads(m.content or "{}") if m.role == "tool" else None
        tool_name = tool_obj.get("tool") if tool_obj and isinstance(tool_obj.get("tool"), str) else None
        tool_kind = None
        if tool_obj:
            if "arguments" in tool_obj:
                tool_kind = "call"
            elif "result" in tool_obj:
                tool_kind = "result"
        try:
            citations = json.loads(getattr(m, "citations_json", "") or "[]")
            if not isinstance(citations, list):
                citations = []
        except Exception:
            citations = []
        return {
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat(),
            "tool": tool_obj,
            "tool_name": tool_name,
            "tool_kind": tool_kind,
            "citations": citations,
            "sender_bot_id": str(m.sender_bot_id) if m.sender_bot_id else None,
            "sender_name": m.sender_name,
            "metrics": {
                "in": m.input_tokens_est,
                "out": m.output_tokens_est,
                "cost": m.cost_usd_est,
                "asr": m.asr_ms,
                "llm1": m.llm_ttfb_ms,
                "llm": m.llm_total_ms,
                "tts1": m.tts_first_audio_ms,
                "total": m.total_ms,
            },
        }

    def _group_conversation_payload(session: Session, conv: Conversation, *, include_messages: bool = True) -> dict:
        bots = _group_bots_from_conv(conv)
        bot_lookup = {b["id"]: b for b in bots}
        messages: list[dict] = []
        if include_messages:
            msgs_raw = list_messages(session, conversation_id=conv.id)
            for m in msgs_raw:
                payload = _group_message_payload(m)
                if payload is not None:
                    messages.append(payload)

        default_bot = bot_lookup.get(str(conv.bot_id))
        individual_map = _ensure_group_individual_conversations(session, conv)
        individual_items = [
            {"bot_id": bid, "conversation_id": cid} for bid, cid in individual_map.items()
        ]
        return {
            "conversation": {
                "id": str(conv.id),
                "title": conv.group_title or "",
                "default_bot_id": str(conv.bot_id),
                "default_bot_name": default_bot.get("name") if default_bot else None,
                "group_bots": bots,
                "individual_conversations": individual_items,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat(),
            },
            "messages": messages,
        }

    @app.get("/api/group-conversations")
    def api_list_group_conversations(session: Session = Depends(get_session)) -> dict:
        stmt = select(Conversation).where(Conversation.is_group == True).order_by(Conversation.updated_at.desc())  # noqa: E712
        convs = list(session.exec(stmt))
        items = []
        for c in convs:
            bots = _group_bots_from_conv(c)
            items.append(
                {
                    "id": str(c.id),
                    "title": c.group_title or "",
                    "default_bot_id": str(c.bot_id),
                    "group_bots": bots,
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                }
            )
        return {"items": items}

    @app.post("/api/group-conversations")
    def api_create_group_conversation(
        payload: GroupConversationCreateRequest = Body(...),
        session: Session = Depends(get_session),
    ) -> dict:
        title = (payload.title or "").strip()
        bot_ids = [str(b).strip() for b in (payload.bot_ids or []) if str(b).strip()]
        default_bot_id = str(payload.default_bot_id or "").strip()
        if not title:
            raise HTTPException(status_code=400, detail="Title is required")
        if not bot_ids:
            raise HTTPException(status_code=400, detail="At least one assistant is required")
        if not default_bot_id:
            raise HTTPException(status_code=400, detail="Default assistant is required")
        if default_bot_id not in bot_ids:
            raise HTTPException(status_code=400, detail="Default assistant must be in the group")

        bots: list[Bot] = []
        for bid in bot_ids:
            try:
                bots.append(get_bot(session, UUID(bid)))
            except Exception:
                raise HTTPException(status_code=404, detail=f"Assistant not found: {bid}")

        used_slugs: set[str] = set()
        group_bots: list[dict[str, str]] = []
        for b in bots:
            base = _slugify(b.name)
            slug = base
            i = 2
            while slug in used_slugs:
                slug = f"{base}-{i}"
                i += 1
            used_slugs.add(slug)
            group_bots.append({"id": str(b.id), "name": b.name, "slug": slug})

        now = dt.datetime.now(dt.timezone.utc)
        conv = Conversation(
            bot_id=UUID(default_bot_id),
            test_flag=bool(payload.test_flag),
            is_group=True,
            group_title=title,
            group_bots_json=json.dumps(group_bots, ensure_ascii=False),
            created_at=now,
            updated_at=now,
        )
        session.add(conv)
        session.commit()
        session.refresh(conv)
        return _group_conversation_payload(session, conv)

    @app.get("/api/group-conversations/{conversation_id}")
    def api_group_conversation_detail(
        conversation_id: UUID,
        include_messages: bool = True,
        session: Session = Depends(get_session),
    ) -> dict:
        conv = get_conversation(session, conversation_id)
        if not bool(conv.is_group):
            raise HTTPException(status_code=404, detail="Group conversation not found")
        return _group_conversation_payload(session, conv, include_messages=include_messages)

    @app.get("/api/group-conversations/{conversation_id}/messages")
    def api_group_conversation_messages(
        conversation_id: UUID,
        since: Optional[str] = None,
        before: Optional[str] = None,
        before_id: Optional[str] = None,
        limit: int = 200,
        order: str = "asc",
        include_tools: bool = False,
        session: Session = Depends(get_session),
    ) -> dict:
        conv = get_conversation(session, conversation_id)
        if not bool(conv.is_group):
            raise HTTPException(status_code=404, detail="Group conversation not found")
        since_dt: dt.datetime | None = None
        if since:
            try:
                since_dt = dt.datetime.fromisoformat(str(since))
            except Exception:
                since_dt = None
        before_dt: dt.datetime | None = None
        if before:
            try:
                before_dt = dt.datetime.fromisoformat(str(before))
            except Exception:
                before_dt = None
        before_uuid: UUID | None = None
        if before_id:
            try:
                before_uuid = UUID(str(before_id))
            except Exception:
                before_uuid = None
        stmt = select(ConversationMessage).where(ConversationMessage.conversation_id == conversation_id)
        if not include_tools:
            stmt = stmt.where(ConversationMessage.role != "tool")
        if since_dt is not None:
            stmt = stmt.where(ConversationMessage.created_at > since_dt)
        if before_dt is not None:
            if before_uuid is not None:
                stmt = stmt.where(
                    or_(
                        ConversationMessage.created_at < before_dt,
                        and_(
                            ConversationMessage.created_at == before_dt,
                            ConversationMessage.id < before_uuid,
                        ),
                    )
                )
            else:
                stmt = stmt.where(ConversationMessage.created_at < before_dt)
        if str(order).lower() == "desc":
            stmt = stmt.order_by(ConversationMessage.created_at.desc(), ConversationMessage.id.desc())
        else:
            stmt = stmt.order_by(ConversationMessage.created_at.asc(), ConversationMessage.id.asc())
        stmt = stmt.limit(min(500, max(1, int(limit))))
        msgs_raw = list(session.exec(stmt))
        messages: list[dict] = []
        for m in msgs_raw:
            payload = _group_message_payload(m)
            if payload is not None:
                messages.append(payload)
        return {"conversation_id": str(conversation_id), "messages": messages}

    @app.post("/api/group-conversations/{conversation_id}/messages")
    async def api_group_conversation_message(
        conversation_id: UUID,
        payload: GroupMessageRequest = Body(...),
        session: Session = Depends(get_session),
    ) -> dict:
        conv = get_conversation(session, conversation_id)
        if not bool(conv.is_group):
            raise HTTPException(status_code=404, detail="Group conversation not found")

        text = str(payload.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text")

        sender_role = str(payload.sender_role or "user").strip().lower()
        if sender_role not in ("user", "assistant"):
            raise HTTPException(status_code=400, detail="Invalid sender_role")

        sender_bot_id: Optional[UUID] = None
        sender_name = (payload.sender_name or "").strip()
        if sender_role == "assistant":
            if not payload.sender_bot_id:
                raise HTTPException(status_code=400, detail="sender_bot_id is required for assistant messages")
            try:
                sender_bot_id = UUID(str(payload.sender_bot_id))
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid sender_bot_id")
            if str(sender_bot_id) not in {b["id"] for b in _group_bots_from_conv(conv)}:
                raise HTTPException(status_code=400, detail="Assistant is not a member of this group")
            if not sender_name:
                try:
                    sender_name = get_bot(session, sender_bot_id).name
                except Exception:
                    sender_name = "Assistant"
        else:
            if not sender_name:
                sender_name = "User"

        msg = add_message_with_metrics(
            session,
            conversation_id=conversation_id,
            role=sender_role,
            content=text,
            sender_bot_id=sender_bot_id,
            sender_name=sender_name,
        )
        if msg.role == "assistant":
            _mirror_group_message(session, conv=conv, msg=msg)

        payload = _group_message_payload(msg)
        if payload:
            await _group_ws_broadcast(conversation_id, {"type": "message", "message": payload})

        targets = _extract_group_mentions(text, conv)
        if sender_bot_id:
            targets = [bid for bid in targets if str(bid) != str(sender_bot_id)]
        if sender_role == "user" and not targets:
            if not conv.bot_id:
                raise HTTPException(status_code=400, detail="Default assistant is not configured")
            targets = [conv.bot_id]

        if targets:
            _schedule_group_bots(conversation_id, targets)

        return _group_conversation_payload(session, conv)

    @app.post("/api/group-conversations/{conversation_id}/reset")
    def api_group_conversation_reset(
        conversation_id: UUID,
        session: Session = Depends(get_session),
    ) -> dict:
        conv = get_conversation(session, conversation_id)
        if not bool(conv.is_group):
            raise HTTPException(status_code=404, detail="Group conversation not found")

        mapping = _ensure_group_individual_conversations(session, conv)
        meta = safe_json_loads(conv.metadata_json or "{}") or {}
        keep = {}
        if isinstance(meta, dict):
            if "demo_seed" in meta:
                keep["demo_seed"] = meta["demo_seed"]
        keep["group_individual_conversations"] = mapping
        _reset_conversation_state(session, conv, keep)

        for bid, cid in mapping.items():
            try:
                child = get_conversation(session, UUID(cid))
            except Exception:
                continue
            child_meta = safe_json_loads(child.metadata_json or "{}") or {}
            keep_child = {}
            if isinstance(child_meta, dict):
                for key in ("group_parent_id", "group_bot_id", "group_bot_name"):
                    if key in child_meta:
                        keep_child[key] = child_meta[key]
            _reset_conversation_state(session, child, keep_child)

        try:
            asyncio.get_running_loop()
            asyncio.create_task(_group_ws_broadcast(conversation_id, {"type": "reset"}))
        except Exception:
            pass
        return {"ok": True}

    @app.delete("/api/group-conversations/{conversation_id}")
    def api_delete_group_conversation(
        conversation_id: UUID,
        session: Session = Depends(get_session),
    ) -> dict:
        conv = get_conversation(session, conversation_id)
        if not bool(conv.is_group):
            raise HTTPException(status_code=404, detail="Group conversation not found")

        mapping = _ensure_group_individual_conversations(session, conv)

        # Delete group messages + conversation.
        session.exec(delete(ConversationMessage).where(ConversationMessage.conversation_id == conv.id))
        session.delete(conv)

        # Delete individual logs for the group.
        for cid in mapping.values():
            try:
                child = get_conversation(session, UUID(cid))
            except Exception:
                continue
            session.exec(delete(ConversationMessage).where(ConversationMessage.conversation_id == child.id))
            session.delete(child)

        session.commit()
        return {"ok": True}

    @app.websocket("/ws/groups/{conversation_id}")
    async def ws_group(conversation_id: UUID, ws: WebSocket) -> None:
        await ws.accept()
        key = str(conversation_id)
        async with group_ws_lock:
            group_ws_clients.setdefault(key, set()).add(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            async with group_ws_lock:
                clients = group_ws_clients.get(key, set())
                clients.discard(ws)
                if clients:
                    group_ws_clients[key] = clients
                else:
                    group_ws_clients.pop(key, None)

    @app.get("/api/conversations/{conversation_id}/data-agent")
    def api_conversation_data_agent(conversation_id: UUID, session: Session = Depends(get_session)) -> dict:
        _ = get_conversation(session, conversation_id)
        meta = _get_conversation_meta(session, conversation_id=conversation_id)
        da = _data_agent_meta(meta)
        container_id = str(da.get("container_id") or "").strip()
        session_id = str(da.get("session_id") or "").strip()
        workspace_dir = str(da.get("workspace_dir") or "").strip() or default_workspace_dir_for_conversation(conversation_id)
        status = get_container_status(conversation_id=conversation_id, container_id=container_id)
        status["conversation_id"] = str(conversation_id)
        status["workspace_dir"] = workspace_dir
        status["session_id"] = session_id
        if container_id and not status.get("container_id"):
            status["container_id"] = container_id
        return status

    @app.post("/api/conversations/{conversation_id}/data-agent/cancel")
    def api_conversation_data_agent_cancel(conversation_id: UUID, session: Session = Depends(get_session)) -> dict:
        _ = get_conversation(session, conversation_id)
        meta = _get_conversation_meta(session, conversation_id=conversation_id)
        da = _data_agent_meta(meta)
        container_id = str(da.get("container_id") or "").strip()
        if not container_id:
            return {"ok": False, "error": "No Isolated Workspace container for this conversation."}
        kill_script = (
            "for p in /proc/[0-9]*; do "
            "cmd=$(tr '\\0' ' ' < \"$p\"/cmdline 2>/dev/null); "
            "case \"$cmd\" in "
            "*codex\\ exec*|*'/codex/codex exec'*|*'/usr/local/bin/codex exec'*|*'@openai/codex'*|*'git clone'*|*'git-upload-pack'*|*'git index-pack'*) "
            "pid=${p##*/}; "
            "if [ \"$pid\" != \"$$\" ]; then kill -9 \"$pid\" 2>/dev/null || true; fi "
            ";; "
            "esac; "
            "done; "
            "echo cancelled"
        )
        res = run_container_command(container_id=container_id, command=kill_script, timeout_s=15.0)
        return {
            "ok": res.ok,
            "exit_code": res.exit_code,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }

    @app.post("/api/bots/{bot_id}/chat/stream")
    def chat_stream(
        bot_id: UUID,
        payload: ChatRequest = Body(...),
        session: Session = Depends(get_session),
    ) -> StreamingResponse:
        bot = get_bot(session, bot_id)
        provider, llm_api_key, llm = _require_llm_client(session, bot=bot)
        speak = bool(payload.speak)
        openai_api_key: Optional[str] = None
        if speak:
            openai_api_key = _get_openai_api_key_for_bot(session, bot=bot)
            if not openai_api_key:
                raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot (needed for TTS).")
        def gen() -> Generator[bytes, None, None]:
            text = (payload.text or "").strip()
            if not text:
                yield _ndjson({"type": "error", "error": "Empty text"})
                return

            messages = [Message(role="system", content=bot.system_prompt), Message(role="user", content=text)]

            delta_q_client: "queue.Queue[Optional[str]]" = queue.Queue()
            delta_q_tts: "queue.Queue[Optional[str]]" = queue.Queue()
            audio_q: "queue.Queue[Optional[tuple[bytes, int]]]" = queue.Queue()
            full_text_parts: list[str] = []
            error_q: "queue.Queue[Optional[str]]" = queue.Queue()

            def llm_thread() -> None:
                try:
                    for delta in llm.stream_text(messages=messages):
                        full_text_parts.append(delta)
                        delta_q_client.put(delta)
                        if speak:
                            delta_q_tts.put(delta)
                except Exception as exc:
                    error_q.put(str(exc))
                finally:
                    delta_q_client.put(None)
                    if speak:
                        delta_q_tts.put(None)

            def tts_thread() -> None:
                if not speak:
                    audio_q.put(None)
                    return
                try:
                    tts_synth = _get_tts_synth_fn(bot, openai_api_key)
                    for text_to_speak in _iter_tts_chunks(delta_q_tts):
                        if not text_to_speak:
                            continue
                        wav, sr = tts_synth(text_to_speak)
                        audio_q.put((wav, sr))
                finally:
                    audio_q.put(None)

            t1 = threading.Thread(target=llm_thread, daemon=True)
            t2 = threading.Thread(target=tts_thread, daemon=True)
            t1.start()
            t2.start()

            open_deltas = True
            open_audio = True
            last_heartbeat = time.time()

            while open_deltas or open_audio:
                sent = False
                try:
                    err = error_q.get_nowait()
                    if err:
                        yield _ndjson({"type": "error", "error": err})
                        open_deltas = False
                        open_audio = False
                        break
                except queue.Empty:
                    pass
                try:
                    d = delta_q_client.get_nowait()
                    if d is None:
                        open_deltas = False
                    else:
                        yield _ndjson({"type": "text_delta", "delta": d})
                    sent = True
                except queue.Empty:
                    pass

                if speak:
                    try:
                        item = audio_q.get_nowait()
                        if item is None:
                            open_audio = False
                        else:
                            wav, sr = item
                            yield _ndjson(
                                {"type": "audio_wav", "wav_base64": base64.b64encode(wav).decode(), "sr": sr}
                            )
                        sent = True
                    except queue.Empty:
                        pass
                else:
                    open_audio = False

                if not sent:
                    if time.time() - last_heartbeat > 10:
                        yield _ndjson({"type": "ping"})
                        last_heartbeat = time.time()
                    time.sleep(0.01)

            t1.join()
            t2.join()
            yield _ndjson({"type": "done", "text": "".join(full_text_parts).strip()})

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/api/bots/{bot_id}/talk/stream")
    def talk_stream(
        bot_id: UUID,
        audio: UploadFile = File(...),
        conversation_id: str = Form(""),
        test_flag: bool = Form(True),
        speak: bool = Form(True),
        session: Session = Depends(get_session),
    ) -> StreamingResponse:
        bot = get_bot(session, bot_id)
        openai_api_key = _get_openai_api_key_for_bot(session, bot=bot)
        if not openai_api_key:
            raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot.")
        provider, llm_api_key, llm = _require_llm_client(session, bot=bot)

        conv_id: Optional[UUID] = UUID(conversation_id) if conversation_id.strip() else None
        if conv_id is None:
            conv = create_conversation(session, bot_id=bot.id, test_flag=bool(test_flag))
            conv_id = conv.id

        wav_bytes = audio.file.read()
        pcm16 = _decode_wav_bytes_to_pcm16_16k(wav_bytes)
        if not pcm16:
            raise HTTPException(status_code=400, detail="Empty audio")

        asr = _get_asr(openai_api_key, bot.openai_asr_model, bot.language).transcribe_pcm16(
            pcm16=pcm16, sample_rate=16000
        )
        user_text = asr.text.strip()
        if not user_text:
            # Return the conversation id so UI can keep the session even if no speech recognized.
            def empty_gen() -> Generator[bytes, None, None]:
                yield _ndjson({"type": "conversation", "id": str(conv_id)})
                yield _ndjson({"type": "asr", "text": ""})
                yield _ndjson({"type": "done", "text": ""})

            return StreamingResponse(empty_gen(), media_type="application/x-ndjson")

        add_message(session, conversation_id=conv_id, role="user", content=user_text)

        def gen() -> Generator[bytes, None, None]:
            yield _ndjson({"type": "conversation", "id": str(conv_id)})
            yield _ndjson({"type": "asr", "text": user_text})

            history = _build_history_budgeted(
                session=session,
                bot=bot,
                conversation_id=conv_id,
                llm_api_key=llm_api_key,
                status_cb=None,
            )

            delta_q_client: "queue.Queue[Optional[str]]" = queue.Queue()
            delta_q_tts: "queue.Queue[Optional[str]]" = queue.Queue()
            audio_q: "queue.Queue[Optional[tuple[bytes, int]]]" = queue.Queue()
            full_text_parts: list[str] = []
            error_q: "queue.Queue[Optional[str]]" = queue.Queue()

            def llm_thread() -> None:
                try:
                    for delta in llm.stream_text(messages=history):
                        full_text_parts.append(delta)
                        delta_q_client.put(delta)
                        if speak:
                            delta_q_tts.put(delta)
                except Exception as exc:
                    error_q.put(str(exc))
                finally:
                    delta_q_client.put(None)
                    if speak:
                        delta_q_tts.put(None)

            def tts_thread() -> None:
                if not speak:
                    audio_q.put(None)
                    return
                try:
                    tts_synth = _get_tts_synth_fn(bot, openai_api_key)
                    for text_to_speak in _iter_tts_chunks(delta_q_tts):
                        if not text_to_speak:
                            continue
                        wav, sr = tts_synth(text_to_speak)
                        audio_q.put((wav, sr))
                finally:
                    audio_q.put(None)

            t1 = threading.Thread(target=llm_thread, daemon=True)
            t2 = threading.Thread(target=tts_thread, daemon=True)
            t1.start()
            t2.start()

            open_deltas = True
            open_audio = True
            last_heartbeat = time.time()

            while open_deltas or open_audio:
                sent = False
                try:
                    err = error_q.get_nowait()
                    if err:
                        yield _ndjson({"type": "error", "error": err})
                        open_deltas = False
                        open_audio = False
                        break
                except queue.Empty:
                    pass
                try:
                    d = delta_q_client.get_nowait()
                    if d is None:
                        open_deltas = False
                    else:
                        yield _ndjson({"type": "text_delta", "delta": d})
                    sent = True
                except queue.Empty:
                    pass

                if speak:
                    try:
                        item = audio_q.get_nowait()
                        if item is None:
                            open_audio = False
                        else:
                            wav, sr = item
                            yield _ndjson(
                                {"type": "audio_wav", "wav_base64": base64.b64encode(wav).decode(), "sr": sr}
                            )
                        sent = True
                    except queue.Empty:
                        pass
                else:
                    open_audio = False

                if not sent:
                    if time.time() - last_heartbeat > 10:
                        yield _ndjson({"type": "ping"})
                        last_heartbeat = time.time()
                    time.sleep(0.01)

            t1.join()
            t2.join()
            final_text = "".join(full_text_parts).strip()
            if final_text:
                add_message(session, conversation_id=conv_id, role="assistant", content=final_text)
            yield _ndjson({"type": "done", "text": final_text})

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/api/bots/{bot_id}/chat")
    def chat_once(
        bot_id: UUID,
        payload: ChatRequest = Body(...),
        session: Session = Depends(get_session),
    ) -> dict:
        bot = get_bot(session, bot_id)
        provider, llm_api_key, llm = _require_llm_client(session, bot=bot)
        text = (payload.text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text")

        messages = [Message(role="system", content=bot.system_prompt), Message(role="user", content=text)]
        out_text = llm.complete_text(messages=messages)

        if not payload.speak:
            return {"text": out_text}

        openai_api_key = _get_openai_api_key_for_bot(session, bot=bot)
        if not openai_api_key:
            raise HTTPException(status_code=400, detail="No OpenAI key configured for this bot (needed for TTS).")
        tts_synth = _get_tts_synth_fn(bot, openai_api_key)
        wav, sr = tts_synth(out_text)
        return {"text": out_text, "audio_wav_base64": base64.b64encode(wav).decode(), "sr": sr}

    if ui_index.exists():
        app.mount("/", SpaStaticFiles(directory=str(ui_dir), html=True, index_file=ui_index), name="studio-ui")

    logger.info("create_app: complete (%.2fs)", time.monotonic() - start)
    return app
