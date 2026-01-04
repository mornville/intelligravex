from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from typing import Any, Optional


def _conv_root_dir(conversation_id: str) -> str:
    cid = (conversation_id or "").strip() or "no_conversation"
    return os.path.join(tempfile.gettempdir(), "igx_codex_one_shot", cid)


def index_jsonl_path(*, conversation_id: str) -> str:
    return os.path.join(_conv_root_dir(conversation_id), "index.jsonl")


def append_saved_run_index(*, conversation_id: str, event: dict[str, Any]) -> str:
    """
    Append a single JSONL event for a Codex-saved run.

    TODO(cleanup): add a retention/cleanup policy for files under igx_codex_one_shot/.
    """
    root = _conv_root_dir(conversation_id)
    os.makedirs(root, exist_ok=True)
    p = index_jsonl_path(conversation_id=conversation_id)
    obj = dict(event or {})
    obj.setdefault("ts_utc", dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"))
    line = json.dumps(obj, ensure_ascii=False)
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return p


def list_saved_runs(*, conversation_id: str, limit: int = 5000) -> list[dict[str, Any]]:
    p = index_jsonl_path(conversation_id=conversation_id)
    if not os.path.exists(p):
        return []
    out: list[dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= int(limit):
                    break
                s = (line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    out.append(obj)
    except Exception:
        return out
    return out


def find_saved_run(
    *,
    conversation_id: str,
    source_tool_name: str,
    source_req_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    tool = (source_tool_name or "").strip()
    if not tool:
        return None
    rid = (source_req_id or "").strip()
    events = list_saved_runs(conversation_id=conversation_id)
    for ev in reversed(events):
        if str(ev.get("tool_name") or "").strip() != tool:
            continue
        if rid and str(ev.get("req_id") or "").strip() != rid:
            continue
        return ev
    return None

