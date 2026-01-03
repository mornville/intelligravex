from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from typing import Any, Optional
from uuid import UUID, uuid4


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def get_conversation_artifacts_dir(conversation_id: UUID) -> str:
    """
    Per-conversation temp directory used for Codex-only artifacts (raw API responses + manifests).

    IMPORTANT: Do not store this path in conversation metadata, since that metadata is exposed to the main LLM history.
    """
    base = os.path.join(tempfile.gettempdir(), "igx_conversations", str(conversation_id))
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(base, "responses"), exist_ok=True)
    os.makedirs(os.path.join(base, "manifests"), exist_ok=True)
    return base


def get_conversation_artifacts_index_path(conversation_id: UUID) -> str:
    return os.path.join(get_conversation_artifacts_dir(conversation_id), "index.json")


def _read_index(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            return obj
    except Exception:
        pass
    return {"version": 1, "items": []}


def _atomic_write_json(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def record_http_response_artifact(
    *,
    conversation_id: UUID,
    tool_name: str,
    method: Optional[str],
    url: Optional[str],
    what_to_search_for: Optional[str],
    why_to_search_for: Optional[str],
    response_json: Any,
    max_index_items: int = 200,
) -> dict[str, str]:
    """
    Writes:
    - responses/<artifact_id>.json (raw response)
    - manifests/<artifact_id>.json (metadata about why it was called)
    - index.json (append-only list of manifest pointers)
    """
    base = get_conversation_artifacts_dir(conversation_id)
    index_path = os.path.join(base, "index.json")

    artifact_id = str(uuid4())
    response_rel = os.path.join("responses", f"{artifact_id}.json")
    manifest_rel = os.path.join("manifests", f"{artifact_id}.json")
    response_path = os.path.join(base, response_rel)
    manifest_path = os.path.join(base, manifest_rel)

    with open(response_path, "w", encoding="utf-8") as f:
        json.dump(response_json, f, ensure_ascii=False, indent=2)

    manifest = {
        "artifact_id": artifact_id,
        "tool_name": str(tool_name or ""),
        "method": (method or "").upper(),
        "url": str(url or ""),
        "called_at": _utc_now_iso(),
        "what_to_search_for": str(what_to_search_for or ""),
        "why_to_search_for": str(why_to_search_for or ""),
        "response_path": response_rel,
    }
    _atomic_write_json(manifest_path, manifest)

    index = _read_index(index_path)
    items = index.get("items")
    if not isinstance(items, list):
        items = []
    items.append(
        {
            "artifact_id": artifact_id,
            "manifest_path": manifest_rel,
            "tool_name": manifest["tool_name"],
            "called_at": manifest["called_at"],
            "what_to_search_for": manifest["what_to_search_for"],
        }
    )
    if max_index_items and len(items) > int(max_index_items):
        items = items[-int(max_index_items) :]
    index["version"] = 1
    index["items"] = items
    _atomic_write_json(index_path, index)

    return {
        "artifacts_dir": base,
        "index_path": index_path,
        "artifact_id": artifact_id,
        "response_path": response_path,
        "manifest_path": manifest_path,
    }

