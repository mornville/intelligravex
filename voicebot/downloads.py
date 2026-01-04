from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Optional
from uuid import uuid4


def _tokens_dir() -> str:
    d = os.path.join(tempfile.gettempdir(), "igx_download_tokens")
    os.makedirs(d, exist_ok=True)
    return d


def create_download_token(
    *,
    file_path: str,
    filename: str,
    mime_type: str,
    conversation_id: str | None = None,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """
    Creates an unguessable download token that maps to a local file path.

    TODO(cleanup): add TTL/cleanup for tokens and exported files.
    """
    token = str(uuid4())
    obj: dict[str, Any] = {
        "token": token,
        "file_path": str(file_path),
        "filename": str(filename),
        "mime_type": str(mime_type or "application/octet-stream"),
        "conversation_id": (str(conversation_id) if conversation_id else None),
        "created_ts": float(time.time()),
        "meta": metadata or {},
    }
    p = os.path.join(_tokens_dir(), f"{token}.json")
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2))
    return token


def load_download_token(*, token: str) -> Optional[dict[str, Any]]:
    t = (token or "").strip()
    if not t:
        return None
    p = os.path.join(_tokens_dir(), f"{t}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.loads(f.read() or "null")
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def is_allowed_download_path(path: str) -> bool:
    """
    Prevents arbitrary file reads by restricting downloads to known temp roots.
    """
    p = os.path.abspath(path or "")
    tmp = os.path.abspath(tempfile.gettempdir())
    allowed = [
        os.path.join(tmp, "igx_exports"),
        os.path.join(tmp, "igx_codex_one_shot"),
    ]
    return any(p.startswith(os.path.abspath(a) + os.sep) or p == os.path.abspath(a) for a in allowed)

