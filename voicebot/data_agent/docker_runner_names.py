from __future__ import annotations

import re
import uuid
from uuid import UUID


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
