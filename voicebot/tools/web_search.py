from __future__ import annotations

from typing import Any


def web_search_tool_def() -> dict[str, Any]:
    """
    OpenAI Responses API built-in web search tool definition.
    """
    return {"type": "web_search"}
